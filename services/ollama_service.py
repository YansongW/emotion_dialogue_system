# Ollama模型服务
#
# 提供与Ollama API的交互功能，支持：
# 1. 模型响应生成
# 2. 上下文管理
# 3. 性能指标收集

import json
import aiohttp
import asyncio
from typing import Dict, List, Optional, Union
from .model_service import BaseModelService
import logging

class OllamaServiceError(Exception):
    """Ollama服务异常类"""
    pass

class OllamaService(BaseModelService):
    """Ollama服务类"""
    
    def __init__(self, model_name: str = "llama3.2"):
        """初始化服务"""
        self.base_url = "http://localhost:11434/api"
        self.model_name = model_name
        self.system_prompt = """你是一个对话机器人。请严格按照以下要求回复:

1. 你的回复必须是一个合法的JSON字符串，包含以下字段:
   - text: 回复的文本内容
   - emotion_type: 情感类型，必须是以下之一：平静、高兴、难过、生气、惊讶
   - action: 动作，必须是以下之一：点头、摇头、微笑、皱眉、挥手
   - speech_speed: 语速，1-10的整数
   - volume: 音量，1-10的整数
   - pitch: 音调，1-10的整数

2. 回复要求:
   - 回答要简短精炼，不超过20个字
   - 表现出适当的情绪
   - 保持对话的自然性和连贯性
   - 使用合适的动作表达

3. 示例回复:
{
    "text": "你好，很高兴见到你",
    "emotion_type": "高兴",
    "action": "微笑",
    "speech_speed": 7,
    "volume": 6,
    "pitch": 7
}

请确保每次回复都是一个完整的、合法的JSON字符串。"""
        # 保存上下文信息
        self.context = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """异步初始化服务"""
        if not self._initialized:
            await self._check_service()
            self._initialized = True
            
    async def _check_service(self) -> None:
        """检查Ollama服务是否可用"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/version") as response:
                    if response.status != 200:
                        raise OllamaServiceError(f"Ollama服务不可用: HTTP {response.status}")
                    await self._check_model()
        except aiohttp.ClientError as e:
            raise OllamaServiceError(f"无法连接到Ollama服务: {str(e)}\n请确保Ollama服务已启动")
            
    async def _check_model(self) -> None:
        """检查模型是否已安装"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/tags") as response:
                    if response.status != 200:
                        raise OllamaServiceError("无法获取模型列表")
                    data = await response.json()
                    models = data.get('models', [])
                    if not any(model.get('name') == self.model_name for model in models):
                        raise OllamaServiceError(
                            f"模型 {self.model_name} 未安装\n"
                            f"可用模型: {', '.join(m.get('name', '') for m in models)}"
                        )
        except aiohttp.ClientError as e:
            raise OllamaServiceError(f"检查模型失败: {str(e)}")
            
    async def _retry_request(self, request_data: Dict, max_retries: int = 3) -> Dict:
        """带重试的请求发送"""
        if not self._initialized:
            await self.initialize()
            
        last_error = None
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/generate",
                        json=request_data,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status != 200:
                            raise OllamaServiceError(f"API请求失败: HTTP {response.status}")
                        return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # 指数退避
                continue
        raise OllamaServiceError(f"请求重试{max_retries}次后仍然失败: {str(last_error)}")
        
    async def _analyze_context(self, memory: List[str]) -> Optional[str]:
        """
        分析对话上下文
        
        Args:
            memory: 对话历史记录列表
            
        Returns:
            Optional[str]: 分析后的上下文信息，如果分析失败则返回None
        """
        try:
            if not memory:
                return None
                
            # 构建分析提示词
            prompt = f"""
            分析以下对话历史，总结对话的主要内容和情感倾向：
            
            {chr(10).join(memory)}
            
            请提供简要分析：
            """
            
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "system": "你是一个对话分析助手，需要简要总结对话内容和情感倾向。",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 100
                }
            }
            
            result = await self._retry_request(request_data)
            return result['response'].strip()
            
        except Exception as e:
            print(f"分析上下文失败: {str(e)}")
            return None
            
    async def _evaluate_emotion(self, context: str) -> Optional[Dict[str, Union[str, int]]]:
        """
        评估情感状态
        
        Args:
            context: 对话上下文
            
        Returns:
            Optional[Dict[str, Union[str, int]]]: 情感状态信息，包含类型和强度
        """
        try:
            if not context:
                return None
                
            # 构建情感分析提示词
            prompt = f"""
            作为一个情感分析助手，请分析以下对话内容的情感状态。

            对话内容：
            {context}

            请严格按照以下JSON格式返回分析结果。可用的情感类型仅限于：平静、高兴、难过、生气、惊讶。
            所有数值参数必须在1-10的范围内。

            输出格式示例：
            {{
                "emotion_type": "平静",
                "speech_speed": 5,
                "volume": 5,
                "pitch": 5
            }}

            请确保：
            1. 情感类型必须是以上五种之一
            2. 所有数值必须是1-10之间的整数
            3. 返回格式必须是合法的JSON

            你的分析结果：
            """
            
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "system": "你是一个专业的情感分析助手。你的任务是分析对话内容的情感状态，并返回规范的JSON格式结果。",
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 100
                }
            }
            
            # 最多重试3次
            for attempt in range(3):
                try:
                    result = await self._retry_request(request_data)
                    response = result['response'].strip()
                    
                    # 处理可能的Markdown格式
                    if response.startswith('```'):
                        # 提取JSON部分
                        json_start = response.find('{')
                        json_end = response.rfind('}') + 1
                        if json_start != -1 and json_end != -1:
                            response = response[json_start:json_end]
                    
                    # 尝试解析JSON响应
                    emotion_data = json.loads(response)
                    
                    # 验证情感类型
                    if emotion_data["emotion_type"] not in ["平静", "高兴", "难过", "生气", "惊讶"]:
                        raise ValueError(f"无效的情感类型: {emotion_data['emotion_type']}")
                    
                    # 验证并规范化数值参数
                    for key in ["speech_speed", "volume", "pitch"]:
                        value = int(emotion_data[key])
                        if not 1 <= value <= 10:
                            emotion_data[key] = max(1, min(10, value))
                    
                    return {
                        "emotion_type": emotion_data["emotion_type"],
                        "speech_speed": emotion_data["speech_speed"],
                        "volume": emotion_data["volume"],
                        "pitch": emotion_data["pitch"]
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败 (尝试 {attempt + 1}/3): {str(e)}")
                    print(f"原始响应: {response}")
                    if attempt == 2:  # 最后一次尝试
                        return {
                            "emotion_type": "平静",
                            "speech_speed": 5,
                            "volume": 5,
                            "pitch": 5
                        }
                except (KeyError, ValueError) as e:
                    print(f"数据验证失败 (尝试 {attempt + 1}/3): {str(e)}")
                    print(f"解析数据: {emotion_data}")
                    if attempt == 2:  # 最后一次尝试
                        return {
                            "emotion_type": "平静",
                            "speech_speed": 5,
                            "volume": 5,
                            "pitch": 5
                        }
                
                # 等待短暂时间后重试
                await asyncio.sleep(0.5)
                
        except Exception as e:
            print(f"评估情感状态失败: {str(e)}")
            # 返回默认值
            return {
                "emotion_type": "平静",
                "speech_speed": 5,
                "volume": 5,
                "pitch": 5
            }
        
    def _build_prompt(self,
                     user_input: str,
                     context: Dict,
                     emotion: Dict,
                     vocabulary_constraints: List[str]) -> str:
        """构建提示词"""
        # 使用父类的_build_prompt方法
        return super()._build_prompt(
            user_input,
            context,
            emotion,
            vocabulary_constraints
        )
        
    async def get_response(self,
                          user_input: str,
                          context: Dict,
                          emotion: Dict,
                          vocabulary_constraints: List[str]) -> Dict:
        """获取Ollama响应"""
        try:
            if not self._initialized:
                await self.initialize()
                
            # 记录输入参数
            logging.info(f"请求参数: user_input={user_input}, context={json.dumps(context, ensure_ascii=False)}, emotion={json.dumps(emotion, ensure_ascii=False)}")
                
            prompt = self._build_prompt(
                user_input,
                context,
                emotion,
                vocabulary_constraints
            )
            
            # 记录构建的提示词
            logging.info(f"构建的提示词: {prompt}")
            
            # 构建请求参数
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "system": self.system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 50,  # 相当于max_tokens
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            # 如果有上下文,添加到请求中
            if self.context is not None:
                request_data["context"] = self.context
            
            # 记录请求数据
            logging.info(f"发送请求: {json.dumps(request_data, ensure_ascii=False)}")
            
            # 发送请求（带重试）
            result = await self._retry_request(request_data)
            
            # 记录原始响应
            logging.info(f"收到响应: {json.dumps(result, ensure_ascii=False)}")
            
            # 保存新的上下文
            if 'context' in result:
                self.context = result['context']
            
            # 提取性能指标
            metrics = {
                'total_duration': result.get('total_duration', 0),
                'eval_count': result.get('eval_count', 0),
                'eval_duration': result.get('eval_duration', 0)
            }
            
            # 计算生成速度(tokens/s)
            if metrics['eval_duration'] > 0:
                metrics['tokens_per_second'] = (
                    metrics['eval_count'] / 
                    (metrics['eval_duration'] / 1e9)  # 转换为秒
                )
            
            # 添加情感相关信息
            try:
                response_data = {
                    'text': result['response'].strip(),
                    'emotion_type': emotion.get('emotion_type', '未检测'),
                    'action': '点头',
                    'speech_speed': emotion.get('speech_speed'),
                    'volume': emotion.get('volume'),
                    'pitch': emotion.get('pitch'),
                    'metrics': metrics
                }
                
                # 记录处理后的响应数据
                logging.info(f"处理后的响应数据: {json.dumps(response_data, ensure_ascii=False)}")
                
                return response_data
                
            except (KeyError, TypeError) as e:
                error_msg = f"构建响应数据失败: {str(e)}"
                logging.error(error_msg)
                return {
                    'text': result['response'].strip(),
                    'emotion_type': '未检测',
                    'action': '摇头',
                    'speech_speed': None,
                    'volume': None,
                    'pitch': None,
                    'metrics': metrics
                }
            
        except OllamaServiceError as e:
            error_msg = f"Ollama服务错误: {str(e)}"
            logging.error(error_msg)
            return {
                'text': "抱歉，服务暂时不可用",
                'emotion_type': '未检测',
                'action': '摇头',
                'speech_speed': None,
                'volume': None,
                'pitch': None,
                'error': str(e)
            }
        except Exception as e:
            error_msg = f"Ollama API调用失败: {str(e)}"
            logging.error(error_msg)
            return {
                'text': "对不起，我现在有点困惑",
                'emotion_type': '未检测',
                'action': '摇头',
                'speech_speed': None,
                'volume': None,
                'pitch': None,
                'error': str(e)
            } 