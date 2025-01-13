"""
OpenAI服务
调用OpenAI API生成对话响应
"""

import os
from typing import Dict, List
from dotenv import load_dotenv
import openai
import json

# 加载环境变量
load_dotenv()

class OpenAIService:
    """OpenAI服务类"""
    
    def __init__(self):
        """初始化服务"""
        # 设置API密钥
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("未设置OPENAI_API_KEY环境变量")
            
        # 加载配置
        self.config = self._load_config()
        
        # 初始化缓存层
        self.cache = {
            "memory": [],  # 对话记忆
            "emotion": {   # 情绪状态
                "type": "平静",
                "speech_speed": 5,
                "volume": 5,
                "pitch": 5
            },
            "scene": {    # 场景状态(预设值)
                "obstacles": "无",
                "temperature": "25°C",
                "lighting": "明亮",
                "safety": "安全"
            }
        }
            
    def _load_config(self):
        """加载配置"""
        try:
            with open('config/config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {
                'system_prompt': '''你是一个情感丰富的对话机器人。在对话中,你需要:
1. 只使用指定的词汇表达
2. 表现出适当的情绪
3. 保持对话的自然性和连贯性
4. 回答要简短精炼''',
                'user_prompt_template': '''用户输入: {user_input}

当前情绪: {emotion_type}

对话类型: {context_type}

可用词汇:
1. 情绪词汇: {emotions}
2. 动作词汇: {actions}
3. 响应词汇:
{responses}

要求:
1. 必须使用以下JSON格式输出:
{
    "话语": "从响应词汇中选择",
    "表情": "从当前情绪词汇中选择",
    "程度": "0-10之间的数字",
    "动作": "从动作词汇中选择"
}

2. 格式要求:
- 话语必须从响应词汇列表中选择
- 表情必须从当前情绪的词汇中选择
- 程度必须是0到10之间的数字
- 动作必须从动作词汇列表中选择

3. 其他要求:
- 回答要简短,不超过20个字
- 保持对话的自然性
- 严格按照JSON格式输出
- 不要添加任何额外说明

请生成回答:''',
                'vocabulary': {
                    'emotions': {},
                    'actions': [],
                    'responses': {}
                }
            }

    def _validate_response(self, response_text: str, vocabulary_constraints: List[str], emotion: Dict) -> Dict:
        """
        验证响应是否符合规则
        
        Args:
            response_text: 响应文本
            vocabulary_constraints: 词汇约束
            emotion: 情绪信息
            
        Returns:
            Dict: 验证通过的响应字典
            
        Raises:
            ValueError: 如果响应不符合规则
        """
        try:
            # 解析JSON回复
            response_dict = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"响应不是有效的JSON格式: {str(e)}\n原始响应: {response_text}")
        
        # 验证所有字段是否存在
        required_fields = ["话语", "表情", "程度", "动作"]
        missing_fields = [field for field in required_fields if field not in response_dict]
        if missing_fields:
            raise ValueError(f"缺少必要的字段: {', '.join(missing_fields)}")
        
        # 获取当前情感类型
        emotion_type = emotion.get('emotion_type', '平静')
        if emotion_type not in self.config['vocabulary']['emotions']:
            raise ValueError(f"未知的情感类型: {emotion_type}")
        
        # 验证话语是否在响应词汇中
        response_word = response_dict["话语"]
        valid_responses = []
        for response_list in self.config['vocabulary']['responses'].values():
            valid_responses.extend(response_list)
        
        if response_word not in valid_responses:
            raise ValueError(f"无效的响应词汇 '{response_word}'。有效词汇: {', '.join(valid_responses)}")
        
        # 验证表情是否在当前情感的词汇中
        emotion_word = response_dict["表情"]
        valid_emotions = self.config['vocabulary']['emotions'][emotion_type]
        if emotion_word not in valid_emotions:
            raise ValueError(f"无效的情绪词汇 '{emotion_word}'。当前情感({emotion_type})的有效词汇: {', '.join(valid_emotions)}")
        
        # 验证程度是否在范围内
        try:
            degree = float(response_dict["程度"])
            if not 0 <= degree <= 10:
                raise ValueError(f"情绪程度 {degree} 超出范围(0-10)")
        except (TypeError, ValueError):
            raise ValueError(f"情绪程度 '{response_dict['程度']}' 必须是0-10之间的数字")
        
        # 验证动作是否在动作词汇中
        action = response_dict["动作"]
        if action not in self.config['vocabulary']['actions']:
            raise ValueError(f"无效的动作词汇 '{action}'。有效动作: {', '.join(self.config['vocabulary']['actions'])}")
        
        return response_dict

    async def _analyze_context(self, messages):
        """分析对话上下文，生成记忆"""
        try:
            prompt = f"""请分析以下对话记录，提炼出核心内容作为记忆：

对话记录：
{messages}

请生成简短的记忆描述，不超过100字。直接返回描述文本，不要添加任何额外说明。"""

            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个对话分析助手，负责提炼对话的核心内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            memory = response.choices[0].message.content.strip()
            self.cache["memory"].append(memory)
            
            # 只保留最近的5条记忆
            if len(self.cache["memory"]) > 5:
                self.cache["memory"] = self.cache["memory"][-5:]
                
            return memory
            
        except Exception as e:
            print(f"分析上下文失败: {str(e)}")
            return None
            
    async def _evaluate_emotion(self, memory):
        """评估情绪状态"""
        try:
            # 构建情绪评估的提示词
            emotions_str = json.dumps(self.config["vocabulary"]["emotions"], ensure_ascii=False)
            prompt = f"""请根据以下记忆内容，评估当前的情绪状态：

记忆内容：{memory}

可选的情绪类型及对应词汇：
{emotions_str}

请按以下JSON格式返回评估结果：
{{
    "type": "情绪类型",
    "speech_speed": "语速(0-10的整数)",
    "volume": "音量(0-10的整数)",
    "pitch": "音高(0-10的整数)"
}}

注意：
1. 情绪类型必须从上述词汇表中选择
2. 语速、音量和音高必须是0-10之间的整数
3. 严格按照JSON格式返回，不要添加任何说明"""

            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个情绪分析助手，负责评估对话的情绪状态。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            self.cache["emotion"] = result
            return result
            
        except Exception as e:
            print(f"评估情绪失败: {str(e)}")
            return None

    async def get_response(self,
                          user_input: str,
                          context: Dict,
                          emotion: Dict,
                          vocabulary_constraints: List[str]) -> Dict:
        """生成对话响应"""
        try:
            # 首先分析上下文并更新记忆
            memory = await self._analyze_context(context.get("messages", []))
            if memory:
                # 评估情绪状态
                new_emotion = await self._evaluate_emotion(memory)
                if new_emotion:
                    self.cache["emotion"] = new_emotion
            
            max_retries = 3  # 最大重试次数
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    # 构建提示词
                    prompt = self._build_prompt(
                        user_input,
                        context,
                        self.cache["emotion"],
                        vocabulary_constraints
                    )
                    
                    # 构建完整的消息历史
                    messages = [
                        {"role": "system", "content": self.config['system_prompt']},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # 如果不是第一次尝试,添加修正提示
                    if retry_count > 0:
                        messages.append({
                            "role": "user",
                            "content": f"你的上一次回复不符合输出规则({last_error})。请严格按照规则重新生成回复。"
                        })
                    
                    # 调用API
                    response = await openai.ChatCompletion.acreate(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=50
                    )
                    
                    # 提取响应文本
                    response_text = response.choices[0].message.content.strip()
                    
                    # 验证响应是否符合规则
                    try:
                        validated_response = self._validate_response(
                            response_text,
                            vocabulary_constraints,
                            self.cache["emotion"]
                        )
                        
                        # 如果情绪发生变化，重新评估情绪状态
                        if validated_response["表情"] != self.cache["emotion"]["type"]:
                            await self._evaluate_emotion(memory + "\n" + user_input)
                        
                        return {
                            'text': validated_response["话语"],
                            'emotion': validated_response["表情"],
                            'degree': float(validated_response["程度"]),
                            'action': validated_response["动作"],
                            'raw_response': response
                        }
                        
                    except Exception as validation_error:
                        last_error = str(validation_error)
                        print(f"第{retry_count + 1}次生成的回复不符合规则: {last_error}")
                        print(f"原始回复: {response_text}")
                        retry_count += 1
                        continue
                        
                except Exception as e:
                    last_error = str(e)
                    print(f"第{retry_count + 1}次调用API失败: {last_error}")
                    retry_count += 1
                    continue
            
            # 如果达到最大重试次数仍未成功,返回错误信息
            error_msg = f"Error: no date (最后一次错误: {last_error})"
            print(f"达到最大重试次数,返回错误信息: {error_msg}")
            return {
                'text': error_msg,
                'error': error_msg,
                'raw_responses': []  # 添加原始响应列表
            }
            
        except Exception as e:
            error_msg = f"Error: no date (错误: {str(e)})"
            print(f"处理请求失败: {error_msg}")
            return {
                'text': error_msg,
                'error': str(e)
            }
            
    def _build_prompt(self,
                     user_input: str,
                     context: Dict,
                     emotion: Dict,
                     vocabulary_constraints: List[str]) -> str:
        """构建提示词"""
        # 获取当前情感类型的词汇
        emotion_type = emotion.get('type', '平静')
        emotion_words = self.config['vocabulary']['emotions'].get(emotion_type, [])
        
        # 构建状态描述
        status = f"""记忆：{' '.join(self.cache['memory'])}
情绪状态：类型={self.cache['emotion']['type']}, 语速={self.cache['emotion']['speech_speed']}, 音量={self.cache['emotion']['volume']}, 音高={self.cache['emotion']['pitch']}
场景状态：障碍物={self.cache['scene']['obstacles']}, 温度={self.cache['scene']['temperature']}, 光照={self.cache['scene']['lighting']}, 安全={self.cache['scene']['safety']}"""
        
        # 构建响应词汇字符串
        responses_str = json.dumps(self.config['vocabulary']['responses'], ensure_ascii=False, indent=2)
        
        # 构建提示词
        prompt = f"""请根据用户输入生成一个完整的回复。

用户输入: {user_input}

当前状态: {status}
可用的情绪词汇: {', '.join(emotion_words)}
可用的动作词汇: {', '.join(self.config['vocabulary']['actions'])}
可用的响应词汇类型及对应词汇: {responses_str}

请生成一个JSON格式的回复,格式如下:
{{
    "话语": "从响应词汇列表中选择的词汇",
    "表情": "从当前情感的情绪词汇中选择",
    "程度": "0-10之间的数字",
    "动作": "从动作词汇列表中选择"
}}

情绪程度(degree)的计算逻辑:
1. 分析用户输入与当前情感的相关程度(relevance): 0-1之间
2. 分析情绪词汇的强烈程度(intensity): 0-10之间
3. 最终程度 = relevance * intensity
4. 结果四舍五入到小数点后一位,确保在0-10范围内

注意事项:
1. 话语必须从响应词汇列表中选择
2. 表情必须从当前情感对应的情绪词汇中选择
3. 程度必须是0到10之间的数字
4. 动作必须从动作词汇列表中选择
5. 回复必须是合法的JSON格式

请直接返回JSON格式的回复,不要添加任何额外说明。"""
        
        return prompt