"""
对话系统核心模块 (Dialogue System Core Module)

本模块实现了对话系统的核心功能，包括：
1. 对话管理：处理用户输入，生成响应
2. 情感管理：分析和更新系统情感状态
3. 安全检查：确保对话内容和场景安全
4. 词汇管理：严格控制系统使用的词汇范围

主要组件：
- DialogueSystem类：核心对话系统类，管理整个对话过程
- 配置管理：使用config_manager加载和管理系统配置
- 情感系统：实现情感状态的分析、转换和表现

依赖库：
- json: 用于配置文件的读写
- random: 用于随机选择词汇和响应
- typing: 提供类型注解支持
- config.settings: 提供配置管理功能

"""

import json
import random
import aiohttp
import openai
from typing import Dict, Any, List, Optional
from config.settings import config_manager

class DialogueSystem:
    """
    对话系统核心类
    
    负责处理用户输入，生成响应，管理系统状态。
    实现了情感分析、安全检查、响应生成等核心功能。
    
    属性:
        config: 系统配置对象
        current_emotion: 当前情感状态
        emotion_history: 情感状态历史记录
    """
    
    def __init__(self):
        """初始化对话系统"""
        self.config = config_manager.config
        
        # 初始化缓存层
        self.cache = {
            "memory": [],  # 对话记忆
            "emotion": {   # 情绪状态
                "emotion_type": "平静",
                "speech_speed": 5,
                "volume": 5,
                "pitch": 5
            },
            "scene": {    # 场景状态(预设值)
                "obstacles": "无",
                "temperature": "25°C",
                "lighting": "明亮",
                "safety": "安全"
            },
            "system": {
                "status": "异常",
                "error": ""
            }
        }
        
        # 初始化模型服务（具体初始化在initialize方法中进行）
        self.model_service = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """异步初始化对话系统"""
        try:
            if self.config.model.model_type == "openai":
                from services.openai_service import OpenAIService
                self.model_service = OpenAIService()
            elif self.config.model.model_type == "ollama":
                from services.ollama_service import OllamaService
                self.model_service = OllamaService(model_name=self.config.model.model_name)
                await self.model_service.initialize()
            else:
                raise ValueError(f"不支持的模型类型: {self.config.model.model_type}")
                
            # 更新系统状态
            self.cache["system"]["status"] = "正常"
            self.cache["system"]["model_type"] = self.config.model.model_type
            self.cache["system"]["model_name"] = self.config.model.model_name
            self._initialized = True
            
        except Exception as e:
            print(f"初始化模型服务失败: {str(e)}")
            self.cache["system"]["status"] = "异常"
            self.cache["system"]["error"] = str(e)
            raise RuntimeError(f"初始化对话系统失败: {str(e)}")
            
    async def ensure_initialized(self) -> None:
        """确保系统已初始化"""
        if not self._initialized:
            await self.initialize()
        
    async def process_input(self, user_input: str, scene_info: Dict[str, Any]) -> Dict:
        """
        处理用户输入，生成响应
        
        完整的对话处理流程：
        1. 进行安全检查
        2. 分析输入的情感倾向
        3. 更新系统情感状态
        4. 生成响应文本
        5. 添加情感表现
        
        Args:
            user_input: 用户输入的文本
            scene_info: 场景信息，包含环境和安全相关数据
            
        Returns:
            Dict: 包含响应文本、情感状态等信息的字典
        """
        try:
            # 确保系统已初始化
            await self.ensure_initialized()
            
            # 安全检查
            if not await self._check_safety(user_input, scene_info):
                return {
                    "status": "rejected",
                    "response": json.dumps({
                        "text": "抱歉，我不能执行这个请求",
                        "emotion_type": "平静",
                        "action": "摇头",
                        "speech_speed": 1,
                        "volume": 1,
                        "pitch": 1
                    }, ensure_ascii=False)
                }
            
            # 分析上下文并更新记忆
            memory = await self.model_service._analyze_context(self.cache["memory"])
            if memory:
                # 评估情绪状态
                new_emotion = await self.model_service._evaluate_emotion(memory)
                if new_emotion:
                    self.cache["emotion"] = new_emotion
            
            # 生成响应
            response = await self.model_service.get_response(
                user_input=user_input,
                context={
                    "memory": self.cache["memory"],
                    "emotion": self.cache.get("emotion", {
                        "emotion_type": "平静",
                        "speech_speed": 5,
                        "volume": 5,
                        "pitch": 5
                    }),
                    "scene": self.cache["scene"]
                },
                emotion=self.cache.get("emotion", {
                    "emotion_type": "平静",
                    "speech_speed": 5,
                    "volume": 5,
                    "pitch": 5
                }),
                vocabulary_constraints=[]
            )
            
            # 如果响应成功，更新缓存
            if "error" not in response:
                # 更新记忆
                self.cache["memory"].append(f"用户: {user_input}")
                self.cache["memory"].append(f"机器人: {response['text']}")
                if len(self.cache["memory"]) > 10:  # 保留最近10条记录
                    self.cache["memory"] = self.cache["memory"][-10:]
                
                # 如果情绪发生变化，重新评估情绪状态
                if response["emotion_type"] != self.cache["emotion"]["emotion_type"]:
                    new_emotion = await self.model_service._evaluate_emotion(
                        memory + "\n用户: " + user_input + "\n机器人: " + response["text"]
                    )
                    if new_emotion:
                        self.cache["emotion"] = new_emotion
                        
                return {
                    "status": "success",
                    "response": json.dumps(response, ensure_ascii=False)
                }
            else:
                return {
                    "status": "error",
                    "error": response["error"],
                    "response": json.dumps({
                        "text": response["text"],
                        "emotion_type": "平静",
                        "action": "摇头",
                        "speech_speed": 1,
                        "volume": 1,
                        "pitch": 1
                    }, ensure_ascii=False)
                }
            
        except Exception as e:
            print(f"处理用户输入失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "response": json.dumps({
                    "text": "抱歉，处理您的请求时出现错误",
                    "emotion_type": "平静",
                    "action": "摇头",
                    "speech_speed": 1,
                    "volume": 1,
                    "pitch": 1
                }, ensure_ascii=False)
            }
    
    async def _check_safety(self, user_input: str, scene_info: Dict[str, Any]) -> bool:
        """
        检查输入和场景是否安全
        
        检查内容：
        1. 检查用户输入是否包含危险关键词
        2. 检查场景中的障碍物距离是否安全
        
        Args:
            user_input: 用户输入文本
            scene_info: 场景信息
            
        Returns:
            bool: 是否安全
        """
        try:
            if not self.config.safety.enable_safety_check:
                return True
                
            # 检查危险关键词
            for keyword in self.config.safety.danger_keywords:
                if keyword in user_input:
                    return False
            
            # 检查场景安全
            if scene_info:
                obstacles = scene_info.get("obstacles", [])
                if isinstance(obstacles, list):
                    for obstacle in obstacles:
                        if isinstance(obstacle, dict) and obstacle.get("distance", float("inf")) < self.config.safety.min_obstacle_distance:
                            return False
                
            return True
            
        except Exception as e:
            print(f"安全检查失败: {str(e)}")
            return False
    
    def _format_vocabulary(self) -> str:
        """
        格式化词汇表信息
        
        将词汇表格式化为清晰的层级结构
        
        Returns:
            str: 格式化后的词汇表信息
        """
        vocabulary_info = """词汇表结构：

1. 情绪词汇：
"""
        for emotion, words in self.config.vocabulary.emotions.items():
            vocabulary_info += f"   - {emotion}: {', '.join(words)}\n"
        
        vocabulary_info += "\n2. 动作词汇：\n"
        vocabulary_info += f"   {', '.join(self.config.vocabulary.actions)}\n"
        
        vocabulary_info += "\n3. 响应词汇：\n"
        for response_type, words in self.config.vocabulary.responses.items():
            vocabulary_info += f"   - {response_type}: {', '.join(words)}\n"
            
        return vocabulary_info
    
    async def _call_model(self, prompt: str, task_type: str = "general") -> Optional[str]:
        """
        调用大模型API
        
        根据配置的模型类型调用相应的API
        
        Args:
            prompt: 提示词
            task_type: 任务类型(general/emotion/response)
            
        Returns:
            Optional[str]: 模型返回的文本,如果调用失败则返回None
        """
        try:
            model_type = self.config.model.model_type
            print(f"Debug - Model Type: {model_type}")
            
            # 获取格式化的词汇表
            vocabulary_info = self._format_vocabulary()
            
            # 根据任务类型构造系统提示词
            if task_type == "emotion":
                system_prompt = f"""你是一个情感分析助手。你的任务是：
1. 理解用户输入的语意
2. 分析用户的情感倾向
3. 从词汇表中选择最合适的情感类型

{vocabulary_info}

注意事项：
1. 只能从词汇表中的情绪词汇中选择
2. 需要考虑情感转换规则
3. 回复时只需返回情感类型，不要解释"""

            elif task_type == "response":
                system_prompt = f"""你是一个对话情感分析助手。你的任务是：
1. 理解用户输入的语意
2. 根据用户输入的语意，选择你认为和用户输入语意最匹配的响应类型
3. 从词汇表中选择对应的响应词汇

{vocabulary_info}

注意事项：
1. 只能从词汇表中的响应类型中选择
2. 回复时只需返回响应类型，不要解释
3. 回复格式要求：
- 必须使用格式：{{情绪词汇}}，{{动作词汇}}，{{响应词汇}}
- 情绪词汇必须与当前情感类型匹配
- 动作词汇必须从动作词汇列表中选择
- 响应词汇必须从对应类型的响应词汇列表中选择
4. 严格按照词汇表中的词汇进行回复"""

            else:
                system_prompt = f"""你是一个对话助手。你的任务是：
1. 理解用户输入的语意
2. 分析用户的情感倾向
3. 选择合适的响应类型
4. 严格按照词汇表中的词汇进行回复

{vocabulary_info}

回复格式要求：
- 必须使用格式：{{情绪词汇}}，{{动作词汇}}，{{响应词汇}}
- 情绪词汇必须与当前情感类型匹配
- 动作词汇必须从动作词汇列表中选择
- 响应词汇必须从对应类型的响应词汇列表中选择

注意：只能使用词汇表中提供的词汇，不能使用其他词汇。"""
            
            if model_type == "ollama":
                # 获取Ollama API地址
                api_base = getattr(self.config.model, "api_base", None)
                if not api_base:
                    api_base = "http://localhost:11434"
                
                # 确保地址格式正确
                if not api_base.startswith(("http://", "https://")):
                    api_base = f"http://{api_base}"
                api_base = api_base.rstrip('/')
                
                api_url = f"{api_base}/api/generate"
                print(f"Debug - API Base: {api_base}")
                print(f"Debug - Full URL: {api_url}")
                
                # 构造请求数据
                request_data = {
                    "model": self.config.model.model_name,
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": self.config.model.temperature,
                        "top_k": self.config.model.top_k,
                        "top_p": self.config.model.top_p
                    }
                }
                print(f"Debug - Request Data: {json.dumps(request_data, ensure_ascii=False)}")
                
                async with aiohttp.ClientSession() as session:
                    try:
                        # 先测试API是否可访问
                        async with session.get(f"{api_base}/api/version") as test_response:
                            if test_response.status != 200:
                                raise Exception(f"Ollama服务未启动或无法访问: {test_response.status}")
                            version_info = await test_response.json()
                            print(f"Debug - Ollama Version: {version_info}")
                    except Exception as e:
                        print(f"Debug - Ollama服务检查失败: {str(e)}")
                        raise Exception(f"无法连接到Ollama服务: {str(e)}")
                    
                    async with session.post(
                        api_url,
                        json=request_data
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Debug - API Error Response: {error_text}")
                            raise Exception(f"API调用失败: {response.status} - {error_text}")
                        
                        result = await response.json()
                        return result["response"].strip()
                        
            elif model_type == "openai":
                openai.api_key = self.config.model.api_key
                # 检查是否配置了自定义API地址
                if hasattr(self.config.model, "api_base"):
                    openai.api_base = self.config.model.api_base
                response = await openai.ChatCompletion.acreate(
                    model=self.config.model.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.model.temperature,
                    top_p=self.config.model.top_p
                )
                return response.choices[0].message.content.strip()
                
            elif model_type == "other":
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.config.model.api_base}/v1/chat/completions",
                        json={
                            "model": self.config.model.model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": self.config.model.temperature,
                            "top_p": self.config.model.top_p
                        }
                    ) as response:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                        
            return None
            
        except Exception as e:
            print(f"调用模型API失败: {str(e)}")
            return None
    
    async def _analyze_emotion(self, text: str) -> str:
        """
        分析文本情感
        
        使用大模型分析文本的情感倾向
        
        Args:
            text: 输入文本
            
        Returns:
            str: 情感类型
        """
        try:
            # 构造提示词
            prompt = f"""请分析以下用户输入的情感倾向。

用户输入: {text}

当前情感: {self.current_emotion}
允许的情感转换: {self.config.emotion.rules.transitions.get(self.current_emotion, [])}

请选择一个最合适的情感类型。"""

            # 调用大模型分析
            response = await self._call_model(prompt, task_type="emotion")
            
            # 如果成功获取响应且情感类型有效
            if response and response in self.config.vocabulary.emotions:
                # 检查情感转换是否允许
                if response in self.config.emotion.rules.transitions.get(self.current_emotion, []):
                    return response
            
            # 如果响应无效或情感转换不允许，保持当前情感
            return self.current_emotion
            
        except Exception as e:
            print(f"情感分析失败: {str(e)}")
            return self.current_emotion
    
    async def _generate_response(self, user_input: str) -> Dict:
        """
        生成响应文本
        
        使用大模型生成符合格式要求的回复
        
        Args:
            user_input: 用户输入
            
        Returns:
            Dict: 响应对象
        """
        try:
            # 构造提示词
            prompt = f"""请根据用户输入生成一个完整的回复。

用户输入: {user_input}

当前情感: {self.current_emotion}
可用的情绪词汇: {self.config.vocabulary.emotions[self.current_emotion]["词汇"]}
可用的动作词汇: {self.config.vocabulary.actions}
可用的响应词汇类型及对应词汇: {json.dumps(self.config.vocabulary.responses, ensure_ascii=False, indent=2)}

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

            # 调用大模型生成回复
            response_text = await self._call_model(prompt)
            
            if response_text:
                try:
                    # 解析JSON回复
                    response_dict = json.loads(response_text)
                    
                    # 验证所有字段是否存在
                    required_fields = ["话语", "表情", "程度", "动作"]
                    if not all(field in response_dict for field in required_fields):
                        raise KeyError("缺少必要的字段")
                    
                    # 验证话语是否在响应词汇中
                    valid_responses = [word for response_list in self.config.vocabulary.responses.values() 
                                    for word in response_list]
                    if response_dict["话语"] not in valid_responses:
                        raise ValueError("无效的响应词汇")
                    
                    # 验证表情是否在当前情感的词汇中
                    if response_dict["表情"] not in self.config.vocabulary.emotions[self.current_emotion]["词汇"]:
                        raise ValueError("无效的情绪词汇")
                    
                    # 验证动作是否有效
                    if response_dict["动作"] not in self.config.vocabulary.actions:
                        raise ValueError("无效的动作词汇")
                    
                    # 验证程度是否在范围内
                    degree = float(response_dict["程度"])
                    if not 0 <= degree <= 10:
                        raise ValueError("情绪程度超出范围")
                    
                    # 构造响应对象
                    return {
                        "text": response_dict["话语"],
                        "emotion": response_dict["表情"],
                        "action": response_dict["动作"],
                        "degree": degree,
                        "speech_speed": self.config.emotion.rules.intensities["语速"][self.current_emotion],
                        "volume": self.config.emotion.rules.intensities["音量"][self.current_emotion],
                        "pitch": self.config.emotion.rules.intensities["音调"][self.current_emotion]
                    }
                    
                except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                    print(f"解析模型回复失败: {str(e)}")
                    # 解析失败时继续使用备选方案

            # 如果生成的回复无效,使用备选方案
            emotion_word = random.choice(self.config.vocabulary.emotions[self.current_emotion]["词汇"])
            response_type = await self._determine_response_type(user_input)
            response_word = random.choice(self.config.vocabulary.responses[response_type])
            action = random.choice(self.config.vocabulary.actions)
            
            return {
                "text": response_word,
                "emotion": emotion_word,
                "action": action,
                "degree": 5.0,  # 默认程度为中等(5.0)
                "speech_speed": self.config.emotion.rules.intensities["语速"][self.current_emotion],
                "volume": self.config.emotion.rules.intensities["音量"][self.current_emotion],
                "pitch": self.config.emotion.rules.intensities["音调"][self.current_emotion]
            }
            
        except Exception as e:
            print(f"生成响应失败: {str(e)}")
            # 如果出现任何错误,返回安全的默认响应
            return {
                "text": "你好",
                "emotion": "平静",
                "action": "点头",
                "degree": 5.0,  # 默认程度为中等(5.0)
                "speech_speed": 1.0,
                "volume": 1.0,
                "pitch": 1.0
            }
    
    async def _determine_response_type(self, user_input: str) -> str:
        """
        确定响应类型
        
        使用大模型分析用户输入的语意,选择合适的响应类型
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            str: 响应类型
        """
        # 构造提示词,引导模型理解用户意图
        prompt = f"""请分析用户输入的语意,选择一个最合适的响应类型。

用户输入: {user_input}

请选择一个最合适的响应类型。"""

        try:
            # 调用大模型进行分析
            response = await self._call_model(prompt, task_type="response")
            
            # 如果成功获取响应且响应类型有效
            if response and response in self.config.vocabulary.responses:
                return response
                
            # 如果响应无效,随机选择一个类型
            available_types = list(self.config.vocabulary.responses.keys())
            return random.choice(available_types)
            
        except Exception as e:
            print(f"确定响应类型失败: {str(e)}")
            return "问候"
    
    def _update_emotion(self, new_emotion: str):
        """
        更新情感状态
        
        根据情感转换规则更新系统的情感状态
        
        Args:
            new_emotion: 新的情感类型
        """
        try:
            # 检查情感转换是否允许
            if new_emotion in self.config.emotion.rules.transitions.get(self.current_emotion, []):
                self.current_emotion = new_emotion
                
            # 更新情感历史
            self.emotion_history.append(self.current_emotion)
            if len(self.emotion_history) > self.config.emotion.emotion_history_size:
                self.emotion_history.pop(0)
                
        except Exception as e:
            print(f"更新情感状态失败: {str(e)}")
            # 如果更新失败，保持当前情感状态