"""
模型服务
支持不同的大语言模型后端
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List
import aiohttp
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

class BaseModelService(ABC):
    """模型服务基类"""
    
    @abstractmethod
    async def get_response(self,
                          user_input: str,
                          context: Dict,
                          emotion: Dict,
                          vocabulary_constraints: List[str]) -> Dict:
        """获取模型响应"""
        pass
        
    def _build_prompt(self,
                     user_input: str,
                     context: Dict,
                     emotion: Dict,
                     vocabulary_constraints: List[str]) -> str:
        """
        构建提示词
        
        Args:
            user_input: 用户输入
            context: 上下文信息
            emotion: 情感状态
            vocabulary_constraints: 词汇约束
            
        Returns:
            str: 构建的提示词
        """
        # 构建提示词
        prompt = f"""你是一个对话机器人。请按照以下要求回复:

1. 输出格式必须是合法的JSON，包含以下字段:
   - text: 回复的文本内容
   - emotion_type: 情感类型，必须是以下之一：平静、高兴、难过、生气、惊讶
   - action: 动作，必须是以下之一：点头、摇头、微笑、皱眉、挥手
   - speech_speed: 语速，1-10的整数
   - volume: 音量，1-10的整数
   - pitch: 音调，1-10的整数

2. 当前情感状态:
   - 类型: {emotion.get('emotion_type', '平静')}
   - 语速: {emotion.get('speech_speed', 5)}
   - 音量: {emotion.get('volume', 5)}
   - 音调: {emotion.get('pitch', 5)}

3. 对话历史:
{chr(10).join(context.get('memory', []))}

4. 用户输入:
{user_input}

请确保你的回复是一个合法的JSON字符串，包含所有必需字段。例如:
{{
    "text": "你好，很高兴见到你",
    "emotion_type": "高兴",
    "action": "微笑",
    "speech_speed": 7,
    "volume": 6,
    "pitch": 7
}}

请根据上述要求生成回复:"""
        
        return prompt

class OpenAIService(BaseModelService):
    """OpenAI服务类"""
    
    def __init__(self):
        """初始化服务"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("未设置OPENAI_API_KEY环境变量")
            
        openai.api_key = self.api_key
        self.system_prompt = """
        你是一个情感丰富的对话机器人。在对话中,你需要:
        1. 只使用指定的词汇表达
        2. 表现出适当的情绪
        3. 保持对话的自然性和连贯性
        4. 回答要简短精炼
        """
        
    async def get_response(self,
                          user_input: str,
                          context: Dict,
                          emotion: Dict,
                          vocabulary_constraints: List[str]) -> Dict:
        """获取OpenAI响应"""
        try:
            prompt = self._build_prompt(
                user_input,
                context,
                emotion,
                vocabulary_constraints
            )
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            return {
                'text': response.choices[0].message.content.strip(),
                'raw_response': response
            }
            
        except Exception as e:
            print(f"OpenAI API调用失败: {str(e)}")
            return {
                'text': "对不起,我现在有点困惑",
                'error': str(e)
            } 