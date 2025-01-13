"""
情绪配置文件
定义系统使用的情绪状态
"""

from enum import Enum
from typing import Dict, List

class EmotionType(Enum):
    """情绪类型枚举"""
    HAPPY = "高兴"
    SAD = "难过"
    ANGRY = "愤怒"
    SURPRISED = "惊讶"
    SCARED = "害怕"
    CALM = "平静"
    EXCITED = "兴奋"
    BORED = "无聊"
    FRIENDLY = "友善"
    SHY = "害羞"

# 情绪参数配置
EMOTION_PARAMS = {
    EmotionType.HAPPY: {
        'speech_speed': 1.2,  # 语速倍率
        'volume': 1.1,  # 音量倍率
        'pitch': 1.1,  # 音调倍率
        'expression': 'smile',  # 表情
        'keywords': ['好', '喜欢', '棒', '开心']  # 触发关键词
    },
    EmotionType.SAD: {
        'speech_speed': 0.8,
        'volume': 0.9,
        'pitch': 0.9,
        'expression': 'sad',
        'keywords': ['不好', '难过', '伤心', '失望']
    },
    EmotionType.ANGRY: {
        'speech_speed': 1.3,
        'volume': 1.2,
        'pitch': 1.2,
        'expression': 'angry',
        'keywords': ['生气', '讨厌', '不要', '别']
    },
    EmotionType.SURPRISED: {
        'speech_speed': 1.1,
        'volume': 1.1,
        'pitch': 1.2,
        'expression': 'surprised',
        'keywords': ['哇', '真的吗', '竟然', '没想到']
    },
    EmotionType.SCARED: {
        'speech_speed': 1.2,
        'volume': 0.8,
        'pitch': 1.1,
        'expression': 'scared',
        'keywords': ['害怕', '可怕', '危险', '小心']
    },
    EmotionType.CALM: {
        'speech_speed': 1.0,
        'volume': 1.0,
        'pitch': 1.0,
        'expression': 'calm',
        'keywords': ['好的', '明白', '知道', '平静']
    },
    EmotionType.EXCITED: {
        'speech_speed': 1.3,
        'volume': 1.2,
        'pitch': 1.2,
        'expression': 'excited',
        'keywords': ['太好了', '太棒了', '好激动', '期待']
    },
    EmotionType.BORED: {
        'speech_speed': 0.9,
        'volume': 0.9,
        'pitch': 0.9,
        'expression': 'bored',
        'keywords': ['无聊', '困', '乏', '懒']
    },
    EmotionType.FRIENDLY: {
        'speech_speed': 1.1,
        'volume': 1.0,
        'pitch': 1.0,
        'expression': 'friendly',
        'keywords': ['朋友', '一起', '帮助', '谢谢']
    },
    EmotionType.SHY: {
        'speech_speed': 0.9,
        'volume': 0.8,
        'pitch': 0.9,
        'expression': 'shy',
        'keywords': ['害羞', '不好意思', '抱歉', '对不起']
    }
}

def get_emotion_params(emotion_type: EmotionType) -> Dict:
    """
    获取情绪参数配置
    
    Args:
        emotion_type: 情绪类型
        
    Returns:
        Dict: 情绪参数配置
    """
    return EMOTION_PARAMS[emotion_type]

def get_emotion_by_keywords(text: str) -> List[EmotionType]:
    """
    根据关键词识别可能的情绪类型
    
    Args:
        text: 输入文本
        
    Returns:
        List[EmotionType]: 可能的情绪类型列表
    """
    possible_emotions = []
    for emotion_type in EmotionType:
        keywords = EMOTION_PARAMS[emotion_type]['keywords']
        if any(keyword in text for keyword in keywords):
            possible_emotions.append(emotion_type)
    return possible_emotions