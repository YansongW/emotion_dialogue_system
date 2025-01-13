"""
配置管理模块
管理系统的所有可配置参数
"""

import os
import json
import threading
from typing import Dict, Any, List
from dataclasses import dataclass, asdict, field

@dataclass
class VocabularyConfig:
    """词汇配置"""
    emotions: Dict[str, Dict[str, List[str]]] = field(default_factory=lambda: {
        "平静": {
            "词汇": ["平静", "安详", "淡定"],
            "emotions": ["calm", "peaceful", "neutral"]
        },
        "高兴": {
            "词汇": ["开心", "快乐", "兴奋", "喜悦"],
            "emotions": ["happy", "joyful", "cheerful", "delighted"]
        },
        "难过": {
            "词汇": ["伤心", "悲伤", "沮丧", "失落"],
            "emotions": ["sad", "sorrowful", "depressed", "disappointed"]
        },
        "生气": {
            "词汇": ["愤怒", "恼火", "不满", "烦躁"],
            "emotions": ["angry", "furious", "irritated", "agitated"]
        },
        "惊讶": {
            "词汇": ["吃惊", "震惊", "意外", "诧异"],
            "emotions": ["surprised", "shocked", "amazed", "astonished"]
        }
    })
    actions: List[str] = field(default_factory=lambda: [
        "点头", "摇头", "微笑", "皱眉", "挥手"
    ])
    responses: Dict[str, List[str]] = field(default_factory=lambda: {
        "问候": ["你好", "早上好", "下午好", "晚上好"],
        "告别": ["再见", "拜拜", "下次见", "回头见"],
        "肯定": ["好的", "没问题", "可以", "当然"],
        "否定": ["抱歉", "不行", "不可以", "恐怕不行"]
    })

@dataclass
class EmotionRules:
    """情绪规则"""
    triggers: Dict[str, List[str]] = field(default_factory=lambda: {
        "高兴": ["表扬", "成功", "帮助"],
        "难过": ["失败", "道歉", "离别"],
        "生气": ["批评", "打扰", "不尊重"],
        "惊讶": ["意外", "突然", "特殊"]
    })
    transitions: Dict[str, List[str]] = field(default_factory=lambda: {
        "平静": ["高兴", "难过", "生气", "惊讶"],
        "高兴": ["平静", "惊讶"],
        "难过": ["平静", "生气"],
        "生气": ["平静", "难过"],
        "惊讶": ["平静", "高兴"]
    })
    intensities: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "语速": {"平静": 1.0, "高兴": 1.2, "难过": 0.8, "生气": 1.3, "惊讶": 1.1},
        "音量": {"平静": 1.0, "高兴": 1.1, "难过": 0.9, "生气": 1.2, "惊讶": 1.1},
        "音调": {"平静": 1.0, "高兴": 1.1, "难过": 0.9, "生气": 1.2, "惊讶": 1.1}
    })

@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str = "ollama"  # ollama、openai 或 other
    model_name: str = "llama2"  # 模型名称
    api_key: str = ""  # API密钥(用于OpenAI)
    api_base: str = ""  # API基础URL(用于其他模型)
    temperature: float = 0.7  # 温度参数
    top_k: int = 40  # top-k采样
    top_p: float = 0.9  # top-p采样
    repeat_penalty: float = 1.1  # 重复惩罚
    num_predict: int = 50  # 生成长度
    stop_sequences: List[str] = field(default_factory=list)  # 停止序列
    context_window: int = 2048  # 上下文窗口大小

@dataclass
class EmotionConfig:
    """情绪配置"""
    default_emotion: str = "平静"  # 默认情绪
    speech_speed_range: tuple = (0.8, 1.3)  # 语速范围
    volume_range: tuple = (0.8, 1.2)  # 音量范围
    pitch_range: tuple = (0.9, 1.2)  # 音调范围
    emotion_history_size: int = 5  # 情绪历史记录大小
    rules: EmotionRules = field(default_factory=EmotionRules)  # 情绪规则

@dataclass
class SafetyConfig:
    """安全配置"""
    enable_safety_check: bool = True  # 启用安全检查
    min_obstacle_distance: float = 1.0  # 最小障碍物距离
    danger_keywords: list = None  # 危险关键词
    max_response_time: int = 5000  # 最大响应时间(毫秒)
    max_conversation_turns: int = 50  # 最大对话轮数
    content_filters: List[str] = field(default_factory=lambda: [
        "暴力", "色情", "政治", "宗教", "歧视"
    ])  # 内容过滤

    def __post_init__(self):
        if self.danger_keywords is None:
            self.danger_keywords = [
                '撞', '跳', '摔', '打', '踢',
                '伤害', '危险', '破坏', '损坏'
            ]

@dataclass
class SystemConfig:
    """系统配置"""
    model: ModelConfig = None
    emotion: EmotionConfig = None
    safety: SafetyConfig = None
    vocabulary: VocabularyConfig = None
    system_prompt: str = ""
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.emotion is None:
            self.emotion = EmotionConfig()
        if self.safety is None:
            self.safety = SafetyConfig()
        if self.vocabulary is None:
            self.vocabulary = VocabularyConfig()
        if not self.system_prompt:
            self.system_prompt = """
            你是一个情感丰富的对话机器人。在对话中,你需要:
            1. 只使用指定的词汇表达
            2. 表现出适当的情绪
            3. 保持对话的自然性和连贯性
            4. 回答要简短精炼
            """

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "config/config.json"):
        """初始化配置管理器"""
        self.config_file = config_file
        self.lock = threading.Lock()  # 添加线程锁
        self.config = self._load_config()
    
    def _load_config(self) -> SystemConfig:
        """加载配置"""
        with self.lock:  # 使用线程锁保护文件读取
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    return self._dict_to_config(data)
                except Exception as e:
                    print(f"加载配置文件失败: {str(e)}")
                    return SystemConfig()
            return SystemConfig()
    
    def _dict_to_config(self, data: Dict) -> SystemConfig:
        """将字典转换为配置对象"""
        try:
            model_data = data.get('model', {})
            emotion_data = data.get('emotion', {})
            safety_data = data.get('safety', {})
            vocabulary_data = data.get('vocabulary', {})
            
            # 创建情绪规则对象
            emotion_rules = EmotionRules(
                triggers=emotion_data.get('rules', {}).get('triggers', {}),
                transitions=emotion_data.get('rules', {}).get('transitions', {}),
                intensities=emotion_data.get('rules', {}).get('intensities', {})
            )
            
            # 创建情绪配置对象
            emotion_config = EmotionConfig(
                default_emotion=emotion_data.get('default_emotion', "平静"),
                speech_speed_range=tuple(emotion_data.get('speech_speed_range', (0.8, 1.3))),
                volume_range=tuple(emotion_data.get('volume_range', (0.8, 1.2))),
                pitch_range=tuple(emotion_data.get('pitch_range', (0.9, 1.2))),
                emotion_history_size=emotion_data.get('emotion_history_size', 5),
                rules=emotion_rules
            )
            
            return SystemConfig(
                model=ModelConfig(**model_data),
                emotion=emotion_config,
                safety=SafetyConfig(**safety_data),
                vocabulary=VocabularyConfig(**vocabulary_data),
                system_prompt=data.get('system_prompt', '')
            )
        except Exception as e:
            print(f"配置转换失败: {str(e)}")
            return SystemConfig()
    
    def save_config(self):
        """保存配置"""
        with self.lock:  # 使用线程锁保护文件写入
            try:
                # 确保配置目录存在
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                
                # 将配置对象转换为字典
                config_dict = {
                    'model': asdict(self.config.model),
                    'emotion': {
                        'default_emotion': self.config.emotion.default_emotion,
                        'speech_speed_range': self.config.emotion.speech_speed_range,
                        'volume_range': self.config.emotion.volume_range,
                        'pitch_range': self.config.emotion.pitch_range,
                        'emotion_history_size': self.config.emotion.emotion_history_size,
                        'rules': asdict(self.config.emotion.rules)
                    },
                    'safety': asdict(self.config.safety),
                    'vocabulary': asdict(self.config.vocabulary),
                    'system_prompt': self.config.system_prompt
                }
                
                # 保存到文件
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=4, ensure_ascii=False)
                    
            except Exception as e:
                print(f"保存配置失败: {str(e)}")
                raise
    
    def validate_config(self) -> tuple[bool, list]:
        """
        验证配置是否有效
        
        Returns:
            tuple[bool, list]: (是否有效, 错误信息列表)
        """
        errors = []
        
        try:
            # 验证模型配置
            if self.config.model.model_type not in ['ollama', 'openai', 'other']:
                errors.append("模型类型必须是'ollama'、'openai'或'other'")
            if not self.config.model.model_name:
                errors.append("模型名称不能为空")
            if not (0 <= self.config.model.temperature <= 1):
                errors.append("temperature必须在0-1之间")
            if self.config.model.model_type == 'openai' and not self.config.model.api_key:
                errors.append("使用OpenAI时必须提供API密钥")
            if self.config.model.model_type == 'other' and not self.config.model.api_base:
                errors.append("使用其他模型时必须提供API基础URL")
                
            # 验证情绪配置
            if not (0 < self.config.emotion.speech_speed_range[0] <= 
                    self.config.emotion.speech_speed_range[1]):
                errors.append("语速范围无效")
            if not (0 < self.config.emotion.volume_range[0] <= 
                    self.config.emotion.volume_range[1]):
                errors.append("音量范围无效")
            if self.config.emotion.default_emotion not in self.config.vocabulary.emotions:
                errors.append("默认情绪必须在情绪词汇表中")
                
            # 验证安全配置
            if self.config.safety.min_obstacle_distance <= 0:
                errors.append("最小障碍物距离必须大于0")
            if self.config.safety.max_response_time <= 0:
                errors.append("最大响应时间必须大于0")
            if self.config.safety.max_conversation_turns <= 0:
                errors.append("最大对话轮数必须大于0")
                
            # 验证词汇配置
            if not self.config.vocabulary.emotions:
                errors.append("情绪词汇表不能为空")
            if not self.config.vocabulary.actions:
                errors.append("动作词汇表不能为空")
            if not self.config.vocabulary.responses:
                errors.append("响应词汇表不能为空")
            
        except Exception as e:
            errors.append(f"配置验证失败: {str(e)}")
        
        return len(errors) == 0, errors
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        更新配置
        
        Args:
            new_config: 新的配置数据
        """
        try:
            self.config = self._dict_to_config(new_config)
            is_valid, errors = self.validate_config()
            if is_valid:
                self.save_config()
            return is_valid, errors
        except Exception as e:
            return False, [f"更新配置失败: {str(e)}"]

# 创建全局配置管理器实例
config_manager = ConfigManager() 