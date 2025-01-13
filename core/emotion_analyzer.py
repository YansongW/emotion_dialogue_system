"""
情绪分析器
分析对话情绪并生成情绪参数
"""

import random
from typing import Dict, List, Optional, Set, Tuple
import json

from config.emotions import (
    EmotionType,
    get_emotion_params,
    get_emotion_by_keywords
)
from utils.logger import logger

class EmotionAnalyzer:
    """情绪分析器类"""
    
    def __init__(self):
        """初始化分析器"""
        self.current_emotion = EmotionType.CALM  # 默认情绪为平静
        self.emotion_history = []  # 情绪历史记录
        self.history_size = 5  # 保留最近5次情绪记录
        
        # 加载配置
        try:
            with open('config/config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = None
        
    def analyze(self, context: Dict) -> Dict:
        """
        分析上下文并生成情绪参数
        
        Args:
            context: 上下文分析结果
            
        Returns:
            Dict: 情绪分析结果
        """
        try:
            # 1. 基于上下文识别情绪
            emotion_type, confidence = self._recognize_emotion(context)
            
            # 2. 更新情绪状态
            self._update_emotion_state(emotion_type)
            
            # 3. 获取情绪参数
            emotion_params = get_emotion_params(emotion_type)
            
            # 4. 根据上下文调整参数
            adjusted_params = self._adjust_params(
                emotion_params,
                context,
                confidence
            )
            
            return {
                'emotion_type': emotion_type.value,
                'params': adjusted_params,
                'history': [e.value for e in self.emotion_history],
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"情绪分析失败: {e}")
            return {
                'emotion_type': EmotionType.CALM.value,
                'params': get_emotion_params(EmotionType.CALM),
                'history': [e.value for e in self.emotion_history],
                'confidence': 0.5
            }
        
    def _recognize_emotion(self, context: Dict) -> Tuple[EmotionType, float]:
        """
        基于上下文识别情绪
        
        Args:
            context: 上下文分析结果
            
        Returns:
            Tuple[EmotionType, float]: 识别出的情绪类型和置信度
        """
        # 1. 从上下文中获取可能的情绪
        possible_emotions = [
            EmotionType(e) for e in context.get('possible_emotions', [])
        ]
        
        # 2. 计算每种情绪的得分
        emotion_scores = {}
        
        # 2.1 基于关键词匹配的得分
        keywords = set(context.get('keywords', []))
        for emotion in EmotionType:
            score = self._calculate_keyword_score(emotion, keywords)
            emotion_scores[emotion] = score
            
        # 2.2 基于上下文类型的得分
        context_type = context.get('context_type', 'chat')
        context_scores = self._get_context_type_scores(context_type)
        for emotion, score in context_scores.items():
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score
            
        # 2.3 基于历史情绪的得分
        history_scores = self._calculate_history_scores()
        for emotion, score in history_scores.items():
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score
            
        # 2.4 基于场景信息的得分
        scene_scores = self._calculate_scene_scores(context.get('scene', {}))
        for emotion, score in scene_scores.items():
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score
            
        # 3. 选择得分最高的情绪
        if emotion_scores:
            max_score = max(emotion_scores.values())
            best_emotions = [
                e for e, s in emotion_scores.items()
                if s == max_score
            ]
            selected_emotion = (
                possible_emotions[0] if possible_emotions
                else random.choice(best_emotions)
            )
            confidence = max_score / (len(emotion_scores) * 4)  # 归一化置信度
        else:
            selected_emotion = self.current_emotion
            confidence = 0.5
            
        return selected_emotion, min(1.0, max(0.0, confidence))
        
    def _calculate_keyword_score(self,
                               emotion: EmotionType,
                               keywords: Set[str]) -> float:
        """
        计算基于关键词的情绪得分
        
        Args:
            emotion: 情绪类型
            keywords: 关键词集合
            
        Returns:
            float: 得分(0-1)
        """
        if not self.config:
            return 0.0
            
        emotion_keywords = set(
            self.config['emotion']['rules']['triggers'].get(
                emotion.value,
                []
            )
        )
        
        if not emotion_keywords:
            return 0.0
            
        matches = keywords & emotion_keywords
        return len(matches) / len(emotion_keywords)
        
    def _get_context_type_scores(self, context_type: str) -> Dict[EmotionType, float]:
        """
        获取基于上下文类型的情绪得分
        
        Args:
            context_type: 上下文类型
            
        Returns:
            Dict[EmotionType, float]: 情绪得分字典
        """
        scores = {}
        
        if context_type == 'greeting':
            scores[EmotionType.FRIENDLY] = 0.8
            scores[EmotionType.HAPPY] = 0.6
            
        elif context_type == 'command':
            scores[EmotionType.EXCITED] = 0.7
            scores[EmotionType.FRIENDLY] = 0.5
            
        elif context_type == 'question':
            scores[EmotionType.SURPRISED] = 0.6
            scores[EmotionType.FRIENDLY] = 0.4
            
        else:  # chat
            scores[EmotionType.CALM] = 0.5
            scores[EmotionType.FRIENDLY] = 0.3
            
        return scores
        
    def _calculate_history_scores(self) -> Dict[EmotionType, float]:
        """
        计算基于历史情绪的得分
        
        Returns:
            Dict[EmotionType, float]: 情绪得分字典
        """
        scores = {}
        
        if not self.emotion_history:
            return scores
            
        # 统计历史情绪频率
        emotion_counts = {}
        total = len(self.emotion_history)
        
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        # 计算得分
        for emotion, count in emotion_counts.items():
            scores[emotion] = count / total * 0.5  # 历史因素的权重为0.5
            
        return scores
        
    def _calculate_scene_scores(self, scene: Dict) -> Dict[EmotionType, float]:
        """
        计算基于场景信息的情绪得分
        
        Args:
            scene: 场景信息
            
        Returns:
            Dict[EmotionType, float]: 情绪得分字典
        """
        scores = {}
        
        # 检查安全状态
        if not scene.get('is_safe', True):
            scores[EmotionType.SCARED] = 0.8
            scores[EmotionType.SURPRISED] = 0.6
            
        # 检查障碍物
        if scene.get('has_obstacles', False):
            scores[EmotionType.SCARED] = scores.get(EmotionType.SCARED, 0) + 0.4
            
        # 检查温度
        temperature = scene.get('temperature', '25°C')
        try:
            temp = float(temperature.replace('°C', ''))
            if temp > 30:
                scores[EmotionType.ANGRY] = 0.3
            elif temp < 10:
                scores[EmotionType.SAD] = 0.3
        except:
            pass
            
        return scores
        
    def _is_emotion_compatible(self,
                             emotion1: EmotionType,
                             emotion2: EmotionType) -> bool:
        """
        检查两种情绪是否兼容
        
        Args:
            emotion1: 第一种情绪
            emotion2: 第二种情绪
            
        Returns:
            bool: 是否兼容
        """
        if not self.config:
            return False
            
        # 从配置文件获取情绪转换规则
        transitions = self.config['emotion']['rules']['transitions']
        
        # 检查是否允许从emotion2转换到emotion1
        allowed_transitions = transitions.get(emotion2.value, [])
        return emotion1.value in allowed_transitions
        
    def _update_emotion_state(self, new_emotion: EmotionType):
        """
        更新情绪状态
        
        Args:
            new_emotion: 新的情绪类型
        """
        # 更新当前情绪
        self.current_emotion = new_emotion
        
        # 更新历史记录
        self.emotion_history.append(new_emotion)
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
            
    def _adjust_params(self,
                      params: Dict,
                      context: Dict,
                      confidence: float) -> Dict:
        """
        根据上下文调整情绪参数
        
        Args:
            params: 原始情绪参数
            context: 上下文信息
            confidence: 情绪识别的置信度
            
        Returns:
            Dict: 调整后的参数
        """
        if not self.config:
            return params
            
        adjusted_params = params.copy()
        
        # 1. 根据置信度调整参数强度
        for key in adjusted_params:
            if key in ['speech_speed', 'volume', 'pitch']:
                # 参数向平静状态靠拢的程度与置信度成反比
                calm_value = 1.0  # 平静状态的参数值
                current_value = adjusted_params[key]
                adjusted_params[key] = (
                    current_value * confidence +
                    calm_value * (1 - confidence)
                )
        
        # 2. 根据是否有危险调整参数
        if not context.get('is_safe', True):
            adjusted_params['speech_speed'] *= 1.2  # 加快语速
            adjusted_params['volume'] *= 0.8  # 降低音量
            
        # 3. 根据上下文类型调整
        context_type = context.get('context_type', 'chat')
        if context_type == 'command':
            adjusted_params['speech_speed'] *= 1.1  # 执行指令时语速略快
            
        # 4. 确保参数在有效范围内
        ranges = self.config['emotion'].get('speech_speed_range', [0.8, 1.3])
        adjusted_params['speech_speed'] = max(
            ranges[0],
            min(ranges[1], adjusted_params['speech_speed'])
        )
        
        ranges = self.config['emotion'].get('volume_range', [0.8, 1.2])
        adjusted_params['volume'] = max(
            ranges[0],
            min(ranges[1], adjusted_params['volume'])
        )
        
        return adjusted_params