"""
上下文分析器
分析用户输入的语义和上下文
"""

import jieba
import jieba.posseg as pseg
from typing import Dict, List, Optional, Set, Tuple
import json
import re

from config.vocabulary import ALL_VOCABULARY, is_valid_word
from config.commands import get_command_by_keywords, CommandType
from config.emotions import get_emotion_by_keywords, EmotionType
from utils.logger import logger

class ContextAnalyzer:
    """上下文分析器类"""
    
    def __init__(self):
        """初始化分析器"""
        # 加载配置
        try:
            with open('config/config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = None
            
        # 初始化分词器
        self._init_tokenizer()
        
        # 初始化语气词典
        self.tone_words = {
            '祈使': {'请', '麻烦', '帮忙', '希望', '建议'},
            '疑问': {'吗', '呢', '吧', '啊', '么'},
            '感叹': {'啊', '哇', '呀', '哦', '诶'},
            '强调': {'一定', '必须', '肯定', '绝对', '确实'},
            '缓和': {'可能', '也许', '大概', '差不多', '稍微'}
        }
        
    def _init_tokenizer(self):
        """初始化分词器"""
        # 将词汇库添加到jieba分词器的自定义词典中
        for word in ALL_VOCABULARY:
            jieba.add_word(word)
            
        # 加载情绪词汇
        if self.config:
            for words in self.config['vocabulary']['emotions'].values():
                for word in words:
                    jieba.add_word(word)
                    
        # 加载动作词汇
        if self.config:
            for word in self.config['vocabulary']['actions']:
                jieba.add_word(word)
                
        # 加载响应词汇
        if self.config:
            for words in self.config['vocabulary']['responses'].values():
                for word in words:
                    jieba.add_word(word)
    
    def analyze(self,
               text: str,
               scene_info: Optional[Dict] = None) -> Dict:
        """
        分析输入文本的上下文
        
        Args:
            text: 输入文本
            scene_info: 场景信息(可选)
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 1. 分词和词性标注
            words, pos_tags = self._segment_text(text)
            
            # 2. 提取关键词
            keywords = self._extract_keywords(words)
            
            # 3. 识别可能的指令
            possible_commands = get_command_by_keywords(text)
            
            # 4. 识别可能的情绪
            possible_emotions = get_emotion_by_keywords(text)
            
            # 5. 分析语境类型
            context_type, confidence = self._analyze_context_type(
                text,
                words,
                pos_tags,
                keywords,
                possible_commands
            )
            
            # 6. 分析语气
            tone_analysis = self._analyze_tone(words, pos_tags)
            
            # 7. 结合场景信息
            context = self._combine_scene_info(
                {
                    'original_text': text,
                    'words': list(words),
                    'pos_tags': list(pos_tags),
                    'keywords': list(keywords),
                    'possible_commands': [cmd.value for cmd in possible_commands],
                    'possible_emotions': [emotion.value for emotion in possible_emotions],
                    'context_type': context_type,
                    'context_confidence': confidence,
                    'tone_analysis': tone_analysis
                },
                scene_info
            )
            
            return context
            
        except Exception as e:
            logger.error(f"分析上下文失败: {e}")
            return {
                'original_text': text,
                'words': [],
                'pos_tags': [],
                'keywords': [],
                'possible_commands': [],
                'possible_emotions': [],
                'context_type': 'chat',
                'context_confidence': 0.5,
                'tone_analysis': {}
            }
    
    def _segment_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        对文本进行分词和词性标注
        
        Args:
            text: 输入文本
            
        Returns:
            Tuple[List[str], List[str]]: 分词结果和词性标注
        """
        # 使用jieba进行词性标注
        words_pos = pseg.cut(text)
        words = []
        pos_tags = []
        
        for word, pos in words_pos:
            words.append(word)
            pos_tags.append(pos)
            
        return words, pos_tags
    
    def _extract_keywords(self, words: List[str]) -> Set[str]:
        """
        从分词结果中提取关键词
        
        Args:
            words: 分词结果
            
        Returns:
            Set[str]: 关键词集合
        """
        return {word for word in words if is_valid_word(word)}
    
    def _analyze_context_type(self,
                            text: str,
                            words: List[str],
                            pos_tags: List[str],
                            keywords: Set[str],
                            possible_commands: List[CommandType]) -> Tuple[str, float]:
        """
        分析语境类型
        
        Args:
            text: 原始文本
            words: 分词结果
            pos_tags: 词性标注
            keywords: 关键词集合
            possible_commands: 可能的指令列表
            
        Returns:
            Tuple[str, float]: 语境类型和置信度
        """
        type_scores = {
            'command': 0.0,
            'question': 0.0,
            'greeting': 0.0,
            'chat': 0.1  # 默认得分
        }
        
        # 1. 基于指令关键词的得分
        if possible_commands:
            type_scores['command'] += 0.6
            
        # 2. 基于词性的得分
        for pos in pos_tags:
            if pos == 'v':  # 动词
                type_scores['command'] += 0.1
            elif pos == 'y':  # 语气词
                type_scores['question'] += 0.1
                
        # 3. 基于标点符号的得分
        if '？' in text or '?' in text:
            type_scores['question'] += 0.4
            
        # 4. 基于疑问词的得分
        question_words = {'是吗', '真的吗', '什么', '为什么', '怎么样'}
        if keywords & question_words:
            type_scores['question'] += 0.4
            
        # 5. 基于问候词的得分
        greeting_words = {'你好', '早上好', '下午好', '晚上好', '再见'}
        if keywords & greeting_words:
            type_scores['greeting'] += 0.6
            
        # 6. 基于句式的得分
        if re.search(r'(请|麻烦|帮忙).*(好吗|可以吗)', text):
            type_scores['command'] += 0.3
            
        # 选择得分最高的类型
        context_type = max(type_scores.items(), key=lambda x: x[1])
        return context_type[0], context_type[1]
    
    def _analyze_tone(self,
                     words: List[str],
                     pos_tags: List[str]) -> Dict:
        """
        分析语气
        
        Args:
            words: 分词结果
            pos_tags: 词性标注
            
        Returns:
            Dict: 语气分析结果
        """
        tone_scores = {
            '祈使': 0.0,
            '疑问': 0.0,
            '感叹': 0.0,
            '强调': 0.0,
            '缓和': 0.0
        }
        
        # 1. 基于语气词的得分
        for word in words:
            for tone, word_set in self.tone_words.items():
                if word in word_set:
                    tone_scores[tone] += 0.3
                    
        # 2. 基于词性的得分
        for pos in pos_tags:
            if pos == 'y':  # 语气词
                tone_scores['感叹'] += 0.1
                tone_scores['疑问'] += 0.1
                
        # 3. 基于句式的得分
        text = ''.join(words)
        if re.search(r'请.*吧', text):
            tone_scores['祈使'] += 0.3
            
        if re.search(r'一定|必须|肯定', text):
            tone_scores['强调'] += 0.3
            
        if re.search(r'可能|也许|大概', text):
            tone_scores['缓和'] += 0.3
            
        # 归一化得分
        max_score = max(tone_scores.values())
        if max_score > 0:
            tone_scores = {
                k: v/max_score for k, v in tone_scores.items()
            }
            
        # 只保留得分大于0.3的语气
        return {
            k: v for k, v in tone_scores.items()
            if v >= 0.3
        }
    
    def _combine_scene_info(self,
                          context: Dict,
                          scene_info: Optional[Dict] = None) -> Dict:
        """
        结合场景信息
        
        Args:
            context: 上下文分析结果
            scene_info: 场景信息
            
        Returns:
            Dict: 完整的上下文信息
        """
        if scene_info:
            context['scene'] = scene_info
            
            # 根据场景信息补充分析
            if 'obstacles' in scene_info:
                context['has_obstacles'] = bool(scene_info['obstacles'])
                context['is_safe'] = self._check_safety(
                    context['possible_commands'],
                    scene_info['obstacles']
                )
            else:
                context['has_obstacles'] = False
                context['is_safe'] = True
                
            # 分析场景对情绪的影响
            context['scene_emotion_impact'] = self._analyze_scene_emotion_impact(
                scene_info
            )
                
        return context
    
    def _check_safety(self,
                     commands: List[str],
                     obstacles: List[Dict]) -> bool:
        """
        检查指令执行的安全性
        
        Args:
            commands: 可能的指令列表
            obstacles: 障碍物信息
            
        Returns:
            bool: 是否安全
        """
        if not commands or not obstacles:
            return True
            
        if not self.config:
            return True
            
        try:
            # 获取安全配置
            safety_config = self.config['safety']
            min_distance = safety_config['min_obstacle_distance']
            
            # 检查每个障碍物
            for obstacle in obstacles:
                distance = obstacle.get('distance', float('inf'))
                if distance < min_distance:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"安全检查失败: {e}")
            return True
            
    def _analyze_scene_emotion_impact(self, scene: Dict) -> Dict:
        """
        分析场景对情绪的影响
        
        Args:
            scene: 场景信息
            
        Returns:
            Dict: 情绪影响分析结果
        """
        impact = {}
        
        try:
            # 分析温度对情绪的影响
            temperature = scene.get('temperature', '25°C')
            try:
                temp = float(temperature.replace('°C', ''))
                if temp > 30:
                    impact['temperature'] = {
                        'type': '不适',
                        'degree': min((temp - 30) / 10, 1.0)
                    }
                elif temp < 10:
                    impact['temperature'] = {
                        'type': '不适',
                        'degree': min((10 - temp) / 10, 1.0)
                    }
            except:
                pass
                
            # 分析光照对情绪的影响
            lighting = scene.get('lighting', '明亮')
            if lighting == '黑暗':
                impact['lighting'] = {
                    'type': '不安',
                    'degree': 0.7
                }
            elif lighting == '昏暗':
                impact['lighting'] = {
                    'type': '警惕',
                    'degree': 0.4
                }
                
            # 分析安全状态对情绪的影响
            safety_status = scene.get('safety_status', '安全')
            if safety_status != '安全':
                impact['safety'] = {
                    'type': '危险',
                    'degree': 0.8
                }
                
            return impact
            
        except Exception as e:
            logger.error(f"分析场景情绪影响失败: {e}")
            return {} 