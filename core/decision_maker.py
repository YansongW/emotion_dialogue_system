"""
决策生成器
基于上下文和情绪分析结果生成决策
"""

import random
from typing import Dict, List, Optional, Set
import json

from config.commands import (
    CommandType,
    get_command_params,
    needs_safety_check
)
from config.vocabulary import ALL_VOCABULARY
from utils.logger import logger

class DecisionMaker:
    """决策生成器类"""
    
    def __init__(self):
        """初始化决策生成器"""
        # 加载配置
        try:
            with open('config/config.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                self.danger_keywords = set(
                    self.config['safety']['danger_keywords']
                )
                self.restricted_areas = set(
                    self.config['safety']['restricted_areas']
                )
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = None
            self.danger_keywords = set()
            self.restricted_areas = set()
            
        # 初始化命令历史
        self.command_history = []
        self.max_history_size = 10
        
    def make_decision(self,
                     context: Dict,
                     emotion: Dict) -> Dict:
        """
        生成决策
        
        Args:
            context: 上下文分析结果
            emotion: 情绪分析结果
            
        Returns:
            Dict: 决策结果
        """
        try:
            # 1. 提取上下文信息
            context_type = context.get('context_type', 'chat')
            possible_commands = context.get('possible_commands', [])
            is_safe = context.get('is_safe', True)
            
            # 2. 如果是指令类型,进行指令处理
            if context_type == 'command' and possible_commands:
                # 处理所有可能的指令
                command_results = []
                for command in possible_commands:
                    result = self._process_command(
                        command,
                        context,
                        emotion
                    )
                    command_results.append(result)
                    
                # 选择最佳结果
                return self._select_best_command_result(command_results)
                
            # 3. 如果是其他类型,生成对话决策
            return self._generate_chat_decision(
                context,
                emotion
            )
            
        except Exception as e:
            logger.error(f"生成决策失败: {e}")
            return {
                'should_execute': False,
                'reject_reason': "抱歉,我现在无法做出决定",
                'action_type': None,
                'vocabulary_constraints': list(ALL_VOCABULARY)
            }
        
    def _process_command(self,
                        command: str,
                        context: Dict,
                        emotion: Dict) -> Dict:
        """
        处理指令类型的决策
        
        Args:
            command: 指令内容
            context: 上下文信息
            emotion: 情绪信息
            
        Returns:
            Dict: 决策结果
        """
        try:
            # 1. 安全性检查
            safety_result = self._check_command_safety(command, context)
            if not safety_result['is_safe']:
                return {
                    'should_execute': False,
                    'reject_reason': self._generate_reject_response(
                        command,
                        safety_result['reason']
                    ),
                    'action_type': None,
                    'vocabulary_constraints': list(ALL_VOCABULARY),
                    'safety_score': safety_result['score']
                }
                
            # 2. 情绪检查
            emotion_type = emotion['emotion_type']
            if emotion_type in ['害怕', '生气']:
                return {
                    'should_execute': False,
                    'reject_reason': self._generate_reject_response(
                        command,
                        '我现在的心情不太好'
                    ),
                    'action_type': None,
                    'vocabulary_constraints': list(ALL_VOCABULARY),
                    'safety_score': 0.5
                }
                
            # 3. 检查历史命令
            if not self._check_command_history(command):
                return {
                    'should_execute': False,
                    'reject_reason': self._generate_reject_response(
                        command,
                        '这个动作最近做得太频繁了'
                    ),
                    'action_type': None,
                    'vocabulary_constraints': list(ALL_VOCABULARY),
                    'safety_score': 0.5
                }
                
            # 4. 生成执行决策
            command_type = CommandType(command)
            command_params = get_command_params(command_type)
            
            # 更新命令历史
            self._update_command_history(command)
            
            return {
                'should_execute': True,
                'action_type': command_params['action'],
                'response_template': self._select_response_template(
                    command_params['response_templates'],
                    emotion
                ),
                'vocabulary_constraints': list(ALL_VOCABULARY),
                'safety_score': safety_result['score']
            }
            
        except Exception as e:
            logger.error(f"处理指令失败: {e}")
            return {
                'should_execute': False,
                'reject_reason': "抱歉,我无法处理这个指令",
                'action_type': None,
                'vocabulary_constraints': list(ALL_VOCABULARY),
                'safety_score': 0.0
            }
        
    def _generate_chat_decision(self,
                              context: Dict,
                              emotion: Dict) -> Dict:
        """
        生成对话类型的决策
        
        Args:
            context: 上下文信息
            emotion: 情绪信息
            
        Returns:
            Dict: 决策结果
        """
        try:
            # 根据上下文类型选择合适的响应模板
            context_type = context.get('context_type', 'chat')
            if context_type == 'greeting':
                response_type = 'greeting'
            elif context_type == 'question':
                response_type = 'question'
            else:
                response_type = 'chat'
                
            # 获取可用的响应词汇
            vocabulary_constraints = self._get_vocabulary_constraints(
                response_type,
                emotion
            )
            
            return {
                'should_execute': True,
                'action_type': None,
                'vocabulary_constraints': vocabulary_constraints
            }
            
        except Exception as e:
            logger.error(f"生成对话决策失败: {e}")
            return {
                'should_execute': True,
                'action_type': None,
                'vocabulary_constraints': list(ALL_VOCABULARY)
            }
        
    def _check_command_safety(self,
                            command: str,
                            context: Dict) -> Dict:
        """
        检查指令的安全性
        
        Args:
            command: 指令内容
            context: 上下文信息
            
        Returns:
            Dict: 安全检查结果
        """
        result = {
            'is_safe': True,
            'reason': None,
            'score': 1.0
        }
        
        try:
            # 1. 检查是否需要安全检查
            command_type = CommandType(command)
            if not needs_safety_check(command_type):
                return result
                
            # 2. 检查上下文是否安全
            if not context.get('is_safe', True):
                result.update({
                    'is_safe': False,
                    'reason': '当前环境不安全',
                    'score': 0.0
                })
                return result
                
            # 3. 检查是否包含危险关键词
            text = context.get('original_text', '')
            danger_words = [word for word in self.danger_keywords if word in text]
            if danger_words:
                result.update({
                    'is_safe': False,
                    'reason': f'包含危险词汇: {", ".join(danger_words)}',
                    'score': 0.0
                })
                return result
                
            # 4. 检查场景信息
            scene = context.get('scene', {})
            
            # 4.1 检查障碍物
            if scene.get('has_obstacles', False):
                obstacles = scene.get('obstacles', [])
                if obstacles:
                    result.update({
                        'is_safe': False,
                        'reason': f'前方有障碍物',
                        'score': 0.0
                    })
                    return result
                    
            # 4.2 检查限制区域
            current_area = scene.get('area', '')
            if current_area in self.restricted_areas:
                result.update({
                    'is_safe': False,
                    'reason': f'当前处于限制区域: {current_area}',
                    'score': 0.0
                })
                return result
                
            # 4.3 检查速度限制
            if 'speed' in scene:
                try:
                    speed = float(scene['speed'])
                    max_speed = self.config['safety']['max_speed']
                    if speed > max_speed:
                        result.update({
                            'is_safe': False,
                            'reason': f'速度超过限制',
                            'score': 0.0
                        })
                        return result
                except:
                    pass
                    
            # 5. 计算安全得分
            safety_checks = self.config['safety']['required_checks']
            passed_checks = sum(
                1 for check in safety_checks
                if self._pass_safety_check(check, scene)
            )
            result['score'] = passed_checks / len(safety_checks)
            
            return result
            
        except Exception as e:
            logger.error(f"安全检查失败: {e}")
            return {
                'is_safe': False,
                'reason': "无法完成安全检查",
                'score': 0.0
            }
        
    def _pass_safety_check(self, check_type: str, scene: Dict) -> bool:
        """
        执行具体的安全检查
        
        Args:
            check_type: 检查类型
            scene: 场景信息
            
        Returns:
            bool: 是否通过检查
        """
        try:
            if check_type == 'distance':
                min_distance = self.config['safety']['min_obstacle_distance']
                current_distance = scene.get('obstacle_distance', float('inf'))
                return current_distance >= min_distance
                
            elif check_type == 'speed':
                max_speed = self.config['safety']['max_speed']
                current_speed = scene.get('speed', 0.0)
                return current_speed <= max_speed
                
            elif check_type == 'obstacles':
                return not scene.get('has_obstacles', False)
                
            elif check_type == 'area':
                current_area = scene.get('area', '')
                return current_area not in self.restricted_areas
                
            return True
            
        except Exception as e:
            logger.error(f"执行安全检查失败: {e}")
            return False
        
    def _check_command_history(self, command: str) -> bool:
        """
        检查命令历史
        
        Args:
            command: 当前命令
            
        Returns:
            bool: 是否允许执行
        """
        # 检查最近的命令历史中是否有过多重复
        recent_commands = self.command_history[-3:]  # 检查最近3条命令
        return recent_commands.count(command) < 2  # 不允许在最近3条命令中重复超过2次
        
    def _update_command_history(self, command: str):
        """
        更新命令历史
        
        Args:
            command: 执行的命令
        """
        self.command_history.append(command)
        if len(self.command_history) > self.max_history_size:
            self.command_history.pop(0)
            
    def _select_response_template(self,
                                templates: List[str],
                                emotion: Dict) -> str:
        """
        选择合适的响应模板
        
        Args:
            templates: 可用的响应模板列表
            emotion: 情绪信息
            
        Returns:
            str: 选中的响应模板
        """
        # 根据情绪类型选择合适的模板
        emotion_type = emotion['emotion_type']
        suitable_templates = []
        
        for template in templates:
            # 检查模板是否适合当前情绪
            if self._is_template_suitable(template, emotion_type):
                suitable_templates.append(template)
                
        if suitable_templates:
            return random.choice(suitable_templates)
        return random.choice(templates)
        
    def _is_template_suitable(self,
                            template: str,
                            emotion_type: str) -> bool:
        """
        检查响应模板是否适合当前情绪
        
        Args:
            template: 响应模板
            emotion_type: 情绪类型
            
        Returns:
            bool: 是否适合
        """
        # 这里可以实现更复杂的匹配逻辑
        return True
        
    def _get_vocabulary_constraints(self,
                                  response_type: str,
                                  emotion: Dict) -> List[str]:
        """
        获取词汇约束
        
        Args:
            response_type: 响应类型
            emotion: 情绪信息
            
        Returns:
            List[str]: 可用词汇列表
        """
        if not self.config:
            return list(ALL_VOCABULARY)
            
        try:
            # 获取响应类型对应的词汇
            responses = self.config['vocabulary']['responses']
            if response_type in responses:
                return responses[response_type]
            return list(ALL_VOCABULARY)
            
        except Exception as e:
            logger.error(f"获取词汇约束失败: {e}")
            return list(ALL_VOCABULARY)
        
    def _select_best_command_result(self,
                                  results: List[Dict]) -> Dict:
        """
        从多个命令结果中选择最佳结果
        
        Args:
            results: 命令处理结果列表
            
        Returns:
            Dict: 选中的结果
        """
        if not results:
            return {
                'should_execute': False,
                'reject_reason': "没有可用的命令",
                'action_type': None,
                'vocabulary_constraints': list(ALL_VOCABULARY)
            }
            
        # 优先选择可执行的命令
        executable_results = [
            r for r in results
            if r['should_execute']
        ]
        
        if executable_results:
            # 选择安全得分最高的结果
            return max(
                executable_results,
                key=lambda x: x.get('safety_score', 0.0)
            )
            
        # 如果没有可执行的命令,返回第一个结果
        return results[0]
        
    def _generate_reject_response(self,
                                command: str,
                                reason: str) -> str:
        """
        生成拒绝响应
        
        Args:
            command: 指令内容
            reason: 拒绝原因
            
        Returns:
            str: 拒绝响应文本
        """
        templates = [
            f"对不起,{reason}",
            f"抱歉,{reason}",
            f"我不能{command},{reason}",
            f"现在不能{command},因为{reason}",
            f"执行{command}可能有风险,{reason}"
        ]
        return random.choice(templates) 