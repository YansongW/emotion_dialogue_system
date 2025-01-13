"""
指令配置文件
定义系统支持的核心指令
"""

from enum import Enum
from typing import Dict, List

class CommandType(Enum):
    """指令类型枚举"""
    COME = "过来"
    TURN = "转身"
    LEFT = "在你左边"
    RIGHT = "在你右边"
    RUN = "快跑"
    STOP = "停下"
    FOLLOW = "跟着我"
    BACK = "后退"
    FORWARD = "向前"
    DANCE = "跳舞"

# 指令参数配置
COMMAND_PARAMS = {
    CommandType.COME: {
        'action': 'move_to_target',
        'safety_check': True,  # 是否需要安全检查
        'keywords': ['过来', '来这里', '到这儿'],  # 触发关键词
        'response_templates': [  # 响应模板
            "好的,我这就过来",
            "马上就到",
            "我来了"
        ]
    },
    CommandType.TURN: {
        'action': 'rotate',
        'safety_check': False,
        'keywords': ['转身', '转过去', '转过来'],
        'response_templates': [
            "好的,我转身了",
            "转过去了",
            "已经转好了"
        ]
    },
    CommandType.LEFT: {
        'action': 'move_left',
        'safety_check': True,
        'keywords': ['左边', '向左', '左面'],
        'response_templates': [
            "好的,向左移动",
            "往左边走",
            "向左边去"
        ]
    },
    CommandType.RIGHT: {
        'action': 'move_right',
        'safety_check': True,
        'keywords': ['右边', '向右', '右面'],
        'response_templates': [
            "好的,向右移动",
            "往右边走",
            "向右边去"
        ]
    },
    CommandType.RUN: {
        'action': 'run',
        'safety_check': True,
        'keywords': ['快跑', '跑起来', '加速'],
        'response_templates': [
            "好的,我开始跑",
            "我跑起来了",
            "加速中"
        ]
    },
    CommandType.STOP: {
        'action': 'stop',
        'safety_check': False,
        'keywords': ['停下', '停止', '别动'],
        'response_templates': [
            "好的,我停下了",
            "已经停止了",
            "不动了"
        ]
    },
    CommandType.FOLLOW: {
        'action': 'follow_target',
        'safety_check': True,
        'keywords': ['跟着我', '跟我来', '跟上'],
        'response_templates': [
            "好的,我跟着你",
            "我跟你走",
            "跟上你了"
        ]
    },
    CommandType.BACK: {
        'action': 'move_backward',
        'safety_check': True,
        'keywords': ['后退', '往后', '向后'],
        'response_templates': [
            "好的,我后退",
            "向后移动",
            "往后走"
        ]
    },
    CommandType.FORWARD: {
        'action': 'move_forward',
        'safety_check': True,
        'keywords': ['向前', '往前', '前进'],
        'response_templates': [
            "好的,我向前",
            "向前移动",
            "往前走"
        ]
    },
    CommandType.DANCE: {
        'action': 'dance',
        'safety_check': False,
        'keywords': ['跳舞', '跳一下', '跳起来'],
        'response_templates': [
            "好的,我跳舞",
            "看我的舞姿",
            "跳起来了"
        ]
    }
}

def get_command_params(command_type: CommandType) -> Dict:
    """
    获取指令参数配置
    
    Args:
        command_type: 指令类型
        
    Returns:
        Dict: 指令参数配置
    """
    return COMMAND_PARAMS[command_type]

def get_command_by_keywords(text: str) -> List[CommandType]:
    """
    根据关键词识别可能的指令类型
    
    Args:
        text: 输入文本
        
    Returns:
        List[CommandType]: 可能的指令类型列表
    """
    possible_commands = []
    for command_type in CommandType:
        keywords = COMMAND_PARAMS[command_type]['keywords']
        if any(keyword in text for keyword in keywords):
            possible_commands.append(command_type)
    return possible_commands

def needs_safety_check(command_type: CommandType) -> bool:
    """
    检查指令是否需要安全检查
    
    Args:
        command_type: 指令类型
        
    Returns:
        bool: 是否需要安全检查
    """
    return COMMAND_PARAMS[command_type]['safety_check'] 