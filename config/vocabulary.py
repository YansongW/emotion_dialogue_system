"""
词汇配置文件
定义系统使用的核心词汇库
"""

# 物品名称
OBJECTS = [
    "苹果", "书本", "椅子", "水杯", "手机",
    "电脑", "眼镜", "钥匙", "衣服", "鞋子"
]

# 疑问词
QUESTIONS = [
    "是吗", "真的吗", "什么", "为什么", "怎么样",
    "在哪里", "谁", "多少", "可以吗", "好吗"
]

# 问候语
GREETINGS = [
    "你好", "早上好", "下午好", "晚上好", "再见"
]

# 态度词
ATTITUDES = [
    "好的", "不好", "喜欢", "不喜欢", "可以",
    "不可以", "同意", "不同意", "明白", "不明白"
]

# 形容词
ADJECTIVES = [
    "大", "小", "高", "低", "快",
    "慢", "热", "冷", "好", "坏",
    "多", "少", "远", "近", "新"
]

# 所有词汇的集合
ALL_VOCABULARY = set(
    OBJECTS +
    QUESTIONS +
    GREETINGS +
    ATTITUDES +
    ADJECTIVES
)

def is_valid_word(word: str) -> bool:
    """
    检查词汇是否在核心词汇库中
    
    Args:
        word: 待检查的词汇
        
    Returns:
        bool: 是否是有效词汇
    """
    return word in ALL_VOCABULARY 