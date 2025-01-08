"""
# 情感对话系统

这是一个基于大语言模型的情感对话系统，能够进行自然的对话交互，并表现出适当的情感状态。

## 系统架构

### 整体架构图
```
+------------------+     +------------------+     +------------------+
|     Web界面      |     |    FastAPI服务    |     |   对话系统核心    |
|  (HTML/JS/CSS)   |<--->|   (server.py)    |<--->|   (main.py)     |
+------------------+     +------------------+     +------------------+
                                                         |
                                                         v
+------------------+     +------------------+     +------------------+
|    配置管理       |     |    模型服务       |     |    日志系统      |
|  (config/*.py)   |<--->|  (services/*.py) |     |  (logging)      |
+------------------+     +------------------+     +------------------+
```

### 核心组件说明

1. DialogueSystem (main.py)
```
核心功能：
├── 初始化
│   ├── 配置加载 (_load_config)
│   ├── 缓存初始化 (cache)
│   └── 模型服务初始化 (initialize)
├── 对话处理
│   ├── 输入处理 (process_input)
│   ├── 安全检查 (_check_safety)
│   ├── 上下文分析 (_analyze_context)
│   └── 响应生成 (_generate_response)
├── 情感管理
│   ├── 情感分析 (_analyze_emotion)
│   ├── 情感更新 (_update_emotion)
│   └── 情感历史记录 (emotion_history)
└── 系统状态
    ├── 模型状态
    ├── 情感状态
    └── 场景状态

主要属性：
├── config: 系统配置对象
├── model_service: 模型服务实例
├── cache: 缓存数据
│   ├── memory: 对话记忆
│   ├── emotion: 情感状态
│   ├── scene: 场景状态
│   └── system: 系统状态
└── _initialized: 初始化状态标志
```

2. Web服务 (server.py)
```
API端点：
├── GET /: 主页
├── GET /settings: 配置页面
├── GET /api/config: 获取配置
├── POST /api/config: 更新配置
├── GET /api/status: 获取状态
├── POST /api/chat: 处理对话
├── GET /api/models/*: 获取模型信息
├── GET /api/logs: 获取系统日志
└── WebSocket /ws: 实时通信

服务功能：
├── 请求处理
│   ├── 输入验证
│   ├── 响应格式化
│   └── 错误处理
├── 状态管理
│   ├── 系统状态更新
│   ├── 配置管理
│   └── 日志记录
└── 服务生命周期
    ├── 启动初始化
    └── 关闭清理
```

3. 配置管理
```
配置文件 (config.json):
├── model: 模型配置
│   ├── model_type: 模型类型(ollama/openai/other)
│   ├── model_name: 模型名称
│   ├── api_key: API密钥
│   ├── api_base: API地址
│   ├── temperature: 温度参数
│   ├── top_k: Top-K采样参数
│   ├── top_p: Top-P采样参数
│   ├── repeat_penalty: 重复惩罚参数
│   ├── num_predict: 预测token数量
│   └── context_window: 上下文窗口大小
├── emotion: 情感配置
│   ├── default_emotion: 默认情感
│   ├── speech_speed_range: 语速范围
│   ├── volume_range: 音量范围
│   ├── pitch_range: 音调范围
│   └── rules: 情感规则
│       ├── triggers: 情感触发词
│       ├── transitions: 情感转换规则
│       └── intensities: 情感强度参数
├── safety: 安全配置
│   ├── enable_safety_check: 启用安全检查
│   ├── min_obstacle_distance: 最小障碍物距离
│   ├── danger_keywords: 危险关键词
│   └── content_filters: 内容过滤规则
└── vocabulary: 词汇配置
    ├── emotions: 情感词汇
    ├── actions: 动作词汇
    └── responses: 响应词汇
```

### 数据结构

1. 对话请求
```json
{
    "message": "用户输入文本",
    "timestamp": 1234567890,
    "scene_info": {
        "obstacles": [],
        "temperature": null,
        "lighting": "未检测",
        "timestamp": 1234567890
    }
}
```

2. 对话响应
```json
{
    "success": true,
    "response": {
        "text": "响应文本",
        "emotion_type": "情感类型",
        "action": "动作",
        "speech_speed": 1.0,
        "volume": 1.0,
        "pitch": 1.0
    },
    "error": null
}
```

3. 系统状态
```json
{
    "emotion": {
        "type": "情感类型",
        "speech_speed": 1.0,
        "volume": 1.0,
        "pitch": 1.0
    },
    "scene": {
        "obstacles": "未检测",
        "temperature": "未检测",
        "lighting": "未检测",
        "safety": "未检测"
    },
    "system": {
        "model_type": "模型类型",
        "model_name": "模型名称",
        "last_response_time": 1234567890,
        "status": "系统状态"
    }
}
```

### 错误处理

1. 错误类型
```
系统错误：
├── 初始化错误
│   ├── 配置加载失败
│   ├── 模型初始化失败
│   └── 服务启动失败
├── 运行时错误
│   ├── API调用失败
│   ├── 响应解析失败
│   └── 状态更新失败
└── 安全错误
    ├── 输入验证失败
    ├── 安全检查失败
    └── 内容过滤失败

错误处理流程：
├── 错误捕获
│   ├── 错误日志记录
│   ├── 错误分类处理
│   └── 错误信息格式化
├── 错误恢复
│   ├── 状态回滚
│   ├── 服务重试
│   └── 降级处理
└── 错误反馈
    ├── 用户提示
    └── 监控报警
```

### 日志系统

1. 日志配置
```python
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler'
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/dialogue.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
})
```

2. 日志记录内容
```
记录类型：
├── 系统日志
│   ├── 启动关闭
│   ├── 配置变更
│   └── 状态更新
├── 对话日志
│   ├── 用户输入
│   ├── 系统响应
│   └── 情感变化
├── 错误日志
│   ├── 异常信息
│   ├── 堆栈跟踪
│   └── 错误上下文
└── 性能日志
    ├── 响应时间
    ├── 资源使用
    └── API调用统计
```

### 部署说明

1. 环境要求
```
系统要求：
├── Python 3.8+
├── Node.js 14+
└── 系统依赖
    ├── uvicorn
    ├── fastapi
    ├── aiohttp
    └── pydantic

配置文件：
├── config.json: 系统配置
├── .env: 环境变量
└── logging.conf: 日志配置
```

2. 安装步骤
```
1. 克隆代码仓库
git clone [仓库地址]

2. 安装依赖
pip install -r requirements.txt

3. 配置环境
cp config.example.json config/config.json
cp .env.example .env

4. 修改配置
- 编辑 config/config.json 设置模型参数
- 编辑 .env 设置环境变量

5. 启动服务
python server.py
```

3. 开发指南
```
项目结构：
├── main.py: 核心系统实现
├── server.py: Web服务实现
├── config/: 配置文件目录
├── services/: 模型服务实现
├── static/: 前端静态文件
└── tests/: 测试用例

开发流程：
1. 创建新分支
2. 编写代码和测试
3. 运行测试确保通过
4. 提交代码和文档
5. 发起合并请求
```
"""
# emotion_dialogue_system
