"""
Web服务器模块 (Web Server Module)

本模块提供Web服务器功能，包括：
1. Web界面：提供用户交互界面
2. API接口：提供后端服务接口
3. 配置管理：处理系统配置的加载和保存
4. 状态管理：维护系统运行状态
5. 日志管理：记录系统运行日志

主要组件：
- FastAPI应用：提供Web服务
- 静态文件服务：提供前端资源
- API路由：处理各类请求
- 配置管理：处理配置文件操作
- 状态管理：维护系统运行状态
- 日志管理：记录系统运行日志

API端点：
- GET /: 主页
- GET /settings: 配置页面
- GET /api/config: 获取配置
- POST /api/config: 更新配置
- GET /api/status: 获取状态
- POST /api/chat: 处理对话
- GET /api/models/*: 获取模型信息
- GET /api/logs: 获取系统日志

依赖库：
- os: 文件系统操作
- json: JSON数据处理
- time: 时间相关功能
- logging: 日志记录
- subprocess: 进程管理
- typing: 类型注解
- pathlib: 路径处理
- uvicorn: ASGI服务器
- fastapi: Web框架
- pydantic: 数据验证

"""

import os
import json
import time
import logging
import subprocess
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from main import DialogueSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="对话系统")

# 全局变量
dialogue_system = None

# 配置文件路径
CONFIG_FILE = "config/config.json"
LOG_FILE = "system.log"

# 默认配置
DEFAULT_CONFIG = {
    "model": {
        "model_type": "ollama",
        "model_name": "llama2",
        "api_key": "",
        "api_base": "",
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_predict": 50,
        "stop_sequences": [],
        "context_window": 2048
    },
    "emotion": {
        "default_emotion": "平静",
        "speech_speed_range": [0.8, 1.3],
        "volume_range": [0.8, 1.2],
        "pitch_range": [0.9, 1.2],
        "emotion_history_size": 5,
        "rules": {
            "triggers": {
                "高兴": ["表扬", "成功", "帮助"],
                "难过": ["失败", "道歉", "离别"],
                "生气": ["批评", "打扰", "不尊重"],
                "惊讶": ["意外", "突然", "特殊"]
            },
            "transitions": {
                "平静": ["高兴", "难过", "生气", "惊讶"],
                "高兴": ["平静", "惊讶"],
                "难过": ["平静", "生气"],
                "生气": ["平静", "难过"],
                "惊讶": ["平静", "高兴"]
            },
            "intensities": {
                "语速": {"平静": 1.0, "高兴": 1.2, "难过": 0.8, "生气": 1.3, "惊讶": 1.1},
                "音量": {"平静": 1.0, "高兴": 1.1, "难过": 0.9, "生气": 1.2, "惊讶": 1.1},
                "音调": {"平静": 1.0, "高兴": 1.1, "难过": 0.9, "生气": 1.2, "惊讶": 1.1}
            }
        }
    },
    "safety": {
        "enable_safety_check": True,
        "min_obstacle_distance": 1.0,
        "danger_keywords": [
            "撞", "跳", "摔", "打", "踢",
            "伤害", "危险", "破坏", "损坏"
        ],
        "max_response_time": 5000,
        "max_conversation_turns": 50,
        "content_filters": [
            "暴力", "色情", "政治", "宗教", "歧视"
        ]
    },
    "vocabulary": {
        "emotions": {
            "平静": ["平静", "安详", "淡定"],
            "高兴": ["开心", "快乐", "兴奋", "喜悦"],
            "难过": ["伤心", "悲伤", "沮丧", "失落"],
            "生气": ["愤怒", "恼火", "不满", "烦躁"],
            "惊讶": ["吃惊", "震惊", "意外", "诧异"]
        },
        "actions": [
            "点头", "摇头", "微笑", "皱眉", "挥手"
        ],
        "responses": {
            "问候": ["你好", "早上好", "下午好", "晚上好"],
            "告别": ["再见", "拜拜", "下次见", "回头见"],
            "肯定": ["好的", "没问题", "可以", "当然"],
            "否定": ["抱歉", "不行", "不可以", "恐怕不行"]
        }
    },
    "system_prompt": "你是一个对话机器人。在对话中,你需要:\n1. 只使用指定的词汇表达\n2. 表现出适当的情绪\n3. 保持对话的自然性和连贯性\n4. 回答要简短精炼"
}

# 请求模型
class ChatRequest(BaseModel):
    message: str
    timestamp: Optional[int] = None

class ConfigUpdate(BaseModel):
    model: Dict
    emotion: Dict
    safety: Dict
    vocabulary: Dict
    system_prompt: str

class VocabularyUpdate(BaseModel):
    """词汇更新请求"""
    category: str  # emotions, actions, responses
    key: Optional[str]  # 对于responses和emotions需要提供
    values: List[str]

# 系统状态
system_status = {
    "emotion": {
        "type": "未检测",
        "speech_speed": None,
        "volume": None,
        "pitch": None
    },
    "scene": {
        "obstacles": "未检测",
        "temperature": "未检测",
        "lighting": "未检测",
        "safety": "未检测"
    },
    "system": {
        "model_type": "未检测",
        "model_name": "未检测",
        "last_response_time": 0,
        "status": "未检测"
    }
}

# 加载配置
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

# 保存配置
def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

# 更新系统状态
def update_system_status(response):
    if response:
        logger.info(f"更新系统状态: {json.dumps(response, ensure_ascii=False)}")
        
        # 更新情绪状态
        system_status["emotion"]["type"] = response.get("emotion_type", "未检测")
        system_status["emotion"]["speech_speed"] = response.get("speech_speed")
        system_status["emotion"]["volume"] = response.get("volume")
        system_status["emotion"]["pitch"] = response.get("pitch")
    
    # 更新系统状态
    config = load_config()
    system_status["system"]["model_type"] = config.get("model", {}).get("model_type", "未检测")
    system_status["system"]["model_name"] = config.get("model", {}).get("model_name", "未检测")
    system_status["system"]["last_response_time"] = int(time.time() * 1000)
    system_status["system"]["status"] = "正常" if dialogue_system and dialogue_system._initialized else "未检测"
    
    logger.info(f"系统状态已更新: {json.dumps(system_status, ensure_ascii=False)}")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    """获取主页"""
    return FileResponse("static/index.html")

@app.get("/settings")
async def get_settings():
    """获取设置页面"""
    return FileResponse("static/settings.html")

@app.get("/api/config")
async def get_config():
    """获取配置"""
    return load_config()

@app.get("/api/models/ollama")
async def get_ollama_models() -> dict:
    """获取Ollama可用模型列表"""
    try:
        # 检查ollama是否已安装,设置5秒超时
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True,
            text=True,
            timeout=5  # 添加5秒超时
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": "Ollama未安装或无法访问",
                "installed": False,
                "models": []
            }
            
        # 解析模型列表
        models = []
        for line in result.stdout.strip().split('\n')[1:]:  # 跳过标题行
            if line.strip():
                name = line.split()[0]
                models.append(name)
                
        return {
            "success": True,
            "installed": True,
            "models": models
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "检查Ollama超时,请确认服务是否正常运行",
            "installed": False,
            "models": []
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "Ollama未安装",
            "installed": False,
            "models": []
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "installed": False,
            "models": []
        }

@app.get("/api/models/openai")
async def verify_openai_key(api_key: str) -> dict:
    """验证OpenAI API密钥"""
    try:
        # 这里应该添加实际的OpenAI API密钥验证逻辑
        # 为了示例，我们只做简单的格式检查
        if not api_key.startswith("sk-") or len(api_key) < 20:
            return {
                "success": False,
                "error": "无效的API密钥格式"
            }
            
        return {
            "success": True,
            "models": [
                "gpt-4",
                "gpt-4-32k",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """更新配置"""
    try:
        # 验证模型配置
        if config.model["model_type"] == "ollama":
            # 检查ollama是否已安装并且模型是否可用
            ollama_status = await get_ollama_models()
            if not ollama_status["success"]:
                raise ValueError(ollama_status["error"])
            if config.model["model_name"] not in ollama_status["models"]:
                raise ValueError(f"Ollama模型 '{config.model['model_name']}' 未安装")
                
        elif config.model["model_type"] == "openai":
            # 验证OpenAI API密钥
            if not config.model.get("api_key"):
                raise ValueError("使用OpenAI时必须提供API密钥")
            key_status = await verify_openai_key(config.model["api_key"])
            if not key_status["success"]:
                raise ValueError(f"无效的OpenAI API密钥: {key_status['error']}")
                
        elif config.model["model_type"] == "other":
            # 验证其他模型的必要配置
            if not config.model.get("api_base"):
                raise ValueError("使用其他模型时必须提供API基础URL")
        else:
            raise ValueError("不支持的模型类型")
            
        # 清理动作词汇中的多余字符
        if "vocabulary" in config.dict() and "actions" in config.dict()["vocabulary"]:
            config.vocabulary["actions"] = [
                action.strip().split('\n')[0].replace('×', '').strip()
                for action in config.vocabulary["actions"]
            ]
            # 去重
            config.vocabulary["actions"] = list(dict.fromkeys(config.vocabulary["actions"]))
            
        # 获取当前配置
        current_config = load_config()
        
        # 构造新配置，保留所有必要字段
        new_config = current_config.copy()
        config_dict = config.dict()
        
        # 更新各个部分
        new_config["model"].update(config_dict["model"])
        new_config["emotion"].update(config_dict["emotion"])
        new_config["safety"].update(config_dict["safety"])
        new_config["vocabulary"].update(config_dict["vocabulary"])
        new_config["system_prompt"] = config_dict["system_prompt"]
        
        # 保存配置
        save_config(new_config)
        
        return {"success": True}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/vocabulary")
async def update_vocabulary(update: VocabularyUpdate):
    """更新词汇配置"""
    try:
        config = load_config()
        
        if update.category == "emotions":
            if not update.key:
                raise ValueError("更新情绪词汇时必须提供情绪类型")
            config["vocabulary"]["emotions"][update.key] = update.values
            
        elif update.category == "actions":
            config["vocabulary"]["actions"] = update.values
            
        elif update.category == "responses":
            if not update.key:
                raise ValueError("更新响应词汇时必须提供响应类型")
            config["vocabulary"]["responses"][update.key] = update.values
            
        else:
            raise ValueError("无效的词汇类别")
        
        # 保存更新后的配置
        save_config(config)
        
        return {"success": True}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/vocabulary/{category}/{key}")
async def delete_vocabulary(category: str, key: str):
    """删除词汇配置"""
    try:
        config = load_config()
        
        if category == "emotions":
            if key == "平静":
                raise ValueError("不能删除默认情绪'平静'")
            if key in config["vocabulary"]["emotions"]:
                del config["vocabulary"]["emotions"][key]
                
        elif category == "responses":
            if key in config["vocabulary"]["responses"]:
                del config["vocabulary"]["responses"][key]
                
        else:
            raise ValueError("无效的词汇类别")
        
        # 保存更新后的配置
        save_config(config)
        
        return {"success": True}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    return system_status

@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """获取系统日志"""
    try:
        if not os.path.exists(LOG_FILE):
            return {"logs": []}
            
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            # 读取最后N行日志
            logs = f.readlines()[-lines:]
            return {"logs": logs}
            
    except Exception as e:
        logger.error(f"获取日志失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """处理对话请求"""
    try:
        # 记录请求
        logger.info(f"收到对话请求: {json.dumps(request.dict(), ensure_ascii=False)}")
        
        # 获取当前时间戳
        current_time = int(time.time() * 1000)
        
        # 处理用户输入
        response = await dialogue_system.process_input(
            request.message,
            {
                "obstacles": [],
                "temperature": None,
                "lighting": "未检测",
                "timestamp": current_time
            }
        )
        
        # 记录原始响应
        logger.info(f"模型原始响应: {json.dumps(response, ensure_ascii=False)}")
        
        # 解析JSON响应
        if response.get("response"):
            response_data = json.loads(response["response"])
        else:
            response_data = {
                "text": "处理请求失败",
                "emotion_type": "未检测",
                "action": "未检测",
                "speech_speed": None,
                "volume": None,
                "pitch": None
            }
        
        # 记录解析后的响应
        logger.info(f"解析后的响应: {json.dumps(response_data, ensure_ascii=False)}")
        
        # 更新系统状态
        update_system_status(response_data)
        
        return {
            "success": response.get("status") == "success",
            "response": response_data,
            "error": response.get("error", None)
        }
        
    except Exception as e:
        error_msg = f"处理对话请求失败: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": str(e),
            "response": {
                "text": "抱歉，处理您的请求时出现错误",
                "emotion_type": "未检测",
                "action": "未检测",
                "speech_speed": None,
                "volume": None,
                "pitch": None
            }
        }

@app.on_event("startup")
async def startup_event():
    """服务启动时的初始化"""
    global dialogue_system
    try:
        dialogue_system = DialogueSystem()
        await dialogue_system.initialize()
        print("对话系统初始化成功")
    except Exception as e:
        print(f"对话系统初始化失败: {str(e)}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时的清理"""
    global dialogue_system
    dialogue_system = None
    print("对话系统已关闭")

def main():
    """启动服务器"""
    # 确保目录存在
    Path("static").mkdir(exist_ok=True)
    Path("config").mkdir(exist_ok=True)
    
    # 启动服务器
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        lifespan="on"  # 启用lifespan事件
    )

if __name__ == "__main__":
    main() 