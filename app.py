"""
Web服务器
提供Web界面和WebSocket通信
"""

import os
import json
import asyncio
from typing import List, Dict
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from main import EmotionDialogueSystem

# 创建FastAPI应用
app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 存储WebSocket连接
websocket_connections: List[WebSocket] = []

# 创建对话系统实例
model_type = os.getenv('MODEL_TYPE', 'ollama')
dialogue_system = EmotionDialogueSystem(model_type)

@app.get("/")
async def get_index():
    """返回主页"""
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接处理"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理输入
            response = await dialogue_system.process_input(
                message['text'],
                message.get('scene_info')
            )
            
            # 发送响应
            await websocket.send_json(response)
            
    except Exception as e:
        print(f"WebSocket错误: {str(e)}")
    finally:
        websocket_connections.remove(websocket)

def run():
    """启动服务器"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run() 