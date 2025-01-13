"""
日志工具
提供统一的日志记录功能
"""

import os
import logging
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional
import json
from datetime import datetime

class Logger:
    """日志记录器"""
    
    _instance: Optional['Logger'] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化日志记录器"""
        if hasattr(self, 'logger'):
            return
            
        # 创建日志目录
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # 创建日志记录器
        self.logger = logging.getLogger('emotion_dialogue')
        self.logger.setLevel(logging.DEBUG)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - '
            '%(filename)s:%(lineno)d - %(funcName)s - '
            '%(message)s'
        )
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 创建文件处理器(按大小轮转)
        file_handler = RotatingFileHandler(
            'logs/app.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 创建错误日志处理器(按时间轮转)
        error_handler = TimedRotatingFileHandler(
            'logs/error.log',
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # 创建性能日志处理器
        perf_handler = RotatingFileHandler(
            'logs/performance.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(perf_handler)
        
        # 初始化日志清理器
        self.cleaner = LogCleaner('logs')
        
    def debug(self, message: str, **kwargs):
        """记录调试信息"""
        self._log('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录一般信息"""
        self._log('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告信息"""
        self._log('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误信息"""
        self._log('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误信息"""
        self._log('critical', message, **kwargs)
        
    def performance(self, operation: str, duration: float, **kwargs):
        """记录性能信息"""
        perf_data = {
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.logger.info(
            f"Performance: {json.dumps(perf_data, ensure_ascii=False)}",
            extra={'type': 'performance'}
        )
        
    def _log(self, level: str, message: str, **kwargs):
        """统一的日志记录方法"""
        # 构建日志上下文
        context = {
            'timestamp': datetime.now().isoformat(),
            'type': kwargs.get('type', 'general'),
            'module': kwargs.get('module', 'unknown'),
            'details': kwargs.get('details', {})
        }
        
        # 格式化消息
        formatted_message = (
            f"{message} - Context: {json.dumps(context, ensure_ascii=False)}"
        )
        
        # 记录日志
        getattr(self.logger, level)(formatted_message)
        
class LogCleaner:
    """日志清理器"""
    
    def __init__(self, log_dir: str):
        """初始化清理器"""
        self.log_dir = log_dir
        self.max_total_size = 100 * 1024 * 1024  # 100MB
        self.max_file_age = 30 * 24 * 60 * 60  # 30天
        self.last_cleanup = 0
        self.cleanup_interval = 24 * 60 * 60  # 24小时
        
    def cleanup(self):
        """执行日志清理"""
        current_time = time.time()
        
        # 检查是否需要清理
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        try:
            total_size = 0
            files_info = []
            
            # 收集日志文件信息
            for filename in os.listdir(self.log_dir):
                if not filename.endswith('.log'):
                    continue
                    
                filepath = os.path.join(self.log_dir, filename)
                file_stat = os.stat(filepath)
                
                files_info.append({
                    'path': filepath,
                    'size': file_stat.st_size,
                    'mtime': file_stat.st_mtime
                })
                
                total_size += file_stat.st_size
                
            # 如果总大小超过限制,删除最旧的文件
            if total_size > self.max_total_size:
                # 按修改时间排序
                files_info.sort(key=lambda x: x['mtime'])
                
                # 删除文件直到总大小小于限制
                while total_size > self.max_total_size and files_info:
                    file_info = files_info.pop(0)
                    try:
                        os.remove(file_info['path'])
                        total_size -= file_info['size']
                    except:
                        pass
                        
            # 删除过期的文件
            current_time = time.time()
            for file_info in files_info:
                if current_time - file_info['mtime'] > self.max_file_age:
                    try:
                        os.remove(file_info['path'])
                    except:
                        pass
                        
            self.last_cleanup = current_time
            
        except Exception as e:
            print(f"日志清理失败: {e}")
            
# 创建全局日志实例
logger = Logger()