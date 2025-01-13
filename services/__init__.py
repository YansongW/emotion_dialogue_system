"""
模型服务包

提供不同类型的大语言模型服务实现
"""

from .model_service import BaseModelService
from .openai_service import OpenAIService
from .ollama_service import OllamaService

__all__ = ['BaseModelService', 'OpenAIService', 'OllamaService'] 