import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from utils import log_title


class ToolCall:
    """工具调用类"""
    def __init__(self, id: str, function: Dict[str, str]):
        self.id = id
        self.function = function  # {'name': str, 'arguments': str}


class ChatOpenAI:
    """OpenAI聊天封装类"""
    
    def __init__(self, model: str, system_prompt: str = '', tools: List[Dict] = None, context: str = ''):
        """
        初始化ChatOpenAI
        
        参数:
            model: 模型名称
            system_prompt: 系统提示词
            tools: 工具列表
            context: 上下文
        """
        self.llm = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        self.model = model
        self.tools = tools or []
        self.messages: List[Dict[str, Any]] = []
        
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        if context:
            self.messages.append({"role": "user", "content": context})

    async def chat(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        发送聊天消息
        
        参数:
            prompt: 用户输入的提示词
            
        返回:
            包含content和tool_calls的字典
        """
        log_title('CHAT')
        
        if prompt:
            self.messages.append({"role": "user", "content": prompt})
        
        stream = self.llm.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True,
            tools=self._get_tools_definition() if self.tools else None
        )
        
        content = ""
        tool_calls: List[Dict[str, Any]] = []
        
        log_title('RESPONSE')
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            
            # 处理普通内容
            if delta.content:
                content_chunk = delta.content or ""
                content += content_chunk
                print(content_chunk, end='', flush=True)
            
            # 处理工具调用
            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    # 第一次要创建一个toolCall
                    while len(tool_calls) <= tool_call_chunk.index:
                        tool_calls.append({
                            'id': '',
                            'function': {'name': '', 'arguments': ''}
                        })
                    
                    current_call = tool_calls[tool_call_chunk.index]
                    
                    if tool_call_chunk.id:
                        current_call['id'] += tool_call_chunk.id
                    if tool_call_chunk.function and tool_call_chunk.function.name:
                        current_call['function']['name'] += tool_call_chunk.function.name
                    if tool_call_chunk.function and tool_call_chunk.function.arguments:
                        current_call['function']['arguments'] += tool_call_chunk.function.arguments
        
        print()  # 换行
        
        # 将消息添加到历史记录
        tool_calls_formatted = [
            {
                'id': call['id'],
                'type': 'function',
                'function': call['function']
            }
            for call in tool_calls
        ] if tool_calls else None
        
        self.messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls_formatted
        })
        
        return {
            'content': content,
            'tool_calls': tool_calls
        }

    def append_tool_result(self, tool_call_id: str, tool_output: str):
        """
        添加工具执行结果到消息历史
        
        参数:
            tool_call_id: 工具调用ID
            tool_output: 工具输出结果
        """
        self.messages.append({
            "role": "tool",
            "content": tool_output,
            "tool_call_id": tool_call_id
        })

    def _get_tools_definition(self) -> List[Dict[str, Any]]:
        """
        获取工具定义
        
        返回:
            OpenAI格式的工具定义列表
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": tool['inputSchema']
                }
            }
            for tool in self.tools
        ]
