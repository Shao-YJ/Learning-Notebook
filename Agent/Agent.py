from MCPClient import MCPClient
from ChatOpenAI import ChatOpenAI
from utils import log_title
from typing import List


class Agent:
    def __init__(self, model: str, mcp_clients: List[MCPClient], system_prompt: str = '', context: str = ''):
        self.mcp_clients = mcp_clients
        self.llm = None
        self.model = model
        self.system_prompt = system_prompt
        self.context = context

    async def init(self):
        log_title('TOOLS')
        for client in self.mcp_clients:
            await client.init()
        tools = []
        for client in self.mcp_clients:
            tools.extend(client.get_tools())
        self.llm = ChatOpenAI(self.model, self.system_prompt, tools, self.context)

    async def close(self):
        for client in self.mcp_clients:
            await client.close()

    async def invoke(self, prompt: str):
        if not self.llm:
            raise Exception('Agent not initialized')
        
        response = await self.llm.chat(prompt)
        
        while True:
            if len(response['tool_calls']) > 0:
                for tool_call in response['tool_calls']:
                    # 查找对应的MCP客户端
                    mcp = None
                    for client in self.mcp_clients:
                        client_tools = client.get_tools()
                        if any(t['name'] == tool_call['function']['name'] for t in client_tools):
                            mcp = client
                            break
                    
                    if mcp:
                        log_title('TOOL USE')
                        print(f"Calling tool: {tool_call['function']['name']}")
                        print(f"Arguments: {tool_call['function']['arguments']}")
                        
                        import json
                        result = await mcp.call_tool(
                            tool_call['function']['name'],
                            json.loads(tool_call['function']['arguments'])
                        )
                        
                        # 将 CallToolResult 转换为可序列化的格式
                        if hasattr(result, 'content'):
                            # MCP CallToolResult 对象有 content 属性
                            result_content = []
                            for item in result.content:
                                if hasattr(item, 'text'):
                                    result_content.append({'type': 'text', 'text': item.text})
                                elif hasattr(item, 'model_dump'):
                                    result_content.append(item.model_dump())
                                else:
                                    result_content.append(str(item))
                            result_str = json.dumps(result_content)
                        else:
                            result_str = json.dumps(result)
                        
                        print(f"Result: {result_str}")
                        self.llm.append_tool_result(tool_call['id'], result_str)
                    else:
                        self.llm.append_tool_result(tool_call['id'], 'Tool not found')
                
                # 工具调用后,继续对话
                response = await self.llm.chat()
                continue
            
            # 没有工具调用,结束对话
            return response['content']
