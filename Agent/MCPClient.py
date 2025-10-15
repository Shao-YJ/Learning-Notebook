from typing import List, Dict, Any, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, name: str, command: str, args: List[str], version: Optional[str] = None):
        """
        初始化MCP客户端
        
        参数:
            name: MCP服务器名称
            command: 启动命令
            args: 命令参数
            version: 版本号
        """
        self.name = name
        self.version = version or "0.0.1"
        self.command = command
        self.args = args
        self.session: Optional[ClientSession] = None
        self.exit_stack = None
        self.tools: List[Dict[str, Any]] = []

    async def init(self):
        """初始化并连接到服务器"""
        await self._connect_to_server()

    async def close(self):
        """关闭MCP客户端连接"""
        if self.exit_stack:
            await self.exit_stack.aclose()

    def get_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        return self.tools

    async def call_tool(self, name: str, params: Dict[str, Any]):
        """调用指定的工具"""
        if not self.session:
            raise Exception("MCP client not initialized")
        
        result = await self.session.call_tool(name=name, arguments=params)
        return result

    async def _connect_to_server(self):
        """连接到MCP服务器"""
        try:
            from contextlib import AsyncExitStack
            
            # 创建服务器参数
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=None
            )
            
            # 使用 AsyncExitStack 管理资源
            self.exit_stack = AsyncExitStack()
            
            # 连接到 stdio 服务器
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport
            
            # 创建客户端会话
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            # 初始化会话
            await self.session.initialize()
            
            # 获取工具列表
            tools_result = await self.session.list_tools()
            self.tools = [
                {
                    'name': tool.name,
                    'description': tool.description,
                    'inputSchema': tool.inputSchema
                }
                for tool in tools_result.tools
            ]
            
            tool_names = [tool['name'] for tool in self.tools]
            print(f"Connected to server with tools: {tool_names}")
            
        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            raise e
