import os
import asyncio
from pathlib import Path
from MCPClient import MCPClient
from Agent import Agent
from EmbeddingRetriever import EmbeddingRetriever
from utils import log_title


URL = 'https://news.ycombinator.com/'
out_path = os.path.join(os.getcwd(), 'output')
TASK = f"""
告诉我Antonette的信息,先从我给你的context中找到相关信息,总结后创作一个关于她的故事
把故事和她的基本信息保存到{out_path}/antonette.md,输出一个漂亮md文件
"""

fetch_mcp = MCPClient("mcp-server-fetch", "uvx", ['mcp-server-fetch'])
file_mcp = MCPClient("mcp-server-file", "npx", ['-y', '@modelcontextprotocol/server-filesystem', out_path])


async def retrieve_context():
    """RAG - 检索相关上下文"""
    embedding_retriever = EmbeddingRetriever("Private-Qwen3-Embedding-4B")
    # knowledge_dir = os.path.join(os.getcwd(), 'knowledge')
    knowledge_dir = '/home/shaoyj/code/Learning-Notebook/Agent/knowledge'
    
    files = os.listdir(knowledge_dir)
    for file in files:
        file_path = os.path.join(knowledge_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        await embedding_retriever.embed_document(content)
    
    context_list = await embedding_retriever.retrieve(TASK, 3)
    context = '\n'.join(context_list)
    log_title('CONTEXT')
    print(context)
    return context


async def main():
    OPENAI_BASE_URL = "https://dd-ai-api.eastmoney.com/v1"
    OPENAI_API_KEY = "sk-acUzWFHVvFTIaINs4fF7Bc2b30F24b45AdC97dAa6aAe8cEa"
    os.environ['OPENAI_BASE_URL'] = OPENAI_BASE_URL
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    # RAG
    context = await retrieve_context()
    
    # Agent
    agent = Agent('gpt-4o-mini', [fetch_mcp, file_mcp], '', context)
    try:
        await agent.init()
        result = await agent.invoke(TASK)
        print(f"\n\n{'='*80}")
        print(f"Final Result:\n{result}")
        print(f"{'='*80}")
    finally:
        await agent.close()


if __name__ == '__main__':
    asyncio.run(main())
