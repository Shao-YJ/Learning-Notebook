import os
import json
from typing import List
import aiohttp
import openai
from utils import log_title
from VectorStore import VectorStore


class EmbeddingRetriever:
    """嵌入检索器,用于文档嵌入和检索"""
    
    def __init__(self, embedding_model: str):
        """
        初始化嵌入检索器
        
        参数:
            embedding_model: 嵌入模型名称
        """
        self.embedding_model = embedding_model
        self.vector_store = VectorStore()

    async def embed_document(self, document: str) -> List[float]:
        """
        嵌入文档
        
        参数:
            document: 要嵌入的文档内容
            
        返回:
            文档的向量表示
        """
        log_title('EMBEDDING DOCUMENT')
        embedding = await self._embed(document)
        await self.vector_store.add_embedding(embedding, document)
        return embedding

    async def embed_query(self, query: str) -> List[float]:
        """
        嵌入查询
        
        参数:
            query: 查询文本
            
        返回:
            查询的向量表示
        """
        log_title('EMBEDDING QUERY')
        embedding = await self._embed(query)
        return embedding

    async def _embed(self, document: str) -> List[float]:
        """
        调用嵌入API
        
        参数:
            document: 要嵌入的文本
            
        返回:
            向量表示
        """
        # embedding_base_url = os.getenv('OPENAI_BASE_URL')
        # embedding_key = os.getenv('OEPNAI_API_KEY')

        client = openai.AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL')
        )
        
        try:
            response =  await client.embeddings.create(
                model = "Private-Qwen3-Embedding-4B",
                input = document
            )
            
            embedding = response.data[0].embedding
            print(f"Embedding length: {len(embedding)}")
            return embedding
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return []
        

    async def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        检索最相关的文档
        
        参数:
            query: 查询文本
            top_k: 返回前k个最相关的文档
            
        返回:
            最相关的文档列表
        """
        query_embedding = await self.embed_query(query)
        return await self.vector_store.search(query_embedding, top_k)
