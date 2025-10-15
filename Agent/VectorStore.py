from typing import List, Dict
import math


class VectorStoreItem:
    """向量存储项"""
    def __init__(self, embedding: List[float], document: str):
        self.embedding = embedding
        self.document = document


class VectorStore:
    """向量存储类,用于存储和检索文档向量"""
    
    def __init__(self):
        self.vector_store: List[VectorStoreItem] = []

    async def add_embedding(self, embedding: List[float], document: str):
        """添加向量和文档到存储"""
        self.vector_store.append(VectorStoreItem(embedding, document))

    async def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        """
        搜索最相似的文档
        
        参数:
            query_embedding: 查询向量
            top_k: 返回前k个最相似的文档
            
        返回:
            最相似的文档列表
        """
        scored = [
            {
                'document': item.document,
                'score': self._cosine_similarity(query_embedding, item.embedding)
            }
            for item in self.vector_store
        ]
        
        # 按分数降序排序并取前top_k个
        scored.sort(key=lambda x: x['score'], reverse=True)
        top_k_documents = [item['document'] for item in scored[:top_k]]
        
        return top_k_documents

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        参数:
            vec_a: 向量A
            vec_b: 向量B
            
        返回:
            余弦相似度值
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
