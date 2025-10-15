# 从零搭建智能Agent：理论与实战分享

## 1. 什么是Agent？

Agent（智能体）是能够自主感知环境、做出决策并采取行动的系统。在AI领域，Agent通常具备如下能力：
- 感知（Perception）：获取外部信息，如文本、图片、API数据等。
- 推理与决策（Reasoning & Decision）：基于感知信息和知识库进行推理，做出决策。
- 行动（Action）：执行操作，如调用工具、生成文本、与外部系统交互。
- 学习（Learning）：通过交互不断优化自身行为。

现代大模型Agent通常结合了大语言模型（LLM）、工具调用（Tool Use）、RAG（检索增强生成）等技术。

---

## 2. 项目结构概览

```
Agent/
  Agent.py                # Agent主流程
  ChatOpenAI.py           # LLM对话封装
  EmbeddingRetriever.py   # 嵌入与检索
  MCPClient.py            # 工具/插件客户端
  VectorStore.py          # 向量存储与相似度检索
  utils.py                # 辅助函数
  index.py                # 主入口，流程串联
```

---

## 3. Agent理论与核心模块详解

### 3.1 RAG（检索增强生成）
- 通过`EmbeddingRetriever`将知识库文档转为向量，存入`VectorStore`。
- 用户提问时，将问题转为向量，检索最相关的文档，作为上下文供LLM生成答案。

### 3.2 LLM对话与工具调用
- `ChatOpenAI`负责与OpenAI API通信，支持多轮对话和工具调用（Function Calling）。
- Agent可根据LLM输出自动调用外部工具（如MCP插件），并将结果反馈给LLM，形成闭环。

### 3.3 工具/插件系统（MCPClient）
- 通过`MCPClient`可动态注册、发现和调用外部工具（如爬虫、文件系统等），实现Agent能力扩展。

---

## 4. 主要代码模块解读

### 4.1 Agent主流程（Agent.py）
- 初始化：注册所有MCP工具，构建LLM。
- invoke：多轮对话，自动检测LLM输出中的工具调用请求，自动分发到对应MCPClient，结果回传LLM。

### 4.2 LLM对话与工具调用（ChatOpenAI.py）
- 支持system prompt、上下文注入。
- 支持OpenAI Function Calling协议，自动解析和分发工具调用。
- 工具调用结果可回写对话历史，实现多轮推理。

### 4.3 RAG检索模块（EmbeddingRetriever.py & VectorStore.py）
- 文档和查询均通过嵌入模型转为向量。
- `VectorStore`支持高效的余弦相似度检索。
- 检索结果作为上下文供LLM生成答案。

### 4.4 工具/插件客户端（MCPClient.py）
- 支持异步连接、发现工具、调用工具。
- 工具描述、参数自动注入到LLM。

---

## 5. 主流程（index.py）

1. 读取知识库文档，全部嵌入并存储。
2. 用户输入问题，检索最相关文档，拼接为上下文。
3. 初始化Agent，注册所有MCP工具。
4. Agent多轮对话，自动调用工具，最终输出结果。

---

## 6. 关键代码片段

### Agent调用流程
```python
agent = Agent('gpt-4o-mini', [fetch_mcp, file_mcp], '', context)
await agent.init()
result = await agent.invoke(TASK)
```

### RAG检索
```python
embedding_retriever = EmbeddingRetriever("Private-Qwen3-Embedding-4B")
await embedding_retriever.embed_document(content)
context_list = await embedding_retriever.retrieve(TASK, 3)
```

### 工具调用
```python
result = await mcp.call_tool(tool_call['function']['name'], json.loads(tool_call['function']['arguments']))
```

---

## 7. Agent应用场景
- 智能问答/知识库助手
- 自动化数据分析与报告
- 多工具协作的智能体（如自动爬取、写入、分析等）

---

## 8. 总结与展望

本项目实现了一个具备RAG、工具调用、插件扩展能力的智能Agent。未来可扩展更多插件、支持多模态输入、引入更强的推理与记忆机制。

---

> **Q&A**
> 欢迎现场提问与交流！
