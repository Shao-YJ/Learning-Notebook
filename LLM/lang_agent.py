from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv

load_dotenv()

def create_agent(llm, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个协作的AI助手，负责执行分配的任务并提供有用的输出。"
                "在完成任务后，请在回复最后加上'FINAL ANSWER:'。以下是你的任务:{system_message}"
            ),
            MessagesPlaceholder(variable_name = 'messages')
        ]
    )
    prompt = prompt.partial(system_message = system_message)
    return prompt | llm

llm = ChatOpenAI(model='gpt-4', temperature = 0)
agent_1 = create_agent(llm, "生成10个随机数字")
agent_2 = create_agent(llm, "请将提供的数字每个都乘以10")


from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: List[BaseMessage]
    sender: str

graph_builder = StateGraph(AgentState)

# 添加节点处理函数
def call_agent_1(state):
    response = agent_1.invoke(state)
    # 更新状态
    return {
        "messages": state["messages"] + [response],
        "sender": "agent_1"
    }

def call_agent_2(state):
    response = agent_2.invoke(state)
    # 更新状态
    return {
        "messages": state["messages"] + [response],
        "sender": "agent_2"
    }

# 添加节点
graph_builder.add_node("agent_1", call_agent_1)
graph_builder.add_node("agent_2", call_agent_2)

# 简化的路由函数
def simple_router(state):
    # 访问字典的键而不是属性
    messages = state["messages"]
    
    if not messages:
        return "agent_1"
    
    last_message = messages[-1]
    current_sender = state["sender"]

    # 如果包含FINAL ANSWER，则转到下一个agent或结束
    if "FINAL ANSWER:" in last_message.content:
        if current_sender == "agent_1":
            return "agent_2"
        else:
            return END
    else:
        # 如果没有FINAL ANSWER，继续当前agent（但这通常不应该发生）
        if current_sender == "agent_1":
            return "agent_1"
        else:
            return "agent_2"

# 设置边
graph_builder.add_conditional_edges("agent_1", simple_router)
graph_builder.add_conditional_edges("agent_2", simple_router)

graph_builder.set_entry_point("agent_1")

graph = graph_builder.compile()

# 运行
initial_state = {
    "messages": [HumanMessage(content="请生成10个随机数字，并将每个数乘以10")],
    "sender": "agent_1"
}

for event in graph.stream(initial_state):
    print(event)
    print("------")