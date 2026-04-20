from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict

# ===== State =====
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ===== 大脑 =====
llm = ChatOpenAI(
    model="qwen-plus",
    api_key="sk-829f5b9d060841edafc109c026cc90f4",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ===== 工具 =====
search = DuckDuckGoSearchRun()
tools = [search]

# 把工具绑定给大脑，让它知道可以用搜索
llm_with_tools = llm.bind_tools(tools)

# ===== 节点 =====
# 思考节点：AI 决定回答还是去搜索
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 工具节点：真正执行搜索
tool_node = ToolNode(tools)

# ===== 搭建图 =====
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

# 关键：思考完之后，需要搜索就去 tools，否则直接结束
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")  # 搜完再回来思考

agent = graph_builder.compile()

# ===== 运行 =====
print("Agent 启动！输入 quit 退出")
while True:
    user_input = input("你：")
    if user_input == "quit":
        break
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print("AI：", result["messages"][-1].content)