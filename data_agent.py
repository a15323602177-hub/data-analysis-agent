import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict

# ===== 大脑 =====
llm = ChatOpenAI(
    model="qwen-plus",
    api_key="sk-829f5b9d060841edafc109c026cc90f4",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ===== 全局变量：存储当前数据 =====
current_df = None

# ===== 工具1：读取数据 =====
@tool
def load_data(filepath: str) -> str:
    """读取 CSV 文件，返回数据基本信息。输入文件路径。"""
    global current_df
    current_df = pd.read_csv(filepath)
    info = f"""
数据加载成功！
- 行数：{current_df.shape[0]}
- 列数：{current_df.shape[1]}
- 列名：{list(current_df.columns)}
- 前3行数据：
{current_df.head(3).to_string()}
    """
    return info

# ===== 工具2：基础统计分析 =====
@tool
def analyze_data(question: str) -> str:
    """对已加载的数据做统计分析。输入你想分析的问题。"""
    global current_df
    if current_df is None:
        return "请先加载数据！"
    stats = f"""
基础统计信息：
{current_df.describe().to_string()}

缺失值情况：
{current_df.isnull().sum().to_string()}

数据类型：
{current_df.dtypes.to_string()}
    """
    return stats

# ===== 工具3：画图 =====
@tool
def plot_chart(column: str) -> str:
    """对指定列画图分析。输入列名。"""
    global current_df
    if current_df is None:
        return "请先加载数据！"
    if column not in current_df.columns:
        return f"列名 {column} 不存在，可用列名：{list(current_df.columns)}"

    plt.figure(figsize=(8, 4))
    if current_df[column].dtype == 'object':
        current_df[column].value_counts().plot(kind='bar')
    else:
        current_df[column].hist(bins=20)

    plt.title(f'{column} 分布图')
    plt.tight_layout()
    plt.savefig(f'{column}_chart.png')
    plt.close()
    return f"{column} 的图表已保存为 {column}_chart.png"

# ===== State =====
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ===== 节点 =====
tools = [load_data, analyze_data, plot_chart]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ===== 搭建图 =====
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
agent = graph_builder.compile()

# ===== 运行 =====
print("数据分析 Agent 启动！输入 quit 退出")
print("提示：先告诉我数据文件的路径，比如：帮我分析 /Users/你的名字/Desktop/train.csv")
while True:
    user_input = input("你：")
    if user_input == "quit":
        break
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })
    print("AI：", result["messages"][-1].content)
    print()