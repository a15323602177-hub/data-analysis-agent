import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
import io

st.set_page_config(page_title="数据分析 Agent", page_icon="📊", layout="wide")
st.title("📊 数据分析 Agent")
st.caption("上传任意 CSV 文件，用自然语言提问，Agent 自动分析并画图")

if "df" not in st.session_state:
    st.session_state.df = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_charts" not in st.session_state:
    st.session_state.last_charts = []

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=st.secrets["ALIYUN_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def build_agent(df, chart_list):

    @tool
    def data_overview() -> str:
        """查看数据的基本信息：行数、列数、列名、数据类型、基础统计、缺失值。"""
        return f"""
数据基本信息：
- 共 {df.shape[0]} 行，{df.shape[1]} 列
- 列名：{list(df.columns)}
- 数据类型：
{df.dtypes.to_string()}

基础统计：
{df.describe().to_string()}

缺失值：
{df.isnull().sum().to_string()}
        """

    @tool
    def plot_relationship(x_column: str, y_column: str) -> str:
        """分析两列之间的关系并画图。自动识别数据类型：
        - 两列都是分类变量：分组柱状图
        - x分类 y数值：分组平均值柱状图
        - x数值 y数值：散点图
        - x数值 y分类：y按x分箱的比例图
        """
        if x_column not in df.columns or y_column not in df.columns:
            return f"列名不存在。可用列名：{list(df.columns)}"
        
        fig, ax = plt.subplots(figsize=(8, 4))
        x_numeric = df[x_column].dtype in ['int64', 'float64']
        y_numeric = df[y_column].dtype in ['int64', 'float64']
        
        if x_numeric and y_numeric:
            ax.scatter(df[x_column], df[y_column], alpha=0.5, color='steelblue')
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(f'{x_column} vs {y_column}')
        elif not x_numeric and y_numeric:
            grouped = df.groupby(x_column)[y_column].mean()
            grouped.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(f'{y_column} by {x_column} (average)')
            ax.set_ylabel(f'avg {y_column}')
            plt.xticks(rotation=0)
        elif x_numeric and not y_numeric:
            bins = pd.cut(df[x_column], bins=8)
            pct = df.groupby(bins, observed=True)[y_column].apply(lambda s: s.value_counts(normalize=True)).unstack().fillna(0)
            pct.index = [f'{int(i.left)}-{int(i.right)}' for i in pct.index]
            pct.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'{y_column} distribution by {x_column}')
            plt.xticks(rotation=0)
        else:
            ct = pd.crosstab(df[x_column], df[y_column], normalize='index') * 100
            ct.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'{y_column} distribution by {x_column} (%)')
            plt.xticks(rotation=0)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_list.append(buf)
        plt.close()
        return f"已生成 {x_column} 与 {y_column} 的关系图。"

    @tool
    def plot_distribution(column: str) -> str:
        """画出单列数据的分布图。数值列自动分箱，分类列显示频次。"""
        if column not in df.columns:
            return f"列名 {column} 不存在。可用列名：{list(df.columns)}"
        fig, ax = plt.subplots(figsize=(8, 4))
        if df[column].dtype in ['int64', 'float64']:
            df[column].dropna().hist(bins=15, ax=ax, color='steelblue', edgecolor='white')
        else:
            df[column].value_counts().head(10).plot(kind='bar', ax=ax, color='steelblue')
            plt.xticks(rotation=0)
        ax.set_title(f'{column} Distribution')
        ax.set_xlabel(column)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_list.append(buf)
        plt.close()
        return f"已生成 {column} 的分布图。"

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tools = [data_overview, plot_relationship, plot_distribution]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    return graph_builder.compile()

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"数据加载成功！共 {st.session_state.df.shape[0]} 行，{st.session_state.df.shape[1]} 列")
        st.dataframe(st.session_state.df.head())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.df is None:
        st.info("👆 请先上传 CSV 文件，再开始提问")
    elif prompt := st.chat_input("比如：分析 Age 和 Survived 的关系"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                charts = []
                agent = build_agent(st.session_state.df, charts)
                result = agent.invoke({
                    "messages": [{"role": "user", "content": prompt}]
                })
                response = result["messages"][-1].content
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.last_charts = charts

with col2:
    st.subheader("📈 图表")
    if st.session_state.last_charts:
        for chart in st.session_state.last_charts:
            st.image(chart)
    else:
        st.info("分析后图表会显示在这里")
