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

st.set_page_config(page_title="Data Analysis Agent", page_icon="📊", layout="wide")
st.title("📊 Data Analysis Agent")
st.caption("Upload any CSV file, ask questions in natural language, and the Agent will analyze and visualize your data automatically.")

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

SYSTEM_PROMPT = "You are a data analysis assistant. Always respond in English. Be concise and insight-driven."

def build_agent(df, chart_list):

    @tool
    def data_overview() -> str:
        """Get basic information about the dataset: row count, column names, data types, statistics, and missing values."""
        return f"""
Dataset Overview:
- Rows: {df.shape[0]}, Columns: {df.shape[1]}
- Column names: {list(df.columns)}
- Data types:
{df.dtypes.to_string()}

Basic Statistics:
{df.describe().to_string()}

Missing Values:
{df.isnull().sum().to_string()}
        """

    @tool
    def plot_relationship(x_column: str, y_column: str) -> str:
        """Analyze and visualize the relationship between two columns. Auto-selects chart type:
        - Both numeric: scatter plot
        - Categorical x, numeric y: grouped bar chart (mean)
        - Numeric x, categorical y: binned stacked bar chart
        - Both categorical: stacked bar chart (percentage)
        """
        if x_column not in df.columns or y_column not in df.columns:
            return f"Column not found. Available columns: {list(df.columns)}"

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
            ax.set_title(f'Average {y_column} by {x_column}')
            ax.set_ylabel(f'Avg {y_column}')
            plt.xticks(rotation=0)
        elif x_numeric and not y_numeric:
            bins = pd.cut(df[x_column], bins=8)
            pct = df.groupby(bins, observed=True)[y_column].apply(lambda s: s.value_counts(normalize=True)).unstack().fillna(0)
            pct.index = [f'{int(i.left)}-{int(i.right)}' for i in pct.index]
            pct.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'{y_column} Distribution by {x_column}')
            plt.xticks(rotation=0)
        else:
            ct = pd.crosstab(df[x_column], df[y_column], normalize='index') * 100
            ct.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title(f'{y_column} Distribution by {x_column} (%)')
            plt.xticks(rotation=0)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_list.append(buf)
        plt.close()
        return f"Chart generated: {x_column} vs {y_column}."

    @tool
    def plot_distribution(column: str) -> str:
        """Plot the distribution of a single column. Auto-bins numeric columns, shows value counts for categorical columns."""
        if column not in df.columns:
            return f"Column '{column}' not found. Available columns: {list(df.columns)}"
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
        return f"Distribution chart generated for {column}."

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    tools = [data_overview, plot_relationship, plot_distribution]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
        response = llm_with_tools.invoke(messages)
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
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success(f"Data loaded successfully! {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns.")
        st.dataframe(st.session_state.df.head())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.df is None:
        st.info("👆 Please upload a CSV file to get started.")
    elif prompt := st.chat_input("e.g. Analyze the relationship between Age and Survived"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
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
    st.subheader("📈 Charts")
    if st.session_state.last_charts:
        for chart in st.session_state.last_charts:
            st.image(chart)
    else:
        st.info("Charts will appear here after analysis.")
