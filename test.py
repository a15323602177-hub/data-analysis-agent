from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen-plus",
    api_key="sk-829f5b9d060841edafc109c026cc90f4",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response = llm.invoke("你好，用一句话介绍你自己")
print(response.content)