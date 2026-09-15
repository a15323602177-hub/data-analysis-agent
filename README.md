# 📊 AI Data Analysis Agent

Upload any CSV file and ask questions in natural language — the agent automatically analyzes the data and generates charts.

🔗 **Live Demo:** https://data-analysis-agent-w3uyoyc63pbrek9n3f3sek.streamlit.app/

## What it does

This is a conversational data analysis tool. Instead of writing pandas code or SQL queries, you can simply ask questions like "What's the trend in monthly sales?" or "Show me the distribution of customer ages," and the agent will run the analysis and return a chart or answer.

## How it works

- **LangGraph** orchestrates the agent's reasoning and tool-calling steps (deciding what analysis to run, which columns to use, and how to visualize the result)
- **Qwen LLM API** parses the natural-language question and translates it into data analysis and visualization logic
- **Streamlit** provides the web interface for uploading files and displaying results
- Deployed on **Streamlit Community Cloud**

## Tech Stack

Python · LangGraph · Qwen API · Streamlit · Pandas

## Run it locally

\`\`\`bash
git clone https://github.com/a15323602177-hub/data-analysis-agent.git
cd data-analysis-agent
pip install -r requirements.txt
streamlit run app.py
\`\`\`

You'll need a Qwen API key set as an environment variable (see `.env.example` if provided, or check `app.py` for the expected variable name).
