# Importing Necessary Libraries
import os
import yfinance as yf
import autogen
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

# Loading API key using .env file
load_dotenv()

# Load LLM API key
config_list = [
    {
        "model": "gemini-pro",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta"
    }
]

# Fetch stock data tool
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    
    if df.empty:
        return None
    
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["Days"] = np.arange(len(df))
    return df.tail(60)

# AI Prediction Tool
def predict_stock_trend(df):
    if df is None:
        return "No data available for trend analysis."
    
    X = df["Days"].values.reshape(-1, 1)
    y = df["Close"].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    trend = "upward" if model.coef_[0][0] > 0 else "downward"
    return f"Stock price is trending {trend} with slope {model.coef_[0][0]:.4f}"

# Interactive Stock Chart with Plotly Tool
def generate_interactive_chart(df, ticker):
    if df is None:
        return "No chart available."
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_20"], mode="lines", name="20-Day MA", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"], mode="lines", name="50-Day MA", line=dict(dash='dot')))
    
    fig.update_layout(title=f"Stock Analysis for {ticker}", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
    return fig

# Define Multi-Agent System to process the stocks
llm_config = {"config_list": config_list, "timeout": 30}
# Agent-1
DataFetcherAgent = autogen.AssistantAgent(
    name="DataFetcher",
    system_message="Fetches historical stock data and computes indicators.",
    llm_config=llm_config
)
# Agent-2
AnalystAgent = autogen.AssistantAgent(
    name="Analyst",
    system_message="Analyzes stock trends using AI and provides trading insights.",
    llm_config=llm_config
)
# Agent-3
ReportAgent = autogen.AssistantAgent(
    name="Reporter",
    system_message="Generates a summary report with analysis and charts.",
    llm_config=llm_config
)
# Agent-4
UserProxyAgent = autogen.UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS"
)

# Multi-Agent Workflow
def run_stock_analysis(ticker):
    print("\n Fetching stock data...")
    df = fetch_stock_data(ticker)
    
    print("\nPerforming AI analysis...")
    trend_analysis = predict_stock_trend(df)
    
    print("\nGenerating interactive stock chart...")
    chart = generate_interactive_chart(df, ticker)
    
    print("\nGenerating report...")
    summary = f"""
    🔹 **Stock:** {ticker}
    🔹 **Trend Analysis:** {trend_analysis}
    """
    
    return summary, chart

# Run AI-Powered Stock Assistant
def stock_assistant():
    print("Welcome to the world of AI-Powered Stock Assistant! Type 'exit' to quit.")
    
    while True:
        ticker = input("\nEnter your stock ticker to analyze the stocks for AAPL,TSLA.... or 'exit': ").upper()
        if ticker == "EXIT":
            break
        
        summary, chart = run_stock_analysis(ticker)
        print("\nAI Report:")
        print(summary)
        
        if isinstance(chart, go.Figure):
            chart.show()

stock_assistant()