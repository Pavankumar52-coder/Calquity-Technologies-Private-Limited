import google.generativeai as genai
import os

os.environ["GOOGLE_API_KEY"] = "Enter your API Key"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

import yfinance as yf
import matplotlib.pyplot as plt

def fetch_stock_chart(ticker: str, period: str = "1mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        return "No data found for the given ticker."

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label=f"{ticker} Closing Price", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{ticker} Stock Price over {period}")
    plt.legend()

    img_buffer = f"{ticker}_chart.png"
    plt.savefig(img_buffer)
    plt.close()

    return f"Stock chart saved at {img_buffer}"

import autogen
from autogen import AssistantAgent, UserProxyAgent

config_list = [
    {
        "model": "gemini-pro",
        "api_key": os.getenv("GOOGLE_API_KEY"),
    }
]

stock_agent = AssistantAgent(
    name="StockAgent",
    system_message="I am a financial assistant. I can fetch and visualize stock data.",
    llm_config={"config_list": config_list}
)

class StockUserAgent(UserProxyAgent):
    def __init__(self, name):
        super().__init__(name=name)
    
    def process_stock_query(self, ticker):
        chart_data = fetch_stock_chart(ticker)
        if isinstance(chart_data, str) and chart_data.startswith("No data found"):
            return chart_data
        return f"Stock chart generated for {ticker} (Base64 Encoded)."

user_agent = StockUserAgent(name="User")

def main():
    print("Stock Analysis Assistant is running. Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter a stock ticker (e.g., AAPL) or 'exit': ").strip().upper()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        response = user_agent.process_stock_query(user_input)
        print("\nAgent Response:\n", response)

if __name__ == "__main__":
    main()