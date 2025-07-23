import os
import io
import sys
import asyncio
import re # Import the re module for regular expressions
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Import PHI components
import phi
from phi.agent import Agent
from phi.model.openai import OpenAIChat # Although not used, keeping as it was in original
from phi.model.groq import Groq # Although not used, keeping as it was in original
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

# Load environment variables from .env file
# Removed duplicate load_dotenv() and os.environ assignments
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") # Keep if you plan to use Groq models
phi.api = os.getenv("PHI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


## Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

## Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    # Added a more descriptive role for better orchestration
    role="Provides current stock prices, analyst recommendations, company fundamentals, and news for a given stock ticker symbol.",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=[
        "Use tables to display the data.",
        "When asked for financial information, always ensure you receive a precise stock ticker symbol (e.g., TSLA, NVDA, AAPL) before calling any tool."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Multi AI Agent with more explicit instructions for orchestration
multi_ai_agent = Agent(
    name="Multi AI Agent",
    role="Uses multiple AI models to answer questions by orchestrating other agents. It will process financial queries by explicitly extracting stock ticker symbols and passing them to the Finance AI Agent.",
    model=Gemini(id="gemini-2.0-flash"),
    team=[web_search_agent, finance_agent],
    instructions=[
        "Always include sources when providing information from searches.",
        "Use tables to display financial data.",
        "When asked to analyze companies for financial data (e.g., stock prices, analyst recommendations, company news), follow these steps for EACH company:",
        "1. Identify the company mentioned (e.g., Tesla, NVIDIA, Apple).",
        "2. Determine its exact stock ticker symbol (e.g., 'TSLA' for Tesla, 'NVDA' for NVIDIA, 'AAPL' for Apple).",
        "3. Call the 'Finance AI Agent' and explicitly pass the identified ticker symbol as the 'symbol' argument to its relevant tools (e.g., `finance_agent.stock_price(symbol='TSLA')`).",
        "4. Gather the requested financial data for that single company.",
        "After gathering information for all specified companies, combine their results into a single, comprehensive response, formatted clearly with tables for financial data."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Function to strip ANSI escape codes
def strip_ansi_codes(text):
    # This regex matches common ANSI escape codes
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

# Helper function to capture printed output from the agent and strip ANSI codes.
def get_agent_response(prompt: str) -> str:
    capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = capture
    try:
        # This method prints the response to stdout.
        multi_ai_agent.print_response(prompt)
    finally:
        sys.stdout = old_stdout
    
    raw_output = capture.getvalue()
    # Strip ANSI escape codes before returning the value for HTML display
    clean_output = strip_ansi_codes(raw_output)
    return clean_output

# Create FastAPI app instance
app = FastAPI()

# Home route returns a simple HTML form for user input
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi AI Agent Search</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; color: #333; }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
            form { background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); max-width: 500px; margin: 0 auto; }
            label { display: block; margin-bottom: 10px; font-weight: bold; }
            input[type=text] { width: calc(100% - 22px); padding: 12px; margin-bottom: 20px; border: 1px solid #ccc; border-radius: 5px; font-size: 16px; }
            button { padding: 12px 25px; background-color: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; display: block; width: 100%; box-sizing: border-box; transition: background-color 0.3s ease; }
            button:hover { background-color: #218838; }
            .result { margin-top: 30px; padding: 25px; border: 1px solid #e0e0e0; border-radius: 10px; background-color: #ffffff; box-shadow: 0 4px 8px rgba(0,0,0,0.05); white-space: pre-wrap; font-family: 'Courier New', Courier, monospace; font-size: 14px; line-height: 1.6; }
            a { display: block; text-align: center; margin-top: 20px; color: #007bff; text-decoration: none; transition: color 0.3s ease; }
            a:hover { color: #0056b3; }
        </style>
    </head>
    <body>
        <h1>Multi AI Agent Search</h1>
        <form action="/search" method="post">
            <label for="query">Enter your enquiry:</label>
            <input type="text" id="query" name="query" placeholder="e.g., Analyze companies like Tesla, NVDA, and Apple" required>
            <button type="submit">Search</button>
        </form>
        <!-- The result div will be populated dynamically if navigated directly to /search -->
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Search route: handles form submission and returns the captured agent response
@app.post("/search", response_class=HTMLResponse)
async def search(query: str = Form(...)):
    loop = asyncio.get_running_loop()
    try:
        # Run the blocking get_agent_response function in an executor.
        result = await loop.run_in_executor(None, get_agent_response, query)
        if not result.strip():
            result = "No response received from the agent."
    except Exception as e:
        result = f"An error occurred: {e}"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; color: #333; }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            .result {{ margin-top: 30px; padding: 25px; border: 1px solid #e0e0e0; border-radius: 10px; background-color: #ffffff; box-shadow: 0 4px 8px rgba(0,0,0,0.05); white-space: pre-wrap; font-family: 'Courier New', Courier, monospace; font-size: 14px; line-height: 1.6; }}
            a {{ display: block; text-align: center; margin-top: 20px; color: #007bff; text-decoration: none; transition: color 0.3s ease; }}
            a:hover {{ color: #0056b3; }}
        </style>
    </head>
    <body>
        <h1>Search Results</h1>
        <div class="result">{result}</div>
        <br>
        <a href="/">&#8592; Back to Search</a>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Standard entry point for Uvicorn
if __name__ == "__main__":
    import uvicorn
    # The string "main:app" tells uvicorn to look for an 'app' object in 'main.py'
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)

