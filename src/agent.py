import os
import time
from dotenv import load_dotenv
import sys

# --- Helper Class for Dual Logging ---
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# LangChain Imports
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_PATH = "./chroma_db"

def setup_agent():
    print("[INFO] Setting up the Financial Analyst Agent...")

    # --- 1. Setup Vector Store ---
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run ingest.py first.")

    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # --- 2. Define Tools ---
    retriever_tool = create_retriever_tool(
        retriever,
        "search_10k_documents", # Renamed tool
        "Useful for searching specific financial details, risk factors, and revenue figures within the company's 10-K Annual Reports. ALWAYS use this tool first."
    )

    search_tool = TavilySearchResults(max_results=3)
    tools = [retriever_tool, search_tool]

    # --- 3. Setup LLM ---
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # --- 4. The System Prompt (FINANCIAL EDITION) ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Senior Wall Street Financial Analyst. Your job is to analyze public company 10-K filings.\n\n"
         "### INSTRUCTIONS:\n"
         "1. If asked about 'Risks', search the 'Item 1A. Risk Factors' section of the documents.\n"
         "2. If asked about 'Market Trends' or 'Competitors' (e.g., BYD vs Tesla), use 'tavily_search_results_json' to find real-time market data.\n"
         "3. Always cite your source: [Source: Tesla 10-K, Page 45] or [Source: Web - Bloomberg].\n"
        ),
        # --- Few-Shot Example ---
        ("human", "What are the primary supply chain risks?"),
        ("ai", "According to the 10-K filing, the company faces significant risks related to semiconductor shortages and reliance on single-source suppliers in Asia. [Source: Apple 10-K, Page 12]"),
        # ------------------------
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    return agent_executor

def evaluate_project(question):
    try:
        agent_executor = setup_agent()
        print(f"\n==================================================")
        print(f"QUESTION: {question}")
        print(f"==================================================")
        
        start_time = time.time()
        result = agent_executor.invoke({"input": question})
        duration = time.time() - start_time
        
        print(f"\n[FINAL ANSWER]:\n{result['output']}\n")
        print(f"[METRICS]:")
        print(f"- Execution Time: {duration:.2f} seconds")
        print(f"- Tool Calls: {len(result['intermediate_steps'])}")

    except Exception as e:
        print(f"[ERROR] Agent failed: {e}")

if __name__ == "__main__":
    sys.stdout = DualLogger("financial_analysis_logs.txt")

    print("------------------------------------------------------------------")
    print("                     STARTING 10-K ANALYSIS                       ")
    print("------------------------------------------------------------------")

    # Q1: Internal Retrieval (Risk Factors)
    evaluate_project("What are the primary risk factors regarding supply chain and international trade mentioned in the documents?")
    
    # Q2: Hybrid (Internal + External Competition)
    evaluate_project("The report mentions competition. How does the company's recent revenue growth compare to its main competitors (like BYD or Huawei) in 2024? Use web search for competitor data.")
    
    print("\n------------------------------------------------------------------")
    print(" [SUCCESS] Execution finished.")