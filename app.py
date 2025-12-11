import os
from dotenv import load_dotenv
import streamlit as st
import plotly.express as px
from google import genai  # Gemini
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
import re

# Optional Graphviz
try:
    from graphviz import Digraph
    graphviz_installed = True
except ModuleNotFoundError:
    graphviz_installed = False

load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Tavily search
search_tool = TavilySearch(max_results=3, api_key=TAVILY_API_KEY)

# Groq LLM
groq_llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")


def tavily_search(query):
    return search_tool.run(query)


def query_agent(startup_input, provider="gemini"):
    search_results = tavily_search(startup_input)
    
    prompt = f"""
You are a startup business consultant. 
Input: {startup_input}
Use search results: {search_results}

Output a clean, actionable, short summary for a non-tech person including:
- Idea
- Market overview
- Budget breakdown
- Step-by-step action plan with registration & setup
- 4-week launch timeline
- Marketing plan with ✅ good / ⚠️ less effective
- Feasibility score & suggested KPIs
- 3-5 unique startup name suggestions (bullet points only)
Limit each section to 1-3 sentences or simple table format.
"""
    provider = provider.lower()
    
    if provider == "gemini":
        chat = client.chats.create(model="gemini-2.5-flash")
        response = chat.send_message(prompt)
        return response.text
    
    else:
        msgs = [
            ("system", "You are a helpful AI assistant."),
            ("human", prompt)
        ]
        out = groq_llm.invoke(msgs)
        return out.content


# Streamlit UI
st.set_page_config(page_title="AI Business Launch Assistant", layout="wide")
st.title("AI Business Launch Assistant")
st.markdown(
    """
Enter your business idea, location, and budget. 
The AI will give you a **short, actionable plan** with market info, budget, marketing, timeline, feasibility, and brand name suggestions.
"""
)

startup_input = st.text_area(
    "💡 Describe your startup idea (include city & budget):",
    "I want to launch a vegan snack startup in Bangalore with $10k",
    key="startup_input_textarea"
)

provider = st.radio(
    "Choose AI provider:",
    ["gemini", "groq"],
    key="ai_provider_radio"
)

if st.button("Generate Full Plan", key="generate_button"):
    with st.spinner("Generating startup plan..."):
        summary = query_agent(startup_input, provider)
    
    st.subheader("Business Summary")
    st.markdown(summary)
    
    st.download_button(
        label=" Download Summary",
        data=summary,
        file_name="startup_summary.txt",
        key="download_summary"
    )
    
    # Timeline Visualization
    st.subheader("📅 4-Week Launch Timeline")
    timeline_data = {
        "Week": ["W1", "W2", "W3", "W4"],
        "Tasks": [
            "Register + source ingredients",
            "Set up kitchen + hire helpers",
            "Test delivery + small batch launch",
            "Full launch + social media marketing"
        ]
    }
    fig = px.timeline(
        timeline_data, 
        x_start=[0, 7, 14, 21], 
        x_end=[7, 14, 21, 28], 
        y="Tasks", 
        text="Tasks", 
        title="4-Week Launch Plan"
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    
    # Mind Map Visualization
    if graphviz_installed:
        st.subheader("🧠 Mind Map of Business Idea")
        dot = Digraph(comment='Startup Ideas')
        dot.node('A', startup_input.split(" ")[0] + " Idea")
        tasks = ["Register", "Kitchen Setup", "Source Ingredients", "Marketing", "Launch"]
        for i, task in enumerate(tasks, start=1):
            dot.node(f'B{i}', task)
            dot.edge('A', f'B{i}')
        st.graphviz_chart(dot)
    else:
        st.warning("Graphviz not installed → Mind Map skipped")
    
    # Extract only name suggestions (without logos)
    # Match any line that looks like a name suggestion (bullet or numbered)
    name_suggestions = re.findall(r"[-\d]*\.?\s*([A-Z][A-Za-z0-9\s]+)", summary)
    name_suggestions = [n.strip() for n in name_suggestions if len(n.strip()) > 2]
    
    if name_suggestions:
        st.subheader("🎨 Business Name Suggestions")
        for i, name in enumerate(name_suggestions[:5], start=1):  # show top 5
            st.markdown(f"**{i}. {name}**")
    else:
        st.info("No name suggestions found in the summary. AI may need stricter formatting.")
