import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Set up the Streamlit app
st.set_page_config(page_title='Text to Math Problem Solver', page_icon="🧮")
st.title("Text to Math Problem Solver using Groq API")
st.secrets["HF_Token"]

# Input Groq API key via sidebar or from environment
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.warning("Please add your Groq API Key to continue.")
    st.stop()

# Initialize the language model
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# Initialize Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search the internet to find various information."
)

# Initialize math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions."
)

# Initialize reasoning tool
prompt_text = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and display it point-wise for the question below.
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt_text
)

reasoning_chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Combine all tools into an agent
agent = initialize_agent(
    tools=[wikipedia_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a Math chatbot who can answer all your math questions."}
    ]

# Display previous messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to generate response
def generate_response(question):
    response = agent.run(input=question)
    return response

# Interaction area
question = st.text_area("Enter your question", "I have 5 bananas and 10 grapes. I eat 2 bananas and give away 3 grapes. How many bananas and grapes do I have now?")

if st.button("Find my answer"):
    if question.strip():
        with st.spinner("Generating response..."):
            st.session_state["messages"].append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            # Streamlit callback handler
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(input=question, callbacks=[st_cb])

            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    else:
        st.warning("Please enter a question.")

