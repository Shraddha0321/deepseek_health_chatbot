import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
import time

# 1. Page Configuration
st.set_page_config(
    page_title="Health Chatbot",
    layout="wide"
)

# 2. Custom CSS Styling (Including Input Box Emoji Styling)
st.markdown("""
<style>
    body {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .main {
        background-color: #1a1a1a;
    }
    .stTextInput textarea {
        color: #ffffff !important;
        background-color: #3d3d3d !important;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Page Title
st.title("💗 Health Chatbot")
st.caption("💊 Your AI Companion for Mental Wellness and Emotional Support")

# 4. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose AI Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🧘 Mindfulness & Meditation  
    - 📓 CBT-Based Techniques  
    - 💡 Lifestyle & Wellness Tips  
    - ❤️ Emotional Support  
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# 5. Initialize Chatbot Memory
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi there! I'm here to support your well-being. How can I help you today? "}
    ]

# 6. Display Chat Messages
for msg in st.session_state.message_log:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. Chat Input (With Emojis in Placeholder)
user_input = st.text_input("💬 Type your health or wellness concern here... ✨", placeholder="e.g., I'm feeling anxious today 😔, can you help?")

# 8. Function to Build Prompt Chain
def build_prompt_chain():
    prompt_sequence = [
        SystemMessagePromptTemplate.from_template(
            "You are a compassionate AI health assistant specializing in mental wellness. "
            "Offer empathetic responses, self-care suggestions, and CBT-style prompts. "
            "Avoid making diagnoses. Always respond in English."
        )
    ]
    
    # Add previous messages to prompt sequence
    for entry in st.session_state.message_log:
        template = HumanMessagePromptTemplate if entry["role"] == "user" else AIMessagePromptTemplate
        prompt_sequence.append(template.from_template(entry["content"]))
    
    return ChatPromptTemplate.from_messages(prompt_sequence)

# 9. Function to Generate AI Response
def generate_ai_response(prompt_chain):
    try:
        # Initialize LLM engine with dynamic model selection
        llm_engine = ChatOllama(
            model=selected_model,
            base_url="http://localhost:11434",
            temperature=0.3
        )
        
        pipeline = prompt_chain | llm_engine | StrOutputParser()
        return pipeline.invoke({})
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# 10. Process User Input
if user_input:
    # Store user message
    st.session_state.message_log.append({"role": "user", "content": user_input})

    # Display user's message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.spinner("AI is thinking... 🤖💭"):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    # Ensure AI response isn't empty
    if not ai_response:
        ai_response = "⚠️ I encountered an issue. Please try again."

    # Display AI response with a typing effect
    with st.chat_message("ai"):
        response_container = st.empty()
        typed_response = ""
        for char in ai_response:
            typed_response += char
            response_container.markdown(typed_response + "▌")
            time.sleep(0.02)
        response_container.markdown(typed_response)

    # Store AI response in session state
    st.session_state.message_log.append({"role": "ai", "content": ai_response})

    # Stop to refresh UI
    st.stop()
