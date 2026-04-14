import streamlit as st
import sys

# Force UTF-8 for Windows console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')




from main import query_system
import os

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="RAG Assistant (Mistral Large 3)",
    page_icon="🤖",
    layout="centered"
)

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

from dotenv import load_dotenv
load_dotenv()

# Initialize Langfuse Session for the person
if "langfuse_session_id" not in st.session_state:
    import uuid
    st.session_state.langfuse_session_id = str(uuid.uuid4())

# Verify API Keys
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OPENAI_API_KEY missing! Please ensure it is set in your .env file.")
    st.stop()

# Support for Aitunnel Fallback
if not os.getenv("AITUNNEL_API_KEY") or os.getenv("AITUNNEL_API_KEY") == "your_key_here":
    st.sidebar.warning("⚠️ Aitunnel API Key not set. Fallback will be disabled.")


# Sidebar for Model Selection
st.sidebar.title("🛠️ Settings")
selected_model = st.sidebar.selectbox(
    "Select LLM Model",
    options=["mistral-large", "mistral-medium", "mistral-small"],
    index=0,
    help="Models are proxied through your local LiteLLM Reverse Proxy or Aitunnel fallback."
)

st.title("📚 Intelligent RAG Assistant")
st.markdown(f"Ask anything about your documents. Primary: **{selected_model}** (Local Proxy) | Fallback: **Aitunnel**.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "contexts" in message and message["contexts"]:
            with st.expander("View Source Contexts"):
                for idx, ctx in enumerate(message["contexts"]):
                    st.markdown(f"**Source {idx + 1}:**\n```text\n{ctx}\n```")

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.status("🔍 Analyzing documents...", expanded=True) as status:
            try:
                # Query the RAG system with session tracking and dynamic model
                answer, contexts = query_system(
                    prompt, 
                    session_id=st.session_state.langfuse_session_id,
                    model_name=selected_model
                )
                
                status.update(label="✅ Answer Generated!", state="complete", expanded=False)
                
                # Display answer
                message_placeholder.markdown(answer)
                
                # Display context in an expander
                if contexts:
                    with st.expander("View Source Contexts"):
                        for idx, ctx in enumerate(contexts):
                            st.markdown(f"**Source {idx + 1}:**\n```text\n{ctx}\n```")
                            
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": answer, "contexts": contexts})
                
            except Exception as e:
                status.update(label="❌ Error Occurred", state="error", expanded=True)
                error_msg = f"An error occurred: {str(e)}"
                message_placeholder.error(error_msg)
                import traceback
                traceback.print_exc()
