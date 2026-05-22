import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import uuid
from main import query_system

st.set_page_config(
    page_title="Corporate Bot (Local)",
    page_icon="🤖",
    layout="centered"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "langfuse_session_id" not in st.session_state:
    st.session_state.langfuse_session_id = str(uuid.uuid4())

st.sidebar.title("🛠️ Settings")
selected_model = st.sidebar.selectbox(
    "Select LLM Model",
    options=["qwen3.5:9b", "qwen2.5:14b", "llama3"],
    index=0,
    help="Models are requested locally from Ollama."
)

st.title("📚 Corporate Bot (Local RAG)")
st.markdown(f"Задайте вопрос по корпоративным документам. Модель: **{selected_model}** (Ollama Local).")

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.status("🔍 Analyzing documents...", expanded=True) as status:
            try:
                answer, contexts = query_system(
                    prompt, 
                    session_id=st.session_state.langfuse_session_id,
                    model_name=selected_model
                )
                
                status.update(label="✅ Answer Generated!", state="complete", expanded=False)
                message_placeholder.markdown(answer)
                
                if contexts:
                    with st.expander("View Source Contexts"):
                        for idx, ctx in enumerate(contexts):
                            st.markdown(f"**Source {idx + 1}:**\n```text\n{ctx}\n```")
                            
                st.session_state.messages.append({"role": "assistant", "content": answer, "contexts": contexts})
                
            except Exception as e:
                status.update(label="❌ Error Occurred", state="error", expanded=True)
                error_msg = f"An error occurred: {str(e)}"
                message_placeholder.error(error_msg)
                import traceback
                traceback.print_exc()
