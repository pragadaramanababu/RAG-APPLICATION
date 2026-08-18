import streamlit as st
from rag import ask_rag, build_vector_database, get_document_count

st.set_page_config(
    page_title="Company Docs AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title {font-size: 38px; font-weight: 700; margin-bottom: 0;}
.subtitle {font-size: 16px; color: #777; margin-bottom: 25px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 Company Docs AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions about company policies, engineering standards, onboarding, products and security.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Settings")

    n_results = st.slider(
        "Retrieved documents",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of document chunks used to answer each question."
    )

    st.divider()
    st.subheader("📚 Knowledge Base")
    st.metric("Indexed Chunks", get_document_count())

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Reading documents and creating embeddings..."):
            try:
                count = build_vector_database()
                st.success(f"Indexed {count} chunks!")
                st.rerun()
            except Exception as error:
                st.error(f"Failed to rebuild: {error}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("RAG Application")
    st.caption("ChromaDB • Sentence Transformers • Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
👋 **Welcome!**

I can answer questions from the company knowledge base.

Try asking:
- What is the work-from-home policy?
- How many annual leave days are available?
- What are the engineering standards?
- What is the onboarding process?
- What are the security policies?
""")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View Sources"):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**Source {index}:** `{source['source']}`")
                    st.caption(source["text"][:300] + "...")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            try:
                result = ask_rag(question, n_results=n_results)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("📚 View Sources"):
                        for index, source in enumerate(sources, start=1):
                            st.markdown(f"**Source {index}:** `{source['source']}`")
                            st.caption(source["text"][:300] + "...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:
                error_message = f"⚠️ I couldn't process your question.\n\n**Error:** `{error}`"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
