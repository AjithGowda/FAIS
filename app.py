import streamlit as st
from server.graph import my_chat_graph
from server.lang_helper import retriever
import time

# Set page config
st.set_page_config(
    page_title="RAG Graph Workflow",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .header-style {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subheader-style {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .step-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .result-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f4f8;
        margin: 1rem 0;
        border-left: 4px solid #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header-style">🔗 RAG Graph Workflow</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-style">Retrieve-Generate Pipeline</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("This app demonstrates a Retrieval-Augmented Generation (RAG) workflow using LangGraph")
    
    # Show graph visualization
    if st.button("📊 Show Workflow Graph"):
        st.subheader("Workflow Graph")
        try:
            from server.graph import my_chat_graph
            mermaid_diagram = my_chat_graph.get_graph().draw_mermaid()
            st.markdown(mermaid_diagram)
        except ImportError as e:
            st.error(f"Graph visualization requires pygraphviz. Install it with: `pip install pygraphviz`")
        except Exception as e:
            st.error(f"Could not generate graph visualization: {e}")

# Main content
tab1, tab2, tab3 = st.tabs(["Chat Interface", "Step-by-Step Execution", "About"])

# Tab 1: Chat Interface
with tab1:
    st.subheader("💬 Ask a Question")
    
    # Input
    user_query = st.text_area(
        "Enter your question:",
        placeholder="Ask anything about your documents...",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        submit_button = st.button("🚀 Send", type="primary", use_container_width=True)
    
    if submit_button and user_query:
        with st.spinner("Processing your query..."):
            try:
                # Run the graph
                start_time = time.time()
                result = my_chat_graph.invoke({"query": user_query, "context": "", "answer": ""})
                execution_time = time.time() - start_time
                
                # Display results
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown("### 📚 Retrieved Context")
                    st.text(result["context"])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown("### 💡 Generated Answer")
                    st.markdown(result["answer"])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.success(f"✅ Completed in {execution_time:.2f} seconds")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    elif submit_button:
        st.warning("Please enter a question first.")

# Tab 2: Step-by-Step Execution
with tab2:
    st.subheader("🔍 Step-by-Step Workflow Execution")
    
    step_query = st.text_input(
        "Enter a query to trace through the workflow:",
        placeholder="Ask a question to see each step..."
    )
    
    if st.button("▶️ Execute Step-by-Step", use_container_width=True):
        if step_query:
            try:
                st.markdown('<div class="step-container">', unsafe_allow_html=True)
                st.markdown("### Step 1: Retrieve")
                st.write("Searching for relevant documents in the vector database...")
                
                with st.spinner("Retrieving documents..."):
                    docs = retriever.invoke(step_query)
                    context = "\n\n".join([doc.page_content for doc in docs])
                
                st.success(f"✅ Found {len(docs)} relevant document(s)")
                st.markdown("**Retrieved Documents:**")
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Document {i}"):
                        st.text(doc.page_content)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="step-container">', unsafe_allow_html=True)
                st.markdown("### Step 2: Generate")
                st.write("Generating answer based on retrieved context...")
                
                with st.spinner("Generating answer..."):
                    start_time = time.time()
                    result = my_chat_graph.invoke({
                        "query": step_query,
                        "context": context,
                        "answer": ""
                    })
                    execution_time = time.time() - start_time
                
                st.success(f"✅ Answer generated in {execution_time:.2f} seconds")
                st.markdown("**Final Answer:**")
                st.markdown(result["answer"])
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error during execution: {str(e)}")
        else:
            st.warning("Please enter a query first.")

# Tab 3: About
with tab3:
    st.subheader("📖 About This App")
    
    st.markdown("""
    ### What is RAG (Retrieval-Augmented Generation)?
    
    RAG is a technique that combines two key components:
    
    1. **Retrieval**: Search for relevant documents/context from a vector database
    2. **Generation**: Use a language model to generate answers based on the retrieved context
    
    ### Workflow Steps
    
    1. **User Query**: You ask a question
    2. **Document Retrieval**: The system searches the FAISS vector database for similar documents
    3. **Context Preparation**: Top-k documents are combined to form the context
    4. **Answer Generation**: An LLM (Groq's Llama-3.1) generates an answer based on the query and context
    
    ### Technology Stack
    
    - **Framework**: Streamlit (for UI)
    - **Graph Framework**: LangGraph (for workflow orchestration)
    - **LLM**: Groq's Llama-3.1-8b
    - **Embeddings**: Ollama's Llama3
    - **Vector Database**: FAISS
    - **Language Chain**: LangChain
    
    ### Components
    
    - **Retriever**: FAISS-based vector database retriever
    - **LLM Chain**: Prompt template + LLM + Output parser
    - **Graph Nodes**:
      - `retrieve`: Fetches relevant documents
      - `generate`: Generates the final answer
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("💡 **Tip**: Use the 'Step-by-Step Execution' tab to understand how the workflow processes your query.")
    
    with col2:
        st.info("📊 **Graph Diagram**: Check the sidebar to visualize the complete workflow graph.")
