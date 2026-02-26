import streamlit as st
import requests
from document_loader import data_load

st.title("RAG AI Chat Bot")
st.warning("This is a demo of a RAG integrated chat bot. Please enter your query below and click 'Send' to get a response from the bot.")

API_URL = "http://localhost:8000/rag/invoke"

# Load vector database once at startup using cache
@st.cache_resource
def load_vector_db():
    with st.spinner("Loading vector database..."):
        return data_load()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vector DB at startup
try:
    vector_db = load_vector_db()
except Exception as e:
    st.error(f"Failed to load vector database: {str(e)}")
    st.stop()

#display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Please enter your query")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    #send the data to llm and get the response
    with st.spinner("Generating response..."):
        try:
            # Retrieve relevant documents from vector store
            retrieved_docs = vector_db.similarity_search(user_input, k=3)
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            print("retrieved docs:", len(retrieved_docs), "documents")
            print("context length:", len(context))

            api_response = requests.post(
                API_URL, 
                json={"input": {"context": context, "question": user_input}},
                timeout=60
            )
            print("API response status:", api_response.status_code)
            
            if api_response.status_code != 200:
                st.error(f"Error: {api_response.status_code} - {api_response.text}")
                response = "Sorry, I couldn't generate a response."
            else:
                api_data = api_response.json()
                output = api_data.get("output", "Sorry, I couldn't generate a response.")
                print("API output type:", type(output))
                # Extract the text content
                if isinstance(output, dict) and "content" in output:
                    response = output["content"]
                else:
                    response = str(output)
        except requests.exceptions.Timeout:
            st.error("Error: Request timed out. The server is taking too long to respond.")
            response = "Sorry, I couldn't generate a response (timeout)."
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to the server. Make sure the FastAPI server is running on port 8000.")
            response = "Sorry, I couldn't generate a response (connection error)."
        except Exception as e:
            st.error(f"Error: {str(e)}")
            print(f"Exception: {e}")
            response = "Sorry, I couldn't generate a response."
    #store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)