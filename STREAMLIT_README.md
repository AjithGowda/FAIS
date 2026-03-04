# RAG Graph Workflow - Streamlit App

A beautiful Streamlit application that visualizes and interacts with a Retrieval-Augmented Generation (RAG) workflow powered by LangGraph.

## Features

✨ **Chat Interface** - Ask questions and get answers based on your documents
🔍 **Step-by-Step Execution** - Trace through the workflow to see retrieval and generation stages
📊 **Workflow Visualization** - View the graph structure in Mermaid format
📚 **Document Inspection** - Explore retrieved documents in detail

## Prerequisites

- Python 3.8+
- FAISS vector database already set up (`faiss_db/` folder)
- Ollama running locally (for embeddings)
- Groq API key (for LLM)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have:
   - Ollama running locally with the llama3 model
   - A `.env` file with your Groq API key:
     ```
     GROQ_API_KEY=your_key_here
     ```

## Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
FAIS/
├── app.py                 # Streamlit app (this file)
├── requirements.txt       # Python dependencies
├── server/
│   ├── graph.py          # LangGraph workflow definition
│   ├── lang_helper.py    # LLM and retriever setup
│   ├── langchain.py      # LangChain utilities
│   └── document_loader.py # Document loading utilities
├── data/                  # Sample documents
│   ├── sample.pdf
│   └── sample.txt
└── faiss_db/             # Vector database
    ├── index.faiss
    └── index.pkl
```

## How It Works

### Workflow Stages

1. **Retrieve Stage**: Uses FAISS to search the vector database for documents similar to the user's query
2. **Generate Stage**: Takes the query and retrieved context, then generates an answer using Groq's Llama-3.1 LLM

### Tabs

- **Chat Interface**: Simple Q&A interface
- **Step-by-Step Execution**: Detailed view of each workflow stage with timing
- **About**: Information about RAG and the tech stack

## Configuration

You can customize the app in the sidebar:
- View the workflow graph diagram
- Monitor execution performance

## Troubleshooting

### "FAISS database not found"
Make sure the `faiss_db/` folder exists with `index.faiss` and `index.pkl`

### "Ollama connection error"
Ensure Ollama is running and has the llama3 model: `ollama pull llama3`

### "Groq API error"
Check your `.env` file has `GROQ_API_KEY` set correctly

## Environment Variables

Required in `.env`:
- `GROQ_API_KEY`: Your Groq API key for accessing Llama-3.1
- `OLLAMA_BASE_URL`: (Optional) Ollama server URL (defaults to localhost:11434)
