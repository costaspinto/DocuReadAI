# DocuReadAI

## Intelligent Document Q&A Assistant

DocuReadAI is an intelligent document understanding assistant built with **LangChain**, **Hugging Face**, **ChromaDB**, **IBM watsonx.ai**, and **Gradio**.

The application allows users to upload PDF documents and ask natural-language questions about their contents. It uses a **Retrieval-Augmented Generation (RAG)** pipeline to retrieve relevant document content and provide context-grounded responses.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Example Use Cases](#example-use-cases)
- [Engineering Highlights](#engineering-highlights)
- [Interview Talking Points](#interview-talking-points)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Screenshots](#screenshots)
- [Security](#security)
- [Author](#author)
- [License](#license)

---

## Overview

Important information is often stored in unstructured documents such as research papers, technical manuals, reports, and other PDF files.

Manually searching through lengthy documents can be time-consuming and inefficient.

DocuReadAI provides a conversational interface for interacting with PDF documents. Instead of manually searching through pages, users can ask questions in natural language and receive responses based on relevant content retrieved from the uploaded document.

The project demonstrates a complete **Retrieval-Augmented Generation (RAG)** workflow, covering document ingestion, text splitting, embedding generation, vector storage, semantic retrieval, and language-model-based response generation.

---

## Problem Statement

Traditional document search often requires users to locate information manually or rely on exact keyword matches.

This becomes difficult when:

- Documents are large.
- Relevant information is distributed across multiple sections.
- Users do not know the exact terminology used in the document.
- The same concept may be expressed using different words.

DocuReadAI addresses this by using semantic embeddings and vector similarity search to retrieve document content that is relevant to a user's question.

---

## Solution

DocuReadAI processes uploaded PDF documents through a RAG pipeline.
<img width="1911" height="862" alt="qa_bot" src="https://github.com/user-attachments/assets/7cd4a4e8-c079-404a-8d16-2b88882d1d1f" />
<img width="547" height="227" alt="load_documents" src="https://github.com/user-attachments/assets/41d778a5-6020-4f67-a995-79cebb0e8b5c" />
<img width="828" height="220" alt="split_text" src="https://github.com/user-attachments/assets/64211367-710d-4731-94fa-225967ce021a" />
<img width="927" height="212" alt="embed_documents" src="https://github.com/user-attachments/assets/974a4a8c-1528-4413-9228-ab0c28a49d12" />
<img width="582" height="191" alt="vector_db" src="https://github.com/user-attachments/assets/6b87bc49-5ff8-4dca-a1ab-b72b892b1859" />
<img width="635" height="210" alt="retriver" src="https://github.com/user-attachments/assets/d07a6e42-5fd3-46eb-8a6e-c37a5153d5c4" />
<img width="4560" height="1113" alt="Blank diagram" src="https://github.com/user-attachments/assets/7b73b50f-7172-4a71-8c75-e6efbdb2c617" />
The high-level workflow is:

```text
PDF Upload
    |
    v
Document Loading
    |
    v
Text Splitting
    |
    v
Embedding Generation
    |
    v
ChromaDB Vector Store
    |
    |
    |       User Question
    |              |
    |              v
    |       Query Embedding
    |              |
    |              v
    +------> Similarity Search
                   |
                   v
          Relevant Document Chunks
                   |
                   v
            Context + Question
                   |
                   v
            IBM watsonx.ai
                   |
                   v
             Final Answer
```

---

## Key Features

### PDF Document Ingestion

Users can upload PDF documents through the Gradio interface.

LangChain's `PyPDFLoader` is used to load the document content.

### Retrieval-Augmented Generation

The application retrieves relevant document content and provides it as context to the language model during answer generation.

### Semantic Search

Document chunks and user queries are converted into vector embeddings, allowing the application to perform similarity-based retrieval.

### Local Vector Storage

ChromaDB is used to store document embeddings and their corresponding text chunks locally.

### IBM watsonx.ai Integration

IBM watsonx.ai is used as the language-model generation layer. The original implementation uses a model such as `google/flan-ul2`.

### Interactive Web Interface

Gradio provides the user interface for uploading documents and asking questions.

---

## Architecture

```text
                    +----------------------+
                    |      PDF Upload      |
                    |      Gradio UI       |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |     PyPDFLoader      |
                    |  Document Extraction |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | RecursiveCharacter    |
                    |   Text Splitter       |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Hugging Face          |
                    | Embedding Model       |
                    | all-MiniLM-L6-v2      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |      ChromaDB         |
                    |    Vector Storage     |
                    +----------+-----------+
                               |
                         User Question
                               |
                               v
                    +----------------------+
                    |    Query Embedding   |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Similarity Search  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |  Relevant Context    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |  IBM watsonx.ai      |
                    |  Language Model      |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |    Final Answer      |
                    +----------------------+
```

---

# How the RAG Pipeline Works

## 1. Document Loading

The workflow begins when a user uploads a PDF document through the Gradio interface.

LangChain's `PyPDFLoader` loads and extracts the document content.

```text
PDF Document
     |
     v
PyPDFLoader
     |
     v
Extracted Document Content
```

---

## 2. Text Splitting

Large documents are divided into smaller sections before embedding and retrieval.

DocuReadAI uses LangChain's:

```text
RecursiveCharacterTextSplitter
```

The text splitter creates smaller, overlapping chunks to help preserve context between adjacent sections.

```text
Large Document
      |
      v
+-------------+
|   Chunk 1   |
+-------------+
|   Chunk 2   |
+-------------+
|   Chunk 3   |
+-------------+
|   Chunk 4   |
+-------------+
```

---

## 3. Embedding Generation

Each document chunk is converted into a numerical vector representation.

The project uses the Hugging Face sentence-transformer model:

```text
all-MiniLM-L6-v2
```

The embedding model represents the semantic meaning of the document chunks.

```text
Document Chunk
      |
      v
Embedding Model
      |
      v
Vector Representation
```

---

## 4. Vector Storage

The generated embeddings and their associated text chunks are stored in:

```text
ChromaDB
```

ChromaDB provides the vector storage and similarity-search layer for the application.

```text
Document Chunk
      |
      v
Embedding
      |
      v
ChromaDB
```

---

## 5. Query Processing

When the user submits a question, the question is converted into an embedding using the same embedding model.

For example:

```text
What are the main findings of this document?
```

The question is transformed into a query vector.

```text
User Question
      |
      v
Embedding Model
      |
      v
Query Vector
```

---

## 6. Context Retrieval

The query vector is compared against the document vectors stored in ChromaDB.

The system retrieves document chunks that are semantically relevant to the user's question.

```text
Query Vector
      |
      v
ChromaDB Similarity Search
      |
      v
Relevant Document Chunks
```

These retrieved chunks form the context used for answer generation.

---

## 7. Answer Generation

The retrieved context is combined with the user's original question.

The resulting prompt is sent to an IBM watsonx.ai language model.

The original implementation uses a model such as:

```text
google/flan-ul2
```

The model generates the final response using the retrieved document context.

```text
Retrieved Context + User Question
                |
                v
        Prompt Construction
                |
                v
        IBM watsonx.ai LLM
                |
                v
          Generated Answer
```

---

# Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **LangChain** | RAG pipeline orchestration |
| **Hugging Face** | Embedding model integration |
| **all-MiniLM-L6-v2** | Sentence-transformer embedding model |
| **ChromaDB** | Vector storage and similarity search |
| **IBM watsonx.ai** | Language-model generation |
| **Gradio** | Interactive web interface |
| **PyPDFLoader** | PDF document loading |

---

# Project Structure

```text
DocuReadAI/
│
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

### File Description

| File | Description |
|---|---|
| `main.py` | Contains the RAG pipeline and Gradio application logic |
| `requirements.txt` | Lists the required Python dependencies |
| `.env` | Stores local API credentials and configuration |
| `.gitignore` | Specifies files and directories that Git should ignore |
| `README.md` | Project documentation |

> **Security:** The `.env` file should never be committed to the repository.

---

# Installation

## Prerequisites

Make sure the following are installed:

- Python 3.x
- Git
- IBM watsonx.ai credentials

---

## 1. Clone the Repository

```bash
git clone https://github.com/costaspinto/DocuReadAI.git
cd DocuReadAI
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Configuration

Create a `.env` file in the root directory of the project.

Add the IBM watsonx.ai configuration:

```env
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://eu-gb.ml.cloud.ibm.com
MODEL_ID=google/flan-ul2
```

Replace the placeholder values with your own credentials.

## Environment Variables

| Variable | Description |
|---|---|
| `WATSONX_API_KEY` | IBM watsonx.ai API key |
| `WATSONX_PROJECT_ID` | IBM watsonx.ai project ID |
| `WATSONX_URL` | IBM watsonx.ai service endpoint |
| `MODEL_ID` | Foundation model used for generation |

---

# Running the Application

Start the application with:

```bash
python main.py
```

After startup, Gradio will provide a local URL in the terminal.

Example:

```text
http://127.0.0.1:7860
```

Open the URL in your browser to access DocuReadAI.

---

# Usage

### Step 1 — Start the Application

Run:

```bash
python main.py
```

### Step 2 — Upload a PDF

Upload the PDF document you want to analyze through the Gradio interface.

### Step 3 — Process the Document

The application:

1. Loads the PDF.
2. Extracts the document content.
3. Splits the content into chunks.
4. Generates embeddings.
5. Stores the embeddings in ChromaDB.

### Step 4 — Ask a Question

Enter a natural-language question about the uploaded document.

Example:

```text
What are the key findings discussed in this document?
```

### Step 5 — Receive the Answer

The application retrieves relevant document chunks and provides them as context to the IBM watsonx.ai model.

The generated answer is then displayed through the Gradio interface.

---

# Example Use Cases

DocuReadAI can be adapted for document-intensive workflows including:

- Research paper analysis
- Technical documentation
- Business reports
- Internal knowledge bases
- Educational material
- Policy documents
- Product documentation
- Long-form PDF analysis

---

# Engineering Highlights

## Retrieval-Augmented Generation

The project demonstrates a practical RAG architecture in which document retrieval is performed before language-model generation.

This allows the generation layer to receive relevant information from the uploaded document as contextual input.

## Semantic Retrieval

The application uses embeddings to retrieve information based on semantic similarity rather than relying only on exact keyword matches.

## Vector Search

ChromaDB provides the vector storage and similarity-search layer required for document retrieval.

## Embedding Pipeline

The `all-MiniLM-L6-v2` model converts document chunks and user queries into vector representations.

## Enterprise LLM Integration

IBM watsonx.ai provides the language-model generation layer.

## AI Application Development

The project combines document processing, vector search, LLM integration, and an interactive Gradio interface into a complete AI application.

---

# Interview Talking Points

DocuReadAI demonstrates practical experience with:

- Retrieval-Augmented Generation
- LangChain
- Large Language Models
- Vector databases
- Semantic search
- Text embeddings
- Hugging Face
- ChromaDB
- IBM watsonx.ai
- PDF document processing
- Prompt construction
- Gradio
- Python
- AI application development

## 60-Second Interview Explanation

> DocuReadAI is a Retrieval-Augmented Generation application for interacting with PDF documents using natural language. I built the pipeline using LangChain, Hugging Face embeddings, ChromaDB, and IBM watsonx.ai. The application first loads the PDF using PyPDFLoader and splits the document into smaller overlapping chunks. Each chunk is converted into an embedding using the all-MiniLM-L6-v2 model and stored in ChromaDB. When a user asks a question, the question is embedded and used for similarity search against the document vectors. The most relevant document chunks are then combined with the user's question and passed to the IBM watsonx.ai language model to generate the final response. The complete workflow is exposed through a Gradio interface.

---

# Technical Design Considerations

### Why RAG?

RAG allows the application to retrieve relevant information from the uploaded document before generating a response.

### Why Embeddings?

Embeddings provide vector representations of text that can be compared using semantic similarity.

### Why ChromaDB?

ChromaDB provides a vector storage and similarity-search layer suitable for the local document retrieval workflow.

### Why `all-MiniLM-L6-v2`?

The project uses `all-MiniLM-L6-v2` as its sentence-transformer embedding model for document and query representations.

### What affects retrieval quality?

Retrieval quality can be affected by:

- Chunk size
- Chunk overlap
- Embedding model
- Number of retrieved chunks
- Document structure
- Query formulation

### Does RAG eliminate hallucinations?

No. RAG can provide relevant external context to a language model, but it does not guarantee hallucination-free responses.

---

# Limitations

The current implementation has several limitations:

- Retrieval quality depends on the embedding model and retrieval configuration.
- Chunk size and overlap can affect the relevance of retrieved content.
- Response quality depends on the selected language model.
- The current implementation is primarily intended for local usage.
- Automated retrieval and generation evaluation is not documented.
- Authentication and multi-user access are not implemented in the documented workflow.
- Large-scale document processing would require additional optimization.
- The current workflow focuses on PDF documents.

---

# Future Improvements

Potential improvements include:

- [ ] Add document citations and source references
- [ ] Support multiple documents simultaneously
- [ ] Add conversation history
- [ ] Implement hybrid keyword and semantic retrieval
- [ ] Add configurable chunk size and overlap
- [ ] Add configurable retrieval parameters
- [ ] Add RAG evaluation metrics
- [ ] Add retrieval precision and recall evaluation
- [ ] Add document metadata filtering
- [ ] Add authentication
- [ ] Add multi-user support
- [ ] Add persistent vector storage
- [ ] Add automated testing
- [ ] Add structured logging
- [ ] Add monitoring and observability
- [ ] Containerize the application with Docker
- [ ] Deploy the application to a cloud environment
- [ ] Add CI/CD
- [ ] Support additional document formats

---

# Screenshots

Add application screenshots to demonstrate the interface and document Q&A workflow.

Recommended structure:

```text
docs/
└── screenshots/
    ├── upload.png
    ├── document-processing.png
    └── question-answering.png
```

Example:

```markdown
![DocuReadAI Interface](docs/screenshots/upload.png)
```

---

# Security

Sensitive credentials must never be committed to GitHub.

Store API credentials using environment variables:

```env
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://eu-gb.ml.cloud.ibm.com
MODEL_ID=google/flan-ul2
```

Ensure `.env` is included in `.gitignore`.

Recommended `.gitignore` entries:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

If credentials are accidentally committed, revoke and regenerate the affected credentials immediately.

---

# Reproducibility

To reproduce the application locally:

```bash
git clone https://github.com/costaspinto/DocuReadAI.git
cd DocuReadAI

python -m venv venv
```

### Windows

```bash
.\venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the required `.env` variables and then run:

```bash
python main.py
```

---

# Skills Demonstrated

```text
Python
LangChain
Retrieval-Augmented Generation
Large Language Models
Hugging Face
Sentence Transformers
Vector Databases
ChromaDB
Semantic Search
Prompt Engineering
IBM watsonx.ai
PDF Processing
Gradio
AI Application Development
```

---

# Project Summary

DocuReadAI demonstrates the implementation of a complete document question-answering workflow using modern AI engineering technologies.

The application combines:

```text
PDF Processing
      +
Text Chunking
      +
Semantic Embeddings
      +
Vector Search
      +
Context Retrieval
      +
LLM Generation
      =
Document Q&A Assistant
```

The project demonstrates practical implementation of a RAG-based AI application from document ingestion through retrieval and language-model-based response generation.

---

# Author

**Costas Pinto**

MCA — Artificial Intelligence & Machine Learning

- GitHub: [Costas Pinto](https://github.com/costaspinto)

---

# License

This project is intended for educational and personal use.

If the project is distributed as open-source software, an explicit open-source license should be added to the repository.
