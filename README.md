RAG Powered AI Chatbot

A web-based Retrieval-Augmented Generation (RAG) chatbot built using Streamlit and LangChain, powered by Groq's LLaMA-3.3-70B model. The chatbot allows users to upload PDF or CSV files and ask questions based on their content using semantic search and vector embeddings.

Features

Upload PDF or CSV documents
Ask questions based on uploaded content
RAG-based contextual answering
Source document preview (for PDF)
Conversational memory
Save and revisit previous chats
Simple Streamlit interface

Project Structure

rag-powered-ai-chatbot/
│
├── Final.py
├── img.png
├── PDF.pdf
├── CSV.csv
├── requirements.txt
├── README.md

Tech Stack

Python, Streamlit, LangChain, FAISS, HuggingFace Embeddings, Groq LLM API, LLaMA 3.3 (70B), PyPDF2, Pandas

How It Works

User uploads a PDF or CSV file
Text is split into chunks
Embeddings are created using sentence-transformers
FAISS vector store is built
User enters a question
Relevant chunks are retrieved
Prompt is sent to Groq-hosted LLaMA 3.3
Model generates answer using retrieved context
Chat history is stored in session
