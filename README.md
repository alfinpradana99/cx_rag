# Customer Service RAG Demo

A Streamlit application that demonstrates a Retrieval-Augmented Generation (RAG) pipeline for customer service responses.

## Features

- Interactive customer service question answering
- Product-specific responses with ingredient information
- Debug view showing retrieved contexts and generated prompts
- Configurable retrieval parameters

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `streamlit run app.py`
4. Enter your API keys in the sidebar

## Deployment

This app is deployed on Streamlit Community Cloud. You'll need:
- Google Generative AI API key
- Pinecone API key

## Technologies Used

- Streamlit for the web interface
- Google Generative AI (Gemini) for embeddings and text generation
- Pinecone for vector database
- RAG pipeline for context-aware responses
