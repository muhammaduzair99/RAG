# RAG
RAG Project
# Gulliver's Travels Question Answering System

This project implements a question-answering system designed to retrieve relevant information from the text corpus of *Gulliver's Travels*. The system uses a combination of retrieval-based techniques and large language models (LLMs) to provide concise answers to user queries. The pipeline includes text embedding, vector similarity search, and generative question-answering, all optimized for accuracy and efficiency.

## Project Overview

The goal of this project is to explore retrieval-augmented generation (RAG) for long-form text documents. Using *Gulliver's Travels* as a sample corpus, this system retrieves relevant chunks of text in response to user questions and generates answers based on those passages.

Key components of the project:
- **Text Embedding**: Embeddings from `sentence-transformers/all-mpnet-base-v2` capture semantic meaning.
- **Retrieval**: `FAISS` for efficient similarity search.
- **Answer Generation**: `google/flan-t5-large` provides fluent answers based on the retrieved content.

## Dataset

The dataset for this project is *Gulliver's Travels*, which has been preprocessed to remove extra elements such as the table of contents and footnotes, making it more suitable for question answering. Text is chunked into 500-character segments with an overlap of 50 characters to ensure context is maintained in responses.

Sourced from Project Gutenberg: https://www.gutenberg.org/ebooks/829

## Platform and Environment

The project is developed and tested on Google Colab, leveraging its GPU capabilities to speed up both retrieval and answer generation. It uses the following tools and libraries:
- **Hugging Face Transformers** for LLMs
- **Sentence-Transformers** for embeddings
- **LangChain** and **FAISS** for efficient vector storage and similarity search.

## Model and Embedding Details

### Models
- **LLM**: `google/flan-t5-large` chosen for its balance between speed and output quality. Other models, such as `bigscience/T0_3B` and `google/flan-t5-xxl`, were tested but excluded from the final pipeline for performance reasons.
- **Summarization (optional)**: `facebook/bart-large-cnn` was integrated and tested as a summarization step, though results showed more accuracy without this preprocessing.

### Embeddings
- `sentence-transformers/all-mpnet-base-v2` is used to generate embeddings for the text chunks. This embedding model was chosen for its accuracy in semantic representation.

## Methodology

1. **Data Loading and Splitting**: The *Gulliver's Travels* text is loaded and split into chunks using `LangChain's RecursiveCharacterTextSplitter`.
2. **Embedding and Vector Storage**: Each chunk is converted into embeddings using `sentence-transformers/all-mpnet-base-v2` and stored in a FAISS vector store.
3. **Question Processing and Retrieval**: When a question is asked, FAISS performs a similarity search to retrieve the most relevant text chunks.
4. **Answer Generation**: Retrieved passages are fed into `google/flan-t5-large`, which generates concise answers based on the query and context.
5. **Performance Optimization**: The pipeline has been optimized to reduce processing time, such as by limiting context length and carefully selecting top-performing models.


## Acknowledgments

Special thanks to Hugging Face for model hosting, Google Colab for providing GPU support, Project Gutenberg for corpus sourcing, and OpenAI's ChatGPT for guidance and assistance throughout the project. This collaboration helped optimize the pipeline and improve answer quality.
