Here's a sample `README.md` file for your Travel Buddy Chatbot application:

---

# BRD to User Story Converting App - Generative AI RAG Application

This project is a **Generative AI-based BRD to User Story Converting App** that uses Retrieval-Augmented Generation (RAG) to provide tips based on information from various file formats (PDF, DOCX, PPTX, Excel). It uses **LangChain**, **OpenAI's GPT model**, and **FAISS** for handling natural language queries and retrieving relevant information.

## Features
- Upload and process files like PDFs, Word documents, PowerPoints, Excel spreadsheets, and plain text files.
- Retrieve travel-related information from embedded text.
- Answer travel-related questions using OpenAI's GPT model.
- Simple user interface built with **Streamlit**.

## Folder Structure
- `data/file_folder/`: Place your documents (PDF, DOCX, PPTX, XLSX, TXT) here for processing.
- `app_secret.py`: Contains your **OpenAI API Key**.
- `requirements.txt`: Contains all necessary libraries to run the application.

## Requirements

Install the required Python packages from the `requirements.txt` file. Here's the list:

```
streamlit==1.24.0
PyPDF2==3.0.1
python-docx==0.8.11
python-pptx==0.6.21
pandas==2.1.1
openai
langchain
langchain_openai
faiss-cpu==1.8.0
langchain-community
langchain-core
```

## Prerequisites

1. **Python 3.8+** installed.
2. An **OpenAI API Key** for using the GPT model. You can sign up for an API key at [OpenAI](https://beta.openai.com/signup/).

## Installation

### 1. Clone the repository:

```bash
git clone https://git.virtusa.com/team-i/backend/-/tree/hackathon
cd chatapp
```

### 2. Install the required dependencies:

Make sure you are in the root of your project directory, then run:

```bash
python -m pip install -r requirements.txt
```

### 3. Set up your OpenAI API Key:

Create a file called `app_secret.py` in the root directory with the following content:

```python
OPENAI_API_KEY = "your-openai-api-key-here"
```

Replace `"your-openai-api-key-here"` with your actual OpenAI API key.

### 4. Add your files for processing:

Place your documents (PDF, DOCX, PPTX, Excel, etc.) inside the `data/file_folder` directory.

### 5. Run the Streamlit Application:

Start the application with the following command:

```bash
streamlit run app.py
```

```
python -m streamlit run E:\MscFinalProjects\mscProj\app.py
```

### 6. Interact with the chatbot:

Once the Streamlit server is up, open your web browser and navigate to `http://localhost:8501/`. Here you can input travel-related queries, and the chatbot will provide answers based on the processed documents.

## How It Works

1. **Document Upload & Processing**: The app reads documents (PDF, DOCX, PPTX, XLSX, TXT) from the `data/file_folder` directory and extracts text.
2. **Text Splitting**: Extracted text is split into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embedding**: Each chunk is embedded into vectors using OpenAI's `Embeddings`.
4. **Vector Store**: FAISS is used to store the vector embeddings for quick retrieval based on user queries.
5. **Query Input**: Users input travel-related questions.
6. **Answer Generation**: The chatbot retrieves the most relevant information from the vector store and uses OpenAI’s GPT model to generate responses.



## Troubleshooting

- If the chatbot does not respond, ensure that the documents are correctly placed in the `data/file_folder` and that they contain travel-related information.
- Make sure your OpenAI API key is valid and correctly set in `app_secret.py`.
- Double-check that all dependencies are installed correctly by reviewing the `requirements.txt`.

---

## Contact

For any issues or questions, feel free to reach out via `dilshantilakaratne29@gmail.com`. 


---

This `README.md` file provides an overview of the project, installation steps, instructions on how to run the application, and contact details.