# ResumeIQ — AI-Powered Resume Screening & Matching System

An AI-powered resume screening and retrieval application built using **Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), ChromaDB, and Streamlit**.

The project helps recruiters retrieve relevant candidate information, evaluate resumes against job descriptions, identify skill gaps, and explore candidate rankings through an interactive dashboard.

## Features

- **Resume Processing:** Load and process resumes in PDF format.
- **Text Chunking:** Split resume documents into smaller chunks for retrieval.
- **Embeddings:** Generate vector embeddings using OpenAI.
- **Vector Database:** Store and retrieve resume chunks using ChromaDB.
- **Semantic Retrieval:** Retrieve relevant resume information based on user queries.
- **AI-Powered Screening:** Evaluate candidate resumes against job descriptions using an LLM.
- **Candidate Ranking:** Rank candidates based on job-description relevance.
- **Skill Gap Analysis:** Identify matching and missing skills.
- **Interactive Dashboard:** Explore screening results using Streamlit.
- **RAG Evaluation:** Evaluate generated answers and retrieved context using Ragas.

## Tech Stack

- **Programming:** Python
- **LLM:** OpenAI
- **Frameworks:** LangChain
- **Retrieval:** Retrieval-Augmented Generation (RAG)
- **Vector Database:** ChromaDB
- **Embeddings:** OpenAI Embeddings
- **Evaluation:** Ragas
- **Dashboard:** Streamlit
- **Data Processing:** Pandas

##  Project Structure

```text
ResumeIQ/
├── main.py                 # Streamlit dashboard
├── resume_project.ipynb    # RAG pipeline and evaluation
├── requirements.txt        # Project dependencies
├── .env                    # API key (not committed)
├── .gitignore              # Files excluded from Git
└── ResumeRAG_1000/         # Local ChromaDB vector store
```

The vector store directory is generated locally and may not be included in the repository. Resume PDFs are not included; provide your own authorized resume dataset.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the placeholders with your GitHub repository URL and folder name.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the OpenAI API key

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_openai_api_key
```

Never upload your actual API key to GitHub.

### 5. Prepare the resume dataset

Place your authorized PDF resumes in a local folder. Update the resume directory path in the notebook to match your local setup.

Run the notebook cells to load resumes, split documents into chunks, generate embeddings, and create the ChromaDB vector store.

### 6. Run the Streamlit dashboard

Ensure the vector store has been created and the dashboard is configured to use it.

```bash
python -m streamlit run main.py
```

Streamlit will display a local URL in the terminal.

## RAG Evaluation

The project uses Ragas to evaluate the quality of the RAG pipeline against reference answers.

Metrics used:

- **Answer Correctness:** Evaluates generated answers against reference answers.
- **Answer Relevancy:** Evaluates whether answers address the questions.
- **Context Precision:** Evaluates the relevance of retrieved context.
- **Context Recall:** Evaluates how well the retrieved context covers the reference answer.

Evaluation scores depend on the questions, reference answers, retrieved documents, and generated responses. Actual results should be reported from the executed evaluation rather than assumed.

## Privacy and Security

- Do not commit API keys or secrets.
- Do not publish private or identifiable resume information.
- Use authorized, anonymized, or synthetic resume data when sharing demonstrations.
- Keep local datasets and sensitive files out of the public repository.

## Project Objective

To explore how RAG and LLMs can support resume information retrieval, job-description matching, candidate screening, and evaluation of generated answers.

##  Author

**Tejaswi Siddagoni**

B.Tech — Artificial Intelligence and Machine Learning

Interested in Generative AI, RAG, LLM applications, and AI Engineering.


