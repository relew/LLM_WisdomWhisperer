# Philosophy Assistant

## Problem Description

In an increasingly complex world, understanding philosophical principles can offer valuable insights into personal growth, decision-making, and well-being. However, accessing and interpreting philosophical wisdom from diverse traditions like Stoicism and Zen can be challenging for many people. 

The **Philosophy Assistant** addresses these challenges by providing a digital assistant that delivers:

- **Accessible Philosophical Guidance:** Many individuals struggle to interpret and apply ancient philosophical teachings to modern life. This LLM-based application offers accessible and actionable insights grounded in Stoic and Zen philosophies, making these teachings more relevant and understandable.
- **Contextualized Answers:** Traditional philosophical texts can be dense and difficult to navigate. The assistant simplifies this by providing clear, contextually relevant answers to user questions based on established philosophical principles.
- **Enriched Understanding through Quotes:** By including relevant quotes from notable philosophers, the assistant not only answers questions but also connects users with original sources, enhancing their understanding and engagement.
- **Educational Value:** Explanations of philosophical quotes and principles promote deeper learning, helping users integrate concepts into their daily lives and decision-making processes.
- **Personalized Philosophy Learning:** Users can interactively explore Stoic and Zen philosophies, receiving personalized responses based on their specific questions and interests.

In essence, the Philosophy Assistant bridges ancient wisdom and contemporary needs, making philosophical teachings more accessible, relevant, and impactful in today's world.

## Dataset Description

The dataset for the Philosophy Assistant consists of curated question-answer pairs categorized by philosophical figures and ideologies. Key components include:

- **Category:** The philosopher or philosophical school (e.g., al-Kindi, Alexander of Aphrodisias, Stoicism).
- **Question:** Specific philosophical queries relating to each category.
- **Answer:** Detailed, contextual responses based on philosophical teachings.
- **Ideology:** The philosophy or tradition associated with the question and answer, such as Stoicism or Zen Buddhism.

This dataset serves as the foundation for delivering accurate, context-rich responses.

## Solution Summary

### Ingestion

To explore the dataset and how it was processed, see the notebook:  
[Dataset Exploration](notebooks/step0_dataset_exploration.ipynb)

### Retrieval Flow

For the code implementing the basic retrieval-augmented generation (RAG) flow, refer to:  
[Basic RAG Flow](notebooks/step1_basic_rag_flow.ipynb)

### Retrieval Evaluation

To evaluate the system’s retrieval accuracy, refer to the notebook:  
[Retrieval Evaluation](notebooks/step3_retrieval_evaluation.ipynb)

#### Evaluation Results:
- **Minsearch without Boosting** (num_results=5):
  - Hit Rate: 0.80
  - MRR: 0.67
- **Weighted Field Parameter Tuning**:
  - Hit Rate: 0.92
  - MRR: 0.76
- **Vector Search (miniLM)**:
  - Hit Rate: 0.96
  - MRR: 0.83
- **TFIDF (Sklearn) with Hyperparameter Tuning**:
  - Hit Rate: 0.92
  - MRR: 0.77

### RAG Evaluation

For the code evaluating retrieval-augmented generation, check:  
[RAG Evaluation](notebooks/step4_rag_evaluation.ipynb)

#### Results:
- **Cosine Similarity** (Answer -> Question -> Answer): 0.82
- **LLM as a Judge**:
  - Answer -> Question -> Answer: 149/150 (99%) RELEVANT
  - Question -> Answer: 145/150 (97%) RELEVANT

## Interface

The application provides an interactive **Streamlit** interface where users can:

- Select an ideology (Stoicism or Zen Buddhism).
- Choose between models (e.g., `ollama/phi3`, `openai/gpt-4o`).
- Choose between text-based or vector search.
- Ask philosophical questions, filter previous questions based on relevance, and provide feedback on answers (positive or negative).
  
The system automatically evaluates answers, categorizing them as **relevant**, **partly relevant**, or **non-relevant** using `openai/gpt-4o-mini`.

## Ingestion Pipeline

The ingestion pipeline for the dataset is automated with a Python script.

## Containerization

The application is containerized using **Docker Compose**, which includes services for:

- Streamlit (application interface)
- PostgreSQL (database)
- Grafana (monitoring)

The `.env` file contains the necessary environmental variables.

## Monitoring

**Grafana** is used to collect user feedback, and a monitoring dashboard is available with over 5 charts visualizing various metrics.

## How to Run the Application

### Prerequisites:

- Python 3.9
- Docker and Docker Compose for containerization
- [Minsearch](philosophy_app/minsearch_xtra.py) for full-text search
- Streamlit of user interface
- Grafana for monitoring and PostgreSQL as the backend for it
- OpenAI as an LLM

## Preparation

Since we use OpenAI, you need to provide the API key:

1. Install `direnv`. If you use Ubuntu, run `sudo apt install direnv` and then `direnv hook bash >> ~/.bashrc`.
2. Set up your `.envrc` file with `OPENAI_API_KEY`.
3. For OpenAI, it's recommended to create a new project and use a separate key.
4. Run `direnv allow` to load the key into your environment.

For dependency management, we use pipenv, so you need to install it:

```bash
pip install pipenv
```

Once installed, you can install the app dependencies:

```bash
pipenv install --dev
```

## Running the application

**Navigate to the project directory:**

### Running with Docker-Compose

The easiest way to run the application is with `docker-compose`:

```bash
docker-compose up
```

### Access the Application

To access the various components of the application, use the following URLs:

- **Streamlit Interface:**  
  Navigate to [http://localhost:8501](http://localhost:8501) to interact with the Streamlit app.

- **Grafana Monitoring:**  
  Open [http://localhost:3000](http://localhost:3000) to view monitoring metrics. Use the default credentials to log in:

  - **Username:** `admin`  
  - **Password:** `admin`

### Grafana Setup

1. **Log in to Grafana** with the default credentials:
   - **Username:** `admin`
   - **Password:** `admin`

2. **Import the Dashboard:**
   - Upload the `grafana.json` file to set up your monitoring dashboard.

Enjoy exploring the application!



