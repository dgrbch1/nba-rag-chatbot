# NBA RAG Chatbot

An easy-to-run Retrieval-Augmented Generation (RAG) chatbot that answers questions about NBA players. This small, portfolio-ready Python project demonstrates building embeddings, fast retrieval, prompt augmentation, and answer generation using an LLM.

Key points:
- Minimal, clear CLI interface (`main.py`).
- Embedding caching to reduce API calls and speed up local development.
- Designed for recruiters: simple setup and run instructions.

## What This Project Does

This chatbot:
- reads player data from `basketball_players.json`
- converts player records into embeddings
- retrieves the most relevant players with similarity search
- sends retrieved context to an OpenAI chat model to generate answers

RAG flow:
1. Retrieve relevant player records
2. Augment the prompt with retrieved context
3. Generate the final answer

## Features

- Embedding cache (`embeddings.pkl`) to reduce repeated API calls
- Fast in-memory retrieval (optional FAISS if installed)
- Query timing logs (retrieval and total response time)
- CLI chatbot (`main.py`)
- Benchmark script (`benchmark.py`)

## Project Structure

```text
nba_rag_chatbot/
├── main.py                  # CLI chatbot entry point
├── rag.py                   # Retrieve + Augment + Generate pipeline
├── retrieval.py             # Vector store and similarity search
├── embeddings.py            # Embedding generation + caching
├── benchmark.py             # Simple benchmarking script
├── basketball_players.json  # Local NBA dataset
├── requirements.txt         # Python dependencies
├── .env.example             # Example env file (safe for GitHub)
├── .gitignore               # Ignores .env and local files
└── README.md
```

## Prerequisites

- Python 3.10+ (3.11 recommended)
- OpenAI API key

Recommended: Python 3.11 for best dependency support and performance.

## Setup (Beginner Friendly)

### Installation

Follow these steps to prepare the project locally.

1. Clone the repository and open the project directory:

```bash
git clone <your-repo-url>
cd nba_rag_chatbot
```

2. Create and activate a virtual environment (Python 3.11 recommended):

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

Create a `.env` from the provided `.env.example` and set your `OPENAI_API_KEY`:

```bash
Copy-Item .env.example .env  # Windows PowerShell
```

Then edit `.env` and add your key (do not commit `.env`):

```env
OPENAI_API_KEY=your_real_openai_api_key
```

```bash
git clone <your-repo-url>
cd nba_rag_chatbot
```


Important:
- Never commit `.env` or real API keys to the repository or history.
- `embeddings.pkl` is ignored by `.gitignore` and generated locally.

## Run the Chatbot

```bash
python main.py
```

Example questions:
- `Who has the most championships?`
- `How many points did Michael Jordan score?`
- `Which player is the tallest?`

Type `exit` to quit.

## Run Benchmarks

Dry benchmark (fast, retrieval-focused):

```bash
python benchmark.py --runs 20 --dry
```

Full benchmark (includes model calls, costs API usage):

```bash
python benchmark.py --runs 5 --full
```

## Notes for GitHub

- `.env` is ignored by `.gitignore`
- `.env.example` is included so others can configure their own key
- `embeddings.pkl` is ignored (generated locally)

## Usage

After installing dependencies and creating `.env`, run the chatbot with:

```bash
python main.py
```

The CLI will prompt `You:` — type a question about NBA players (e.g., `Which player is the tallest?`). Type `exit` to quit.

Example:

```
You: Who won the most championships?

Chatbot: Michael Jordan (6 championships) ...
```

## How It Works

This project follows a standard RAG pipeline:

- **Retrieval:** Precompute embeddings for each player record, store them in a vector store, and perform a nearest-neighbor search to find the top-k relevant records for a user query.
- **Augmentation:** The retrieved records are formatted and included as context in the prompt sent to the LLM so it can ground its response on factual data.
- **Generation:** The LLM receives the augmented prompt and generates the final conversational answer.

The pipeline in `main.py` delegates embedding generation to `embeddings.py`, retrieval to `retrieval.py`, and the prompt + generation logic to `rag.py`.

## Performance Improvements

- **Embedding caching:** Embeddings are cached locally (`embeddings.pkl`) to avoid recomputing them on every run and to reduce API cost and latency.
- **Reduced API calls:** Only the final generation call is made per query; embeddings are reused from cache.
- **Faster retrieval:** Uses a small, in-memory vector store for quick nearest-neighbor search. Optional FAISS integration can be added for larger datasets.

## Architecture (text diagram)

User → Embedding → Retrieval → Context → LLM → Response

```
User
  |
  v
[Query text] --(embed)--> [Vector Store] --(retrieve top-k)--> [Context]
  |
  v
[Augmented Prompt] --> [LLM/API] --> [Generated Answer]
```

## Troubleshooting

- If you get API key errors:
  - check `.env` exists
  - check key name is exactly `OPENAI_API_KEY`
- If dependency install fails:
  - use Python 3.11
  - upgrade pip: `python -m pip install --upgrade pip`

## Publishing to GitHub

To create a public GitHub repository and link this project, follow one of the methods below.

- Create the repo with the GitHub CLI (recommended):

```bash
# install GitHub CLI if you don't have it: https://cli.github.com/
gh auth login
gh repo create <your-username>/nba_rag_chatbot --public --source=. --remote=origin --push
```

- Or create the repo on the GitHub website and push from local Git:

```bash
git init
git add .
git commit -m "Initial commit"
# create a repo on https://github.com/new (name it `nba_rag_chatbot`), then:
git remote add origin https://github.com/<your-username>/nba_rag_chatbot.git
git branch -M main
git push -u origin main
```

Once pushed, your project will be available at:

https://github.com/<your-username>/nba_rag_chatbot

Replace `<your-username>` with your GitHub username. If you'd like, I can run `git init` and make the initial commit here, and/or create the GitHub repo for you if you provide a GitHub username or allow me to use the `gh` CLI (and authenticate). 

