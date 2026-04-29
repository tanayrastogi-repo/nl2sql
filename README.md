# NL2SQL Agent

AI agent that converts natural language instructions to SQLite SQL queries. Returns structured JSON output with SQL and parameterized values — no query execution (generation-only MVP).

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Verbose Mode](#verbose-mode)
  - [Examples](#examples)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [Development](#development)
  - [Code Quality](#code-quality)
  - [LangSmith Tracing](#langsmith-tracing)

## Project Overview

NL2SQL is a local-first AI agent that translates natural language questions into SQLite SQL queries. It uses a two-step LLM approach (table selection → SQL generation) with a strict 5-step validation pipeline to ensure safe, correct SQL output.

**Example Input:**
> "Get the total revenue for Adventure Works Cycles. Include the contact information as well."

**Success Output:**
```json
{
    "sql": "SELECT c.company, c.city, c.email, SUM(o.total) AS revenue FROM customers c INNER JOIN orders o ON c.id = o.customer_id WHERE c.company = ? GROUP BY c.company, c.city, c.email",
    "param_values": ["Adventure Works Cycles"],
    "status": "success"
}
```

## Features

- **Local LLM**: Runs entirely on your machine via Ollama (zero API costs, full privacy)
- **Two-Step Generation**: Selects relevant tables first, then generates SQL with narrowed schema
- **Strict Validation**: 5-step `sqlglot` pipeline (syntax, security, table/column allowlist, param count, literal check)
- **Retry Logic**: Automatic retries with error feedback for both table selection (max 2) and SQL generation (max 3)
- **LangGraph Orchestration**: Stateful multi-step workflow with conditional edges
- **LangSmith Integration**: Optional tracing of all LLM calls and graph transitions

## Project Structure

```
nl2sql/
├── src/
│   ├── __init__.py
│   ├── models.py          # AgentState TypedDict + SQLQuery Pydantic model
│   ├── schema.py          # SQLite schema introspection via PRAGMA
│   ├── validator.py       # 5-step SQL validation pipeline
│   └── graph.py          # LangGraph StateGraph workflow
├── tests/
│   └── test.db          # Sample SQLite database for testing
├── .env                  # Environment variables (not committed)
├── nl2sql.py             # CLI entry point
├── pyproject.toml        # Project configuration and dependencies
├── SPEC.md               # Full design document
└── README.md
```

## Getting Started

### Prerequisites

- **Python** ≥ 3.11
- **uv** (package manager) — [installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Ollama** — [download](https://ollama.com/download) and install
- **`phi4-mini-reasoning:latest`** model pulled in Ollama

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd nl2sql
   ```

2. Install dependencies with `uv`:
   ```bash
   uv sync
   ```

3. Pull the required Ollama model:
   ```bash
   ollama pull phi4-mini-reasoning:latest
   ```

### Environment Setup

Create a `.env` file in the project root (optional, defaults work out of the box):

```bash
# LangSmith Tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=nl2sql-dev
LANGCHAIN_API_KEY=your_langsmith_api_key

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_API_KEY=
```

## Usage

### Basic Usage

```bash
uv run nl2sql --db tests/test.db "Your question here"
```

### Verbose Mode

```bash
uv run nl2sql --db tests/test.db --verbose "Your question here"
```

### Examples

**Success case:**
```bash
uv run nl2sql --db tests/test.db "Get the total revenue for Adventure Works Cycles"
```

Output:
```json
{
  "sql": "SELECT SUM(o.total) AS revenue FROM customers c INNER JOIN orders o ON c.id = o.customer_id WHERE c.company = ? GROUP BY c.company",
  "param_values": ["Adventure Works Cycles"],
  "status": "success"
}
```

**Failure case (max attempts exceeded):**
```bash
uv run nl2sql --db tests/test.db "Get data from non_existent_table"
```

Output:
```json
{
  "sql": null,
  "param_values": null,
  "status": "error",
  "error": "Validation failed after 2 attempts. Last error: ..."
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | (none) | LangSmith API key |
| `LANGCHAIN_PROJECT` | `nl2sql-dev` | LangSmith project name |

**Note:** The LLM model (`phi4-mini-reasoning:latest`) is hardcoded in `src/graph.py`.

## Development

### Code Quality

After any code changes, run the mandatory linting and type-checking:

```bash
# Ruff (lint + format check)
uv run ruff check src/ nl2sql.py
uv run ruff format --check src/ nl2sql.py

# MyPy (type checking)
uv run mypy src/ nl2sql.py
```

To auto-format with Ruff:
```bash
uv run ruff format src/ nl2sql.py
```

### LangSmith Tracing

To enable tracing of all LangGraph nodes and LLM calls:

1. Set `LANGCHAIN_TRACING_V2=true` in `.env`
2. Add your `LANGCHAIN_API_KEY`
3. Run your query — traces will appear in the LangSmith dashboard

No code changes required — LangChain/LangGraph auto-integrate via env vars.
