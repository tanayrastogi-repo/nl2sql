import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from typing import Literal  # noqa: E402

from langchain_ollama import ChatOllama  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402

from src.models import AgentState, SQLQuery  # noqa: E402
from src.schema import get_schema  # noqa: E402
from src.validator import validate_sql as validate_sql_func  # noqa: E402


def _get_llm() -> ChatOllama:
    model = "phi4-mini-reasoning:latest"
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return ChatOllama(model=model, base_url=base_url)


def _extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        return json_match.group(1).strip()
    return text.strip()


def select_tables(state: AgentState) -> AgentState:
    llm = _get_llm()

    prompt = f"""You are an expert at analyzing database schemas.
Given a question and full database schema, identify the minimum set of tables needed to answer the question.

Schema:
{state["db_schema"]}

Question: {state["question"]}

Return ONLY a JSON array of table names, e.g., ["customers", "orders"]"""

    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else ""
    clean = _extract_json(content)

    try:
        tables = json.loads(clean)
        if not isinstance(tables, list) or not tables:
            raise ValueError("Must be a non-empty list")
        for t in tables:
            if not isinstance(t, str):
                raise ValueError("All table names must be strings")
        for t in tables:
            prefix = t + " ("
            if not any(
                line.startswith(prefix) for line in state["db_schema"].splitlines()
            ):
                raise ValueError(f"Table '{t}' not found in schema")
    except Exception as e:
        new_attempts = state["select_attempts"] + 1
        if new_attempts >= 2:
            return {
                **state,
                "select_attempts": new_attempts,
                "error": f"Validation failed after {new_attempts} attempts. Last error: {e}",
                "selected_tables": None,
            }
        return {
            **state,
            "select_attempts": new_attempts,
            "error": f"Invalid JSON response from table selection: {e}",
        }

    narrowed_schema = get_schema(state["db_path"], tables)
    return {
        **state,
        "selected_tables": tables,
        "db_schema": narrowed_schema,
        "error": None,
    }


def generate_sql(state: AgentState) -> AgentState:
    llm = _get_llm()

    system = """You are an expert SQLite SQL generator.

Rules:
- SELECT statements only (no INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE)
- Use ? placeholders for literal values in the question
- Build param_values array with the literal values in order
- No SELECT * — only select needed columns
- Use explicit GROUP BY when using aggregate functions
- Use table aliases for readability

Output Format: Return ONLY valid JSON with keys: "sql" and "param_values"
Example:
Question: "Get the total revenue for Adventure Works Cycles. Include the contact information."
{
    "sql": "SELECT c.company, c.city, c.email, SUM(o.total) AS revenue FROM customers c INNER JOIN orders o ON c.id = o.customer_id WHERE c.company = ? GROUP BY c.company, c.city, c.email",
    "param_values": ["Adventure Works Cycles"]
}"""

    if state["attempts"] == 0:
        prompt = f"""{system}

Schema (selected tables only):
{state["db_schema"]}

Question: {state["question"]}"""
    else:
        prompt = f"""{system}

PREVIOUS SQL: {state["sql_query"]}
VALIDATION ERROR: {state["error"]}
INSTRUCTION: Fix the SQL above based on the validation error and regenerate.

Schema (selected tables only):
{state["db_schema"]}

Question: {state["question"]}"""

    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else ""
    clean = _extract_json(content)

    try:
        parsed = SQLQuery.model_validate_json(clean)
    except Exception as e:
        new_attempts = state["attempts"] + 1
        if new_attempts >= 3:
            return {
                **state,
                "attempts": new_attempts,
                "error": f"Validation failed after {new_attempts} attempts. Last error: {e}",
            }
        return {
            **state,
            "attempts": new_attempts,
            "error": f"Parse error: {e}",
        }

    return {
        **state,
        "sql_query": parsed.sql,
        "param_values": parsed.param_values,
        "error": None,
    }


def validate_sql_node(state: AgentState) -> AgentState:
    valid, error = validate_sql_func(
        state["sql_query"] or "", state["param_values"] or [], state["db_schema"]
    )
    if not valid:
        return {
            **state,
            "error": error,
            "attempts": state["attempts"] + 1,
        }
    return {
        **state,
        "error": None,
    }


def should_retry_select(state: AgentState) -> Literal["retry", "continue", "end"]:
    if state.get("selected_tables"):
        return "continue"
    if state["select_attempts"] >= 2:
        return "end"
    return "retry"


def should_retry_sql(state: AgentState) -> Literal["retry", "end"]:
    if state.get("error"):
        if state["attempts"] >= 3:
            return "end"
        return "retry"
    return "end"


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("select_tables", select_tables)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate_sql_node)

    graph.add_edge(START, "select_tables")

    graph.add_conditional_edges(
        "select_tables",
        should_retry_select,
        {
            "retry": "select_tables",
            "continue": "generate_sql",
            "end": END,
        },
    )

    graph.add_edge("generate_sql", "validate_sql")

    graph.add_conditional_edges(
        "validate_sql",
        should_retry_sql,
        {
            "retry": "generate_sql",
            "end": END,
        },
    )

    return graph.compile()
