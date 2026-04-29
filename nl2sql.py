import argparse
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from src.graph import build_graph  # noqa: E402
from src.models import AgentState  # noqa: E402
from src.schema import get_schema  # noqa: E402

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert natural language to SQL queries."
    )
    parser.add_argument("--db", required=True, help="Path to SQLite .db file")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    parser.add_argument("question", help="Natural language question")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print(
            json.dumps(
                {"status": "error", "error": f"Database file not found: {args.db}"}
            )
        )
        sys.exit(1)

    db_schema = get_schema(args.db)
    initial_state: AgentState = {
        "question": args.question,
        "db_schema": db_schema,
        "db_path": args.db,
        "select_attempts": 0,
        "attempts": 0,
        "selected_tables": None,
        "sql_query": None,
        "param_values": None,
        "error": None,
    }

    graph = build_graph()
    result = graph.invoke(initial_state)

    if result.get("error"):
        attempts = result.get("attempts", 0)
        select_attempts = result.get("select_attempts", 0)
        if select_attempts >= 2:
            error_msg = f"Validation failed after {select_attempts} attempts. Last error: {result['error']}"
        else:
            error_msg = f"Validation failed after {attempts} attempts. Last error: {result['error']}"
        output = {
            "sql": result.get("sql_query", ""),
            "param_values": result.get("param_values", []),
            "status": "error",
            "error": error_msg,
        }
    else:
        output = {
            "sql": result.get("sql_query", ""),
            "param_values": result.get("param_values", []),
            "status": "success",
        }

    print(json.dumps(output, indent=2))

    if args.verbose:
        print("\n--- DEBUG INFO ---", file=sys.stderr)
        print(
            f"Final state: {json.dumps(result, indent=2, default=str)}", file=sys.stderr
        )


if __name__ == "__main__":
    main()
