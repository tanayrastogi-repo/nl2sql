import sqlglot  # noqa: E402
from sqlglot import expressions as exp  # noqa: E402


def validate_sql(
    sql: str, param_values: list[str], schema_str: str
) -> tuple[bool, str]:
    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        return False, f"Syntax error: {e}"

    if not isinstance(ast, exp.Select):
        return False, f"Only SELECT statements allowed, got: {type(ast).__name__}"

    for node in ast.walk():
        if isinstance(
            node,
            (
                exp.Insert,
                exp.Update,
                exp.Delete,
                exp.Drop,
                exp.Alter,
                exp.Create,
                exp.TruncateTable,
            ),
        ):
            return False, f"Disallowed SQL operation: {type(node).__name__}"

    tables = {t.name for t in ast.find_all(exp.Table)}
    columns = {c.name for c in ast.find_all(exp.Column)}

    allowed_tables, allowed_columns = _parse_schema(schema_str)

    unauthorized_tables = tables - allowed_tables
    if unauthorized_tables:
        return False, f"Unauthorized table(s): {', '.join(sorted(unauthorized_tables))}"

    unauthorized_columns = columns - allowed_columns
    if unauthorized_columns:
        return (
            False,
            f"Unauthorized column(s): {', '.join(sorted(unauthorized_columns))}",
        )

    placeholders = list(ast.find_all(exp.Placeholder))
    placeholder_count = len(placeholders)
    if placeholder_count != len(param_values):
        return (
            False,
            f"Placeholder count ({placeholder_count}) does not match param_values length ({len(param_values)})",
        )

    literals = list(ast.find_all(exp.Literal))
    if literals:
        values = [lit.this for lit in literals]
        return (
            False,
            f"Hardcoded literal(s) found: {values}. Use ? placeholders instead.",
        )

    return True, ""


def _parse_schema(schema_str: str) -> tuple[set[str], set[str]]:
    tables: set[str] = set()
    columns: set[str] = set()

    for line in schema_str.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        paren_idx = line.find("(")
        if paren_idx == -1:
            continue
        table_name = line[:paren_idx].strip()
        tables.add(table_name)

        body = line[paren_idx + 1 : line.rfind(")")]
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            if part.startswith("FOREIGN KEY") or part.startswith("PRIMARY KEY"):
                continue
            tokens = part.split()
            if tokens:
                columns.add(tokens[0])

    return tables, columns
