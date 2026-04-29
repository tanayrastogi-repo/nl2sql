import sqlite3
from typing import Optional  # noqa: E402


def get_schema(db_path: str, tables: Optional[list[str]] = None) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if tables:
        table_names = tables
    else:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = [row[0] for row in cursor.fetchall()]

    lines: list[str] = []

    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = cursor.fetchall()

        cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
        fks = cursor.fetchall()

        col_parts: list[str] = []
        pk_cols: list[str] = []

        for cid, name, type_, notnull, dflt_value, pk in columns:
            col_def = f"{name} {type_}"
            if pk:
                pk_cols.append(name)
            else:
                if notnull:
                    col_def += " NOT NULL"
            col_parts.append(col_def)

        if pk_cols:
            pk_str = f"PRIMARY KEY ({', '.join(pk_cols)})"
            col_parts.insert(len(pk_cols), pk_str)

        for id_, seq, ref_table, from_col, to_col, on_update, on_delete, match in fks:
            col_parts.append(
                f"FOREIGN KEY ({from_col}) REFERENCES {ref_table}({to_col})"
            )

        lines.append(f"{table_name} ({', '.join(col_parts)})")

    conn.close()
    return "\n".join(lines)
