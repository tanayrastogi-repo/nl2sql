from typing import TypedDict

from pydantic import BaseModel, ConfigDict, field_validator  # noqa: E402


class SQLQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: str
    param_values: list[str]

    @field_validator("sql")
    @classmethod
    def sql_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sql must not be empty")
        return v


class AgentState(TypedDict):
    question: str
    db_schema: str
    db_path: str

    select_attempts: int
    attempts: int

    selected_tables: list[str] | None
    sql_query: str | None
    param_values: list[str] | None
    error: str | None
