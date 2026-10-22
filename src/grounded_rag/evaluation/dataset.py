"""Evaluation question set: loading and validation."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class ExpectedSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    doc_name: str
    page: int = Field(ge=1)


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    expected_sources: tuple[ExpectedSource, ...] = ()
    answerable: bool = True

    @model_validator(mode="after")
    def _unanswerable_has_no_sources(self) -> "EvalCase":
        if not self.answerable and self.expected_sources:
            raise ValueError(f"{self.id}: unanswerable cases must not list expected_sources")
        return self


_CASES = TypeAdapter(tuple[EvalCase, ...])


def parse_cases(raw: str) -> tuple[EvalCase, ...]:
    cases = _CASES.validate_python(json.loads(raw))
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case ids: {', '.join(duplicates)}")
    return cases


def load_cases(path: Path) -> tuple[EvalCase, ...]:
    return parse_cases(path.read_text(encoding="utf-8"))
