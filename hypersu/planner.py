"""Standalone query planner for decomposing complex questions into sub-queries.

This module is intentionally not wired into HyperSU's main retrieval or QA
pipeline. It can be imported independently or used as a small CLI utility.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from hypersu.utils import LLM_Model


PLANNER_SYSTEM_PROMPT = """
You are a query planning agent for multi-hop retrieval.

Your job is to decide whether the user query should be decomposed into smaller
sub-queries, and if so, produce a short, high-value plan for retrieval.

Rules:
- Return valid JSON only.
- Preserve the original meaning of the query.
- Decompose only when it improves retrieval or reasoning.
- Prefer 2-5 sub-queries for complex questions.
- Each sub-query should be independently searchable.
- Avoid trivial paraphrases or overly overlapping sub-queries.
- Keep sub-queries concrete and evidence-seeking.
- The final answer must still be synthesized from all relevant evidence.

Return a JSON object with this schema:
{
  "is_complex": true,
  "reasoning": "short explanation",
  "sub_queries": [
    {
      "id": "sq1",
      "query": "sub-question text",
      "purpose": "why this helps",
      "answer_type": "entity|event|cause|comparison|attribute|timeline|other"
    }
  ],
  "synthesis_instruction": "how to combine the evidence"
}

If the query is not complex enough to decompose, return:
{
  "is_complex": false,
  "reasoning": "short explanation",
  "sub_queries": [
    {
      "id": "sq1",
      "query": "the original query",
      "purpose": "answer directly",
      "answer_type": "other"
    }
  ],
  "synthesis_instruction": "answer directly from the best supporting evidence"
}
""".strip()


@dataclass
class PlannedSubQuery:
    id: str
    query: str
    purpose: str
    answer_type: str = "other"


@dataclass
class QueryPlan:
    original_query: str
    is_complex: bool
    reasoning: str
    sub_queries: list[PlannedSubQuery] = field(default_factory=list)
    synthesis_instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "is_complex": self.is_complex,
            "reasoning": self.reasoning,
            "sub_queries": [asdict(item) for item in self.sub_queries],
            "synthesis_instruction": self.synthesis_instruction,
        }


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = _strip_code_fences(text)
    if not text:
        return None

    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _fallback_plan(query: str, reason: str) -> QueryPlan:
    return QueryPlan(
        original_query=query,
        is_complex=False,
        reasoning=reason,
        sub_queries=[
            PlannedSubQuery(
                id="sq1",
                query=query,
                purpose="answer directly",
                answer_type="other",
            )
        ],
        synthesis_instruction="answer directly from the best supporting evidence",
    )


def _normalize_sub_queries(items: Any, original_query: str) -> list[PlannedSubQuery]:
    if not isinstance(items, list):
        return [
            PlannedSubQuery(
                id="sq1",
                query=original_query,
                purpose="answer directly",
                answer_type="other",
            )
        ]

    normalized: list[PlannedSubQuery] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        normalized.append(
            PlannedSubQuery(
                id=str(item.get("id") or f"sq{idx}"),
                query=query,
                purpose=str(item.get("purpose") or "gather supporting evidence").strip(),
                answer_type=str(item.get("answer_type") or "other").strip(),
            )
        )

    if normalized:
        return normalized

    return [
        PlannedSubQuery(
            id="sq1",
            query=original_query,
            purpose="answer directly",
            answer_type="other",
        )
    ]


class QueryPlanner:
    """Standalone planner agent for complex-query decomposition."""

    def __init__(self, llm_model_name: str = "gpt-4o-mini", max_subqueries: int = 5):
        self.llm_model = LLM_Model(llm_model_name)
        self.max_subqueries = max(1, max_subqueries)

    def _build_messages(self, query: str, extra_context: str | None = None) -> list[dict[str, str]]:
        user_payload = {
            "query": query,
            "max_subqueries": self.max_subqueries,
        }
        if extra_context:
            user_payload["extra_context"] = extra_context
        return [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

    def plan(self, query: str, extra_context: str | None = None) -> QueryPlan:
        query = (query or "").strip()
        if not query:
            return _fallback_plan("", "empty query")

        raw_response = self.llm_model.infer(self._build_messages(query, extra_context))
        payload = _extract_json_object(raw_response)
        if payload is None:
            return _fallback_plan(query, "planner returned non-JSON output")

        sub_queries = _normalize_sub_queries(payload.get("sub_queries"), query)[: self.max_subqueries]
        return QueryPlan(
            original_query=query,
            is_complex=bool(payload.get("is_complex", len(sub_queries) > 1)),
            reasoning=str(payload.get("reasoning") or "").strip() or "planner generated a query plan",
            sub_queries=sub_queries,
            synthesis_instruction=(
                str(payload.get("synthesis_instruction") or "").strip()
                or "combine the sub-query evidence to answer the original question"
            ),
        )


def plan_query(query: str, llm_model_name: str = "gpt-4o-mini",
               max_subqueries: int = 5, extra_context: str | None = None) -> QueryPlan:
    """Convenience wrapper for one-off planning."""
    planner = QueryPlanner(llm_model_name=llm_model_name, max_subqueries=max_subqueries)
    return planner.plan(query=query, extra_context=extra_context)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone planner agent for decomposing complex queries."
    )
    parser.add_argument("--query", required=True, help="Query to decompose")
    parser.add_argument("--model", default="gpt-4o-mini", help="Planner LLM model name")
    parser.add_argument("--max-subqueries", type=int, default=5, help="Maximum sub-queries to keep")
    parser.add_argument(
        "--context",
        default=None,
        help="Optional extra planning context, such as corpus/domain hints",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan = plan_query(
        query=args.query,
        llm_model_name=args.model,
        max_subqueries=args.max_subqueries,
        extra_context=args.context,
    )
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
