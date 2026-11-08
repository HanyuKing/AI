import json
from pathlib import Path

from .retriever import PolicyRetriever


def evaluate(retriever: PolicyRetriever, cases: list[dict], top_k: int = 3) -> dict:
    details = []
    for case in cases:
        hits = retriever.search(case["query"], case["tenant"], top_k=top_k)
        retrieved_ids = [hit["id"] for hit in hits]
        relevant = set(case["expected_ids"])
        details.append({"query": case["query"], "expected_ids": sorted(relevant),
                        "retrieved_ids": retrieved_ids,
                        "hit": bool(relevant & set(retrieved_ids))})
    return {"queries": len(details), "recall_at_k": sum(row["hit"] for row in details) / len(details)
            if details else 1.0, "top_k": top_k, "details": details}


if __name__ == "__main__":
    here = Path(__file__).parent
    retriever = PolicyRetriever.from_file(here / "documents.json")
    cases = json.loads((here / "cases.json").read_text(encoding="utf-8"))
    print(json.dumps(evaluate(retriever, cases), ensure_ascii=False, indent=2))
