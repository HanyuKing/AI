"""可解释的教学检索器：租户/时间过滤、中文二元词、BM25 风格词频与引用。"""

import json
import math
import re
from datetime import date
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """示范中文字符二元词和英文词；生产检索应使用成熟 tokenizer。"""
    terms: list[str] = []
    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            terms.extend(token[i:i + 2] for i in range(len(token) - 1))
            if len(token) == 1:
                terms.append(token)
        else:
            terms.append(token)
    return terms


class PolicyRetriever:
    def __init__(self, documents: list[dict]):
        self.documents = documents

    @classmethod
    def from_file(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def search(self, query: str, tenant: str, top_k: int = 3,
               as_of: str | None = None) -> list[dict]:
        if not tenant:
            raise ValueError("tenant is required")
        if top_k < 1:
            return []
        today = as_of or date.today().isoformat()
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        # ACL 与生效期在召回打分前执行，未授权文档不参与任何候选或上下文。
        candidates = [doc for doc in self.documents
                      if doc["tenant"] == tenant
                      and doc["effective"] <= today
                      and (not doc.get("expires") or today < doc["expires"])]
        document_frequency: dict[str, int] = {}
        tokenized = []
        for doc in candidates:
            terms = tokenize(f"{doc.get('topic', '')} {doc['text']}")
            tokenized.append((doc, terms))
            for term in set(terms):
                document_frequency[term] = document_frequency.get(term, 0) + 1

        avg_length = sum(len(terms) for _, terms in tokenized) / max(1, len(tokenized))
        ranked = []
        for doc, terms in tokenized:
            frequencies = {term: terms.count(term) for term in query_terms if term in terms}
            score = 0.0
            for term, frequency in frequencies.items():
                idf = math.log(1 + (len(candidates) - document_frequency[term] + 0.5)
                               / (document_frequency[term] + 0.5))
                length_norm = 1 - 0.75 + 0.75 * len(terms) / max(1, avg_length)
                score += idf * (frequency * 2.2) / (frequency + 1.2 * length_norm)
            # 标题命中加权，易解释；不能替代语义召回或交叉编码器重排。
            if any(term in tokenize(doc.get("topic", "")) for term in query_terms):
                score += 1.5
            if score > 0:
                ranked.append((score, doc))
        ranked.sort(key=lambda row: (row[0], row[1]["version"], row[1]["id"]), reverse=True)
        return [{**doc, "score": round(score, 4),
                 "citation": {"document_id": doc["id"], "version": doc["version"],
                              "effective": doc["effective"]}}
                for score, doc in ranked[:top_k]]


def recall_at_k(results: list[list[str]], expected: list[set[str]], k: int) -> float:
    """以每个问题的相关文档是否进入 top-k 计算 hit-style Recall@k。"""
    if not expected:
        return 1.0
    hits = sum(bool(set(result[:k]) & relevant) for result, relevant in zip(results, expected))
    return hits / len(expected)
