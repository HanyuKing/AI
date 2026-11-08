from .retriever import PolicyRetriever


def answer_policy(retriever: PolicyRetriever, query: str, tenant: str) -> dict:
    hits = retriever.search(query, tenant)
    if not hits:
        return {"answer": "暂未找到适用规则，请转人工核实。", "sources": [], "grounded": False}
    return {"answer": hits[0]["text"],
            "sources": [hit["citation"] for hit in hits],
            "grounded": True,
            "notice": "政策说明不代表订单实时状态或最终资格判定。"}
