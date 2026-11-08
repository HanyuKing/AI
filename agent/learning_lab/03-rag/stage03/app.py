from pathlib import Path
from .answer import answer_policy
from .retriever import PolicyRetriever

if __name__ == "__main__":
    retriever = PolicyRetriever.from_file(Path(__file__).with_name("documents.json"))
    print(answer_policy(retriever, "签收七天内退货规则", "shop-a"))
