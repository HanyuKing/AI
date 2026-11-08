"""从原始政策记录切块，并以稳定来源 ID 保留版本元数据。"""

import re


def chunk_document(document: dict, max_chars: int = 180) -> list[dict]:
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|(?<=[。！？])", document["text"]) if part.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        if current and len(current) + len(paragraph) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}{paragraph}" if not current else f"{current} {paragraph}"
    if current:
        chunks.append(current)
    return [{**document, "id": f"{document['id']}#chunk-{index + 1}",
             "parent_id": document["id"], "text": chunk}
            for index, chunk in enumerate(chunks)]


def replace_document(index: list[dict], document: dict) -> list[dict]:
    """新版本取代同一逻辑文档旧版本；历史可另存审计区。"""
    updated = [row for row in index if row.get("logical_id", row["id"].split("#")[0]) != document["id"]]
    source = {**document, "logical_id": document["id"]}
    return updated + chunk_document(source)
