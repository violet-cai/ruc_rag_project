import importlib


def get_retriever(config):
    return getattr(importlib.import_module("rag.reranker"), "Reranker")(config)
