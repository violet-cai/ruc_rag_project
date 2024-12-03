import importlib


def get_reranker(config):
    return getattr(importlib.import_module("rag.reranker"), "Reranker")(config)
