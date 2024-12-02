import importlib


def get_retriever(config):
    return getattr(importlib.import_module("rag.retriever"), "Retriever")(config)
