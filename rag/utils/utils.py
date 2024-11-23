import importlib

from rag.config.configuration import Config


def get_modifier(config: Config):
    return getattr(importlib.import_module("rag.retriever"), "Modifier")(config)


def get_judger(config: Config):
    return getattr(importlib.import_module("rag.retriever"), "Judger")(config)


def get_retriever(config: Config):
    return getattr(importlib.import_module("rag.retriever"), "Retriever")(config)


def get_reranker(config: Config):
    return getattr(importlib.import_module("rag.reranker"), "Reranker")(config)


def get_refiner(config: Config):
    return getattr(importlib.import_module("rag.reranker"), "Refiner")(config)


def get_generator(config: Config):
    return getattr(importlib.import_module("rag.generator"), "Generator")(config)
