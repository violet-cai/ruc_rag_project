import importlib


def get_generator(config):
    return getattr(importlib.import_module("rag.generator"), "Generator")(config)
