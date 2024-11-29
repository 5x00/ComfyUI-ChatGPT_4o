from .tagger import GPT4oNode

NODE_CLASS_MAPPINGS = {
    "GPT4o Tagger" : GPT4oNode,
}

__all__ = ['NODE_CLASS_MAPPINGS']