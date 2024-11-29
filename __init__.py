from .tagger import vlmnode

NODE_CLASS_MAPPINGS = {
    "GPT4o Tagger" : vlmnode,
}

__all__ = ['NODE_CLASS_MAPPINGS']