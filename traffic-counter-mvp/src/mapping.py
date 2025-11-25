# src/mapping.py
from typing import List

def build_class_map(class_names: List[str]):
    """
    Return a dict mapping integer id -> class name
    """
    return {i: name for i, name in enumerate(class_names)}
