__all__ = ["CLASSES", "CLASS_TO_ID"]
CLASSES = ("frame", "text", "balloon", "onomatopoeia")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
