from dataclasses import dataclass

@dataclass
class ImageConfig:
    width: int
    height: int
    bbox_padding: float