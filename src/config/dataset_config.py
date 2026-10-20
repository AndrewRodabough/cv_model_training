from dataclasses import dataclass
from src.config.coco_wholebody_config import CocoWholeBodyConfig

@dataclass
class DatasetConfig:
    name: str
    annotation_version: str
    annotation_file: str
    coco_wholebody: CocoWholeBodyConfig | None = None