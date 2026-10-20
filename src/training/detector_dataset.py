"""Model-agnostic detector samples backed by cleaned detector XML."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
import xml.etree.ElementTree as ET


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectorPerson:
    bbox: Box
    track_id: str
    person_id: str | None = None
    role: str | None = None
    couple_id: str | None = None


@dataclass(frozen=True)
class DetectorSample:
    image_path: Path
    task_id: str
    frame_number: int
    persons: tuple[DetectorPerson, ...]

    @property
    def boxes(self) -> tuple[Box, ...]:
        return tuple(person.bbox for person in self.persons)


def _optional_attr(element: ET.Element, name: str) -> str | None:
    value = element.get(name)
    return value if value not in (None, '') else None


def _required_attr(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if value is None:
        raise ValueError(f'Detector bbox is missing {name}')
    return value


def _parse_box(element: ET.Element) -> Box:
    values: Box = (
        float(_required_attr(element, 'x1')),
        float(_required_attr(element, 'y1')),
        float(_required_attr(element, 'x2')),
        float(_required_attr(element, 'y2')),
    )
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Invalid detector box: {values}')
    return values


def load_detector_samples(
    annotations_file: Path,
    *,
    require_images: bool = True,
) -> list[DetectorSample]:
    """Load every valid bbox person from a detector cleaned XML file.

    The XML is the source of truth. No YOLO labels, COCO files, or split
    directories are created or required.
    """
    annotations_file = Path(annotations_file)
    if not annotations_file.is_file():
        raise FileNotFoundError(f'Missing detector annotations: {annotations_file}')

    root = ET.parse(annotations_file).getroot()
    samples: list[DetectorSample] = []
    for video in root.findall('./project/videos/video'):
        task_id = video.get('id')
        if task_id is None:
            raise ValueError('Detector annotation video is missing its task ID')
        for frame in video.findall('frame'):
            frame_number = frame.get('number')
            image_name = frame.get('image')
            if frame_number is None or image_name is None:
                raise ValueError(
                    f'Missing frame number or image for task {task_id}'
                )
            image_path = (annotations_file.parent / image_name).resolve()
            if require_images and not image_path.is_file():
                raise FileNotFoundError(f'Missing detector image: {image_path}')

            persons: list[DetectorPerson] = []
            for person in frame.findall('person'):
                bbox_element = person.find('bbox')
                if bbox_element is None:
                    raise ValueError(
                        f'Missing bbox for task {task_id}, frame {frame_number}'
                    )
                try:
                    bbox = _parse_box(bbox_element)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f'Invalid bbox for task {task_id}, frame {frame_number}'
                    ) from exc
                persons.append(
                    DetectorPerson(
                        bbox=bbox,
                        track_id=person.get('track_id', ''),
                        person_id=_optional_attr(person, 'person_id'),
                        role=_optional_attr(person, 'role'),
                        couple_id=_optional_attr(person, 'couple_id'),
                    )
                )
            if persons:
                samples.append(
                    DetectorSample(
                        image_path=image_path,
                        task_id=task_id,
                        frame_number=int(frame_number),
                        persons=tuple(persons),
                    )
                )
    return samples


def split_detector_samples(
    samples: list[DetectorSample],
    *,
    validation_ratio: float = 0.2,
    seed: int = 44,
) -> tuple[list[DetectorSample], list[DetectorSample]]:
    """Split samples by task without creating train/validation directories."""
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError(
            f'validation_ratio must be in [0, 1), got {validation_ratio}'
        )
    task_ids = sorted({sample.task_id for sample in samples})
    if len(task_ids) < 2 or validation_ratio == 0.0:
        return list(samples), []

    shuffled = list(task_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_ratio))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation_tasks = set(shuffled[:validation_count])
    train = [sample for sample in samples if sample.task_id not in validation_tasks]
    validation = [sample for sample in samples if sample.task_id in validation_tasks]
    return train, validation
