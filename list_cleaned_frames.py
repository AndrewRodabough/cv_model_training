#!/usr/bin/env python3
"""List cleaned frames for one task in a dataset annotation version.

Uses the same selection rules as dataset_parser.py:
  - frame_cleaned=true → include
  - cleaned=true + cleaned_completion=N → every Nth relative frame
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

from dataset_parser import (
    IMAGES_DIR,
    attribute_is_true,
    attribute_value,
    resolve_dataset_dir,
    shape_should_include,
    track_attribute_value,
    track_has_selectable_shapes,
)


def task_start_frames(tasks: dict[str, ET.Element]) -> dict[str, int]:
    starts = {}
    global_offset = 0
    for task_id in sorted(tasks, key=lambda value: int(value)):
        starts[task_id] = global_offset
        global_offset += int(tasks[task_id].findtext('size') or 0)
    return starts


def list_cleaned_frames(annotations_file, task_id: str):
    root = ET.parse(annotations_file).getroot()
    meta = root.find('meta')
    if meta is None:
        raise ValueError('annotations XML missing meta')
    project = next(p for p in meta.findall('project') if p.findtext('id') == '1')
    tasks = {task.findtext('id'): task for task in project.findall('./tasks/task')}
    if task_id not in tasks:
        known = ', '.join(sorted(tasks, key=lambda value: int(value)))
        raise ValueError(f'Task {task_id} not found. Known task ids: {known}')

    start_frame = task_start_frames(tasks)[task_id]
    task_name = tasks[task_id].findtext('name', '')

    # frame_rel -> list of (track_id, person_id, reason)
    by_frame: dict[int, list[tuple[str, str | None, str]]] = defaultdict(list)

    for track in root.findall('track'):
        if track.get('task_id') != task_id or track.get('label') != 'skeleton':
            continue
        if not track_has_selectable_shapes(track):
            continue

        track_id = track.get('id', '')
        person_id = track_attribute_value(track, 'person_id')
        for shape in track:
            if shape.tag in ('attribute',) or shape.get('frame') is None:
                continue
            if not shape_should_include(shape, start_frame, track_id, task_id=task_id):
                continue

            global_frame = int(shape.get('frame'))
            relative_frame = global_frame - start_frame
            if attribute_is_true(shape, 'frame_cleaned'):
                reason = 'frame_cleaned'
            else:
                completion = attribute_value(shape, 'cleaned_completion') or '?'
                reason = f'cleaned_completion/{completion}'
            by_frame[relative_frame].append((track_id, person_id, reason))

    return task_name, start_frame, dict(sorted(by_frame.items()))


def main():
    parser = argparse.ArgumentParser(
        description='List cleaned frames for a task in an annotation version.'
    )
    parser.add_argument('version', help="Annotation version, e.g. '1.5.0'")
    parser.add_argument('task', help='CVAT task id, e.g. 7 or task_7')
    parser.add_argument(
        '--source',
        choices=('annotations', 'cleaned'),
        default='annotations',
        help='Read raw annotations.xml (default) or cleaned_annotations.xml',
    )
    args = parser.parse_args()

    task_id = args.task.removeprefix('task_')
    dataset_dir = resolve_dataset_dir(args.version)
    if args.source == 'cleaned':
        annotations_file = dataset_dir / 'cleaned_annotations.xml'
        if not annotations_file.is_file():
            raise FileNotFoundError(
                f'Missing {annotations_file}; run: python dataset_parser.py {args.version}'
            )
        # cleaned XML already stores relative frame numbers and persons.
        root = ET.parse(annotations_file).getroot()
        video = root.find(f'./project/videos/video[@id="{task_id}"]')
        if video is None:
            known = [v.get('id') for v in root.findall('./project/videos/video')]
            raise ValueError(f'Task {task_id} not in cleaned XML. Known: {known}')
        frames = []
        for frame in video.findall('frame'):
            rel = int(frame.get('number'))
            people = [
                (person.get('track_id'), person.get('person_id'), 'cleaned')
                for person in frame.findall('person')
            ]
            frames.append((rel, people))
        task_name = video.get('name', '')
        print(f'version={args.version} task={task_id} ({task_name}) source=cleaned')
        print(f'cleaned_frames={len(frames)}')
        for rel, people in frames:
            image = f'task_{task_id}_frame_{rel:06d}.jpg'
            people_s = ', '.join(
                f'track={tid}/person={pid}' for tid, pid, _ in people
            )
            print(f'{rel:6d}  {image}  {people_s}')
        return

    annotations_file = dataset_dir / 'annotations.xml'
    if not annotations_file.is_file():
        raise FileNotFoundError(f'Missing annotations XML: {annotations_file}')

    task_name, start_frame, by_frame = list_cleaned_frames(annotations_file, task_id)
    print(
        f'version={args.version} task={task_id} ({task_name}) '
        f'start_frame={start_frame} source=annotations'
    )
    print(f'cleaned_frames={len(by_frame)}')
    for rel, entries in by_frame.items():
        image = f'task_{task_id}_frame_{rel:06d}.jpg'
        exists = 'ok' if (IMAGES_DIR / image).exists() else 'MISSING'
        people_s = ', '.join(
            f'track={tid}/person={pid}/{reason}' for tid, pid, reason in entries
        )
        print(f'{rel:6d}  {image}  [{exists}]  {people_s}')


if __name__ == '__main__':
    main()
