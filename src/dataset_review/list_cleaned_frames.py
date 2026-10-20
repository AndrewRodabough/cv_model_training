#!/usr/bin/env python3
"""List cleaned frames for one task in a dataset annotation version.

Uses the same selection rules as dataset_parser.py:
  - frame_cleaned=true → include
  - cleaned=true + cleaned_completion=N → every Nth relative frame
"""


import argparse
import xml.etree.ElementTree as ET
import yaml
from src.util.get_frame import list_cleaned_frames
from src.util.get_version import (
    get_export_file_path,
    get_pose_estimator_version_folder,
)
from src.util.dir import IMAGES_DIR


def main():
    parser = argparse.ArgumentParser(
        description='List cleaned frames for a task in an annotation version.'
    )
    parser.add_argument('--version', help="Annotation version, e.g. '1.5.0'")
    parser.add_argument('--task', help='CVAT task id, e.g. 7 or task_7')
    parser.add_argument(
        '--source',
        choices=('annotations', 'cleaned'),
        default='annotations',
        help='Read raw annotations.xml (default) or cleaned_annotations.xml',
    )
    args = parser.parse_args()

    task_id = args.task.removeprefix('task_')
    dataset_dir = get_pose_estimator_version_folder(args.version)
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


    # annotations source
    manifest_file = dataset_dir / 'manifest.yaml'
    with manifest_file.open(encoding='utf-8') as handle:
        manifest = yaml.load(handle, Loader=yaml.BaseLoader)
    name = manifest['export_version']
    annotations_file = get_export_file_path(name)

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
