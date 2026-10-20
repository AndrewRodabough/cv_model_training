from argparse import ArgumentParser
from datetime import date, datetime
from os.path import exists
from pathlib import Path
from shutil import move
import os
import xml.etree.ElementTree as ET

import yaml

from src.util.dir import DETECTOR_VERSION_DIR, EXPORT_DIR, IMAGES_DIR
from src.util.get_version import get_latest_detector_version, parse_version
from src.dataset_review.dataset_parser import (
    attribute_value,
    load_project,
    shape_should_include,
    track_attribute_value,
)


class _ManifestDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _write_manifest(path, manifest):
    with open(path, 'w', encoding='utf-8') as manifest_file:
        yaml.dump(
            manifest,
            manifest_file,
            Dumper=_ManifestDumper,
            default_flow_style=False,
            sort_keys=False,
            width=4096,
        )


def _get_export_file(input_file):
    input_path = Path(input_file).resolve()
    export_dir = EXPORT_DIR.resolve()
    try:
        input_path.relative_to(export_dir)
    except ValueError:
        formatted_date = datetime.now().strftime('%Y-%m-%d')
        index = 0
        while exists(EXPORT_DIR / f'{formatted_date}_{index:03d}.xml'):
            index += 1
            if index > 999:
                raise ValueError(f'Index {index} is too large')
        destination = EXPORT_DIR / f'{formatted_date}_{index:03d}.xml'
        move(input_path, destination)
        return destination, destination.stem
    return input_path, input_path.stem


def build_bbox_annotations(
    annotations_file: Path,
    output_file: Path,
    images_dir: Path = IMAGES_DIR,
):
    """Build a cleaned project annotation file containing cleaned bbox tracks."""
    annotations_file = Path(annotations_file)
    output_file = Path(output_file)
    images_dir = Path(images_dir)

    if not annotations_file.is_file():
        raise FileNotFoundError(f'Missing annotations XML: {annotations_file}')

    root, project, tasks = load_project(annotations_file)
    task_start_frames = {}
    global_offset = 0
    for task_id in sorted(tasks, key=lambda value: int(value)):
        task_start_frames[task_id] = global_offset
        global_offset += int(tasks[task_id].findtext('size') or 0)

    selected_images = set()
    selected_shapes = []
    selected_tracks = set()
    selected_track_frames = 0

    for track in root.findall('track'):
        if track.get('label') != 'bbox':
            continue
        task_id = track.get('task_id')
        if task_id not in tasks:
            continue

        start_frame = task_start_frames[task_id]
        track_id = track.get('id', '')
        for shape in track:
            if shape.tag != 'box' or shape.get('frame') is None:
                continue
            if shape.get('outside') == '1':
                continue
            if not shape_should_include(shape, start_frame, track_id, task_id=task_id):
                continue

            frame = int(shape.get('frame'))
            relative_frame = frame - start_frame
            image_name = f'task_{task_id}_frame_{relative_frame:06d}.jpg'
            image_path = images_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f'Missing image-store file: {image_path}')

            selected_tracks.add(track_id)
            selected_track_frames += 1
            selected_images.add((task_id, relative_frame, image_name))
            selected_shapes.append((task_id, relative_frame, track, shape))

    output_root = ET.Element('dataset')
    project_element = ET.SubElement(
        output_root,
        'project',
        id='1',
        name=project.findtext('name', ''),
    )
    videos_element = ET.SubElement(project_element, 'videos')
    image_by_frame = {
        (task_id, frame): image
        for task_id, frame, image in selected_images
    }
    frames_by_task = {}
    for task_id, frame, track, shape in selected_shapes:
        frames_by_task.setdefault(task_id, {}).setdefault(frame, []).append(
            (track, shape)
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    for task_id, frames in frames_by_task.items():
        video_element = ET.SubElement(
            videos_element,
            'video',
            id=task_id,
            name=tasks[task_id].findtext('name', ''),
        )
        for frame_number, people in sorted(frames.items()):
            frame_element = ET.SubElement(
                video_element,
                'frame',
                number=str(frame_number),
                image=os.path.relpath(
                    images_dir / image_by_frame[(task_id, frame_number)],
                    output_file.parent,
                ),
            )
            for track, shape in people:
                person_attrs = {
                    'track_id': track.get('id', ''),
                    'person_id': (
                        attribute_value(shape, 'person_id')
                        or track_attribute_value(track, 'person_id')
                        or ''
                    ),
                }
                role = (
                    attribute_value(shape, 'role')
                    or track_attribute_value(track, 'role')
                )
                if role is not None:
                    person_attrs['role'] = role.lower()

                person = ET.SubElement(frame_element, 'person', **person_attrs)
                ET.SubElement(
                    person,
                    'bbox',
                    x1=shape.get('xtl', ''),
                    y1=shape.get('ytl', ''),
                    x2=shape.get('xbr', ''),
                    y2=shape.get('ybr', ''),
                )

    output_tree = ET.ElementTree(output_root)
    ET.indent(output_tree, space='  ')
    output_tree.write(output_file, encoding='utf-8', xml_declaration=True)
    return len(selected_images), len(selected_tracks), selected_track_frames


def main():
    parser = ArgumentParser(description='Create a cleaned detector annotation version')
    parser.add_argument('--input-file', type=str, required=True, help='The input CVAT XML file')
    parser.add_argument(
        '--version-type',
        choices=('major', 'minor', 'patch'),
        required=True,
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=True,
        help='Collect a description and change log interactively',
    )
    args = parser.parse_args()

    if not exists(args.input_file):
        raise FileNotFoundError(f'Input file {args.input_file} not found')

    description = ''
    changes = []
    if args.interactive:
        description = input('Enter Description: ')
        while True:
            change = input('Enter Change (enter for done): ')
            if not change:
                break
            changes.append(change)

    latest_version = get_latest_detector_version()
    major, minor, patch = parse_version(latest_version)
    if args.version_type == 'major':
        major += 1
    elif args.version_type == 'minor':
        minor += 1
    else:
        patch += 1

    version = f'{major}.{minor}.{patch}'
    version_folder = DETECTOR_VERSION_DIR / f'{major}.X' / f'{major}.{minor}.X' / version
    version_folder.mkdir(parents=True, exist_ok=False)

    image_count, track_count, track_frame_count = build_bbox_annotations(
        annotations_file=Path(args.input_file),
        output_file=version_folder / 'cleaned_annotations.xml',
        images_dir=IMAGES_DIR,
    )

    destination, export_version = _get_export_file(args.input_file)
    formatted_date = datetime.now().strftime('%Y-%m-%d')

    manifest = {
        'annotation_version': version,
        'image_version': 0,
        'export_version': export_version,
        'date': date.today(),
        'description': description,
        'changes': changes,
        'statistics': {
            'frames_used': image_count,
            'tasks_used': 'x',
            'tracks': track_count,
            'track_frames': track_frame_count,
        },
    }
    _write_manifest(version_folder / 'manifest.yaml', manifest)


if __name__ == '__main__':
    main()
