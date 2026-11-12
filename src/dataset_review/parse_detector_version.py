"""Create a cleaned detector annotation version from a CVAT project export.

Main vs side people: a bbox person is ``main`` when the same task has a skeleton
track with the same ``person_id``; otherwise ``side`` (background / no keypoints).
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import date, datetime
from os.path import exists
from pathlib import Path
from shutil import move
import os
import tempfile
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
from src.dataset_review.parse_pose_estimator_version import (
    ingest_task_export,
    resolve_project_annotations,
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


def _get_export_file(input_file: Path):
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
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        destination = EXPORT_DIR / f'{formatted_date}_{index:03d}.xml'
        move(str(input_path), destination)
        return destination, destination.stem
    return input_path, input_path.stem


def skeleton_person_ids_by_task(root: ET.Element) -> dict[str, set[str]]:
    """task_id → set of person_id strings that have a skeleton track."""
    by_task: dict[str, set[str]] = {}
    for track in root.findall('track'):
        if track.get('label') != 'skeleton':
            continue
        task_id = track.get('task_id')
        if task_id is None:
            continue
        person_id = track_attribute_value(track, 'person_id')
        if person_id is None or person_id == '':
            continue
        by_task.setdefault(task_id, set()).add(person_id)
    return by_task


def build_bbox_annotations(
    annotations_file: Path,
    output_file: Path,
    images_dir: Path = IMAGES_DIR,
):
    """Build cleaned detector XML; tag each person main|side via skeleton match."""
    annotations_file = Path(annotations_file)
    output_file = Path(output_file)
    images_dir = Path(images_dir)

    if not annotations_file.is_file():
        raise FileNotFoundError(f'Missing annotations XML: {annotations_file}')

    root, project, tasks = load_project(annotations_file)
    skeleton_ids = skeleton_person_ids_by_task(root)
    task_start_frames = {}
    global_offset = 0
    for task_id in sorted(tasks, key=lambda value: int(value)):
        task_start_frames[task_id] = global_offset
        global_offset += int(tasks[task_id].findtext('size') or 0)

    selected_images = set()
    selected_shapes = []
    selected_tracks = set()
    selected_track_frames = 0
    n_main = 0
    n_side = 0

    for track in root.findall('track'):
        if track.get('label') != 'bbox':
            continue
        task_id = track.get('task_id')
        if task_id not in tasks:
            continue

        start_frame = task_start_frames[task_id]
        track_id = track.get('id', '')
        track_person_id = track_attribute_value(track, 'person_id')
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
            selected_shapes.append((task_id, relative_frame, track, shape, track_person_id))

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
    frames_by_task: dict[str, dict[int, list]] = {}
    for task_id, frame, track, shape, track_person_id in selected_shapes:
        frames_by_task.setdefault(task_id, {}).setdefault(frame, []).append(
            (track, shape, track_person_id)
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    for task_id, frames in frames_by_task.items():
        video_element = ET.SubElement(
            videos_element,
            'video',
            id=task_id,
            name=tasks[task_id].findtext('name', ''),
        )
        main_pids = skeleton_ids.get(task_id, set())
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
            for track, shape, track_person_id in people:
                person_id = (
                    attribute_value(shape, 'person_id')
                    or track_person_id
                    or ''
                )
                tier = 'main' if person_id in main_pids else 'side'
                if tier == 'main':
                    n_main += 1
                else:
                    n_side += 1
                person_attrs = {
                    'track_id': track.get('id', ''),
                    'person_id': person_id,
                    'tier': tier,
                }
                couple_id = (
                    attribute_value(shape, 'couple_id')
                    or track_attribute_value(track, 'couple_id')
                )
                if couple_id is not None:
                    person_attrs['couple_id'] = couple_id
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
    tasks_used = {task_id for task_id, _, _ in selected_images}
    return {
        'frames': len(selected_images),
        'tracks': len(selected_tracks),
        'track_frames': selected_track_frames,
        'tasks': len(tasks_used),
        'main_boxes': n_main,
        'side_boxes': n_side,
    }


def main():
    parser = ArgumentParser(description='Create a cleaned detector annotation version')
    parser.add_argument(
        '--input-file',
        '--annotations',
        dest='input_file',
        type=str,
        required=True,
        help='Project annotations: .xml or .zip containing annotations.xml',
    )
    parser.add_argument(
        '--task',
        dest='tasks',
        type=Path,
        nargs='*',
        default=[],
        help='Optional task export zip(s)/folders with images to ingest into image_store',
    )
    parser.add_argument(
        '--version-type',
        choices=('major', 'minor', 'patch'),
        required=True,
    )
    parser.add_argument(
        '--description',
        type=str,
        default='',
        help='Manifest description (non-interactive)',
    )
    parser.add_argument(
        '--change',
        dest='changes',
        action='append',
        default=[],
        help='Manifest change line (repeatable)',
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=False,
        help='Collect description/changes interactively',
    )
    args = parser.parse_args()

    annotations_path = resolve_project_annotations(Path(args.input_file))
    # Keep a durable copy under dataset/exports before building.
    if annotations_path.parent != EXPORT_DIR.resolve() and not str(
        annotations_path
    ).startswith(str(EXPORT_DIR.resolve())):
        # resolve_project_annotations may return a temp xml from a zip — copy contents.
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        formatted_date = datetime.now().strftime('%Y-%m-%d')
        index = 0
        while exists(EXPORT_DIR / f'{formatted_date}_{index:03d}.xml'):
            index += 1
        durable = EXPORT_DIR / f'{formatted_date}_{index:03d}.xml'
        durable.write_bytes(Path(annotations_path).read_bytes())
        export_version = durable.stem
        annotations_for_build = durable
    else:
        annotations_for_build, export_version = _get_export_file(annotations_path)

    ingested = 0
    for task_path in args.tasks:
        ingested += ingest_task_export(
            task_path=Path(task_path),
            image_store=IMAGES_DIR,
            project_annotations=annotations_for_build,
        )
    if args.tasks:
        print(f'Ingested {ingested} images from {len(args.tasks)} task export(s)')

    description = args.description
    changes = list(args.changes)
    if args.interactive:
        description = input('Enter Description: ') or description
        while True:
            change = input('Enter Change (enter for done): ')
            if not change:
                break
            changes.append(change)

    latest_version = get_latest_detector_version()
    major, minor, patch = parse_version(latest_version)
    if args.version_type == 'major':
        major, minor, patch = major + 1, 0, 0
    elif args.version_type == 'minor':
        minor, patch = minor + 1, 0
    else:
        patch += 1

    version = f'{major}.{minor}.{patch}'
    version_folder = DETECTOR_VERSION_DIR / f'{major}.X' / f'{major}.{minor}.X' / version
    version_folder.mkdir(parents=True, exist_ok=False)

    stats = build_bbox_annotations(
        annotations_file=annotations_for_build,
        output_file=version_folder / 'cleaned_annotations.xml',
        images_dir=IMAGES_DIR,
    )

    manifest = {
        'annotation_version': version,
        'image_version': 0,
        'export_version': export_version,
        'date': date.today(),
        'description': description
        or 'Detector bboxes with main/side tiers (side = no skeleton track)',
        'changes': changes
        or [
            'Ingest tasks 58/59 images',
            'Tag main vs side by skeleton person_id match',
            'Include side/background people for recall-weighted training',
        ],
        'statistics': {
            'frames_used': stats['frames'],
            'tasks_used': stats['tasks'],
            'tracks': stats['tracks'],
            'track_frames': stats['track_frames'],
            'main_boxes': stats['main_boxes'],
            'side_boxes': stats['side_boxes'],
            'images_ingested': ingested,
        },
    }
    _write_manifest(version_folder / 'manifest.yaml', manifest)
    print(f'Wrote detector version {version} → {version_folder}')
    print(
        f"frames={stats['frames']} tracks={stats['tracks']} "
        f"main={stats['main_boxes']} side={stats['side_boxes']} "
        f"tasks={stats['tasks']}"
    )


if __name__ == '__main__':
    main()
