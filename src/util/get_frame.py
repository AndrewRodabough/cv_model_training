import xml.etree.ElementTree as ET
from collections import defaultdict


from src.dataset_review.dataset_parser import (
    attribute_is_true,
    attribute_value,
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
