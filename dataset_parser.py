import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

DATASET_DIR = Path('dataset/versions/1.X/1.3.X/1.3.0')
ANNOTATIONS_FILE = DATASET_DIR / 'annotations.xml'
IMAGES_DIR = Path('dataset/image_store')
OUTPUT_ANNOTATIONS_FILE = DATASET_DIR / 'cleaned_annotations.xml'
CVAT_MAPPING_FILE = Path('mapping/cvat_dance_28.yaml')


def attribute_value(element, name):
    attribute = element.find(f"./attribute[@name='{name}']")
    return attribute.text.strip() if attribute is not None and attribute.text else None


def attribute_is_true(element, name):
    value = attribute_value(element, name)
    return value is not None and value.lower() in ('true', '1', 'yes')


def track_attribute_value(track, name):
    value = attribute_value(track, name)
    if value is not None:
        return value
    for shape in track:
        value = attribute_value(shape, name)
        if value is not None:
            return value
    return None


def track_has_selectable_shapes(track):
    return any(
        attribute_is_true(shape, 'cleaned') or attribute_is_true(shape, 'frame_cleaned')
        for shape in track
        if shape.tag not in ('attribute',)
    )


def shape_should_include(shape, start_frame, track_id):
    """Include a skeleton frame if it is on the cleaned_completion grid or manually marked.

    - Track/shape `cleaned` + `cleaned_completion`: keep every Nth relative frame.
    - Per-frame `frame_cleaned`: always include that frame (union with the grid).
    """
    if attribute_is_true(shape, 'frame_cleaned'):
        return True

    if not attribute_is_true(shape, 'cleaned'):
        return False

    completion = int(attribute_value(shape, 'cleaned_completion') or '0')
    if completion <= 0:
        raise ValueError(f'Invalid cleaned_completion on track {track_id}.')
    frame = int(shape.get('frame'))
    relative_frame = frame - start_frame
    return relative_frame % completion == 0


def load_cvat_label_to_standard_id(mapping_path: Path = CVAT_MAPPING_FILE) -> dict[int, int]:
    """Map CVAT skeleton point labels (1-based) to standard_id.

    cvat_dance_28.yaml uses 0-based local ids; CVAT exports those points as labels 1..N.
    """
    with open(mapping_path, encoding='utf-8') as mapping_file:
        mapping = yaml.safe_load(mapping_file)

    label_to_standard = {}
    for name, meta in mapping['keypoints'].items():
        cvat_label = int(meta['id']) + 1
        standard_id = int(meta['standard_id'])
        if cvat_label in label_to_standard:
            raise ValueError(
                f'Duplicate CVAT label {cvat_label} in {mapping_path} '
                f'({label_to_standard[cvat_label]} vs {standard_id} for {name})'
            )
        label_to_standard[cvat_label] = standard_id
    return label_to_standard


def image_for_task_frame(task_id, frame_number):
    image_name = f'task_{task_id}_frame_{frame_number:06d}.jpg'
    image_path = IMAGES_DIR / image_name
    if not image_path.exists():
        raise FileNotFoundError(f'Missing image-store file: {image_path}')
    return image_name


def load_project():
    root = ET.parse(ANNOTATIONS_FILE).getroot()
    meta = root.find('meta')
    if meta is None:
        raise ValueError('The annotations XML does not contain a meta element.')
    project = next(project for project in meta.findall('project') if project.findtext('id') == '1')
    tasks = {task.findtext('id'): task for task in project.findall('./tasks/task')}
    return root, project, tasks


def build_annotations():
    root, project, tasks = load_project()
    cvat_label_to_standard_id = load_cvat_label_to_standard_id()
    task_start_frames = {
        task_id: min(
            int(shape.get('frame'))
            for track in root.findall('track')
            if track.get('task_id') == task_id
            for shape in track
        )
        for task_id in tasks
        if any(track.get('task_id') == task_id for track in root.findall('track'))
    }

    bbox_tracks_by_task = {}
    skeleton_tracks_by_task = {}
    for track in root.findall('track'):
        task_id = track.get('task_id')
        if track.get('label') == 'bbox':
            bbox_tracks_by_task.setdefault(task_id, []).append(track)
        elif track.get('label') == 'skeleton' and track_has_selectable_shapes(track):
            skeleton_tracks_by_task.setdefault(task_id, []).append(track)

    bbox_for_skeleton = {}
    for task_id, skeleton_tracks in skeleton_tracks_by_task.items():
        bbox_by_person = {}
        for bbox_track in bbox_tracks_by_task.get(task_id, []):
            person_id = track_attribute_value(bbox_track, 'person_id')
            if person_id is None:
                raise ValueError(f'BBox track {bbox_track.get("id")} is missing person_id.')
            key = (task_id, person_id)
            if key in bbox_by_person:
                raise ValueError(f'Duplicate bbox person_id {person_id} in task {task_id}.')
            bbox_by_person[key] = bbox_track
        for skeleton_track in skeleton_tracks:
            person_id = track_attribute_value(skeleton_track, 'person_id')
            bbox_track = bbox_by_person.get((task_id, person_id))
            if person_id is None or bbox_track is None:
                raise ValueError(f'No matching bbox for person_id {person_id} in task {task_id}.')
            bbox_for_skeleton[skeleton_track.get('id')] = bbox_track

    selected_images = set()
    selected_shapes = []
    selected_tracks = []
    selected_via_completion = 0
    selected_via_frame_cleaned = 0
    for track in root.findall('track'):
        if track.get('label') != 'skeleton' or track.get('task_id') not in tasks:
            continue
        if track.get('id') not in bbox_for_skeleton:
            continue
        task_id = track.get('task_id')
        start_frame = task_start_frames.get(task_id, 0)
        filtered_track = copy.copy(track)
        filtered_track.clear()
        has_selected_shape = False
        for shape in track:
            if shape.tag in ('attribute',):
                continue
            if not shape_should_include(shape, start_frame, track.get('id')):
                continue

            frame = int(shape.get('frame'))
            relative_frame = frame - start_frame
            bbox_track = bbox_for_skeleton[track.get('id')]
            bbox_shape = next((candidate for candidate in bbox_track if candidate.get('frame') == str(frame)), None)
            if bbox_shape is None:
                raise ValueError(f'Missing bbox for skeleton track {track.get("id")} at frame {frame}.')

            if attribute_is_true(shape, 'frame_cleaned'):
                selected_via_frame_cleaned += 1
            else:
                selected_via_completion += 1

            filtered_track.append(copy.deepcopy(shape))
            has_selected_shape = True
            image_name = image_for_task_frame(task_id, relative_frame)
            selected_images.add((task_id, relative_frame, image_name))
            selected_shapes.append((task_id, relative_frame, track, shape, bbox_shape))
        if has_selected_shape:
            selected_tracks.append(filtered_track)

    output_root = ET.Element('dataset')
    project_element = ET.SubElement(output_root, 'project', id='1', name=project.findtext('name', ''))
    videos_element = ET.SubElement(project_element, 'videos')
    image_by_frame = {(task_id, frame): image for task_id, frame, image in selected_images}
    frames_by_task = {}
    for task_id, frame, track, shape, bbox_shape in selected_shapes:
        frames_by_task.setdefault(task_id, {}).setdefault(frame, []).append((track, shape, bbox_shape))

    for task_id, frames in frames_by_task.items():
        video_element = ET.SubElement(videos_element, 'video', id=task_id, name=tasks[task_id].findtext('name', ''))
        for frame_number, people in sorted(frames.items()):
            frame_element = ET.SubElement(
                video_element,
                'frame',
                number=str(frame_number),
                image=os.path.relpath(IMAGES_DIR / image_by_frame[(task_id, frame_number)], OUTPUT_ANNOTATIONS_FILE.parent),
            )
            for track, shape, bbox_shape in people:
                person = ET.SubElement(frame_element, 'person', track_id=track.get('id', ''), person_id=track_attribute_value(track, 'person_id'))
                ET.SubElement(person, 'bbox', x1=bbox_shape.get('xtl', ''), y1=bbox_shape.get('ytl', ''), x2=bbox_shape.get('xbr', ''), y2=bbox_shape.get('ybr', ''))
                for point in shape.findall('points'):
                    cvat_label = int(point.get('label'))
                    if cvat_label not in cvat_label_to_standard_id:
                        raise ValueError(
                            f'Unknown CVAT keypoint label {cvat_label} on track {track.get("id")}; '
                            f'expected one of {sorted(cvat_label_to_standard_id)}'
                        )
                    standard_id = cvat_label_to_standard_id[cvat_label]
                    x, y = point.get('points', '').split(',')
                    visibility = '0' if point.get('outside') == '1' else ('1' if point.get('occluded') == '1' else '2')
                    ET.SubElement(
                        person,
                        'keypoint',
                        id=str(standard_id),
                        x=x,
                        y=y,
                        visibility=visibility,
                    )

    output_tree = ET.ElementTree(output_root)
    ET.indent(output_tree, space='  ')
    output_tree.write(OUTPUT_ANNOTATIONS_FILE, encoding='utf-8', xml_declaration=True)
    return (
        len(selected_images),
        len(selected_tracks),
        selected_via_completion,
        selected_via_frame_cleaned,
    )


def main():
    image_count, track_count, via_completion, via_frame_cleaned = build_annotations()
    print(f'Created {image_count} unique images.')
    print(f'Created {track_count} cleaned tracks.')
    print(f'Selected shapes via cleaned_completion grid: {via_completion}')
    print(f'Selected shapes via frame_cleaned: {via_frame_cleaned}')
    print(f'Output: {OUTPUT_ANNOTATIONS_FILE}')
    print(f'Keypoint ids written as standard_id via {CVAT_MAPPING_FILE}')


if __name__ == '__main__':
    main()
