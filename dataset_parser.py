import argparse
import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

IMAGES_DIR = Path('dataset/image_store')
CVAT_MAPPING_FILE = Path('mapping/cvat_dance_28.yaml')

# Set in main() from --version / positional version.
DATASET_DIR: Path
ANNOTATIONS_FILE: Path
OUTPUT_ANNOTATIONS_FILE: Path


def resolve_dataset_dir(annotation_version: str) -> Path:
    """Map 'X.Y.Z' -> dataset/versions/X.X/X.Y.X/X.Y.Z (same layout as train.py)."""
    parts = annotation_version.split('.')
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"Expected semantic version format 'X.Y.Z', got: {annotation_version}"
        )
    major, minor, _ = parts
    dataset_dir = (
        Path('dataset')
        / 'versions'
        / f'{major}.X'
        / f'{major}.{minor}.X'
        / annotation_version
    )
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f'Dataset version directory not found: {dataset_dir}')
    return dataset_dir


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


VALID_ROLES = frozenset({'lead', 'follow'})


def project_defines_role(project) -> bool:
    """True if bbox or skeleton label schema includes a role attribute."""
    for label in project.findall('./labels/label'):
        for attribute in label.findall('./attributes/attribute'):
            if attribute.findtext('name') == 'role':
                return True
    return False


def resolve_track_role(skeleton_track, bbox_track) -> str:
    """Track-constant role from skeleton (preferred); warn if bbox disagrees."""
    skeleton_role = track_attribute_value(skeleton_track, 'role')
    bbox_role = track_attribute_value(bbox_track, 'role') if bbox_track is not None else None
    if skeleton_role is None:
        raise ValueError(
            f'Skeleton track {skeleton_track.get("id")} is missing required role. '
            f'Set role to one of {sorted(VALID_ROLES)} in CVAT.'
        )
    role = skeleton_role.lower()
    if role not in VALID_ROLES:
        raise ValueError(
            f'Skeleton track {skeleton_track.get("id")} has invalid role '
            f'{skeleton_role!r}; expected one of {sorted(VALID_ROLES)}.'
        )
    if bbox_role is not None and bbox_role.lower() != role:
        print(
            f'Warning: role mismatch on skeleton track {skeleton_track.get("id")}: '
            f'skeleton={role!r} bbox={bbox_role.lower()!r}; using skeleton.'
        )
    return role


def track_has_selectable_shapes(track):
    return any(
        attribute_is_true(shape, 'cleaned') or attribute_is_true(shape, 'frame_cleaned')
        for shape in track
        if shape.tag not in ('attribute',)
    )


def shape_should_include(shape, start_frame, track_id, task_id=None):
    """Include a skeleton frame if it is on the cleaned_completion grid or manually marked.

    - Track/shape `cleaned` + `cleaned_completion`: keep every Nth relative frame.
    - Per-frame `frame_cleaned`: always include that frame (union with the grid).
    """
    if attribute_is_true(shape, 'frame_cleaned'):
        return True

    if not attribute_is_true(shape, 'cleaned'):
        return False

    raw_completion = attribute_value(shape, 'cleaned_completion')
    try:
        completion = int(raw_completion) if raw_completion is not None else 0
    except ValueError:
        completion = None
    frame = int(shape.get('frame'))
    relative_frame = frame - start_frame
    task_bit = f' in task {task_id}' if task_id is not None else ''
    if completion is None or completion <= 0:
        shown = 'missing' if raw_completion is None else repr(raw_completion)
        raise ValueError(
            f'Skeleton track {track_id}{task_bit} is marked cleaned=true at '
            f'global frame {frame} (task-local frame {relative_frame}), but '
            f'cleaned_completion is {shown}. '
            f'When cleaned=true, cleaned_completion must be a positive integer N '
            f'(keep every Nth frame, e.g. 1 = every frame). '
            f'Either set cleaned_completion to N in CVAT, or set cleaned=false and '
            f'mark individual frames with frame_cleaned=true instead.'
        )
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
    require_role = project_defines_role(project)
    # CVAT project exports use global frame indices across tasks in id order.
    # Do not use min(track frame): a task can start before its first annotation.
    task_start_frames = {}
    global_offset = 0
    for task_id in sorted(tasks, key=lambda value: int(value)):
        task_start_frames[task_id] = global_offset
        global_offset += int(tasks[task_id].findtext('size') or 0)

    bbox_tracks_by_task = {}
    skeleton_tracks_by_task = {}
    for track in root.findall('track'):
        task_id = track.get('task_id')
        if track.get('label') == 'bbox':
            bbox_tracks_by_task.setdefault(task_id, []).append(track)
        elif track.get('label') == 'skeleton' and track_has_selectable_shapes(track):
            skeleton_tracks_by_task.setdefault(task_id, []).append(track)

    def track_frame_set(track):
        return {
            int(shape.get('frame'))
            for shape in track
            if shape.tag not in ('attribute',) and shape.get('frame') is not None
        }

    bbox_for_skeleton = {}
    for task_id, skeleton_tracks in skeleton_tracks_by_task.items():
        bboxes_by_person = {}
        for bbox_track in bbox_tracks_by_task.get(task_id, []):
            person_id = track_attribute_value(bbox_track, 'person_id')
            if person_id is None:
                raise ValueError(
                    f'BBox track {bbox_track.get("id")} in task {task_id} is missing '
                    f'person_id. Set person_id on the bbox track in CVAT.'
                )
            bboxes_by_person.setdefault(person_id, []).append(bbox_track)

        for person_id, bbox_tracks in bboxes_by_person.items():
            for index, bbox_track in enumerate(bbox_tracks):
                frames = track_frame_set(bbox_track)
                for other in bbox_tracks[index + 1:]:
                    overlap = frames & track_frame_set(other)
                    if overlap:
                        raise ValueError(
                            f'Task {task_id} has overlapping bbox tracks for person_id '
                            f'{person_id} (tracks {bbox_track.get("id")} and '
                            f'{other.get("id")}), e.g. at global frame {min(overlap)}. '
                            f'Each person_id may only have one bbox track per frame.'
                        )

        for skeleton_track in skeleton_tracks:
            person_id = track_attribute_value(skeleton_track, 'person_id')
            if person_id is None:
                raise ValueError(
                    f'Skeleton track {skeleton_track.get("id")} in task {task_id} is '
                    f'missing person_id. Set person_id so it can be matched to a bbox.'
                )
            skeleton_frames = track_frame_set(skeleton_track)
            matches = [
                bbox_track
                for bbox_track in bboxes_by_person.get(person_id, [])
                if track_frame_set(bbox_track) & skeleton_frames
            ]
            if len(matches) != 1:
                raise ValueError(
                    f'Skeleton track {skeleton_track.get("id")} in task {task_id} '
                    f'(person_id={person_id}) matched {len(matches)} bbox tracks; '
                    f'expected exactly 1 with overlapping frames. '
                    f'Check that a bbox with the same person_id covers this skeleton.'
                )
            bbox_for_skeleton[skeleton_track.get('id')] = matches[0]

    selected_images = set()
    selected_shapes = []
    selected_tracks = []
    selected_via_completion = 0
    selected_via_frame_cleaned = 0
    role_by_track = {}
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
            if not shape_should_include(
                shape, start_frame, track.get('id'), task_id=task_id
            ):
                continue

            frame = int(shape.get('frame'))
            relative_frame = frame - start_frame
            bbox_track = bbox_for_skeleton[track.get('id')]
            bbox_shape = next((candidate for candidate in bbox_track if candidate.get('frame') == str(frame)), None)
            if bbox_shape is None:
                # BBox tracks can have gaps (e.g. after outside=1) even when the skeleton
                # was marked frame_cleaned; skip rather than fail the whole build.
                print(
                    f'Warning: missing bbox for skeleton track {track.get("id")} '
                    f'at frame {frame}; skipping.'
                )
                continue
            if bbox_shape.get('outside') == '1':
                print(
                    f'Warning: outside bbox for skeleton track {track.get("id")} '
                    f'at frame {frame}; skipping.'
                )
                continue

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
                person_attrs = {
                    'track_id': track.get('id', ''),
                    'person_id': track_attribute_value(track, 'person_id') or '',
                }
                track_id = track.get('id')
                if require_role:
                    if track_id not in role_by_track:
                        role_by_track[track_id] = resolve_track_role(
                            track, bbox_for_skeleton.get(track_id)
                        )
                    person_attrs['role'] = role_by_track[track_id]
                else:
                    role = track_attribute_value(track, 'role')
                    if role is not None:
                        person_attrs['role'] = role.lower()
                person = ET.SubElement(frame_element, 'person', **person_attrs)
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
    global DATASET_DIR, ANNOTATIONS_FILE, OUTPUT_ANNOTATIONS_FILE, IMAGES_DIR, CVAT_MAPPING_FILE

    parser = argparse.ArgumentParser(
        description='Build cleaned_annotations.xml for a dataset annotation version.'
    )
    parser.add_argument(
        'version',
        help="Annotation version, e.g. '1.4.0' -> dataset/versions/1.X/1.4.X/1.4.0",
    )
    parser.add_argument(
        '--images',
        type=Path,
        default=Path('dataset/image_store'),
        help='Image store directory (default: dataset/image_store)',
    )
    parser.add_argument(
        '--mapping',
        type=Path,
        default=Path('mapping/cvat_dance_28.yaml'),
        help='CVAT label mapping YAML (default: mapping/cvat_dance_28.yaml)',
    )
    args = parser.parse_args()

    DATASET_DIR = resolve_dataset_dir(args.version)
    ANNOTATIONS_FILE = DATASET_DIR / 'annotations.xml'
    OUTPUT_ANNOTATIONS_FILE = DATASET_DIR / 'cleaned_annotations.xml'
    IMAGES_DIR = args.images
    CVAT_MAPPING_FILE = args.mapping

    if not ANNOTATIONS_FILE.is_file():
        raise FileNotFoundError(f'Missing annotations XML: {ANNOTATIONS_FILE}')

    print(f'Dataset dir: {DATASET_DIR}')
    image_count, track_count, via_completion, via_frame_cleaned = build_annotations()
    print(f'Created {image_count} unique images.')
    print(f'Created {track_count} cleaned tracks.')
    print(f'Selected shapes via cleaned_completion grid: {via_completion}')
    print(f'Selected shapes via frame_cleaned: {via_frame_cleaned}')
    print(f'Output: {OUTPUT_ANNOTATIONS_FILE}')
    print(f'Keypoint ids written as standard_id via {CVAT_MAPPING_FILE}')


if __name__ == '__main__':
    main()
