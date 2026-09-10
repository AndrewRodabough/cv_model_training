import copy
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

DATASET_DIR = Path('datasets/dataset_1.1_9-9')
ANNOTATIONS_FILE = DATASET_DIR / 'annotations.xml'
IMAGES_DIR = DATASET_DIR / 'images' / 'default'
OUTPUT_DIR = Path('datasets/cleaned_dataset_1.0_9-9')
OUTPUT_IMAGES_DIR = OUTPUT_DIR / 'images'
OUTPUT_ANNOTATIONS_DIR = OUTPUT_DIR / 'annotations'


def attribute_value(element, name):
    attribute = element.find(f"./attribute[@name='{name}']")
    return attribute.text.strip() if attribute is not None and attribute.text else None


def track_attribute_value(track, name):
    value = attribute_value(track, name)
    if value is not None:
        return value
    for shape in track:
        value = attribute_value(shape, name)
        if value is not None:
            return value
    return None


tree = ET.parse(ANNOTATIONS_FILE)
root = tree.getroot()
meta = root.find('meta')

if meta is None:
    raise ValueError('The annotations XML does not contain a meta element.')

project = next(
    project
    for project in meta.findall('project')
    if project.findtext('id') == '1'
)

tasks = {
    task.findtext('id'): task
    for task in project.findall('./tasks/task')
}
task_order = list(tasks)


def image_suffix(path):
    match = re.fullmatch(r'frame_\d{6}(?:_(\d+))?\.([^.]+)', path.name)
    if match is None:
        return -1
    return int(match.group(1) or 0)


def image_for_task_frame(task_id, relative_frame):
    candidates = sorted(
            [
                path
                for path in IMAGES_DIR.glob(f'frame_{relative_frame:06d}*')
                if re.fullmatch(
                    rf'frame_{relative_frame:06d}(?:_\d+)?\.(?:jpg|jpeg|png)',
                    path.name,
                    re.IGNORECASE,
                )
            ],
            key=image_suffix,
        )
    active_tasks = [
        current_task_id
        for current_task_id in task_order
        if relative_frame < int(tasks[current_task_id].findtext('size'))
    ]
    if len(candidates) != len(active_tasks):
        raise FileNotFoundError(
            f'Frame {relative_frame} has {len(candidates)} exported images but '
            f'{len(active_tasks)} active tasks.'
        )
    try:
        image_path = candidates[active_tasks.index(task_id)]
    except ValueError as error:
        raise KeyError(f'Task {task_id} is not active at frame {relative_frame}.') from error
    return image_path.name


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
for annotation_track in root.findall('track'):
    task_id = annotation_track.get('task_id')
    if annotation_track.get('label') == 'bbox':
        bbox_tracks_by_task.setdefault(task_id, []).append(annotation_track)
    elif (
        annotation_track.get('label') == 'skeleton'
        and any(
            attribute_value(shape, 'cleaned') == 'true'
            for shape in annotation_track
        )
    ):
        skeleton_tracks_by_task.setdefault(task_id, []).append(annotation_track)

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
        if person_id is None:
            raise ValueError(
                f'Skeleton track {skeleton_track.get("id")} is missing person_id.'
            )
        bbox_track = bbox_by_person.get((task_id, person_id))
        if bbox_track is None:
            raise ValueError(
                f'No bbox track for person_id {person_id} in task {task_id}.'
            )
        bbox_for_skeleton[skeleton_track.get('id')] = bbox_track

selected_tracks = []
selected_images = set()
selected_shapes = []

for track in root.findall('track'):
    if track.get('label') != 'skeleton':
        continue
    task_id = track.get('task_id')
    task = tasks.get(task_id)
    if task is None:
        continue

    start_frame = task_start_frames.get(task_id, 0)
    filtered_track = copy.copy(track)
    filtered_track.clear()
    track_has_cleaned_shape = False

    for shape in track:
        if shape.find("./attribute[@name='cleaned']") is None:
            continue
        if attribute_value(shape, 'cleaned') != 'true':
            continue

        completion = int(attribute_value(shape, 'cleaned_completion') or '0')
        if completion <= 0:
            raise ValueError(
                f"Invalid cleaned_completion on track {track.get('id')} "
                f"at frame {shape.get('frame')}"
            )

        frame = int(shape.get('frame'))
        relative_frame = frame - start_frame
        if relative_frame % completion != 0:
            continue

        bbox_track = bbox_for_skeleton[track.get('id')]
        bbox_shape = next(
            (candidate for candidate in bbox_track if candidate.get('frame') == str(frame)),
            None,
        )
        if bbox_shape is None:
            raise ValueError(
                f'Missing bbox for skeleton track {track.get("id")} at frame {frame}'
            )

        filtered_track.append(copy.deepcopy(shape))
        track_has_cleaned_shape = True
        image_name = image_for_task_frame(task_id, relative_frame)
        selected_images.add((task_id, relative_frame, image_name))
        selected_shapes.append((task_id, relative_frame, track, shape, bbox_shape))

    if track_has_cleaned_shape:
        selected_tracks.append(filtered_track)

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

for task_id, relative_frame, image_name in sorted(selected_images):
    source_image = IMAGES_DIR / image_name
    if not source_image.exists():
        raise FileNotFoundError(f'Missing source image: {source_image}')
    shutil.copy2(source_image, OUTPUT_IMAGES_DIR / image_name)

output_root = ET.Element('dataset')
project_element = ET.SubElement(
    output_root,
    'project',
    id='1',
    name=project.findtext('name', ''),
)
videos_element = ET.SubElement(project_element, 'videos')

selected_images_by_frame = {
    (task_id, relative_frame): image_name
    for task_id, relative_frame, image_name in selected_images
}
frames_by_task = {}
for task_id, relative_frame, track, shape, bbox_shape in selected_shapes:
    frames_by_task.setdefault(task_id, {}).setdefault(relative_frame, []).append(
        (track, shape, bbox_shape)
    )

for task_id, frames in frames_by_task.items():
    video_element = ET.SubElement(
        videos_element,
        'video',
        id=task_id,
        name=tasks[task_id].findtext('name', ''),
    )
    for relative_frame in sorted(frames):
        frame_element = ET.SubElement(
            video_element,
            'frame',
            number=str(relative_frame),
            image=f"../images/{selected_images_by_frame[(task_id, relative_frame)]}",
        )
        for track, shape, bbox_shape in frames[relative_frame]:
            person_element = ET.SubElement(
                frame_element,
                'person',
                track_id=track.get('id', ''),
                person_id=track_attribute_value(track, 'person_id'),
                bbox_x1=bbox_shape.get('xtl', ''),
                bbox_y1=bbox_shape.get('ytl', ''),
                bbox_x2=bbox_shape.get('xbr', ''),
                bbox_y2=bbox_shape.get('ybr', ''),
            )
            for point in shape.findall('points'):
                x, y = point.get('points', '').split(',')
                visibility = '0' if point.get('outside') == '1' else (
                    '1' if point.get('occluded') == '1' else '2'
                )
                ET.SubElement(
                    person_element,
                    'keypoint',
                    id=point.get('label', ''),
                    x=x,
                    y=y,
                    visibility=visibility,
                )

output_tree = ET.ElementTree(output_root)
ET.indent(output_tree, space='  ')
output_tree.write(
    OUTPUT_ANNOTATIONS_DIR / 'annotations.xml',
    encoding='utf-8',
    xml_declaration=True,
)

print(f'Created {len(selected_images)} unique images.')
print(f'Created {len(selected_tracks)} cleaned tracks.')
print(f'Output: {OUTPUT_DIR}')

