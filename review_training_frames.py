import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt


SKELETON_CONNECTIONS = (
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (15, 16),
    (16, 17),
    (18, 19),
    (19, 20),
    (17, 21),
    (17, 22),
    (17, 23),
    (17, 24),
    (20, 25),
    (20, 26),
    (20, 27),
    (20, 28),
)


def person_color(person_id):
    digest = hashlib.sha256(str(person_id).encode('utf-8')).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))


def draw_person(image, person):
    person_id = person.get('person_id', person.get('track_id', 'unknown'))
    color = person_color(person_id)
    bbox = tuple(float(person.get(name)) for name in ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'))
    x1, y1, x2, y2 = (round(value) for value in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image,
        f'person_id={person_id}',
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    points = {}
    for keypoint in person.findall('keypoint'):
        point_id = int(keypoint.get('id'))
        visibility = int(keypoint.get('visibility', '0'))
        x = float(keypoint.get('x'))
        y = float(keypoint.get('y'))
        points[point_id] = (round(x), round(y), visibility)
        if visibility == 0:
            continue
        radius = 5 if visibility == 2 else 4
        cv2.circle(image, (round(x), round(y)), radius, color, -1)
        cv2.putText(
            image,
            str(point_id),
            (round(x) + 6, round(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    for first_id, second_id in SKELETON_CONNECTIONS:
        first = points.get(first_id)
        second = points.get(second_id)
        if first is None or second is None or first[2] == 0 or second[2] == 0:
            continue
        cv2.line(image, first[:2], second[:2], color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description='Review training frames one at a time.')
    parser.add_argument(
        '--annotations',
        type=Path,
        default=Path('datasets/cleaned_dataset_1.0_9-9/annotations/annotations.xml'),
    )
    parser.add_argument('--start', type=int, default=0, help='Frame position at which to start.')
    parser.add_argument('--video-id', help='Review only one video/task ID.')
    args = parser.parse_args()

    root = ET.parse(args.annotations).getroot()
    frames = [
        (video, frame)
        for video in root.findall('./project/videos/video')
        if args.video_id is None or video.get('id') == args.video_id
        for frame in video.findall('frame')
    ]
    if not frames:
        raise ValueError(f'No frames found in {args.annotations}')
    if not 0 <= args.start < len(frames):
        raise ValueError(f'--start must be between 0 and {len(frames) - 1}')

    figure, axis = plt.subplots(figsize=(12, 8))
    state = {'position': args.start}

    def show_frame():
        position = state['position']
        if position >= len(frames):
            plt.close(figure)
            return

        video, frame = frames[position]
        image_path = (args.annotations.parent / frame.get('image')).resolve()
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Could not read image: {image_path}')

        for person in frame.findall('person'):
            draw_person(image, person)

        axis.clear()
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(
            f'Video {video.get("id")}: {video.get("name")} | '
            f'Frame {frame.get("number")} | {Path(image_path).name} | '
            f'{position + 1}/{len(frames)} | '
            'Space/Enter: next | Q/Esc: quit'
        )
        axis.axis('off')
        figure.canvas.draw_idle()

    def handle_key(event):
        if event.key in (' ', 'enter'):
            state['position'] += 1
            show_frame()
        elif event.key in ('q', 'escape'):
            plt.close(figure)

    figure.canvas.mpl_connect('key_press_event', handle_key)
    show_frame()
    plt.show()
    if state['position'] >= len(frames):
        print('Finished reviewing all training frames.')


if __name__ == '__main__':
    main()
