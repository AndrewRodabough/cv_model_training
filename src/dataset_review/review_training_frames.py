import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt

from src.util.get_version import get_pose_estimator_version_folder
from src.util.drawing import draw_person

def main():
    parser = argparse.ArgumentParser(description='Review training frames one at a time.')
    parser.add_argument(
        '--version',
        default=None,
        help="Annotation version, e.g. '1.4.0'",
    )
    parser.add_argument(
        '--annotations',
        type=Path,
        default=None,
        help='Override path to cleaned_annotations.xml (default: <version dir>/cleaned_annotations.xml)',
    )
    parser.add_argument('--start', type=int, default=0, help='Frame position at which to start.')
    parser.add_argument('--video-id', help='Review only one video/task ID.')
    args = parser.parse_args()

    annotations = args.annotations or (
        get_pose_estimator_version_folder(args.version) / 'cleaned_annotations.xml'
    )
    if not annotations.is_file():
        raise FileNotFoundError(f'Missing cleaned annotations: {annotations}')

    root = ET.parse(annotations).getroot()
    frames = [
        (video, frame)
        for video in root.findall('./project/videos/video')
        if args.video_id is None or video.get('id') == args.video_id
        for frame in video.findall('frame')
    ]
    if not frames:
        raise ValueError(f'No frames found in {annotations}')
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
        image_path = (annotations.parent / frame.get('image')).resolve()
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
    print(f'Annotations: {annotations}')
    show_frame()
    plt.show()
    if state['position'] >= len(frames):
        print('Finished reviewing all training frames.')


if __name__ == '__main__':
    main()
