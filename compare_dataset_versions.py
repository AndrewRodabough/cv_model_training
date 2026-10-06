#!/usr/bin/env python3
"""Compare two annotation versions and review frames added in the newer one.

Default base version:
  - 1.6.5 → 1.6.4  (previous patch in the same minor line)
  - 1.6.0 → last version under 1.5.X (e.g. 1.5.0)

Examples:
  python compare_dataset_versions.py 1.6.5
  python compare_dataset_versions.py 1.6.0
  python compare_dataset_versions.py 1.6.5 --base 1.6.3
  python compare_dataset_versions.py 1.6.5 --list-only
  python compare_dataset_versions.py 1.6.5 --show removed
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from dataset_parser import resolve_dataset_dir
from review_training_frames import draw_person


@dataclass(frozen=True)
class FrameRef:
    video_id: str
    video_name: str
    frame_number: str
    image_name: str
    image_rel: str


def parse_version(version: str) -> tuple[int, int, int]:
    parts = version.split('.')
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Expected semantic version 'X.Y.Z', got: {version}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def discover_versions(root: Path = Path('dataset/versions')) -> list[str]:
    versions: list[str] = []
    for manifest in root.rglob('manifest.yaml'):
        name = manifest.parent.name
        try:
            parse_version(name)
        except ValueError:
            continue
        versions.append(name)
    return sorted(versions, key=parse_version)


def previous_version(version: str) -> str:
    """Pick the natural predecessor: prior patch, or last build in prior minor line."""
    major, minor, patch = parse_version(version)
    all_versions = discover_versions()
    if not all_versions:
        raise FileNotFoundError('No dataset versions found under dataset/versions')

    if patch > 0:
        same_minor_earlier = [
            ver
            for ver in all_versions
            if parse_version(ver)[:2] == (major, minor)
            and parse_version(ver) < (major, minor, patch)
        ]
        if same_minor_earlier:
            return same_minor_earlier[-1]
        raise FileNotFoundError(
            f'No earlier patch than {version} in {major}.{minor}.X'
        )

    # X.Y.0 → last version in X.(Y-1).X
    if minor <= 0:
        earlier = [ver for ver in all_versions if parse_version(ver) < (major, minor, patch)]
        if earlier:
            return earlier[-1]
        raise FileNotFoundError(f'No version before {version}')

    prev_minor = [
        ver
        for ver in all_versions
        if parse_version(ver)[0] == major and parse_version(ver)[1] == minor - 1
    ]
    if prev_minor:
        return prev_minor[-1]
    raise FileNotFoundError(
        f'No versions found under {major}.{minor - 1}.X to compare with {version}'
    )


def load_frame_index(annotations: Path) -> dict[str, FrameRef]:
    """Index frames by image basename (stable across version dirs)."""
    root = ET.parse(annotations).getroot()
    index: dict[str, FrameRef] = {}
    for video in root.findall('./project/videos/video'):
        video_id = video.get('id', '')
        video_name = video.get('name', '')
        for frame in video.findall('frame'):
            image_rel = frame.get('image') or ''
            image_name = Path(image_rel).name
            if not image_name:
                continue
            index[image_name] = FrameRef(
                video_id=video_id,
                video_name=video_name,
                frame_number=frame.get('number', ''),
                image_name=image_name,
                image_rel=image_rel,
            )
    return index


def annotations_path(version: str) -> Path:
    path = resolve_dataset_dir(version) / 'cleaned_annotations.xml'
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {path}; run: python dataset_parser.py {version}'
        )
    return path


def find_frame_element(annotations: Path, image_name: str):
    root = ET.parse(annotations).getroot()
    for video in root.findall('./project/videos/video'):
        for frame in video.findall('frame'):
            if Path(frame.get('image') or '').name == image_name:
                return video, frame
    return None, None


def print_summary(
    new_version: str,
    base_version: str,
    added: list[FrameRef],
    removed: list[FrameRef],
    shared: int,
) -> None:
    print(f'Compare {base_version} → {new_version}')
    print(f'  shared={shared}  added={len(added)}  removed={len(removed)}')
    if added:
        print('\nAdded frames:')
        for ref in added:
            print(
                f'  + task_{ref.video_id} frame_{ref.frame_number} '
                f'{ref.image_name} ({ref.video_name})'
            )
    if removed:
        print('\nRemoved frames:')
        for ref in removed:
            print(
                f'  - task_{ref.video_id} frame_{ref.frame_number} '
                f'{ref.image_name} ({ref.video_name})'
            )


def review_frames(
    refs: list[FrameRef],
    annotations: Path,
    *,
    title_prefix: str,
) -> None:
    if not refs:
        print('Nothing to review.')
        return

    # Cache parsed tree for drawing annotations from the newer/selected XML.
    root = ET.parse(annotations).getroot()
    frame_by_image: dict[str, tuple[ET.Element, ET.Element]] = {}
    for video in root.findall('./project/videos/video'):
        for frame in video.findall('frame'):
            frame_by_image[Path(frame.get('image') or '').name] = (video, frame)

    figure, axis = plt.subplots(figsize=(12, 8))
    state = {'position': 0}

    def show_frame() -> None:
        position = state['position']
        if position < 0:
            state['position'] = 0
            position = 0
        if position >= len(refs):
            plt.close(figure)
            return

        ref = refs[position]
        image_path = (annotations.parent / ref.image_rel).resolve()
        if not image_path.is_file():
            # Fallback to shared image_store by basename.
            image_path = Path('dataset/image_store') / ref.image_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f'Could not read image: {image_path}')

        pair = frame_by_image.get(ref.image_name)
        if pair is not None:
            _video, frame = pair
            for person in frame.findall('person'):
                draw_person(image, person)

        axis.clear()
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(
            f'{title_prefix} | task_{ref.video_id} frame {ref.frame_number} | '
            f'{ref.image_name} | {position + 1}/{len(refs)}\n'
            f'{ref.video_name} | ←/→ or A/D: prev/next | Space/Enter: next | Q/Esc: quit'
        )
        axis.axis('off')
        figure.canvas.draw_idle()

    def handle_key(event) -> None:
        if event.key in (' ', 'enter', 'right', 'd'):
            state['position'] += 1
            show_frame()
        elif event.key in ('left', 'a'):
            state['position'] = max(0, state['position'] - 1)
            show_frame()
        elif event.key in ('q', 'escape'):
            plt.close(figure)

    figure.canvas.mpl_connect('key_press_event', handle_key)
    print(f'Reviewing {len(refs)} frame(s). Annotations: {annotations}')
    show_frame()
    plt.show()
    if state['position'] >= len(refs):
        print('Finished reviewing all listed frames.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Find and review frames added between dataset annotation versions.'
    )
    parser.add_argument(
        'version',
        help="Newer annotation version, e.g. '1.6.5'",
    )
    parser.add_argument(
        '--base',
        default=None,
        help='Older version to compare against (default: previous patch / last prior minor).',
    )
    parser.add_argument(
        '--show',
        choices=('added', 'removed', 'both'),
        default='added',
        help='Which side of the diff to review interactively (default: added).',
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Print the diff summary and exit without opening the viewer.',
    )
    args = parser.parse_args()

    new_version = args.version
    base_version = args.base or previous_version(new_version)
    if parse_version(base_version) >= parse_version(new_version):
        raise ValueError(
            f'Base version {base_version} must be older than {new_version}'
        )

    new_ann = annotations_path(new_version)
    base_ann = annotations_path(base_version)
    new_index = load_frame_index(new_ann)
    base_index = load_frame_index(base_ann)

    added_names = sorted(set(new_index) - set(base_index))
    removed_names = sorted(set(base_index) - set(new_index))
    shared = len(set(new_index) & set(base_index))

    def sort_key(ref: FrameRef):
        try:
            return (int(ref.video_id), int(ref.frame_number), ref.image_name)
        except ValueError:
            return (ref.video_id, ref.frame_number, ref.image_name)

    added = sorted((new_index[name] for name in added_names), key=sort_key)
    removed = sorted((base_index[name] for name in removed_names), key=sort_key)

    print_summary(new_version, base_version, added, removed, shared)
    if args.list_only:
        return

    if args.show in ('added', 'both'):
        review_frames(
            added,
            new_ann,
            title_prefix=f'ADDED in {new_version} (vs {base_version})',
        )
    if args.show in ('removed', 'both'):
        review_frames(
            removed,
            base_ann,
            title_prefix=f'REMOVED after {base_version} (gone in {new_version})',
        )


if __name__ == '__main__':
    main()
