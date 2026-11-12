from argparse import ArgumentParser
from datetime import datetime
from os.path import exists
from pathlib import Path
from shutil import move, rmtree
import tempfile
import zipfile

import yaml

from src.util.dir import EXPORT_DIR, MAPPING_DIR, POSE_ESTIMATOR_VERSION_DIR, IMAGES_DIR
from src.util.mapping import convert_keypoint_ids, CVAT_DEFAULT_MAPPING_NAME
from src.util.get_version import get_latest_pose_estimator_version, parse_version
from src.dataset_review.dataset_parser import build_annotations
from src.dataset_review.rename_image_store import rename_image_store


class _FlowStyleList(list):
    pass


class _ManifestDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _represent_flow_style_list(dumper, value):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=True)


_ManifestDumper.add_representer(_FlowStyleList, _represent_flow_style_list)


def _is_zip(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == '.zip'


def _annotation_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        return [
            name
            for name in zip_ref.namelist()
            if Path(name).name == 'annotations.xml' and not name.endswith('/')
        ]


def _find_annotations_xml(root: Path) -> Path:
    direct = root / 'annotations.xml'
    if direct.is_file():
        return direct
    matches = sorted(path for path in root.rglob('annotations.xml') if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f'No annotations.xml found under {root}')
    joined = ', '.join(str(path) for path in matches)
    raise ValueError(f'Multiple annotations.xml files under {root}: {joined}')


def _resolve_images_dir(folder: Path) -> Path:
    """Return the directory that contains CVAT frame_*.jpg files."""
    for candidate in (folder / 'images' / 'default', folder / 'images'):
        if candidate.is_dir() and any(candidate.glob('frame_*.*')):
            return candidate
    images = folder / 'images'
    if images.is_dir():
        return images
    raise FileNotFoundError(
        f'Task export at {folder} is missing an images directory with frame_*.* files'
    )


def _extract_xml_from_zip(zip_path: Path, destination: Path) -> Path:
    members = _annotation_members(zip_path)
    if len(members) != 1:
        raise ValueError(
            f'Expected exactly one annotations.xml in {zip_path}, found {len(members)}'
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref, zip_ref.open(members[0]) as source:
        destination.write_bytes(source.read())
    return destination


def resolve_project_annotations(input_path: Path) -> Path:
    """Resolve project annotations from an .xml file or a .zip containing one."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f'Input file {input_path} not found')
    if input_path.is_file() and input_path.suffix.lower() == '.xml':
        return input_path.resolve()
    if _is_zip(input_path):
        temp_xml = Path(
            tempfile.mkstemp(prefix='pose_annotations_', suffix='.xml')[1]
        )
        return _extract_xml_from_zip(input_path, temp_xml).resolve()
    raise ValueError(
        f'Project annotations must be an .xml file or .zip containing annotations.xml; '
        f'got {input_path}'
    )


def _materialize_task_dir(task_path: Path) -> tuple[Path, Path | None]:
    """Return (folder, cleanup_dir). cleanup_dir is removed by the caller when set."""
    task_path = Path(task_path)
    if not task_path.exists():
        raise FileNotFoundError(f'Task export not found: {task_path}')
    if task_path.is_dir():
        return task_path.resolve(), None
    if not _is_zip(task_path):
        raise ValueError(
            f'Task export must be a directory or .zip with annotations + images: {task_path}'
        )
    temp_dir = Path(tempfile.mkdtemp(prefix='pose_task_'))
    with zipfile.ZipFile(task_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir, temp_dir


def ingest_task_export(
    task_path: Path,
    image_store: Path,
    project_annotations: Path,
) -> int:
    """Ingest one task export (folder or zip) into the shared image store.

    Returns the number of images written.
    """
    folder, cleanup_dir = _materialize_task_dir(task_path)
    try:
        annotations = _find_annotations_xml(folder)
        images_dir = _resolve_images_dir(annotations.parent)
        mapping = rename_image_store(
            images=images_dir,
            annotations=annotations,
            output=image_store,
            apply=True,
            project_annotations=project_annotations,
        )
        print(f'Ingested {len(mapping)} images from {task_path} -> {image_store}')
        return len(mapping)
    finally:
        if cleanup_dir is not None and cleanup_dir.exists():
            rmtree(cleanup_dir)


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
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        move(input_path, destination)
        return destination, destination.stem
    return input_path, input_path.stem


def main():
    parser = ArgumentParser(
        description=(
            'Create a pose-estimator dataset version (cleaned annotations + manifest). '
            'Pass project annotations alone, or also task export(s) with images to ingest.'
        )
    )
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
        help=(
            'Optional task export(s): each is a folder or .zip containing '
            'annotations.xml and images/ to upload into the image store'
        ),
    )
    parser.add_argument(
        '--version-type',
        choices=('major', 'minor', 'patch'),
        type=str,
        required=True,
    )
    parser.add_argument(
        '--keypoints-used',
        type=str,
        required=True,
        help='comma separated list of keypoints used e.e. 1, 2. 3 or name of mapping i.e. "dance_23"',
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=True,
        help='Collect a description and change log interactively',
    )
    parser.add_argument(
        '--non-interactive',
        dest='interactive',
        action='store_false',
        help='Skip description and change prompts',
    )
    args = parser.parse_args()

    annotations_file = resolve_project_annotations(Path(args.input_file))

    ingested = 0
    for task_path in args.tasks:
        ingested += ingest_task_export(
            task_path=task_path,
            image_store=IMAGES_DIR,
            project_annotations=annotations_file,
        )
    if args.tasks:
        print(f'Total images ingested into {IMAGES_DIR}: {ingested}')

    # check if keypoints used is a list of integers or a mapping name
    if len(args.keypoints_used.split(',')) > 1:
        keypoints_used = [int(k) for k in args.keypoints_used.split(',')]
    else:
        # convert keypoints to the CVAT mapping
        keypoints_used = convert_keypoint_ids(args.keypoints_used, CVAT_DEFAULT_MAPPING_NAME)

    description = ''
    changes = []
    if args.interactive:
        description = input('Enter Description: ')

        while True:
            change = input('Enter Change (enter for done): ')
            if not change:
                break
            changes.append(change)

    # parse the file and save it at the version folder
    latest_version = get_latest_pose_estimator_version()
    major, minor, patch = parse_version(latest_version)
    if args.version_type == 'major':
        major, minor, patch = major + 1, 0, 0
    elif args.version_type == 'minor':
        minor, patch = minor + 1, 0
    elif args.version_type == 'patch':
        patch += 1
    else:
        raise ValueError(f'Invalid version type: {args.version_type}')

    version = f'{major}.{minor}.{patch}'
    version_folder = Path(f'{POSE_ESTIMATOR_VERSION_DIR}/{major}.X/{major}.{minor}.X/{version}')

    # create the version folder
    version_folder.mkdir(parents=True, exist_ok=True)

    image_count, track_count, via_completion, via_frame_cleaned = build_annotations(
        annotations_file=annotations_file,
        output_file=version_folder / 'cleaned_annotations.xml',
        images_dir=IMAGES_DIR,
        mapping_file=MAPPING_DIR / f'{CVAT_DEFAULT_MAPPING_NAME}.yaml',
        output_mapping_file=MAPPING_DIR / f'{args.keypoints_used}.yaml',
    )
    # create the manifest
    destination, export_version = _get_export_file(annotations_file)
    formatted_date = datetime.now().strftime('%Y-%m-%d')
    manifest = {
        'annotation_version': version,
        'image_version': 0,
        'export_version': export_version,
        'date': formatted_date,
        'description': description,
        'changes': changes,
        'statistics': {
            'frames_used': image_count,
            'tasks_used': 'x',
            'tracks': track_count,
            'track_frames': via_frame_cleaned + via_completion,
        },
        'keypoints_used': _FlowStyleList(keypoints_used),
    }
    with open(version_folder / 'manifest.yaml', 'w') as f:
        yaml.dump(
            manifest,
            f,
            Dumper=_ManifestDumper,
            default_flow_style=False,
            sort_keys=False,
            width=4096,
        )
    print(f'Wrote {destination}')
    print(f'Wrote {version_folder / "cleaned_annotations.xml"}')
    print(f'Wrote {version_folder / "manifest.yaml"}')


if __name__ == '__main__':
    main()
