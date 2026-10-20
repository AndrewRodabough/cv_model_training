from argparse import ArgumentParser
from datetime import date, datetime
from os.path import exists
from pathlib import Path
from shutil import move
import yaml

from src.util.dir import EXPORT_DIR, MAPPING_DIR, POSE_ESTIMATOR_VERSION_DIR, IMAGES_DIR
from src.util.mapping import convert_keypoint_ids, CVAT_DEFAULT_MAPPING_NAME
from src.util.get_version import get_latest_pose_estimator_version, parse_version
from src.dataset_review.dataset_parser import build_annotations


class _FlowStyleList(list):
    pass


class _ManifestDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _represent_flow_style_list(dumper, value):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', value, flow_style=True)


_ManifestDumper.add_representer(_FlowStyleList, _represent_flow_style_list)


def _get_export_file(input_file):
    input_path = Path(input_file).resolve()
    export_dir = EXPORT_DIR.resolve()
    try:
        input_path.relative_to(export_dir)
    except ValueError:
        formatted_date = datetime.now().strftime("%Y-%m-%d")
        index = 0
        while exists(EXPORT_DIR / f'{formatted_date}_{index:03d}.xml'):
            index += 1
            if index > 999:
                raise ValueError(f'Index {index} is too large')
        destination = EXPORT_DIR / f'{formatted_date}_{index:03d}.xml'
        move(input_path, destination)
        return destination, destination.stem
    return input_path, input_path.stem


def main():
    parser = ArgumentParser(description='Create a manifest file for a dataset')
    parser.add_argument('--input-file', type=str, help='The input file')
    parser.add_argument('--version-type', choices=('major', 'minor', 'patch'),type=str, required=True)
    parser.add_argument('--keypoints-used', type=str, required=True, help='comma separated list of keypoints used e.e. 1, 2. 3 or name of mapping i.e. "dance_23"')
    parser.add_argument('--interactive', type=bool, default=True, help='if true, will ask for input for description and changes')
    args = parser.parse_args()

    # check if input file exists
    if not exists(args.input_file):
        raise FileNotFoundError(f'Input file {args.input_file} not found')


    # check if keypoints used is a list of integers or a mapping name
    if len(args.keypoints_used.split(',')) > 1:
        keypoints_used = [int(k) for k in args.keypoints_used.split(',')]
    else:
        # convert keypoints to the CVAT mapping
        keypoints_used = convert_keypoint_ids(args.keypoints_used, CVAT_DEFAULT_MAPPING_NAME)

    # if interactive, ask for description and change log
    if args.interactive:
        description = input('Enter Description: ')

        change = None
        changes = []
        while change != '':
            change = input('Enter Change (enter for done): ')
            if change:
                changes.append(change)

    # parse the file and save it at the version folder
    latest_version = get_latest_pose_estimator_version()
    major, minor, patch = parse_version(latest_version)
    if args.version_type == 'major':
        major += 1
    elif args.version_type == 'minor':
        minor += 1
    elif args.version_type == 'patch':
        patch += 1
    else:
        raise ValueError(f'Invalid version type: {args.version_type}')

    version = f'{major}.{minor}.{patch}'
    version_folder = Path(f'{POSE_ESTIMATOR_VERSION_DIR}/{major}.X/{major}.{minor}.X/{version}')

    # create the version folder
    version_folder.mkdir(parents=True, exist_ok=True)

    image_count, track_count, via_completion, via_frame_cleaned = build_annotations(
        annotations_file=args.input_file,
        output_file=version_folder/'cleaned_annotations.xml',
        images_dir=IMAGES_DIR,
        mapping_file=MAPPING_DIR / f'{CVAT_DEFAULT_MAPPING_NAME}.yaml',
        output_mapping_file=MAPPING_DIR / f'{args.keypoints_used}.yaml',
    )
    # create the manifest
    destination, export_version = _get_export_file(args.input_file)
    formatted_date = datetime.now().strftime("%Y-%m-%d")
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
    with open(version_folder/'manifest.yaml', 'w') as f:
        yaml.dump(
            manifest,
            f,
            Dumper=_ManifestDumper,
            default_flow_style=False,
            sort_keys=False,
            width=4096,
        )

if __name__ == '__main__':
    main()