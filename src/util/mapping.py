import yaml
from src.util.dir import MAPPING_DIR

CVAT_DEFAULT_MAPPING_NAME = 'cvat_dance_28'
STANDARD_MAPPING_NAME = 'standard'

def mapping_exists(mapping_name):
    return (MAPPING_DIR / f'{mapping_name}.yaml').is_file()

def convert_keypoint_ids(input_mapping_name, output_mapping_name):
    if not mapping_exists(input_mapping_name):
        raise ValueError(f'Mapping {input_mapping_name} not found')
    if not mapping_exists(output_mapping_name):
        raise ValueError(f'Mapping {output_mapping_name} not found')

    with open(MAPPING_DIR / f'{input_mapping_name}.yaml', encoding='utf-8') as f:
        input_mapping = yaml.safe_load(f)
        input_keypoints = input_mapping['keypoints']
    with open(MAPPING_DIR / f'{output_mapping_name}.yaml', encoding='utf-8') as f:
        output_mapping = yaml.safe_load(f)
        output_keypoints = output_mapping['keypoints']

    standard_keypoint_ids = {
        keypoint['standard_id']
        for keypoint in input_keypoints.values()
    }

    return [
        keypoint['id']
        for keypoint in output_keypoints.values()
        if keypoint['standard_id'] in standard_keypoint_ids
    ]