# Dance Model Training

code for training a full top-down 3D human pose estimation on dance videos.

## Current State

- CVAT annotations are processed into cleaned, task-specific training annotations.
- Bounding boxes are paired with skeletons using task-local `person_id` values.
- The shared image store uses task/frame image names.
- Training uses DINOv3 with a SimCC keypoint head and 23 selected keypoints.
- Training and inference code are still under active development.