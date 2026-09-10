# Dance Model Training

Research code for training a top-down 2D human pose estimator on dance videos.

## Current State

- CVAT annotations are processed into cleaned, task-specific training annotations.
- Bounding boxes are paired with skeletons using task-local `person_id` values.
- The shared image store uses task/frame image names.
- Training uses DINOv2 with a SimCC keypoint head and 20 selected keypoints.
- Training and inference code are still under active development.