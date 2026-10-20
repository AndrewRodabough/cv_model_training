import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image


IMAGE_PATTERN = re.compile(
    r"frame_(?P<frame>\d+)(?:_(?P<suffix>\d+))?\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


def parse_image_name(path):
    match = IMAGE_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group("frame")), int(match.group("suffix") or 0)


def _parse_task_elements(task_elements):
    task_data = []
    for task in task_elements:
        task_id = task.findtext("id")
        size = task.findtext("size")
        if task_id is None or not task_id.strip() or size is None:
            raise ValueError("Every task must contain id and size metadata.")
        if not task_id.isdigit() or int(task_id) < 1:
            raise ValueError(f"Invalid CVAT task ID: {task_id!r}")
        task_data.append((task_id, int(size)))
    return task_data


def _load_project_task_list(project_annotations):
    root = ET.parse(project_annotations).getroot()
    tasks = root.findall("./meta/project/tasks/task")
    if not tasks:
        raise ValueError(
            f"No project tasks found in {project_annotations}. "
            "Expected meta/project/tasks/task entries."
        )
    return _parse_task_elements(tasks)


def _resolve_job_task(job_element, project_annotations):
    size_text = job_element.findtext("size")
    if size_text is None:
        raise ValueError("CVAT job export is missing size metadata.")
    size = int(size_text)
    project_tasks = _load_project_task_list(project_annotations)
    matches = [(task_id, task_size) for task_id, task_size in project_tasks if task_size == size]
    if len(matches) != 1:
        raise ValueError(
            f"CVAT job export (size={size}) matched {len(matches)} project tasks in "
            f"{project_annotations}; need exactly one match to choose task_id."
        )
    return matches


def load_tasks(annotation_path, project_annotations=None):
    annotation_path = Path(annotation_path)
    root = ET.parse(annotation_path).getroot()

    project_tasks = root.findall("./meta/project/tasks/task")
    if project_tasks:
        return _parse_task_elements(project_tasks)

    task = root.find("./meta/task")
    if task is not None:
        return _parse_task_elements([task])

    job = root.find("./meta/job")
    if job is not None:
        if project_annotations is None:
            raise ValueError(
                f"CVAT job export in {annotation_path} does not include task id "
                "(only job/segment id). Pass project annotations to resolve task_id."
            )
        return _resolve_job_task(job, project_annotations)

    raise ValueError(f"No CVAT tasks found in {annotation_path}")


def build_mapping(images_dir, tasks):
    images_by_frame = {}
    for image_path in images_dir.iterdir():
        if not image_path.is_file():
            continue
        parsed = parse_image_name(image_path)
        if parsed is None:
            continue
        frame_number, suffix = parsed
        images_by_frame.setdefault(frame_number, []).append((suffix, image_path))

    mapping = []
    for frame_number, candidates in sorted(images_by_frame.items()):
        candidates.sort(key=lambda item: item[0])
        active_tasks = [
            task_id
            for task_id, size in tasks
            if frame_number < size
        ]
        if len(candidates) != len(active_tasks):
            raise ValueError(
                f"Frame {frame_number}: found {len(candidates)} images but "
                f"{len(active_tasks)} active tasks. CVAT export order cannot be "
                "reconstructed safely."
            )
        for (_, source_path), task_id in zip(candidates, active_tasks):
            output_name = f"task_{task_id}_frame_{frame_number:06d}.jpg"
            mapping.append((source_path, output_name))
    return mapping


def rename_image_store(
    images, annotations, output=None, apply=False, project_annotations=None
):
    """Map CVAT collision-suffixed frames to task/frame names.

    Returns the (source_path, output_name) mapping. When apply is True, also
    writes JPEGs into output (default: <images>/named).
    """
    images = Path(images)
    annotations = Path(annotations)
    if not images.is_dir():
        raise NotADirectoryError(images)
    if not annotations.is_file():
        raise FileNotFoundError(annotations)

    output_dir = Path(output) if output is not None else images / "named"
    tasks = load_tasks(annotations, project_annotations=project_annotations)
    mapping = build_mapping(images, tasks)

    print(f"Tasks: {len(tasks)}")
    print(f"Images: {len(mapping)}")
    print(f"Output: {output_dir}")
    if not apply:
        print("Dry run only. Pass apply=True to write JPEG files.")
        for source_path, output_name in mapping[:20]:
            print(f"{source_path.name} -> {output_name}")
        if len(mapping) > 20:
            print(f"... {len(mapping) - 20} more mappings")
        return mapping

    missing_sources = [
        source_path
        for source_path, _ in mapping
        if not source_path.exists()
    ]
    if missing_sources:
        examples = "\n".join(str(path) for path in missing_sources[:10])
        raise FileNotFoundError(
            f"{len(missing_sources)} source images are missing. Examples:\n{examples}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for source_path, output_name in mapping:
        destination = output_dir / output_name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        with Image.open(source_path) as image:
            image.convert("RGB").save(destination, format="JPEG", quality=95)
        written += 1

    print(f"Wrote {written} JPEG files.")
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Convert CVAT collision-suffixed images to task/frame names."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="CVAT image directory, such as images/default.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="CVAT annotations.xml file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory. Defaults to <images>/named.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write files. Without this flag, only print the planned mapping.",
    )
    parser.add_argument(
        "--project-annotations",
        type=Path,
        help="Project annotations.xml (required for CVAT job exports).",
    )
    args = parser.parse_args()
    rename_image_store(
        images=args.images,
        annotations=args.annotations,
        output=args.output,
        apply=args.apply,
        project_annotations=args.project_annotations,
    )


if __name__ == "__main__":
    main()
