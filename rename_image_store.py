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


def load_tasks(annotation_path):
    root = ET.parse(annotation_path).getroot()
    tasks = root.findall("./meta/project/tasks/task")
    if not tasks:
        tasks = root.findall(".//task")
    if not tasks:
        raise ValueError(f"No CVAT tasks found in {annotation_path}")

    task_data = []
    for task in tasks:
        task_id = task.findtext("id")
        size = task.findtext("size")
        if task_id is None or not task_id.strip() or size is None:
            raise ValueError("Every task must contain id and size metadata.")
        if not task_id.isdigit() or int(task_id) < 1:
            raise ValueError(f"Invalid CVAT task ID: {task_id!r}")
        task_data.append((task_id, int(size)))
    return task_data


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
    args = parser.parse_args()

    if not args.images.is_dir():
        raise NotADirectoryError(args.images)
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)

    output_dir = args.output or args.images / "named"
    tasks = load_tasks(args.annotations)
    mapping = build_mapping(args.images, tasks)

    print(f"Tasks: {len(tasks)}")
    print(f"Images: {len(mapping)}")
    print(f"Output: {output_dir}")
    if not args.apply:
        print("Dry run only. Add --apply to write PNG files.")
        for source_path, output_name in mapping[:20]:
            print(f"{source_path.name} -> {output_name}")
        if len(mapping) > 20:
            print(f"... {len(mapping) - 20} more mappings")
        return

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

    print(f"Wrote {written} PNG files.")


if __name__ == "__main__":
    main()
