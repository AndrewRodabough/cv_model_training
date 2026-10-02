import argparse
import shutil
import zipfile
from pathlib import Path

from dataset_parser import build_annotations, resolve_dataset_dir
from rename_image_store import rename_image_store


def folder_has_annotations(folder: Path) -> bool:
    return (folder / 'annotations.xml').is_file()


def resolve_images_dir(folder: Path) -> Path | None:
    """Return the directory that contains CVAT frame_*.jpg files, if any."""
    for candidate in (folder / 'images' / 'default', folder / 'images'):
        if candidate.is_dir() and any(candidate.glob('frame_*.*')):
            return candidate
    images = folder / 'images'
    if images.is_dir():
        return images
    return None


def zip_contains_annotations(zip_path: Path) -> bool:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        return any(Path(name).name == 'annotations.xml' for name in zip_ref.namelist())


def extract_annotation_zips(version_dir: Path) -> None:
    """Extract zips that contain annotations.xml into <stem>/ and remove the zip."""
    for zip_path in sorted(version_dir.glob('*.zip')):
        if not zip_contains_annotations(zip_path):
            continue
        dest = version_dir / zip_path.stem
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest)
        zip_path.unlink()
        print(f'Extracted {zip_path.name} -> {dest}')


def discover_annotation_sources(version_dir: Path):
    """Classify annotation sources by whether they include images.

    Location does not matter: only presence of an images tree decides the role.
    Returns (project_folders_without_images, folders_with_images).
    folders_with_images entries are (folder, images_dir).
    """
    extract_annotation_zips(version_dir)

    candidates = []
    if folder_has_annotations(version_dir):
        candidates.append(version_dir)
    for child in sorted(version_dir.iterdir()):
        if child.is_dir() and folder_has_annotations(child):
            candidates.append(child)

    without_images = []
    with_images = []
    for folder in candidates:
        images_dir = resolve_images_dir(folder)
        if images_dir is None:
            without_images.append(folder)
        else:
            with_images.append((folder, images_dir))
    return without_images, with_images


def ingest_images(folder: Path, images_dir: Path, image_store: Path) -> None:
    annotations = folder / 'annotations.xml'
    rename_image_store(
        images=images_dir,
        annotations=annotations,
        output=image_store,
        apply=True,
    )
    images_root = folder / 'images'
    if images_root.is_dir():
        shutil.rmtree(images_root)
        print(f'Removed source images at {images_root}')


KEEP_VERSION_FILES = frozenset({
    'annotations.xml',
    'cleaned_annotations.xml',
    'manifest.yaml',
})


def ensure_project_annotations_at_root(version_dir: Path, annotations_file: Path) -> Path:
    """Make sure project annotations.xml lives at the version root.

    Does not remove version_dir itself if annotations were already there.
    """
    target = version_dir / 'annotations.xml'
    source = Path(annotations_file).resolve()
    if source == target.resolve():
        return target
    if not source.is_file():
        raise FileNotFoundError(f'Missing project annotations: {annotations_file}')
    if target.exists() and target.resolve() != source:
        raise FileExistsError(
            f'Cannot move {source} to {target}: destination already exists.'
        )
    shutil.move(str(source), str(target))
    print(f'Moved project annotations to {target}')
    return target


def cleanup_version_dir(version_dir: Path) -> None:
    """Leave only annotations.xml, cleaned_annotations.xml, and manifest.yaml.

    Removes leftover export dirs/files inside version_dir. Never deletes version_dir.
    """
    version_dir = Path(version_dir).resolve()
    removed = []
    for child in sorted(version_dir.iterdir()):
        if child.name in KEEP_VERSION_FILES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed.append(child.name)
    if removed:
        print(f'Cleaned version folder; removed: {", ".join(removed)}')
    else:
        print('Version folder already clean.')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Ingest CVAT exports for a dataset version (optional task images), '
            'then build cleaned_annotations.xml via dataset_parser.'
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--version',
        metavar='VERSION',
        help="Annotation version, e.g. '1.4.0' -> dataset/versions/1.X/1.4.X/1.4.0",
    )
    mode.add_argument(
        '--input-folder',
        type=Path,
        help='Version folder to parse (alternative to --version)',
    )
    parser.add_argument(
        '--output-folder',
        type=Path,
        help='Folder to write cleaned_annotations.xml (default: version folder)',
    )
    parser.add_argument(
        '--images',
        type=Path,
        default=Path('dataset/image_store'),
        help='Image store directory (default: dataset/image_store)',
    )
    parser.add_argument(
        '--mapping',
        type=Path,
        default=Path('mapping/cvat_dance_28.yaml'),
        help='CVAT label mapping YAML (default: mapping/cvat_dance_28.yaml)',
    )
    args = parser.parse_args()

    if args.input_folder is not None:
        version_dir = Path(args.input_folder)
    else:
        version_dir = resolve_dataset_dir(args.version)

    if not version_dir.is_dir():
        raise FileNotFoundError(f'Version directory not found: {version_dir}')

    if args.output_folder is not None:
        output_file = Path(args.output_folder) / 'cleaned_annotations.xml'
    else:
        output_file = version_dir / 'cleaned_annotations.xml'

    without_images, with_images = discover_annotation_sources(version_dir)
    print(
        f'Found {len(without_images)} annotation source(s) without images, '
        f'{len(with_images)} with images.'
    )

    # 1) Single source with images: full project export + images.
    if len(without_images) == 0 and len(with_images) == 1:
        folder, images_dir = with_images[0]
        annotations_file = folder / 'annotations.xml'
        ingest_images(folder, images_dir, args.images)

    # 2) Single source without images: project annotations only.
    elif len(without_images) == 1 and len(with_images) == 0:
        annotations_file = without_images[0] / 'annotations.xml'

    # 3) One project (no images) + any task exports (with images).
    elif len(without_images) == 1 and len(with_images) >= 1:
        annotations_file = without_images[0] / 'annotations.xml'
        for folder, images_dir in with_images:
            ingest_images(folder, images_dir, args.images)

    elif len(without_images) > 1:
        paths = ', '.join(str(path / 'annotations.xml') for path in without_images)
        raise ValueError(
            f'Expected exactly one project annotations file (no images); '
            f'found {len(without_images)}: {paths}'
        )
    elif len(without_images) == 0 and len(with_images) > 1:
        paths = ', '.join(str(folder / 'annotations.xml') for folder, _ in with_images)
        raise ValueError(
            f'Multiple annotation sources with images but no project annotations '
            f'without images. Add one project annotations.xml (no images), or keep '
            f'only a single full project export with images. Found: {paths}'
        )
    else:
        raise ValueError(
            f'No annotations.xml found under {version_dir}. '
            f'Place a project annotations.xml (and optional task folders/zips with images).'
        )

    image_count, track_count, via_completion, via_frame_cleaned = build_annotations(
        annotations_file=annotations_file,
        output_file=output_file,
        images_dir=args.images,
        mapping_file=args.mapping,
    )

    annotations_file = ensure_project_annotations_at_root(version_dir, annotations_file)
    cleaned_at_root = version_dir / 'cleaned_annotations.xml'
    if output_file.resolve() != cleaned_at_root.resolve():
        shutil.copy2(output_file, cleaned_at_root)
        print(f'Copied cleaned annotations to {cleaned_at_root}')
    cleanup_version_dir(version_dir)

    print(f'Dataset dir: {version_dir}')
    print(f'Project annotations: {annotations_file}')
    print(f'Created {image_count} unique images.')
    print(f'Created {track_count} cleaned tracks.')
    print(f'Selected shapes via cleaned_completion grid: {via_completion}')
    print(f'Selected shapes via frame_cleaned: {via_frame_cleaned}')
    print(f'Output: {cleaned_at_root}')
    print(f'Keypoint ids written as standard_id via {args.mapping}')


if __name__ == '__main__':
    main()
