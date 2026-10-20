from pathlib import Path
from src.util.dir import POSE_ESTIMATOR_VERSION_DIR, DETECTOR_VERSION_DIR, EXPORT_DIR

def get_version_subpath(annotation_version: str) -> Path:
    """Return the relative 'X.X/X.Y.X/X.Y.Z' tail for a semantic version."""
    parts = annotation_version.split('.')
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"Expected semantic version format 'X.Y.Z', got: {annotation_version}"
        )
    major, minor, patch = parts
    return Path(f'{major}.X') / f'{major}.{minor}.X' / f'{major}.{minor}.{patch}'


def get_version_folder(base_dir: Path, annotation_version: str) -> Path:
    """return the full path to the version directory for a semantic version."""
    version_dir = base_dir / get_version_subpath(annotation_version)
    if not version_dir.is_dir():
        raise FileNotFoundError(f'Dataset version directory not found: {version_dir}')
    return version_dir


def get_pose_estimator_version_folder(annotation_version: str) -> Path:
    return get_version_folder(POSE_ESTIMATOR_VERSION_DIR, annotation_version)


def get_detector_version_folder(annotation_version: str) -> Path:
    return get_version_folder(DETECTOR_VERSION_DIR, annotation_version)


def get_detector_annotation_file(annotation_version: str) -> Path:
    annotation_file = get_detector_version_folder(annotation_version) / 'cleaned_annotations.xml'
    if not annotation_file.is_file():
        raise FileNotFoundError(f'Detector annotations not found: {annotation_file}')
    return annotation_file


def parse_version(version: str) -> tuple[int, int, int]:
    parts = version.split('.')
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Expected semantic version 'X.Y.Z', got: {version}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def get_all_versions(base_dir: Path) -> list[str]:
    versions: list[str] = []
    for manifest in base_dir.rglob('manifest.yaml'):
        name = manifest.parent.name
        try:
            parse_version(name)
        except ValueError:
            continue
        versions.append(name)
    return sorted(versions, key=parse_version)


def get_all_pose_estimator_versions() -> list[str]:
    return get_all_versions(POSE_ESTIMATOR_VERSION_DIR)


def get_all_detector_versions() -> list[str]:
    return get_all_versions(DETECTOR_VERSION_DIR)


def get_previous_version(base_dir: Path, version: str) -> str:
    """Pick the natural predecessor: prior patch, or last build in prior minor line."""
    major, minor, patch = parse_version(version)
    all_versions = get_all_versions(base_dir)
    if not all_versions:
        raise FileNotFoundError('No dataset versions found')

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


def get_previous_pose_estimator_version(version: str) -> str:
    return get_previous_version(POSE_ESTIMATOR_VERSION_DIR, version)


def get_previous_detector_version(version: str) -> str:
    return get_previous_version(DETECTOR_VERSION_DIR, version)


def get_export_file_path(name: str) -> Path:
    file = EXPORT_DIR / f'{name}.xml'
    if not file.exists():
        raise FileNotFoundError(f'Export file not found: {file}')
    return file


def get_latest_pose_estimator_version() -> str:
    return get_all_pose_estimator_versions()[-1]

def get_latest_detector_version() -> str:
    return get_all_detector_versions()[-1]