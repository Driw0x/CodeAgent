import hashlib
import json
from pathlib import Path

MANIFEST_FILE = "manifest.json"


def project_relative_path(path: str | Path, project_path: str | Path) -> str:
    path = Path(path).resolve()
    project_path = Path(project_path).resolve()

    try:
        return path.relative_to(project_path).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(files: list[dict], project_path: str | Path) -> dict[str, str]:
    manifest = {}

    for file in files:
        relative_path = project_relative_path(file["path"], project_path)
        file_hash = hashlib.sha256(file["content"].encode("utf-8")).hexdigest()
        manifest[relative_path] = file_hash

    return manifest


def manifest_exists(directory: str | Path) -> bool:
    return (Path(directory) / MANIFEST_FILE).exists()


def save_manifest(manifest: dict[str, str], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest(directory: str | Path) -> dict[str, str]:
    return json.loads(
        (Path(directory) / MANIFEST_FILE).read_text(encoding="utf-8")
    )


def compare_manifests(
    previous: dict[str, str],
    current: dict[str, str],
) -> dict[str, list[str]]:
    previous_files = set(previous)
    current_files = set(current)
    common_files = previous_files & current_files

    return {
        "added": sorted(current_files - previous_files),
        "modified": sorted(
            path for path in common_files if previous[path] != current[path]
        ),
        "deleted": sorted(previous_files - current_files),
        "unchanged": sorted(
            path for path in common_files if previous[path] == current[path]
        ),
    }