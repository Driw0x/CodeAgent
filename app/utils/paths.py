# Gestion des chemins
import hashlib
from pathlib import Path


def project_memory_dir(
    project_path: str | Path,
    memory_root: str | Path,
) -> Path:
    project_path = Path(project_path).resolve()

    project_id = hashlib.sha256(
        str(project_path).encode("utf-8")
    ).hexdigest()[:8]

    return Path(memory_root) / f"{project_path.name}-{project_id}"