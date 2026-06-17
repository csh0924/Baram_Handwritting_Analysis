from pathlib import Path

def find_project_root(marker_dir: str = "Analysis_Module") -> Path:
    current = Path.cwd()
    while current != current.parent:
        if (current / marker_dir).exists():
            return current
        current = current.parent
    raise FileNotFoundError(f"Could not find project root containing '{marker_dir}'")

def get_analysis_data_root(project_root) -> Path:
    project_root = Path(project_root) 
    return project_root / "Analysis_Data"

def get_handwriting_dir(project_root, handwriting_id: str) -> Path:
    return get_analysis_data_root(project_root) / handwriting_id
