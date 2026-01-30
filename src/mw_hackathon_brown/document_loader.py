from pathlib import Path


def load_markdown_documents(directory: Path) -> dict[str, str]:
    documents: dict[str, str] = {}

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for md_file in sorted(directory.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        documents[md_file.name] = content

    return documents


def get_preseason_documents_path() -> Path:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root / "documents" / "preseason"
