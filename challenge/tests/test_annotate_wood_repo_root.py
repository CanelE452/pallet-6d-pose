"""Wood annotator resolves the repository, not its parent directory."""

from pathlib import Path

from scripts.annotate import annotate_wood


def test_annotate_wood_repo_root_contains_required_markers() -> None:
    repo = Path(annotate_wood._REPO)
    assert annotate_wood.find_repo_root(Path(annotate_wood.__file__).parent) == repo
    assert (repo / ".git").exists()
    assert (repo / "challenge").is_dir()
    assert (repo / "scripts").is_dir()
    assert repo.name == "pallet-pose"
