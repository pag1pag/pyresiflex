import tempfile
from pathlib import Path

import pytest

from pyresiflex.misc.utils import get_path_to_data


def test_get_path_to_data_returns_data_folder(monkeypatch):
    # Patch get_root to a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        data_dir.mkdir()
        monkeypatch.setattr("pyresiflex.misc.utils.get_root", lambda: root)
        path = get_path_to_data()
        assert path == data_dir.resolve()
        assert path.exists()


def test_get_path_to_data_with_subfolder(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        subfolder = root / "data" / "sub"
        subfolder.mkdir(parents=True)
        monkeypatch.setattr("pyresiflex.misc.utils.get_root", lambda: root)
        path = get_path_to_data("sub")
        assert path == subfolder.resolve()
        assert path.exists()


def test_get_path_to_data_file_not_found(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "data").mkdir()
        monkeypatch.setattr("pyresiflex.misc.utils.get_root", lambda: root)
        with pytest.raises(FileNotFoundError):
            get_path_to_data("nonexistent")


def test_get_path_to_data_force_return(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "data").mkdir()
        monkeypatch.setattr("pyresiflex.misc.utils.get_root", lambda: root)
        path = get_path_to_data("nonexistent", force_return=True)
        assert path == (root / "data" / "nonexistent").resolve()
        # Path does not exist, but should still return
        assert not path.exists()


if __name__ == "__main__":
    pytest.main([__file__])
