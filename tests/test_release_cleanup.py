"""Testes da limpeza do clone temporário utilizado na validação final."""

import stat
from pathlib import Path
from unittest.mock import Mock

import pytest
from scripts.validate_clean_clone import (
    _retry_readonly_removal,
    cleanup_temporary_directory,
)


def test_readonly_file_is_made_writable_before_removal(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache"
    cache_file.write_text("cache DVC", encoding="utf-8")
    cache_file.chmod(stat.S_IREAD)
    removal = Mock()

    _retry_readonly_removal(removal, str(cache_file), PermissionError("sem acesso"))

    assert cache_file.stat().st_mode & stat.S_IWRITE
    removal.assert_called_once_with(str(cache_file))


def test_temporary_clone_is_removed(tmp_path: Path) -> None:
    directory = tmp_path / "release"
    directory.mkdir()
    (directory / "model.joblib").write_bytes(b"modelo")

    cleanup_temporary_directory(directory)

    assert not directory.exists()


def test_cleanup_error_does_not_invalidate_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def deny_removal(*args: object, **kwargs: object) -> None:
        raise PermissionError("arquivo protegido")

    monkeypatch.setattr("scripts.validate_clean_clone.shutil.rmtree", deny_removal)

    cleanup_temporary_directory(tmp_path / "release")

    assert "[validacao] AVISO: limpeza incompleta" in capsys.readouterr().err
