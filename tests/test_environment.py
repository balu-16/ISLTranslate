import importlib.metadata

import pytest

from isl_slt.utils.environment import check_managed_environment


def test_managed_environment_rejects_missing_dependency(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: ["not-installed-package>=1"])
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ValueError, match="Managed environment mismatch"):
        check_managed_environment()
