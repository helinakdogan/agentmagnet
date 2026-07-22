import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _project_metadata():
    return tomllib.loads((ROOT / "sdk" / "pyproject.toml").read_text())["project"]


def test_registry_versions_match_python_package():
    project_version = _project_metadata()["version"]
    server = json.loads((ROOT / "server.json").read_text())

    assert server["version"] == project_version
    assert server["packages"][0]["version"] == project_version


def test_http_server_runtime_dependency_is_packaged():
    dependencies = _project_metadata()["dependencies"]
    dependency_names = {
        re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0].strip().lower()
        for dependency in dependencies
    }

    assert "uvicorn" in dependency_names
