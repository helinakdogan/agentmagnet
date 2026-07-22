import json

import pytest

from magnet.cli import ConfigReadError, _write_mcp_only


@pytest.mark.parametrize("content", ["{broken", "[]"])
def test_existing_invalid_config_is_preserved(tmp_path, content):
    config_path = tmp_path / "mcp.json"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigReadError, match="Cannot safely update existing config"):
        _write_mcp_only(config_path, "alice")

    assert config_path.read_text(encoding="utf-8") == content


def test_valid_config_preserves_unrelated_settings(tmp_path):
    config_path = tmp_path / "mcp.json"
    original = {
        "theme": "dark",
        "mcpServers": {"other-server": {"command": "other-command"}},
    }
    config_path.write_text(json.dumps(original), encoding="utf-8")

    _write_mcp_only(config_path, "alice")

    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert updated["theme"] == "dark"
    assert updated["mcpServers"]["other-server"] == {
        "command": "other-command"
    }
    assert updated["mcpServers"]["agent-magnet"]["env"] == {
        "MAGNET_USER_ID": "alice"
    }
    assert list(tmp_path.glob(".mcp.json.*.tmp")) == []
