import pytest

from aria import cli
from aria.voice import minimal


def test_minimal_cli_is_preview_without_credentials_or_optional_factories(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("minimal preview must not load credentials or services")
    monkeypatch.setattr(cli, "load_local_env", forbidden)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-must-not-enable-models")
    monkeypatch.setenv("ARIA_USER_MEMORY_DIR", str(tmp_path / "private-do-not-touch"))
    seen = {}

    class Server:
        server_port = 8099

        def __init__(self, app):
            self.app = app

        def serve_forever(self):
            seen["mode"] = self.app.mode
            state = self.app.start()["state"]
            self.app.submit("纸风车转起来了", "cli-turn", state["session_id"])

        def server_close(self):
            self.app.close()
            seen["state"] = self.app.state()

    monkeypatch.setattr(minimal, "make_minimal_server", lambda app, *a: Server(app))
    assert cli.web_main(["--minimal-data-dir", str(tmp_path / "demo")]) == 0
    assert seen["mode"] == "local_preview"
    assert seen["state"]["session_id"] is None
    assert seen["state"]["offline"]["pending_count"] == 1
    assert not (tmp_path / "private-do-not-touch").exists()
    assert "fake-key" not in capsys.readouterr().out


@pytest.mark.parametrize("args", [["--enable-model-calls"],
                                 ["--minimal", "--offline", "--enable-model-calls"]])
def test_model_switch_requires_unambiguous_minimal_mode(args, monkeypatch):
    monkeypatch.setattr(cli, "load_local_env", lambda: pytest.fail("must validate before credentials"))
    with pytest.raises(SystemExit) as error:
        cli.web_main(args)
    assert error.value.code == 2


def test_minimal_cli_help_advertises_opt_in_and_data_isolation(capsys):
    with pytest.raises(SystemExit) as result:
        cli.web_main(["--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "--minimal" in output and "--minimal-data-dir" in output
    assert "--enable-model-calls" in output


def test_minimal_cli_reports_close_failure_without_claiming_sealed(tmp_path, monkeypatch, capsys):
    class Server:
        server_port = 8099

        def serve_forever(self):
            return None

        def server_close(self):
            raise RuntimeError("private-storage-path-must-not-be-disclosed")

    monkeypatch.setattr(minimal, "make_minimal_server", lambda *a: Server())
    assert cli.web_main(["--minimal", "--minimal-data-dir", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "封存失败" in output
    assert "已停止；本地封存记录保留" not in output
    assert "private-storage-path" not in output


def test_care_setup_is_explicit_and_never_overwrites_existing_account(tmp_path, monkeypatch, capsys):
    class Server:
        server_port = 8099

        def __init__(self, app):
            self.app = app

        def serve_forever(self):
            assert self.app.care.login("nurse", "local-fixture-password")

        def server_close(self):
            self.app.close()

    monkeypatch.setattr(minimal, "make_minimal_server", lambda app, *a: Server(app))
    monkeypatch.setattr(cli, "getpass", lambda prompt: "local-fixture-password")
    options = ["--minimal", "--minimal-data-dir", str(tmp_path), "--care-setup-admin", "nurse"]
    assert cli.web_main(options) == 0
    assert cli.web_main(options) == 1
    assert "local-fixture-password" not in capsys.readouterr().out


def test_care_setup_requires_matching_passwords(tmp_path, monkeypatch):
    passwords = iter(["local-fixture-password", "different-fixture-password"])
    monkeypatch.setattr(cli, "getpass", lambda prompt: next(passwords))
    monkeypatch.setattr(minimal, "make_minimal_server", lambda *a: pytest.fail("must not start server"))
    assert cli.web_main(["--minimal", "--minimal-data-dir", str(tmp_path), "--care-setup-admin", "nurse"]) == 1
