"""Auth-handler tests: `hermes auth add|status|logout antigravity-oauth` without touching the host.

agy is replaced by a fake executable script so the real Google session is never read or used.
"""

import io
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import auth  # noqa: E402

SIGNED_IN = "#!/bin/sh\nif [ \"$1\" = models ]; then echo 'Fetching available models...' >&2; " \
            "printf 'gemini-3.8-flash-high\\tGemini\\nclaude-sonnet-4-6\\tClaude\\n'; exit 0; fi\necho OK\n"
SIGNED_OUT = "#!/bin/sh\nif [ \"$1\" = models ]; then echo 'Error: Please sign in to view available models.' >&2; " \
             "exit 1; fi\necho OK\n"


@unittest.skipIf(os.name == "nt", "fake agy is a POSIX shell script")
class AuthHandlerTests(unittest.TestCase):
    def _fake_agy(self, body: str) -> str:
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        p = Path(d) / "agy"
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return str(p)

    def _patch(self, agy: str | None, keychain: bool):
        for target, value in (("auth.resolve_agy_command", agy or "agy"), ("auth.is_authenticated", keychain)):
            p = patch(target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        if agy is None:
            p = patch("auth.shutil.which", return_value=None)
            p.start()
            self.addCleanup(p.stop)

    def test_probe_missing_agy_is_unavailable_with_install_hint(self):
        self._patch(None, keychain=False)
        status = auth.probe()
        self.assertFalse(status["available"])
        self.assertIn("antigravity.google/cli", status["detail"])

    def test_probe_signed_in_is_verified_live(self):
        self._patch(self._fake_agy(SIGNED_IN), keychain=True)
        status = auth.probe()
        self.assertTrue(status["logged_in"])
        self.assertIn("2 models", status["detail"])

    def test_stale_keychain_item_is_not_trusted(self):
        # Keychain item exists but agy rejects the session (revoked/expired): must read logged out.
        self._patch(self._fake_agy(SIGNED_OUT), keychain=True)
        status = auth.probe()
        self.assertFalse(status["logged_in"])
        self.assertIn("hermes auth add antigravity-oauth", status["detail"])
        self.assertEqual(status["login_command"][1:3], ["-p", auth._LOGIN_PROMPT])

    def test_add_when_already_signed_in_does_not_launch_login(self):
        self._patch(self._fake_agy(SIGNED_IN), keychain=True)
        with patch("auth._run_login") as login, redirect_stdout(io.StringIO()) as out:
            self.assertTrue(auth.antigravity_auth_handler("add", SimpleNamespace()))
        login.assert_not_called()
        self.assertIn("already signed in", out.getvalue())

    def test_add_non_tty_refuses_instead_of_hanging(self):
        self._patch(self._fake_agy(SIGNED_OUT), keychain=False)
        with patch("auth.sys.stdin.isatty", return_value=False), self.assertRaises(SystemExit) as cm:
            auth.antigravity_auth_handler("add", SimpleNamespace())
        self.assertIn("interactive terminal", str(cm.exception))

    def test_add_runs_agy_login_then_verifies(self):
        agy = self._fake_agy(SIGNED_OUT)
        self._patch(agy, keychain=False)

        def fake_login(path):
            Path(path).write_text(SIGNED_IN)  # the sign-in "succeeds"
            return 0

        with patch("auth.sys.stdin.isatty", return_value=True), patch("auth._run_login", side_effect=fake_login), \
                redirect_stdout(io.StringIO()) as out:
            auth.antigravity_auth_handler("add", SimpleNamespace())
        self.assertIn("signed in", out.getvalue())
        self.assertIn("-m gemini-3.8-flash", out.getvalue())

    def test_add_failed_login_exits_nonzero(self):
        self._patch(self._fake_agy(SIGNED_OUT), keychain=False)
        with patch("auth.sys.stdin.isatty", return_value=True), patch("auth._run_login", return_value=1), \
                redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
            auth.antigravity_auth_handler("add", SimpleNamespace())
        self.assertIn("did not complete", str(cm.exception))

    def test_status_and_logout_are_owned(self):
        self._patch(self._fake_agy(SIGNED_IN), keychain=True)
        with redirect_stdout(io.StringIO()) as out:
            self.assertTrue(auth.antigravity_auth_handler("status", SimpleNamespace()))
            self.assertTrue(auth.antigravity_auth_handler("logout", SimpleNamespace()))
        self.assertIn("antigravity-oauth: logged in", out.getvalue())
        self.assertIn("/logout", out.getvalue())

    def test_unknown_action_falls_through_to_core(self):
        self.assertFalse(auth.antigravity_auth_handler("rotate", SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
