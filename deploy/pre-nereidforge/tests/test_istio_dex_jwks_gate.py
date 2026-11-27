# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Dex/JWKS bootstrap gates.

The dex-jwt RequestAuthentication must not be applied while Dex cannot serve
OIDC keys: istiod then caches its built-in public-only JWKS (a key without a
kid) and every Dex-issued JWT fails envoy verification until a
backoff-delayed refresh recovers. The two gates in scripts/lib/common.sh
prevent that race and detect the fallback after the rule lands.
"""

import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import textwrap
import unittest

PRE_ROOT = Path(__file__).resolve().parents[1]
COMMON = PRE_ROOT / "scripts" / "lib" / "common.sh"
KUBEFLOW_INSTALLER = PRE_ROOT / "scripts" / "30_install-kubeflow.sh"

FAKE_KUBECTL = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_CALLS"
case "$1" in
  rollout)
    exit "${FAKE_ROLLOUT_RC:-0}"
    ;;
  exec)
    if [[ "$*" == *"/dex/keys"* ]]; then
      [[ "${FAKE_DEX_KEYS_RC:-0}" == 0 ]] || exit 1
      cat "$FAKE_DEX_KEYS_FILE"
      exit 0
    fi
    if [[ "$*" == *"config_dump"* ]]; then
      if [[ "${FAKE_DUMP_ALWAYS_EMPTY:-0}" == 1 ]]; then
        exit 0
      fi
      remaining="$(cat "$FAKE_STATE" 2>/dev/null || echo 0)"
      if [[ "$remaining" =~ ^[0-9]+$ && "$remaining" -gt 0 ]]; then
        echo "$((remaining - 1))" > "$FAKE_STATE"
        exit 0
      fi
      cat "$FAKE_DUMP_FILE"
      exit 0
    fi
    echo "unexpected fake kubectl exec: $*" >&2
    exit 2
    ;;
  *)
    echo "unexpected fake kubectl call: $*" >&2
    exit 2
    ;;
esac
"""

GOOD_DEX_KEYS = {"keys": [{"kty": "RSA", "kid": "3c9c66fb", "n": "t96vA4", "e": "AQAB"}]}
FALLBACK_DEX_KEYS = {"keys": [{"kty": "RSA", "n": "0xObjM0", "e": "AQAB"}]}
EMPTY_DEX_KEYS = {"keys": []}


def gateway_dump(keys_document):
    inline = json.dumps(keys_document)
    return {
        "configs": [
            {
                "dynamic_listeners": [
                    {
                        "active_state": {
                            "listener": {
                                "filter_chains": [
                                    {
                                        "filters": [
                                            {
                                                "typed_config": {
                                                    "http_filters": [
                                                        {
                                                            "typed_config": {
                                                                "providers": {
                                                                    "origins-0": {
                                                                        "local_jwks": {
                                                                            "inline_string": inline
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                ]
            }
        ]
    }


class IstioDexJwksGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.bin = root / "bin"
        self.bin.mkdir()
        (self.bin / "kubectl").write_text(FAKE_KUBECTL, encoding="utf-8")
        (self.bin / "kubectl").chmod(0o755)
        self.calls = root / "calls"
        self.state = root / "state"
        self.state.write_text("0")
        self.dex_keys = root / "dex-keys.json"
        self.gw_dump = root / "gw-dump.json"

    def _env(self, dex_keys, dump, *, rollout_rc=0, dex_keys_rc=0,
             empty_dumps=0, always_empty_dump=False, timeout=2, poll=0):
        self.dex_keys.write_text(json.dumps(dex_keys), encoding="utf-8")
        self.gw_dump.write_text(json.dumps(dump), encoding="utf-8")
        self.state.write_text(str(empty_dumps))
        env = os.environ.copy()
        env.update(
            PATH=f"{self.bin}:{env['PATH']}",
            FAKE_CALLS=str(self.calls),
            FAKE_STATE=str(self.state),
            FAKE_DEX_KEYS_FILE=str(self.dex_keys),
            FAKE_DUMP_FILE=str(self.gw_dump),
            FAKE_ROLLOUT_RC=str(rollout_rc),
            FAKE_DEX_KEYS_RC=str(dex_keys_rc),
            FAKE_DUMP_ALWAYS_EMPTY="1" if always_empty_dump else "0",
            DEX_READY_TIMEOUT=str(timeout),
            DEX_READY_POLL=str(poll),
            ISTIO_JWKS_VERIFY_TIMEOUT=str(timeout),
            ISTIO_JWKS_VERIFY_POLL=str(poll),
        )
        return env

    def run_bash(self, body, env):
        script = textwrap.dedent(
            f"""\
            set -euo pipefail
            source {shlex.quote(str(COMMON))}
            {body}
            """
        )
        return subprocess.run(
            ["bash", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def call_lines(self):
        return self.calls.read_text(encoding="utf-8").splitlines()

    # ---- wait_for_dex_readiness ----

    def test_dex_gate_passes_when_oidc_keys_are_served(self):
        env = self._env(GOOD_DEX_KEYS, gateway_dump(GOOD_DEX_KEYS))
        result = self.run_bash("wait_for_dex_readiness", env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.call_lines()
        self.assertTrue(any(c.startswith("rollout status deployment/dex") for c in calls))
        self.assertTrue(any("/dex/keys" in c for c in calls))

    def test_dex_gate_fails_closed_when_keys_have_no_kid(self):
        env = self._env(FALLBACK_DEX_KEYS, gateway_dump(GOOD_DEX_KEYS))
        result = self.run_bash("wait_for_dex_readiness", env)
        self.assertNotEqual(result.returncode, 0)
        combined = result.stdout + result.stderr
        self.assertIn("did not become OIDC-ready", combined)
        self.assertIn("dex-jwt RequestAuthentication", combined)

    def test_dex_gate_fails_closed_when_rollout_never_completes(self):
        env = self._env(GOOD_DEX_KEYS, gateway_dump(GOOD_DEX_KEYS), rollout_rc=1)
        result = self.run_bash("wait_for_dex_readiness", env)
        self.assertNotEqual(result.returncode, 0)
        # The short-circuit must not probe a Dex pod that is not rolled out.
        self.assertFalse(any("/dex/keys" in c for c in self.call_lines()))

    def test_dex_gate_fails_closed_on_empty_key_set(self):
        env = self._env(EMPTY_DEX_KEYS, gateway_dump(GOOD_DEX_KEYS), dex_keys_rc=1)
        result = self.run_bash("wait_for_dex_readiness", env)
        self.assertNotEqual(result.returncode, 0)

    # ---- verify_istiod_dex_jwks ----

    def test_jwks_verify_passes_with_a_real_key_id(self):
        env = self._env(GOOD_DEX_KEYS, gateway_dump(GOOD_DEX_KEYS))
        result = self.run_bash("verify_istiod_dex_jwks", env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("real Dex JWKS", result.stdout + result.stderr)
        self.assertTrue(any("config_dump" in c for c in self.call_lines()))

    def test_jwks_verify_fails_closed_on_the_public_only_fallback(self):
        env = self._env(GOOD_DEX_KEYS, gateway_dump(FALLBACK_DEX_KEYS))
        result = self.run_bash("verify_istiod_dex_jwks", env)
        self.assertNotEqual(result.returncode, 0)
        combined = result.stdout + result.stderr
        self.assertIn("public-only fallback", combined)
        self.assertIn("restart istiod", combined)

    def test_jwks_verify_rides_out_a_transient_empty_dump(self):
        env = self._env(
            GOOD_DEX_KEYS, gateway_dump(GOOD_DEX_KEYS), empty_dumps=2, timeout=10
        )
        result = self.run_bash("verify_istiod_dex_jwks", env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_jwks_verify_fails_when_gateway_never_answers(self):
        env = self._env(
            GOOD_DEX_KEYS, gateway_dump(GOOD_DEX_KEYS),
            always_empty_dump=True, timeout=1,
        )
        result = self.run_bash("verify_istiod_dex_jwks", env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "did not cache a real Dex JWKS", result.stdout + result.stderr
        )

    # ---- installer wiring ----

    def test_kubeflow_installer_hooks_both_gates_into_the_ordered_loop(self):
        installer = KUBEFLOW_INSTALLER.read_text(encoding="utf-8")
        dex_hook = installer.index('dex)')
        oauth_hook = installer.index('oauth2)')
        self.assertIn("wait_for_dex_readiness", installer[dex_hook:oauth_hook])
        self.assertIn("verify_istiod_dex_jwks", installer[oauth_hook:])
        # A failed target must skip its gate instead of masking the failure.
        self.assertIn("continue", installer)
        common = COMMON.read_text(encoding="utf-8")
        self.assertIn("wait_for_dex_readiness() {", common)
        self.assertIn("verify_istiod_dex_jwks() {", common)


if __name__ == "__main__":
    unittest.main()
