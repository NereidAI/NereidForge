#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the application-owned OCI chart publication helper."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "lib" / "nf_publication.sh"
PUBLISH_DEV = ROOT / ".github" / "workflows" / "publish-dev.yml"


FAKE_TOOL = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys


tool = Path(sys.argv[0]).name
args = sys.argv[1:]
state_root = Path(os.environ["FAKE_REGISTRY_STATE"])
state_file = state_root / "state.json"
log_file = Path(os.environ["FAKE_TOOL_LOG"])


def record():
    with log_file.open("a") as stream:
        stream.write(json.dumps({"tool": tool, "args": args}) + "\n")


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


def load_state():
    return json.loads(state_file.read_text())


def save_state(state):
    state_file.write_text(json.dumps(state, sort_keys=True))


def object_dir(digest):
    return state_root / "objects" / digest.split(":", 1)[1]


def store(files):
    hasher = hashlib.sha256()
    for name, data in sorted(files.items()):
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(data)
    digest = "sha256:" + hasher.hexdigest()
    destination = object_dir(digest)
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (destination / name).write_bytes(data)
    return digest


def object_files(digest):
    directory = object_dir(digest)
    if not directory.is_dir():
        fail("Error: object is missing: " + digest)
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def resolve(state, reference):
    forced_error = os.environ.get("FAKE_RESOLVE_ERROR")
    if forced_error:
        fail(forced_error)
    if "FAKE_RESOLVE_OUTPUT" in os.environ:
        print(os.environ["FAKE_RESOLVE_OUTPUT"])
        return ""
    try:
        return state["tags"][reference]
    except KeyError:
        fail("Error: " + reference + ": not found")


record()
state = load_state()
if tool == "oras":
    command = args[0]
    if command == "resolve":
        digest = resolve(state, args[-1])
        if digest:
            print(digest)
    elif command == "tag":
        source, target = args[-2:]
        if "@" not in source:
            fail("Error: aliases must use an immutable digest")
        repository, digest = source.rsplit("@", 1)
        object_files(digest)
        tag_digest = digest
        if os.environ.get("FAKE_TAG_DIGEST_ONCE") and not state.get("bad_tag_injected"):
            tag_digest = os.environ["FAKE_TAG_DIGEST_ONCE"]
            state["bad_tag_injected"] = True
        state["tags"][repository + ":" + target] = tag_digest
        save_state(state)
    else:
        fail("Error: unsupported fake oras command: " + command)
elif tool == "helm":
    command = args[0]
    if command == "push":
        if os.environ.get("FAKE_PUSH_ERROR"):
            fail(os.environ["FAKE_PUSH_ERROR"])
        package = Path(args[1])
        name = os.environ["FAKE_CHART_NAME"]
        prefix = name + "-"
        if not package.name.startswith(prefix) or package.suffix != ".tgz":
            fail("Error: invalid fake chart package name: " + package.name)
        version = package.name[len(prefix):-len(".tgz")]
        repository = args[2].removeprefix("oci://") + "/" + name
        digest = store({package.name: package.read_bytes()})
        state["tags"][repository + ":" + version] = digest
        save_state(state)
    elif command == "pull":
        if os.environ.get("FAKE_PULL_ERROR"):
            fail(os.environ["FAKE_PULL_ERROR"])
        repository = args[1].removeprefix("oci://")
        requested = args[args.index("--version") + 1]
        destination = Path(args[args.index("--destination") + 1])
        digest = resolve(state, repository + ":" + requested)
        files = object_files(digest)
        if len(files) != 1:
            fail("Error: fake chart object must have one archive")
        destination.mkdir(parents=True, exist_ok=True)
        name = repository.rsplit("/", 1)[1]
        (destination / (name + "-" + requested + ".tgz")).write_bytes(next(iter(files.values())))
    else:
        fail("Error: unsupported fake helm command: " + command)
else:
    fail("Error: unknown fake tool: " + tool)
'''


class HelperOwnershipTests(unittest.TestCase):
    def test_helper_is_self_contained_and_nf_scoped(self):
        helper = HELPER.read_text(encoding="utf-8")
        for function in (
            "pre_oras_resolve",
            "pre_oras_resolve_required",
            "pre_helm_pull_chart",
            "pre_oras_alias_digest",
            "pre_publish_chart",
        ):
            self.assertIn(function + "() {", helper)
        for forbidden in (
            "pre_publish_installer",
            "source ",
            "config.sh",
            "Secret",
            "SECRET",
            "oras push",
            "_pre_oras pull",
            "deploy/pre-nereidforge",
        ):
            self.assertNotIn(forbidden, helper)

    def test_publish_dev_imports_only_the_nf_helper(self):
        workflow = PUBLISH_DEV.read_text(encoding="utf-8")
        self.assertEqual(workflow.count("source scripts/lib/nf_publication.sh"), 2)
        self.assertNotIn("source deploy/pre-nereidforge/scripts/lib/publication.sh", workflow)
        self.assertIn("pre_oras_resolve", workflow)
        self.assertIn("pre_publish_chart", workflow)


class FakeRegistryPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="nf-publication-")
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.state = self.base / "registry"
        self.state.mkdir()
        (self.state / "state.json").write_text('{"tags": {}}', encoding="utf-8")
        self.log = self.base / "commands.jsonl"
        for name in ("helm", "oras"):
            executable = self.bin / name
            executable.write_text(FAKE_TOOL, encoding="utf-8")
            executable.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            FAKE_REGISTRY_STATE=str(self.state),
            FAKE_TOOL_LOG=str(self.log),
            HELM=str(self.bin / "helm"),
            ORAS=str(self.bin / "oras"),
            HELM_REGISTRY_CONFIG=str(self.base / "helm-config.json"),
            ORAS_REGISTRY_CONFIG=str(self.base / "oras-config.json"),
            FAKE_CHART_NAME="demo",
        )
        for variable in (
            "FAKE_RESOLVE_ERROR",
            "FAKE_RESOLVE_OUTPUT",
            "FAKE_TAG_DIGEST_ONCE",
            "FAKE_PUSH_ERROR",
            "FAKE_PULL_ERROR",
        ):
            self.env.pop(variable, None)

    def run_helper(self, function, *args, succeed=True, extra_env=None):
        env = self.env.copy()
        if extra_env:
            env.update(extra_env)
        script = textwrap.dedent(
            """
            set -euo pipefail
            source "$1"
            function_name="$2"
            shift 2
            "$function_name" "$@"
            """
        )
        result = subprocess.run(
            ["bash", "-c", script, "_", str(HELPER), function, *map(str, args)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if succeed:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def commands(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def clear_commands(self):
        self.log.write_text("", encoding="utf-8")

    def tags(self):
        return json.loads((self.state / "state.json").read_text(encoding="utf-8"))["tags"]

    def make_chart(self, directory, version, marker="local chart bytes"):
        package = self.base / directory / f"demo-{version}.tgz"
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_bytes(marker.encode())
        return package

    def test_missing_chart_publishes_then_identical_content_reuses_without_push(self):
        immutable = "0.0.0-dev-abcdef1"
        floating = "0.0.0-dev"
        repository = "registry.invalid/charts/demo"
        package = self.make_chart("charts", immutable)

        self.run_helper("pre_publish_chart", repository, package, immutable, floating)
        first = self.commands()
        pushes = [item for item in first if item["tool"] == "helm" and item["args"][0] == "push"]
        self.assertEqual(len(pushes), 1)
        self.assertIn("oci://registry.invalid/charts", pushes[0]["args"])
        pull_index = next(
            index for index, item in enumerate(first)
            if item["tool"] == "helm" and item["args"][0] == "pull"
        )
        tag_index = next(
            index for index, item in enumerate(first)
            if item["tool"] == "oras" and item["args"][0] == "tag"
        )
        self.assertLess(pull_index, tag_index)
        immutable_digest = self.tags()[repository + ":" + immutable]
        self.assertEqual(self.tags()[repository + ":" + floating], immutable_digest)
        self.assertEqual(first[tag_index]["args"][-2], repository + "@" + immutable_digest)

        self.clear_commands()
        self.run_helper("pre_publish_chart", repository, package, immutable, floating)
        second = self.commands()
        self.assertFalse(any(item["args"][0] == "push" for item in second))
        self.assertTrue(any(item["tool"] == "helm" and item["args"][0] == "pull" for item in second))

    def test_immutable_collision_fails_before_push_or_alias(self):
        immutable = "0.0.0-dev-abcdef1"
        floating = "0.0.0-dev"
        repository = "registry.invalid/charts/demo"
        package = self.make_chart("collision", immutable)
        self.run_helper("pre_publish_chart", repository, package, immutable, floating)
        before = self.tags().copy()
        package.write_bytes(b"different bytes must never replace the immutable chart")
        self.clear_commands()

        result = self.run_helper(
            "pre_publish_chart", repository, package, immutable, floating, succeed=False,
        )

        self.assertIn("Immutable chart collision", result.stderr)
        self.assertFalse(any(item["args"][0] in ("push", "tag") for item in self.commands()))
        self.assertEqual(self.tags(), before)

    def test_non_absence_resolve_errors_fail_before_any_mutation(self):
        immutable = "0.0.0-dev-abcdef1"
        repository = "registry.invalid/charts/demo"
        package = self.make_chart("errors", immutable)
        for error in (
            "Error: registry unavailable",
            "Error: unauthorized",
            "Error: permission denied: not found",
            "Error: network connection failed: not found",
            "Error: x509 certificate failure: not found",
            "bash: oras: command not found",
        ):
            with self.subTest(error=error):
                self.clear_commands()
                self.run_helper(
                    "pre_publish_chart",
                    repository,
                    package,
                    immutable,
                    "0.0.0-dev",
                    succeed=False,
                    extra_env={"FAKE_RESOLVE_ERROR": error},
                )
                commands = self.commands()
                self.assertEqual(len(commands), 1)
                self.assertEqual(commands[0]["args"][0], "resolve")
                self.assertEqual(self.tags(), {})

    def test_only_an_explicit_missing_manifest_is_absence(self):
        for message in ("manifest unknown", "name unknown", "Error: tag not found"):
            with self.subTest(message=message, absent=True):
                self.run_helper("pre_oras_is_not_found_error", message)
        for message in (
            "",
            "unauthorized: not found",
            "authentication failed",
            "connection refused: not found",
            "command not found",
            "no such file or directory",
            "tls timeout: manifest unknown",
        ):
            with self.subTest(message=message, absent=False):
                self.run_helper("pre_oras_is_not_found_error", message, succeed=False)

    def test_resolve_distinguishes_absence_from_other_failures(self):
        missing = self.run_helper(
            "pre_oras_resolve", "registry.invalid/charts/demo:missing", succeed=False,
        )
        self.assertEqual(missing.returncode, 10)
        self.clear_commands()
        failure = self.run_helper(
            "pre_oras_resolve",
            "registry.invalid/charts/demo:missing",
            succeed=False,
            extra_env={"FAKE_RESOLVE_ERROR": "registry unavailable"},
        )
        self.assertEqual(failure.returncode, 1)
        self.assertEqual(len(self.commands()), 1)

    def test_resolve_rejects_empty_and_multiline_digests(self):
        for output in ("", "sha256:first\nsha256:second"):
            with self.subTest(output=output):
                self.clear_commands()
                result = self.run_helper(
                    "pre_oras_resolve",
                    "registry.invalid/charts/demo:0.0.0-dev-abcdef1",
                    succeed=False,
                    extra_env={"FAKE_RESOLVE_OUTPUT": output},
                )
                self.assertIn("Invalid empty or multiline digest", result.stderr)
                self.assertEqual(len(self.commands()), 1)

    def test_missing_package_and_failed_push_or_pull_never_move_an_alias(self):
        immutable = "0.0.0-dev-abcdef1"
        repository = "registry.invalid/charts/demo"
        missing = self.base / "missing" / f"demo-{immutable}.tgz"
        self.run_helper(
            "pre_publish_chart", repository, missing, immutable, "0.0.0-dev", succeed=False,
        )
        self.assertEqual(self.commands(), [])

        package = self.make_chart("failed-operations", immutable)
        for variable in ("FAKE_PUSH_ERROR", "FAKE_PULL_ERROR"):
            with self.subTest(variable=variable):
                self.clear_commands()
                result = self.run_helper(
                    "pre_publish_chart",
                    repository,
                    package,
                    immutable,
                    "0.0.0-dev",
                    succeed=False,
                    extra_env={variable: "injected operation failure"},
                )
                self.assertIn("failed", result.stderr.lower())
                self.assertFalse(any(item["args"][0] == "tag" for item in self.commands()))
                self.assertNotIn(repository + ":0.0.0-dev", self.tags())

    def test_new_alias_resolution_mismatch_restores_immutable_digest(self):
        immutable = "0.0.0-dev-abcdef1"
        floating = "0.0.0-dev"
        repository = "registry.invalid/charts/demo"
        package = self.make_chart("new-alias-mismatch", immutable)

        result = self.run_helper(
            "pre_publish_chart",
            repository,
            package,
            immutable,
            floating,
            succeed=False,
            extra_env={"FAKE_TAG_DIGEST_ONCE": "sha256:wrong"},
        )

        self.assertIn("resolved to sha256:wrong", result.stderr)
        self.assertNotIn("Operator recovery is required", result.stderr)
        self.assertEqual(
            self.tags()[repository + ":" + floating],
            self.tags()[repository + ":" + immutable],
        )
        tag_commands = [
            item for item in self.commands()
            if item["tool"] == "oras" and item["args"][0] == "tag"
        ]
        self.assertEqual(len(tag_commands), 2)
        self.assertEqual(
            tag_commands[-1]["args"][-2],
            repository + "@" + self.tags()[repository + ":" + immutable],
        )

    def test_existing_alias_resolution_mismatch_restores_previous_digest(self):
        previous = "0.0.0-dev-old0001"
        immutable = "0.0.0-dev-new0002"
        floating = "0.0.0-dev"
        repository = "registry.invalid/charts/demo"
        previous_package = self.make_chart("previous-alias", previous, "previous chart bytes")
        package = self.make_chart("existing-alias-mismatch", immutable, "new chart bytes")
        self.run_helper("pre_publish_chart", repository, previous_package, previous, floating)
        previous_digest = self.tags()[repository + ":" + previous]
        self.assertEqual(self.tags()[repository + ":" + floating], previous_digest)
        self.clear_commands()

        result = self.run_helper(
            "pre_publish_chart",
            repository,
            package,
            immutable,
            floating,
            succeed=False,
            extra_env={"FAKE_TAG_DIGEST_ONCE": "sha256:wrong"},
        )

        self.assertIn("resolved to sha256:wrong", result.stderr)
        self.assertNotIn("Operator recovery is required", result.stderr)
        self.assertEqual(self.tags()[repository + ":" + floating], previous_digest)
        self.assertNotEqual(
            self.tags()[repository + ":" + floating],
            self.tags()[repository + ":" + immutable],
        )
        tag_commands = [
            item for item in self.commands()
            if item["tool"] == "oras" and item["args"][0] == "tag"
        ]
        self.assertEqual(len(tag_commands), 2)
        self.assertEqual(tag_commands[-1]["args"][-2], repository + "@" + previous_digest)

    def test_chart_without_floating_argument_never_aliases(self):
        version = "1.2.3"
        repository = "registry.invalid/charts/demo"
        package = self.make_chart("formal", version)
        self.run_helper("pre_publish_chart", repository, package, version)
        self.clear_commands()
        self.run_helper("pre_publish_chart", repository, package, version)
        commands = self.commands()
        self.assertFalse(any(item["args"][0] in ("push", "tag") for item in commands))
        self.assertTrue(any(item["tool"] == "helm" and item["args"][0] == "pull" for item in commands))

    def test_floating_pull_uses_requested_filename_with_immutable_bytes(self):
        immutable = "0.0.0-dev-abcdef1"
        floating = "0.0.0-dev"
        repository = "registry.invalid/charts/demo"
        package = self.make_chart("requested-name", immutable)
        self.run_helper("pre_publish_chart", repository, package, immutable, floating)
        destination = self.base / "anonymous-floating"

        self.run_helper("pre_helm_pull_chart", repository, floating, destination)

        pulled = destination / f"demo-{floating}.tgz"
        self.assertTrue(pulled.is_file())
        self.assertFalse((destination / f"demo-{immutable}.tgz").exists())
        self.assertEqual(pulled.read_bytes(), package.read_bytes())

    def test_oras_uses_only_the_explicit_registry_configuration(self):
        self.run_helper(
            "pre_oras_resolve", "registry.invalid/charts/demo:missing", succeed=False,
        )
        command = self.commands()[0]
        self.assertEqual(command["tool"], "oras")
        self.assertEqual(command["args"][0], "resolve")
        self.assertEqual(
            command["args"][1:3],
            ["--registry-config", self.env["ORAS_REGISTRY_CONFIG"]],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
