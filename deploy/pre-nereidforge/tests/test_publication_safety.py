#!/usr/bin/env python3
# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import textwrap
import unittest


PRE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PRE_ROOT.parents[1]
HELPER = PRE_ROOT / "scripts" / "lib" / "publication.sh"
WORKFLOWS = REPOSITORY / ".github" / "workflows"


FAKE_TOOL = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


tool = Path(sys.argv[0]).name
args = sys.argv[1:]
state_root = Path(os.environ["FAKE_REGISTRY_STATE"])
state_file = state_root / "state.json"
log_file = Path(os.environ["FAKE_TOOL_LOG"])


def load_state():
    return json.loads(state_file.read_text())


def save_state(state):
    state_file.write_text(json.dumps(state, sort_keys=True))


def record():
    with log_file.open("a") as stream:
        stream.write(json.dumps({"tool": tool, "args": args}) + "\n")


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


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
    try:
        return state["tags"][reference]
    except KeyError:
        fail("Error: " + reference + ": not found")


record()
state = load_state()
if tool == "helm":
    command = args[0]
    if command == "push":
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
        repository = args[1].removeprefix("oci://")
        requested = args[args.index("--version") + 1]
        destination = Path(args[args.index("--destination") + 1])
        digest = resolve(state, repository + ":" + requested)
        files = object_files(digest)
        if len(files) != 1:
            fail("Error: fake chart object must have one archive")
        data = next(iter(files.values()))
        destination.mkdir(parents=True, exist_ok=True)
        # Match Helm OCI behavior: the filename uses the requested tag, not the
        # immutable Chart.yaml version embedded in the archive bytes.
        name = repository.rsplit("/", 1)[1]
        (destination / (name + "-" + requested + ".tgz")).write_bytes(data)
    else:
        fail("Error: unsupported fake helm command: " + command)
elif tool == "oras":
    command = args[0]
    if command == "resolve":
        print(resolve(state, args[-1]))
    elif command == "tag":
        source, target = args[-2:]
        if "@" in source:
            repository, digest = source.rsplit("@", 1)
            object_files(digest)
        else:
            repository, source_tag = source.rsplit(":", 1)
            digest = resolve(state, repository + ":" + source_tag)
        state["tags"][repository + ":" + target] = digest
        save_state(state)
    elif command == "pull":
        reference = args[-1]
        repository, tag = reference.rsplit(":", 1)
        digest = resolve(state, repository + ":" + tag)
        output = Path(args[args.index("--output") + 1])
        source = object_dir(digest)
        if not source.is_dir():
            fail("Error: object is missing: " + digest)
        # Preserve nested files and symlinks so reuse tests can model malformed
        # or unexpectedly extended remote artifacts instead of flattening them.
        shutil.copytree(source, output, dirs_exist_ok=True, symlinks=True)
    elif command == "push":
        positional = []
        options_with_values = {"--registry-config", "--artifact-type", "--annotation"}
        index = 1
        while index < len(args):
            if args[index] in options_with_values:
                index += 2
            elif args[index].startswith("--"):
                index += 1
            else:
                positional.append(args[index])
                index += 1
        reference = positional[0]
        if "," in reference:
            fail("Error: immutable and floating tags must not share one push")
        repository, tag = reference.rsplit(":", 1)
        files = {}
        for specification in positional[1:]:
            filename = specification.rsplit(":", 1)[0]
            path = Path(filename)
            files[path.name] = path.read_bytes()
        digest = store(files)
        state["tags"][reference] = digest
        save_state(state)
        print(digest)
    else:
        fail("Error: unsupported fake oras command: " + command)
else:
    fail("Error: unknown fake tool: " + tool)
'''


FAKE_GH = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys


args = sys.argv[1:]
log_file = Path(os.environ["FAKE_GH_LOG"])
with log_file.open("a") as stream:
    stream.write(json.dumps(args) + "\n")
call_index = len(log_file.read_text().splitlines()) - 1
if os.environ.get("FAKE_GH_API_ERROR"):
    print("fake Actions API failure", file=sys.stderr)
    raise SystemExit(1)
if len(args) != 5 or args[:2] != ["api", "--paginate"] or args[3] != "--jq":
    print("unexpected fake gh arguments", file=sys.stderr)
    raise SystemExit(2)
polls = json.loads(Path(os.environ["FAKE_GH_POLLS"]).read_text())["polls"]
poll = polls[min(call_index, len(polls) - 1)]
for page in poll:
    # Execute the workflow's real jq filter against each fake paginated response.
    result = subprocess.run(
        ["jq", "-cr", args[4]],
        input=json.dumps(page),
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
'''


def workflow_step(text, marker):
    start = text.index(marker)
    end = text.find("\n      - name:", start + len(marker))
    return text[start:] if end == -1 else text[start:end]


def workflow_run_block(text, marker):
    lines = workflow_step(text, marker).splitlines()
    run_start = next(index for index, line in enumerate(lines) if line.strip() == "run: |")
    return textwrap.dedent("\n".join(lines[run_start + 1 :])) + "\n"


@unittest.skipUnless(WORKFLOWS.is_dir(), "requires a combined NereidForge checkout")
class PublicationWorkflowSafetyTests(unittest.TestCase):
    def test_workflows_delegate_immutable_publication_to_safe_helper(self):
        formal = (WORKFLOWS / "pre-publish.yml").read_text()
        development = (WORKFLOWS / "pre-publish-dev.yml").read_text()
        for text, marker in (
            (formal, "- name: Publish only the five pre charts"),
            (development, "- name: Publish immutable pre charts and ORAS floating aliases"),
        ):
            with self.subTest(marker=marker):
                step = workflow_step(text, marker)
                self.assertIn("source deploy/pre-nereidforge/scripts/lib/publication.sh", step)
                self.assertIn("pre_publish_chart", step)
                self.assertNotIn("helm push", step)
                self.assertNotIn("oras tag", step)

        login = workflow_step(formal, "- name: Log in to GHCR with isolated publisher configuration")
        self.assertIn("ORAS_REGISTRY_CONFIG", login)
        self.assertIn("oras login", login)

    def test_formal_publication_uses_per_tag_group_and_oldest_first_api_wait(self):
        formal = (WORKFLOWS / "pre-publish.yml").read_text()
        self.assertIn("group: pre-nereidforge-publish-${{ github.ref }}\n", formal)
        self.assertNotIn("group: pre-nereidforge-publish\n", formal)
        self.assertIn("cancel-in-progress: false", formal)
        self.assertEqual(formal.count("\nconcurrency:"), 1)

        job_header = formal.split("  publish:\n", 1)[1].split("    steps:", 1)[0]
        self.assertIn("actions: read", job_header)
        marker = "- name: Wait for older formal tag runs before publication"
        wait = workflow_step(formal, marker)
        self.assertIn("GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}", wait)
        self.assertIn("gh api --paginate", wait)
        self.assertIn("actions/workflows/pre-publish.yml/runs?event=push&per_page=100", wait)
        self.assertIn('.event == "push"', wait)
        self.assertIn('.status != "completed"', wait)
        self.assertIn("(.id < (env.GITHUB_RUN_ID | tonumber))", wait)
        self.assertIn("while :; do", wait)
        self.assertIn("sleep 15", wait)
        self.assertLess(formal.index(marker), formal.index("- name: Log in to GHCR"))
        self.assertLess(formal.index(marker), formal.index("- name: Publish only the five pre charts"))
        self.assertLess(formal.index(marker), formal.index("- name: Publish the pre release"))

    def run_formal_wait(self, directory, polls, api_error=False):
        root = Path(directory)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(FAKE_GH)
        gh.chmod(0o755)
        sleep = bin_dir / "sleep"
        sleep.write_text('#!/bin/bash\nprintf \'%s\\n\' "$*" >> "$FAKE_SLEEP_LOG"\n')
        sleep.chmod(0o755)
        gh_log = root / "gh.jsonl"
        sleep_log = root / "sleep.log"
        poll_file = root / "polls.json"
        poll_file.write_text(json.dumps({"polls": polls}))
        script = workflow_run_block(
            (WORKFLOWS / "pre-publish.yml").read_text(),
            "- name: Wait for older formal tag runs before publication",
        )
        env = os.environ.copy()
        env.update(
            PATH=str(bin_dir) + os.pathsep + env["PATH"],
            FAKE_GH_LOG=str(gh_log),
            FAKE_GH_POLLS=str(poll_file),
            FAKE_SLEEP_LOG=str(sleep_log),
            GITHUB_EVENT_NAME="push",
            GITHUB_REPOSITORY="NereidAI/NereidForge",
            GITHUB_RUN_ID="300",
            GH_TOKEN="local-test-token-must-not-leak",
        )
        if api_error:
            env["FAKE_GH_API_ERROR"] = "1"
        else:
            env.pop("FAKE_GH_API_ERROR", None)
        result = subprocess.run(
            ["bash", "-c", script],
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        calls = [json.loads(line) for line in gh_log.read_text().splitlines()] if gh_log.exists() else []
        sleeps = sleep_log.read_text().splitlines() if sleep_log.exists() else []
        return result, calls, sleeps

    @unittest.skipUnless(shutil.which("jq"), "requires jq to execute the workflow API filter")
    def test_formal_wait_pages_filters_and_polls_until_no_older_run_is_active(self):
        polls = [
            [
                {
                    "workflow_runs": [
                        {"id": 300, "event": "push", "status": "in_progress", "head_branch": "current"},
                        {"id": 299, "event": "push", "status": "completed", "head_branch": "completed"},
                        {"id": 301, "event": "push", "status": "queued", "head_branch": "newer"},
                    ]
                },
                {
                    "workflow_runs": [
                        {"id": 200, "event": "push", "status": "in_progress", "head_branch": "older-200"},
                        {"id": 150, "event": "push", "status": "waiting", "head_branch": "older-150"},
                        {"id": 100, "event": "workflow_dispatch", "status": "queued", "head_branch": "other-event"},
                    ]
                },
            ],
            [
                {
                    "workflow_runs": [
                        {"id": 200, "event": "push", "status": "completed", "head_branch": "older-200"},
                        {"id": 150, "event": "push", "status": "completed", "head_branch": "older-150"},
                        {"id": 300, "event": "push", "status": "in_progress", "head_branch": "current"},
                    ]
                }
            ],
        ]
        with tempfile.TemporaryDirectory(prefix="pre-formal-queue-") as directory:
            result, calls, sleeps = self.run_formal_wait(directory, polls)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, ["15"])
        self.assertIn("200\tolder-200\tin_progress", result.stdout)
        self.assertIn("150\tolder-150\twaiting", result.stdout)
        for excluded in ("current", "completed\n", "newer", "other-event"):
            self.assertNotIn(excluded, result.stdout)
        self.assertNotIn("local-test-token-must-not-leak", result.stdout + result.stderr)
        for call in calls:
            self.assertEqual(call[:2], ["api", "--paginate"])
            self.assertEqual(
                call[2],
                "repos/NereidAI/NereidForge/actions/workflows/pre-publish.yml/runs?event=push&per_page=100",
            )

    @unittest.skipUnless(shutil.which("jq"), "requires jq to execute the workflow API filter")
    def test_formal_wait_fails_closed_on_api_and_jq_errors(self):
        for mode in ("api", "jq"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="pre-formal-queue-error-") as directory:
                polls = [[{"message": "malformed response"}]] if mode == "jq" else [[{"workflow_runs": []}]]
                result, calls, sleeps = self.run_formal_wait(directory, polls, api_error=mode == "api")
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("refusing to publish", result.stdout + result.stderr)
                self.assertNotIn("local-test-token-must-not-leak", result.stdout + result.stderr)
                self.assertEqual(len(calls), 1)
                self.assertEqual(sleeps, [])

    def test_development_installer_push_and_alias_contract_is_static(self):
        text = (WORKFLOWS / "pre-publish-dev.yml").read_text()
        step = workflow_step(
            text,
            "- name: Publish the pre installer artifact with both development tags",
        )
        self.assertIn("source deploy/pre-nereidforge/scripts/lib/publication.sh", step)
        self.assertIn("pre_publish_installer", step)
        self.assertNotIn("oras push", step)
        self.assertNotIn("${IMMUTABLE},${FLOATING}", text)
        helper = HELPER.read_text()
        self.assertIn('"${repository}:${immutable}" \\', helper)
        self.assertIn('_pre_oras tag "${repository}@${immutable_digest}" "$floating"', helper)
        installer_helper = helper.split("pre_publish_installer() {", 1)[1]
        self.assertLess(
            installer_helper.index("_pre_oras pull --output"),
            installer_helper.index("pre_oras_alias_digest \"$repository\" \"$immutable_digest\" \"$floating\""),
        )

    def test_floating_chart_verification_uses_requested_filename_and_immutable_metadata(self):
        text = (WORKFLOWS / "pre-publish-dev.yml").read_text()
        step = workflow_step(text, "- name: Anonymously verify both chart identities and record digests")
        self.assertIn('floating_package="${floating_dir}/${name}-${FLOATING}.tgz"', step)
        self.assertNotIn('floating_package="${floating_dir}/${name}-${IMMUTABLE}.tgz"', step)
        self.assertIn('cmp "$immutable_package" "$floating_package"', step)
        self.assertIn('floating_metadata="$(helm show chart "$floating_package")"', step)
        self.assertIn('"$floating_metadata" | tr -d \'"\')" = "$IMMUTABLE"', step)


class FakeRegistryPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="pre-publication-safety-")
        self.addCleanup(self.tempdir.cleanup)
        self.base = Path(self.tempdir.name)
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.state = self.base / "registry"
        self.state.mkdir()
        (self.state / "state.json").write_text('{"tags": {}}')
        self.log = self.base / "commands.jsonl"
        for name in ("helm", "oras"):
            executable = self.bin / name
            executable.write_text(FAKE_TOOL)
            executable.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            FAKE_REGISTRY_STATE=str(self.state),
            FAKE_TOOL_LOG=str(self.log),
            HELM=str(self.bin / "helm"),
            ORAS=str(self.bin / "oras"),
            HELM_REGISTRY_CONFIG=str(self.base / "helm-config.json"),
            ORAS_REGISTRY_CONFIG=str(self.base / "oras-config.json"),
        )

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
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if succeed:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def commands(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def clear_commands(self):
        self.log.write_text("")

    def tags(self):
        return json.loads((self.state / "state.json").read_text())["tags"]

    def make_chart(self, path, version, marker=""):
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = f"apiVersion: v2\nname: demo\nversion: {version}\nappVersion: 1.2.3\n# {marker}\n".encode()
        info = tarfile.TarInfo("demo/Chart.yaml")
        info.size = len(metadata)
        info.mtime = 0
        with tarfile.open(path, "w:gz") as bundle:
            bundle.addfile(info, io.BytesIO(metadata))

    def make_installer(self, immutable):
        release = self.base / "release"
        release.mkdir(exist_ok=True)
        archive = f"pre-nereidforge-installer-{immutable}.tar.gz"
        names = (
            archive,
            archive + ".sha256",
            f"pre-nereidforge-installer-{immutable}.lock.json",
            f"pre-nereidforge-charts-{immutable}.txt",
        )
        for index, name in enumerate(names):
            (release / name).write_bytes(f"artifact-{index}-{immutable}\n".encode())
        return release, names

    def test_chart_missing_publishes_then_identical_reuses_and_mismatch_fails(self):
        immutable = "0.0.0-dev-abcdef1"
        floating = "0.0.0-dev"
        repository = "registry.invalid/charts/demo"
        package = self.base / "charts" / f"demo-{immutable}.tgz"
        self.make_chart(package, immutable)
        self.env["FAKE_CHART_NAME"] = "demo"

        self.run_helper("pre_publish_chart", repository, package, immutable, floating)
        first = self.commands()
        self.assertEqual(sum(item["tool"] == "helm" and item["args"][0] == "push" for item in first), 1)
        self.assertLess(
            next(index for index, item in enumerate(first) if item["tool"] == "helm" and item["args"][0] == "pull"),
            next(index for index, item in enumerate(first) if item["tool"] == "oras" and item["args"][0] == "tag"),
        )
        self.assertEqual(self.tags()[repository + ":" + immutable], self.tags()[repository + ":" + floating])

        self.clear_commands()
        self.run_helper("pre_publish_chart", repository, package, immutable, floating)
        second = self.commands()
        self.assertFalse(any(item["args"][0] == "push" for item in second))
        self.assertTrue(any(item["tool"] == "helm" and item["args"][0] == "pull" for item in second))

        before = self.tags().copy()
        package.write_bytes(b"different bytes must never replace the immutable chart")
        self.clear_commands()
        result = self.run_helper(
            "pre_publish_chart", repository, package, immutable, floating, succeed=False
        )
        self.assertIn("Immutable chart collision", result.stderr)
        self.assertFalse(any(item["args"][0] in ("push", "tag") for item in self.commands()))
        self.assertEqual(self.tags(), before)

    def test_formal_chart_reuses_fixed_upstream_version_without_alias_or_push(self):
        version = "1.2.3"
        repository = "registry.invalid/charts/demo"
        package = self.base / "formal" / f"demo-{version}.tgz"
        self.make_chart(package, version, marker="formal")
        self.env["FAKE_CHART_NAME"] = "demo"
        self.run_helper("pre_publish_chart", repository, package, version)
        self.clear_commands()
        self.run_helper("pre_publish_chart", repository, package, version)
        commands = self.commands()
        self.assertFalse(any(item["args"][0] in ("push", "tag") for item in commands))
        self.assertTrue(any(item["tool"] == "helm" and item["args"][0] == "pull" for item in commands))

    def test_non_not_found_resolve_errors_fail_before_chart_or_installer_push(self):
        immutable = "0.0.0-dev-abcdef1"
        repository = "registry.invalid/charts/demo"
        package = self.base / "errors" / f"demo-{immutable}.tgz"
        self.make_chart(package, immutable)
        self.env["FAKE_CHART_NAME"] = "demo"
        release, _ = self.make_installer(immutable)
        for function, args in (
            ("pre_publish_chart", (repository, package, immutable, "0.0.0-dev")),
            (
                "pre_publish_installer",
                (release, immutable, "0.0.0-dev", "registry.invalid/installer", "abcdef1"),
            ),
        ):
            for error in (
                "Error: registry unavailable",
                "Error: unauthorized",
                "Error: permission denied: not found",
                "Error: network connection failed: not found",
                "bash: oras: command not found",
            ):
                with self.subTest(function=function, error=error):
                    self.clear_commands()
                    self.run_helper(
                        function,
                        *args,
                        succeed=False,
                        extra_env={"FAKE_RESOLVE_ERROR": error},
                    )
                    commands = self.commands()
                    self.assertEqual(len(commands), 1)
                    self.assertEqual(commands[0]["args"][0], "resolve")

    def test_installer_reuse_compares_every_artifact_and_never_combines_push_tags(self):
        immutable = "0.0.0-dev-abcdef1"
        floating = "0.0.0-dev"
        repository = "registry.invalid/installer"
        release, artifacts = self.make_installer(immutable)

        self.run_helper("pre_publish_installer", release, immutable, floating, repository, "abcdef1")
        first = self.commands()
        pushes = [item for item in first if item["tool"] == "oras" and item["args"][0] == "push"]
        self.assertEqual(len(pushes), 1)
        self.assertIn(repository + ":" + immutable, pushes[0]["args"])
        self.assertNotIn(repository + ":" + immutable + "," + floating, pushes[0]["args"])
        push_index = first.index(pushes[0])
        pull_index = next(
            index for index, item in enumerate(first) if item["tool"] == "oras" and item["args"][0] == "pull"
        )
        tag_index = next(
            index for index, item in enumerate(first) if item["tool"] == "oras" and item["args"][0] == "tag"
        )
        self.assertLess(push_index, pull_index)
        self.assertLess(pull_index, tag_index)
        digest = self.tags()[repository + ":" + immutable]
        self.assertEqual(first[tag_index]["args"][-2], repository + "@" + digest)

        self.clear_commands()
        self.run_helper("pre_publish_installer", release, immutable, floating, repository, "abcdef1")
        self.assertFalse(any(item["args"][0] == "push" for item in self.commands()))

        for artifact in artifacts:
            with self.subTest(artifact=artifact):
                before = self.tags().copy()
                path = release / artifact
                original = path.read_bytes()
                path.write_bytes(original + b"mismatch\n")
                self.clear_commands()
                result = self.run_helper(
                    "pre_publish_installer",
                    release,
                    immutable,
                    floating,
                    repository,
                    "abcdef1",
                    succeed=False,
                )
                self.assertIn("Immutable installer collision", result.stderr)
                commands = self.commands()
                self.assertFalse(any(item["args"][0] in ("push", "tag") for item in commands))
                self.assertEqual(self.tags(), before)
                registry_copy = self.state / "objects" / digest.split(":", 1)[1] / artifact
                self.assertEqual(registry_copy.read_bytes(), original)
                path.write_bytes(original)

    def test_installer_reuse_rejects_extra_remote_file_without_push_or_alias(self):
        immutable = "0.0.0-dev-extra01"
        floating = "0.0.0-dev"
        repository = "registry.invalid/installer"
        release, _ = self.make_installer(immutable)
        self.run_helper("pre_publish_installer", release, immutable, floating, repository, "abcdef1")
        digest = self.tags()[repository + ":" + immutable]
        remote_object = self.state / "objects" / digest.split(":", 1)[1]
        (remote_object / "unexpected-layer.txt").write_bytes(b"unexpected remote layer\n")
        before = self.tags().copy()
        floating_digest = before[repository + ":" + floating]
        secret = "local-registry-config-secret-must-not-leak"
        self.env["ORAS_REGISTRY_CONFIG"] = str(self.base / secret)
        self.clear_commands()

        result = self.run_helper(
            "pre_publish_installer",
            release,
            immutable,
            floating,
            repository,
            "abcdef1",
            succeed=False,
        )

        self.assertIn("unexpected or nested file", result.stderr)
        self.assertNotIn(secret, result.stdout + result.stderr)
        commands = self.commands()
        self.assertTrue(any(item["tool"] == "oras" and item["args"][0] == "pull" for item in commands))
        self.assertFalse(any(item["args"][0] in ("push", "tag") for item in commands))
        self.assertEqual(self.tags(), before)
        self.assertEqual(self.tags()[repository + ":" + floating], floating_digest)

    def test_installer_reuse_rejects_nested_missing_and_symlink_entries(self):
        repository = "registry.invalid/installer"
        floating = "0.0.0-dev"
        for index, mode in enumerate(("nested", "missing", "symlink")):
            with self.subTest(mode=mode):
                immutable = f"0.0.0-dev-shape{index}"
                release, artifacts = self.make_installer(immutable)
                self.run_helper("pre_publish_installer", release, immutable, floating, repository, "abcdef1")
                digest = self.tags()[repository + ":" + immutable]
                remote_object = self.state / "objects" / digest.split(":", 1)[1]
                if mode == "nested":
                    (remote_object / "nested").mkdir()
                    (remote_object / "nested" / artifacts[0]).write_bytes(b"nested duplicate\n")
                elif mode == "missing":
                    (remote_object / artifacts[0]).unlink()
                else:
                    (remote_object / "unexpected-link").symlink_to(release / artifacts[0])
                before = self.tags().copy()
                self.clear_commands()

                result = self.run_helper(
                    "pre_publish_installer",
                    release,
                    immutable,
                    floating,
                    repository,
                    "abcdef1",
                    succeed=False,
                )

                self.assertIn("expected exactly four files", result.stderr)
                commands = self.commands()
                self.assertFalse(any(item["args"][0] in ("push", "tag") for item in commands))
                self.assertEqual(self.tags(), before)

    def test_floating_helm_pull_uses_requested_tag_filename_but_keeps_immutable_bytes(self):
        immutable = "0.0.0-dev-abcdef1"
        floating = "0.0.0-dev"
        repository = "registry.invalid/charts/demo"
        package = self.base / "floating" / f"demo-{immutable}.tgz"
        self.make_chart(package, immutable, marker="floating-name")
        self.env["FAKE_CHART_NAME"] = "demo"
        self.run_helper("pre_publish_chart", repository, package, immutable, floating)

        destination = self.base / "anonymous-floating"
        self.run_helper("pre_helm_pull_chart", repository, floating, destination)
        pulled = destination / f"demo-{floating}.tgz"
        self.assertTrue(pulled.is_file())
        self.assertFalse((destination / f"demo-{immutable}.tgz").exists())
        self.assertEqual(pulled.read_bytes(), package.read_bytes())
        with tarfile.open(pulled, "r:gz") as bundle:
            metadata = bundle.extractfile("demo/Chart.yaml").read().decode()
        self.assertIn(f"version: {immutable}\n", metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
