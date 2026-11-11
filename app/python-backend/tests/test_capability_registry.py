# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Capability announcement registry + entitlement gate (ADR-015).

The point of these tests is the *direction* of the dependency: the platform must
derive task lists, default architectures and lock states from what a component
announces, and must not keep a second copy of the truth.
"""

from pathlib import Path

import pytest

from app.services import entitlements, trainer_profiles
from app.services.images import BUILTIN_SEED, _img_dict
from app.services.jobs import JobService


def _seed(repo_key: str) -> dict:
    # Seeds resolve to flat public refs (ghcr.io/nereidai/<category>-<name>),
    # so accept both "trainer/torch-cpu" and "trainer-torch-cpu".
    needle = repo_key.replace("/", "-")
    for entry in BUILTIN_SEED:
        base = entry["repository"].rsplit(":", 1)[0].rsplit("/", 1)[-1]
        if base == needle:
            return entry
    raise AssertionError(f"no built-in seed for {repo_key}")


# ---------------------------------------------------------------------------
# Reading the announcement from the component
# ---------------------------------------------------------------------------
def test_torch_seed_announces_the_six_vision_tasks():
    spec = _seed("trainer/torch-cpu")
    assert spec["task_types"] == [
        "classification", "multilabel", "detection", "instance_seg", "semantic_seg", "pose",
    ]
    assert spec["profile"]["schema"] == trainer_profiles.PROFILE_SCHEMA
    assert spec["profile"]["devices"] == ["cpu", "cuda", "rocm", "xpu"]


def test_paddle_seed_announces_only_detection():
    spec = _seed("trainer/paddle-cpu")
    # The pre-2026-10 catalog claimed four Paddle tasks while only one was
    # implemented; the announcement is now generated from the component.
    assert spec["task_types"] == ["detection"]
    assert spec["profile"]["tasks"][0]["default_arch"] == "ppyoloe_plus_s"


def test_vendor_seeds_share_the_family_announcement():
    cpu = _seed("trainer/torch-cpu")["profile"]
    cu = _seed("trainer/torch-cu")["profile"]
    assert cpu["tasks"] == cu["tasks"]


def test_entrypoint_hint_comes_from_the_announcement():
    spec = _seed("trainer/torch-cpu")
    assert spec["entrypoint_hint"].startswith("python /app/train.py")
    assert "evaluate" in spec["entrypoint_hint"]


def test_llm_and_prediction_seeds_have_no_trainer_profile():
    # No announcement: the platform must not invent one.
    assert _seed("trainer/llm-cu")["profile"] == {}
    assert _seed("ml-backend")["profile"] == {}


def test_profile_path_never_guesses_outside_the_repo():
    assert trainer_profiles.profile_path("") == ""
    assert trainer_profiles.profile_path("trainer/torch-cpu") == ""  # legacy layout
    assert trainer_profiles.profile_path("builders/trainer-torch/Dockerfile.cu").endswith(
        "app/builders/trainer-torch/trainer-profile.json"
    )


def test_backend_image_packages_every_builtin_profile():
    """Production seeding must see the same announcements as source tests."""
    dockerfile = (Path(trainer_profiles.repo_root()) / "app/python-backend/Dockerfile").read_text()
    contexts = {spec["build_context"] for spec in BUILTIN_SEED if spec.get("profile")}
    required = set()
    for context in contexts:
        directories = [part for part in context.split("/") if not part.startswith("Dockerfile")]
        relative = Path("app", *directories, trainer_profiles.PROFILE_FILENAME).as_posix()
        required.add(f"COPY {relative} /{relative}")
    missing = sorted(copy for copy in required if copy not in dockerfile)
    assert not missing, f"backend image omits built-in profiles: {missing}"


def test_unreadable_announcement_degrades_instead_of_raising(tmp_path, monkeypatch):
    broken = tmp_path / "trainer-profile.json"
    broken.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(trainer_profiles, "profile_path", lambda ctx: str(broken))
    assert trainer_profiles.load_profile("builders/trainer-torch") == {}


def test_normalise_rejects_junk_without_raising():
    assert trainer_profiles.normalise_profile(None) == {}
    assert trainer_profiles.normalise_profile([]) == {}
    normalised = trainer_profiles.normalise_profile({
        "schema": "x", "tasks": [{"id": "Det"}, {"nope": 1}, "junk"],
        "capabilities": [{"id": "a", "tier": "extended"}],
    })
    assert [t["id"] for t in normalised["tasks"]] == ["det"]
    assert normalised["tasks"][0]["tier"] == "core"
    assert normalised["capabilities"][0]["tier"] == "extended"


def test_unknown_task_ids_do_not_become_platform_facts():
    profile = trainer_profiles.normalise_profile({
        "tasks": [{"id": "detection", "architectures": [{"id": "a"}]},
                  {"id": "quantum", "architectures": [{"id": "q"}]}],
    })
    assert trainer_profiles.task_ids(profile) == ["detection"]


# ---------------------------------------------------------------------------
# Derived facts
# ---------------------------------------------------------------------------
def test_default_arch_is_announced_per_task():
    profile = _seed("trainer/torch-cpu")["profile"]
    assert trainer_profiles.default_arch(profile, "detection") == "fasterrcnn_mobilenet_v3_large_fpn"
    assert trainer_profiles.default_arch(profile, "pose") == "keypointrcnn_resnet50_fpn"
    assert trainer_profiles.default_arch(profile) == "resnet50"
    assert trainer_profiles.default_arch({}, "detection") == ""


def test_architectures_are_announced_with_labels_and_licences():
    profile = _seed("trainer/paddle-cpu")["profile"]
    archs = trainer_profiles.architectures(profile, "detection")
    assert {a["id"] for a in archs} >= {"ppyoloe_plus_s", "rtdetr_l"}
    assert all(a.get("license") for a in archs)


def test_tier_and_entitlement_lookup_is_most_specific():
    profile = _seed("trainer/torch-cpu")["profile"]
    assert trainer_profiles.tier_for(profile, task="detection") == "core"
    assert trainer_profiles.tier_for(profile, capability="optimization.tensorrt") == "extended"
    assert trainer_profiles.requirements_for(profile, capability="optimization.tensorrt") == [
        "extended.optimization"
    ]
    assert trainer_profiles.missing_entitlements(
        profile, [], capability="optimization.tensorrt"
    ) == ["extended.optimization"]
    assert trainer_profiles.missing_entitlements(
        profile, ["extended.optimization"], capability="optimization.tensorrt"
    ) == []


def test_extended_capabilities_are_listed_for_sale():
    profile = _seed("trainer/torch-cpu")["profile"]
    extended = trainer_profiles.extended_capabilities(profile)
    assert [e["id"] for e in extended] == ["optimization.tensorrt"]
    assert extended[0]["requires"] == ["extended.optimization"]


def test_profile_summary_is_api_safe():
    summary = trainer_profiles.profile_summary(_seed("trainer/paddle-cpu")["profile"])
    assert summary["name"] == "vision/paddle"
    assert summary["tasks"][0]["architectures"][0]["id"] == "ppyoloe_plus_s"
    assert trainer_profiles.profile_summary({}) == {}


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
class _Row:
    """Minimal stand-in for a ContainerImage row."""

    id = "img-1"
    name = "trainer"
    category = "train"
    repository = "ghcr.io/nereidai/trainer-torch-cpu"
    tag = "0.0.1"
    description = ""
    build_context = "builders/trainer-torch"
    entrypoint_hint = ""
    task_types = ["detection"]
    profile_at = None
    profile_error = ""
    is_builtin = True
    created_by = "system"
    created_at = None
    updated_at = None

    def __init__(self, profile):
        self.profile = profile


def test_img_dict_exposes_profile_defaults_and_locks():
    profile = _seed("trainer/torch-cpu")["profile"]
    payload = _img_dict(_Row(profile))
    assert payload["default_archs"]["detection"] == "fasterrcnn_mobilenet_v3_large_fpn"
    assert [e["id"] for e in payload["extended"]] == ["optimization.tensorrt"]
    assert payload["profile"]["schema"] == trainer_profiles.PROFILE_SCHEMA


def test_img_dict_survives_an_image_without_an_announcement():
    payload = _img_dict(_Row({}))
    assert payload["task_types"] == ["detection"]
    assert payload["default_archs"] == {}
    assert payload["extended"] == []


# ---------------------------------------------------------------------------
# Entitlement gate
# ---------------------------------------------------------------------------
def test_capability_locked_error_is_explicit():
    error = entitlements.CapabilityLocked(["extended.face"], task="face_recognition")
    assert "extended capability" in str(error)
    assert "extended.face" in str(error)
    assert error.missing == ["extended.face"]


def test_check_passes_for_core_and_for_unknown_images():
    profile = _seed("trainer/torch-cpu")["profile"]
    entitlements.check(profile, [], task="detection")          # core: free
    entitlements.check({}, [], task="face_recognition")        # no announcement: not ours to gate
    entitlements.check(profile, [], capability="export.onnx")  # core capability


def test_check_blocks_extended_without_the_entitlement():
    profile = _seed("trainer/torch-cpu")["profile"]
    with pytest.raises(entitlements.CapabilityLocked):
        entitlements.check(profile, [], capability="optimization.tensorrt")
    entitlements.check(profile, ["extended.optimization"], capability="optimization.tensorrt")


def test_describe_splits_available_and_locked():
    profile = _seed("trainer/torch-cpu")["profile"]
    state = entitlements.describe(profile, [])
    assert [e["id"] for e in state["locked"]] == ["optimization.tensorrt"]
    assert state["locked"][0]["locked"] is True
    unlocked = entitlements.describe(profile, ["extended.optimization"])
    assert unlocked["locked"] == []
    assert [e["id"] for e in unlocked["available"]] == ["optimization.tensorrt"]


def test_granted_reads_the_platform_config_and_tolerates_a_broken_db():
    class _Boom:
        def execute(self, *_args, **_kwargs):
            raise RuntimeError("db down")

    assert entitlements.granted(_Boom()) == []


# ---------------------------------------------------------------------------
# jobs.py consumption
# ---------------------------------------------------------------------------
class _Db:
    """Stub session returning one catalog row."""

    def __init__(self, profile):
        self._profile = profile

    def scalar(self, *_args, **_kwargs):
        return _Row(self._profile) if self._profile is not None else None


def test_job_default_arch_prefers_the_announcement():
    profile = trainer_profiles.normalise_profile(
        {"tasks": [{"id": "detection", "default_arch": "announced-arch",
                    "architectures": [{"id": "announced-arch"}]}]}
    )
    svc = JobService(_Db(profile), None)  # type: ignore[arg-type]
    assert svc._default_arch("detection", "ghcr.io/x/y:v1") == "announced-arch"


def test_job_default_arch_falls_back_when_nothing_is_announced():
    svc = JobService(_Db(None), None)  # type: ignore[arg-type]
    assert svc._default_arch("detection", "ghcr.io/x/y:v1") == JobService.DEFAULT_ARCHS["detection"]
    assert svc._default_arch("unknown-task", "ghcr.io/x/y:v1") == "fasterrcnn_mobilenet_v3_large_fpn"


def test_job_profile_lookup_never_raises():
    class _Boom:
        def scalar(self, *_args, **_kwargs):
            raise RuntimeError("db down")

    svc = JobService(_Boom(), None)  # type: ignore[arg-type]
    assert svc._image_profile("ghcr.io/x/y:v1") == {}


def test_check_capability_is_a_noop_for_unannounced_images(monkeypatch):
    svc = JobService(_Db(None), None)  # type: ignore[arg-type]
    svc.check_capability("default", {}, {})  # must not raise


def test_check_capability_uses_explicit_architecture_with_checkpoint_weights():
    """A checkpoint URI must not hide the selected extended architecture."""
    profile = trainer_profiles.normalise_profile({
        "tasks": [{
            "id": "detection",
            "architectures": [{
                "id": "locked-arch", "tier": "extended",
                "requires": ["extended.vision"],
            }],
        }],
    })
    svc = JobService(_Db(profile), None)  # type: ignore[arg-type]

    with pytest.raises(entitlements.CapabilityLocked):
        svc.check_capability(
            "default",
            {"custom_image": "ghcr.io/x/y:v1"},
            {
                "task_type": "detection",
                "architecture": "locked-arch",
                "base_model": "s3://nereidforge/base-models/cv/checkpoint.pt",
            },
        )


# ---------------------------------------------------------------------------
# GET /capabilities aggregation
# ---------------------------------------------------------------------------
class _ScalarDb:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self, *_args, **_kwargs):
        class _Q:
            @staticmethod
            def all():
                return self._rows
        return _Q()

    def execute(self, *_args, **_kwargs):
        class _R:
            @staticmethod
            def scalar():
                return None
        return _R()


def test_capabilities_aggregates_tasks_devices_and_locks():
    from app.services.images import ImageService

    profile = _seed("trainer/torch-cpu")["profile"]
    svc = ImageService.__new__(ImageService)
    svc.db = _ScalarDb([_Row(profile)])
    out = svc.capabilities("default")
    (image,) = out["images"]
    assert image["devices"] == ["cpu", "cuda", "rocm", "xpu"]
    assert image["announced"] is True
    assert [t["id"] for t in image["tasks"]] == [
        "classification", "multilabel", "detection", "instance_seg", "semantic_seg", "pose",
    ]
    assert all(not t["locked"] for t in image["tasks"])
    trt = next(c for c in image["capabilities"] if c["id"] == "optimization.tensorrt")
    assert trt["locked"] is True
    assert trt["missing"] == ["extended.optimization"]


def test_capabilities_keeps_legacy_tasks_for_unannounced_custom_images():
    """Custom catalog images must remain selectable without a profile."""
    from app.services.images import ImageService

    svc = ImageService.__new__(ImageService)
    svc.db = _ScalarDb([_Row({})])
    (image,) = svc.capabilities("default")["images"]
    assert image["announced"] is False
    assert image["tasks"] == []
    assert image["task_types"] == ["detection"]


def test_capabilities_handles_no_images():
    from app.services.images import ImageService

    svc = ImageService.__new__(ImageService)
    svc.db = _ScalarDb([])
    assert svc.capabilities("default") == {"entitlements": [], "images": []}
