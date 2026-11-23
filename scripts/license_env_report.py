# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Classify installed Python distributions without misreading bundled notices.

The report distinguishes declared package licenses from license texts bundled
inside metadata. Strong-copyleft declarations and standalone license titles
block the gate; every GPL-family match remains visible for policy review.
"""

import importlib.metadata as md
from pathlib import Path, PurePosixPath
import re
import sys


BLOCK_PATTERN = re.compile(
    r"(AGPL|AFFERO|SSPL|SERVER SIDE PUBLIC|COMMONS CLAUSE|BUSL|BUSINESS SOURCE)",
    re.IGNORECASE,
)
BLOCK_TITLE_PATTERN = re.compile(
    r"^\s*(GNU AFFERO GENERAL PUBLIC LICENSE|SERVER SIDE PUBLIC LICENSE|"
    r"COMMONS CLAUSE|BUSINESS SOURCE LICENSE).*$",
    re.IGNORECASE | re.MULTILINE,
)
REVIEW_PATTERN = re.compile(
    r"(\bLGPL\b|\bGPL\b|\bMPL\b|\bEPL\b|MOZILLA PUBLIC|ECLIPSE PUBLIC)",
    re.IGNORECASE,
)
SHORT_LICENSE_LIMIT = 512
NATIVE_LIBRARY_LICENSES = {
    "libgfortran": "GPL-3.0-or-later WITH GCC-exception-3.1",
    "libquadmath": "LGPL-2.1-or-later",
}


def load_denylist(path):
    """load_denylist reads package-name rules and ignores comments or blanks."""
    names = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            name, _, rest = line.partition("|")
            license_id = rest.split("|", 1)[0].strip()
            names[name.strip().lower()] = license_id
    return names


def installed_license_texts(dist):
    """installed_license_texts reads legal files omitted from core metadata."""
    texts = []
    for file in dist.files or []:
        path = PurePosixPath(str(file))
        if not any(part.endswith(".dist-info") for part in path.parts):
            continue
        if not path.name.upper().startswith(("LICENSE", "COPYING")):
            continue
        try:
            texts.append(Path(dist.locate_file(file)).read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            texts.append(f"UNREADABLE LICENSE FILE {path}: {exc}")
    return texts


def bundled_native_license_evidence(dist):
    """bundled_native_license_evidence identifies shipped GCC runtime libraries."""
    evidence = []
    for file in dist.files or []:
        path = PurePosixPath(str(file))
        for prefix, license_id in NATIVE_LIBRARY_LICENSES.items():
            if path.name.startswith(prefix) and ".so" in path.name:
                evidence.append(f"{path} ({license_id})")
    return evidence


def declared_license(dist):
    """declared_license returns full notices and the primary declaration.

    Long License fields often contain complete third-party license texts. They
    remain searchable for review and standalone prohibited titles, but arbitrary
    words inside those texts cannot become the package's declared license.
    """
    meta = dist.metadata
    expression = str(meta.get("License-Expression") or "").strip()
    license_field = str(meta.get("License") or "").strip()
    classifiers = [
        value
        for value in (meta.get_all("Classifier") or [])
        if value.startswith("License ::")
    ]
    full = " | ".join(
        part for part in [expression, license_field, *classifiers, *installed_license_texts(dist)] if part
    )
    primary_parts = [expression, *classifiers]
    if len(license_field) <= SHORT_LICENSE_LIMIT and "\n" not in license_field:
        primary_parts.insert(0, license_field)
    primary = " | ".join(part for part in primary_parts if part)
    return full.strip(), primary.strip()


def match_excerpt(text, match, radius=100):
    """match_excerpt shows the actual evidence instead of a metadata prefix."""
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    excerpt = re.sub(r"\s+", " ", text[start:end]).strip()
    return f"{match.group(0)!r} near {excerpt!r}"


def report_distribution(dist, denied):
    """report_distribution returns blocked, reviewed, and diagnostic text."""
    name = str(dist.metadata.get("Name") or "").strip()
    if not name:
        return None
    version = str(dist.version or "?")
    license_text, primary = declared_license(dist)
    if name.lower() in denied:
        return 1, 0, f"FAIL  {name} {version} -> denylisted ({denied[name.lower()]})"

    block_match = BLOCK_PATTERN.search(primary)
    evidence_text = primary
    if not block_match:
        block_match = BLOCK_TITLE_PATTERN.search(license_text)
        evidence_text = license_text
    if block_match:
        evidence = match_excerpt(evidence_text, block_match)
        return 1, 0, f"FAIL  {name} {version} -> prohibited license {evidence}"

    review_matches = list(REVIEW_PATTERN.finditer(license_text))
    unique = []
    for match in review_matches:
        if match.group(0).lower() not in {item.group(0).lower() for item in unique}:
            unique.append(match)
    review_details = [match_excerpt(license_text, match) for match in unique[:4]]
    native_evidence = bundled_native_license_evidence(dist)
    if native_evidence:
        review_details.append("bundled native libraries: " + ", ".join(native_evidence))
    if "UNREADABLE LICENSE FILE" in license_text:
        review_details.append("one or more installed license files could not be read")
    if review_details:
        evidence = "; ".join(review_details)
        return 0, 1, f"WARN  {name} {version} -> license review required: {evidence}"
    if not license_text:
        return 0, 1, f"WARN  {name} {version} -> no license metadata"
    if not primary:
        return 0, 0, f"INFO  {name} {version} -> installed license file reviewed; declaration absent"
    return 0, 0, None


def main(denylist_path, distributions=None):
    """main reports every distribution and fails only on actual blockers."""
    denied = load_denylist(denylist_path)
    blocked = reviewed = total = 0
    for dist in md.distributions() if distributions is None else distributions:
        result = report_distribution(dist, denied)
        if result is None:
            continue
        total += 1
        dist_blocked, dist_reviewed, diagnostic = result
        blocked += dist_blocked
        reviewed += dist_reviewed
        if diagnostic:
            print(diagnostic)
    print(f"      scanned {total} distributions: {blocked} blocked, {reviewed} need review")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
