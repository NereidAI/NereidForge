# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for installed-distribution license classification."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from infra.scripts import license_env_report


class FakeMetadata(dict):
    """FakeMetadata supplies the get_all API used by importlib.metadata."""

    def get_all(self, key):
        """get_all returns every classifier value for the requested key."""
        value = self.get(key)
        return value if isinstance(value, list) else ([value] if value is not None else [])


class FakeDistribution:
    """FakeDistribution carries only metadata consumed by the report."""

    def __init__(self, name, license_text, classifiers=None, expression="", files=None, root=None):
        """FakeDistribution builds an installed-package metadata substitute."""
        self.metadata = FakeMetadata(
            Name=name,
            License=license_text,
            Classifier=classifiers or [],
            **{"License-Expression": expression},
        )
        self.version = "1.0"
        self.files = files or []
        self.root = root

    def locate_file(self, file):
        """locate_file resolves fixture files like importlib.metadata does."""
        if self.root is None:
            raise OSError(f"no fixture root for {file}")
        return self.root / file


class LicenseEnvironmentReportTest(unittest.TestCase):
    """LicenseEnvironmentReportTest protects block and review distinctions."""

    def run_report(self, distributions):
        """run_report captures diagnostics using an empty package denylist."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as denylist:
            output = io.StringIO()
            with redirect_stdout(output):
                status = license_env_report.main(denylist.name, distributions)
        return status, output.getvalue()

    def test_numpy_bundled_gpl_text_is_review_not_agpl_block(self):
        """test_numpy_bundled_gpl_text_is_review_not_agpl_block covers wheel notices."""
        license_text = """Copyright (c) NumPy Developers. All rights reserved.
Redistribution and use in source and binary forms are permitted.
13. Use with the GNU Affero General Public License.
The libgfortran runtime is GPL-3.0-with-GCC-exception.
The libquadmath runtime is LGPL-2.1-or-later.
"""
        dist = FakeDistribution(
            "numpy",
            license_text,
            ["License :: OSI Approved :: BSD License"],
        )
        status, output = self.run_report([dist])
        self.assertEqual(0, status)
        self.assertNotIn("FAIL  numpy", output)
        self.assertIn("WARN  numpy", output)
        self.assertIn("GPL", output)
        self.assertIn("LGPL", output)
        self.assertIn("1 need review", output)

    def test_numpy_spdx_expression_does_not_hide_bundled_native_libraries(self):
        """test_numpy_spdx_expression_does_not_hide_bundled_native_libraries covers PEP 639 wheels."""
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            legal_path = root_path / "numpy-2.5.4.dist-info/licenses/LICENSE.txt"
            legal_path.parent.mkdir(parents=True)
            legal_path.write_text(
                "GNU GENERAL PUBLIC LICENSE\n"
                "13. Use with the GNU Affero General Public License.\n"
                "Name: libgfortran\nLicense: GPL-3.0-or-later WITH GCC-exception-3.1\n"
                "Name: libquadmath\nLicense: LGPL-2.1-or-later\n",
                encoding="utf-8",
            )
            dist = FakeDistribution(
                "numpy",
                "",
                expression="BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
                files=[
                    "numpy-2.5.4.dist-info/licenses/LICENSE.txt",
                    "numpy.libs/libgfortran-example.so.5.0.0",
                    "numpy.libs/libquadmath-example.so.0.0.0",
                ],
                root=root_path,
            )
            status, output = self.run_report([dist])
        self.assertEqual(0, status)
        self.assertNotIn("FAIL  numpy", output)
        self.assertIn("WARN  numpy", output)
        self.assertIn("libgfortran", output)
        self.assertIn("libquadmath", output)
        self.assertIn("LGPL", output)

    def test_named_dist_info_nested_license_file_is_read(self):
        """test_named_dist_info_nested_license_file_is_read covers PEP 639 paths."""
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            legal_path = root_path / "google_crc32c-1.8.0.dist-info/licenses/LICENSE"
            legal_path.parent.mkdir(parents=True)
            legal_path.write_text("Apache License\nVersion 2.0, January 2004\n", encoding="utf-8")
            dist = FakeDistribution(
                "google-crc32c",
                "",
                files=["google_crc32c-1.8.0.dist-info/licenses/LICENSE"],
                root=root_path,
            )
            status, output = self.run_report([dist])
        self.assertEqual(0, status)
        self.assertIn("INFO  google-crc32c", output)
        self.assertIn("installed license file reviewed", output)
        self.assertNotIn("WARN  google-crc32c", output)
        self.assertIn("0 need review", output)

    def test_agpl_classifier_remains_a_blocker(self):
        """test_agpl_classifier_remains_a_blocker protects declared AGPL detection."""
        dist = FakeDistribution(
            "affero-package",
            "",
            ["License :: OSI Approved :: GNU Affero General Public License v3"],
        )
        status, output = self.run_report([dist])
        self.assertEqual(1, status)
        self.assertIn("FAIL  affero-package", output)
        self.assertIn("Affero", output)

    def test_agpl_license_title_remains_a_blocker_without_classifiers(self):
        """test_agpl_license_title_remains_a_blocker_without_classifiers covers text metadata."""
        dist = FakeDistribution(
            "affero-text-package",
            "GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3, 19 November 2007\n",
        )
        status, output = self.run_report([dist])
        self.assertEqual(1, status)
        self.assertIn("FAIL  affero-text-package", output)
        self.assertIn("GNU AFFERO GENERAL PUBLIC LICENSE", output)


if __name__ == "__main__":
    unittest.main()
