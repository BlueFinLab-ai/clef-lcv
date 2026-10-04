"""Exercise sample publication guards and benchmark label isolation without a GPU."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from benchmark_emails import DEFAULT_DATASET, load_dataset, make_request, summarize


class EmailBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.categories, cls.guide, cls.manifest = load_dataset(DEFAULT_DATASET)

    def test_references_are_not_sent(self):
        row = copy.deepcopy(self.rows[0])
        row["reference_category"] = "PRIVATE_REFERENCE_SENTINEL"
        row["reference_reason"] = "PRIVATE_REASON_SENTINEL"
        row["id"] = "PRIVATE_ID_SENTINEL"
        request = make_request(row, "clef-flash", self.categories, self.guide, True)
        encoded = json.dumps(request)
        self.assertNotIn("SENTINEL", encoded)
        self.assertEqual(request["state"], {"email_to_classify": row["email"]})
        self.assertEqual(request["context"], self.guide)

    def test_publication_guards_reject_accidental_identifiers(self):
        for suffix in (" contact somebody@invalid-mail.invalid", " https://invalid.example/private-link",
                       " Account: 123456789012", " Call 202-555-0199"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                for name in self.manifest["sha256"]:
                    shutil.copyfile(DEFAULT_DATASET / name, folder / name)
                data = json.loads((folder / "dataset.json").read_text())
                data["records"][0]["email"]["body"] += suffix
                (folder / "dataset.json").write_text(json.dumps(data))
                manifest = copy.deepcopy(self.manifest)
                manifest["sha256"]["dataset.json"] = hashlib.sha256((folder / "dataset.json").read_bytes()).hexdigest()
                (folder / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_dataset(folder)

    def test_checksum_rejects_unrecorded_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name in (*self.manifest["sha256"], "manifest.json"):
                shutil.copyfile(DEFAULT_DATASET / name, folder / name)
            with (folder / "category-guide.txt").open("a") as stream:
                stream.write("changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_dataset(folder)

    def test_failed_requests_do_not_inflate_agreement(self):
        rows = [
            {"id": "a", "reference_category": "marketing", "choice": "marketing", "usage": {}, "client_wall_ms": 10},
            {"id": "b", "reference_category": "finance", "choice": "marketing", "usage": {}, "client_wall_ms": 20},
            {"id": "c", "reference_category": "finance", "error": "HTTP 503", "client_wall_ms": 30},
        ]
        summary = summarize(rows, 1.0, self.categories)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["reference_agreement_percent"], 50)
        self.assertEqual(summary["emails_per_second"], 2)
        self.assertEqual(summary["confusion_matrix"]["finance"]["marketing"], 1)


if __name__ == "__main__":
    unittest.main()
