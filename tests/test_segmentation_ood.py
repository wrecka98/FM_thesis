"""Run with: python -m unittest discover -s tests -p 'test_segmentation_ood.py'."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

def notebook_definitions(filename):
    """Test the definitions actually shipped in the Jupyter notebooks."""
    path = Path(__file__).resolve().parents[1] / "pipelines_and_experiments" / filename
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "notebook_test"}
    for cell in notebook["cells"]:
        if "pipeline-definitions" in cell.get("metadata", {}).get("tags", []):
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    return namespace


versa = notebook_definitions("SEG_OOD_VersaMammo.ipynb")
nnunet = notebook_definitions("SEG_OOD_nnUNet.ipynb")
for name in (
    "VersaMammoAdapter", "as_2d", "attach_ood_scores", "binary_metrics",
    "collect_paired_cases", "collect_zgt_cases", "evaluate_cases",
    "extract_reference_features", "load_reference_features", "normalize_scores",
    "read_case", "save_analysis",
):
    globals()[name] = versa[name]
NNUNetAdapter = nnunet["NNUNetAdapter"]


class SegmentationOODTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_case(self, name, value, foreground=True):
        folder = self.root / name
        folder.mkdir(parents=True)
        image = np.full((8, 12), value, dtype=np.uint8)
        mask = np.zeros((8, 12), dtype=np.uint8)
        if foreground:
            mask[2:6, 3:9] = 255
        np.savez(folder / "image.npz", image=image, unrelated=np.zeros(1))
        np.savez(folder / "mask.npz", mask=mask)
        return image, mask

    def test_unique_nested_cases_and_npz_role_keys(self):
        self.make_case("patient_a/slice_1", 10)
        self.make_case("patient_b/slice_1", 20)
        cases = collect_paired_cases(self.root, "mri")
        self.assertEqual(len({c["case_id"] for c in cases}), 2)
        image, mask = read_case(cases[0])
        self.assertEqual(int(image.mean()), 10)
        self.assertEqual(mask.sum(), 24)
        np.save(self.root / "patient_a/slice_1/image.npy", image)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            collect_paired_cases(self.root, "mri")

    def test_prefixed_pairs_in_one_directory_match_by_case(self):
        for case_id, value in [("case_a", 10), ("case_b", 20)]:
            np.savez(self.root / f"{case_id}_image.npz", image=np.full((8, 12), value))
            np.savez(self.root / f"{case_id}_mask.npz", mask=np.full((8, 12), value))
        for definitions in (versa, nnunet):
            cases = definitions["collect_paired_cases"](self.root, "inbreast")
            self.assertEqual(len(cases), 2)
            for case in cases:
                image_stem = Path(case["image_path"]).stem.removesuffix("_image")
                mask_stem = Path(case["mask_path"]).stem.removesuffix("_mask")
                self.assertEqual(image_stem, mask_stem)
        (self.root / "case_a_mask.npz").unlink()
        with self.assertRaisesRegex(ValueError, "matching case_a_mask"):
            collect_paired_cases(self.root, "inbreast")

    def test_prefixed_img_and_ambiguous_masks(self):
        np.savez(self.root / "123_img.npz", image=np.zeros((8, 12)))
        np.savez(self.root / "123_mask.npz", mask=np.zeros((8, 12)))
        self.assertEqual(len(collect_paired_cases(self.root, "inbreast")), 1)
        np.save(self.root / "123_mask.npy", np.zeros((8, 12)))
        with self.assertRaisesRegex(ValueError, "2 masks"):
            collect_paired_cases(self.root, "inbreast")

    def test_zgt_multiple_annotations_union_and_mass_filter(self):
        self.make_case("zgt", 50)
        extra = np.zeros((8, 12), dtype=np.uint8)
        extra[0, 0] = 1
        np.save(self.root / "extra.npy", extra)
        metadata = self.root / "metadata.csv"
        with metadata.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ImagePath", "ROIPath", "AbnormalityType"], delimiter=";")
            writer.writeheader()
            for mask, abnormality in [("zgt/mask.npz", "Mass"), ("extra.npy", "Mass"), ("extra.npy", "Calcification")]:
                writer.writerow({"ImagePath": "zgt/image.npz", "ROIPath": mask, "AbnormalityType": abnormality})
        cases = collect_zgt_cases(self.root, metadata)
        self.assertEqual(len(cases), 1)
        self.assertEqual(read_case(cases[0])[1].sum(), 25)

    def test_mask_metrics_and_empty_cases(self):
        empty = np.zeros((8, 12), dtype=bool)
        positive = empty.copy()
        positive[0, 0] = True
        self.assertEqual(binary_metrics(empty, empty)["dice"], 1)
        self.assertEqual(binary_metrics(empty, positive)["dice"], 0)
        self.assertEqual(binary_metrics(positive, positive)["iou"], 1)
        with self.assertRaises(ValueError):
            binary_metrics(empty, empty.T)

    def test_grayscale_channel_conversion_preserves_intensity_dtype(self):
        image = np.full((8, 12), 100, dtype=np.uint8)
        for array in [image, image[None], image[..., None]]:
            result = as_2d(array)
            np.testing.assert_array_equal(result, image)
            self.assertEqual(result.dtype, np.uint8)
        with self.assertRaises(ValueError):
            as_2d(np.zeros((6, 8, 12)))

    def test_case_feature_metric_alignment_and_resized_gt(self):
        self.make_case("b", 20, foreground=False)
        self.make_case("a", 10)
        cases = collect_paired_cases(self.root, "test")
        def features(image):
            return torch.tensor([[image.mean(), 1]], dtype=torch.float32)
        def predict(image):
            gt = np.zeros((4, 6), dtype=bool)
            if image.mean() == 10:
                gt[1:3, 2:5] = True
            return gt, (4, 6)
        rows, vectors = evaluate_cases(cases, features, predict, self.root / "results", save_masks=True)
        self.assertEqual([r["case_id"] for r in rows], [c["case_id"] for c in cases])
        self.assertEqual(vectors[:, 0].tolist(), [10, 20])
        self.assertEqual([r["dice"] for r in rows], [1, 1])
        self.assertEqual(rows[0]["gt_area"], 6)
        self.assertEqual(rows[0]["original_gt_area"], 24)
        self.assertTrue(all(Path(r["prediction_path"]).is_file() for r in rows))

    def test_reference_signature_and_legacy_validation(self):
        self.make_case("id", 10)
        cases = collect_paired_cases(self.root, "id")
        path = self.root / "features.pt"
        signature = {"encoder": "test", "fold": 0}
        features = extract_reference_features(cases, lambda image: torch.ones(1, 4), path, signature)
        torch.testing.assert_close(load_reference_features(path, signature), features)
        with self.assertRaisesRegex(ValueError, "signature differs"):
            load_reference_features(path, {"encoder": "test", "fold": 1})
        torch.save(features, path)
        with self.assertRaisesRegex(ValueError, "Legacy"):
            load_reference_features(path, signature)

    def test_actual_knn_joint_scaling_and_pairing(self):
        # Near-ID feature should score lower than the orthogonal feature.
        reference = torch.tensor([[1., 0.], [1., 0.1], [1., -0.1]])
        queries = torch.tensor([[0., 1.], [1., 0.]])
        rows = [{"dataset": "mri", "case_id": "far"}, {"dataset": "mammo", "case_id": "near"}]
        scaling = attach_ood_scores(rows, reference, queries, k=1)
        self.assertEqual([r["ood_score"] for r in rows], [1., 0.])
        self.assertGreater(rows[0]["ood_score_raw"], rows[1]["ood_score_raw"])
        self.assertGreater(scaling["max"], scaling["min"])
        with self.assertRaisesRegex(ValueError, "reference count"):
            attach_ood_scores(rows, reference, queries, k=50)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            attach_ood_scores(rows, reference, torch.ones(2, 3), k=1)
        norm, _ = normalize_scores([2, 2])
        np.testing.assert_array_equal(norm, [0, 0])

    def test_analysis_nonconstant_constant_and_two_sample_groups(self):
        rows = [{"dataset": "a", "case_id": str(i), "dice": dice,
                 "ood_score": score, "ood_score_raw": score * 2}
                for i, (score, dice) in enumerate([(1., 0.), (0., 1.)])]
        rows += [{"dataset": "b", "case_id": str(i), "dice": .5,
                  "ood_score": .3, "ood_score_raw": .6} for i in range(3)]
        fig, summary = save_analysis(rows, self.root, "Smoke test", {})
        self.assertAlmostEqual(summary["a"]["pearson_r"], -1.)
        self.assertIsNone(summary["a"]["spearman_p"])
        self.assertIsNone(summary["b"]["pearson_r"])
        json.loads((self.root / "summary.json").read_text())
        with (self.root / "ranked_by_ood.csv").open() as handle:
            ranking = list(csv.DictReader(handle))
        self.assertEqual(ranking[0]["ood_score_raw"], "0.0")
        self.assertTrue((self.root / "dice_vs_ood.png").is_file())
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_versa_thresholds_probabilities_once(self):
        adapter = object.__new__(VersaMammoAdapter)
        adapter.size, adapter.threshold = (8, 12), .5
        adapter.preprocess = lambda image: torch.zeros(1, 3, 8, 12)
        adapter.model = lambda tensor: torch.full((1, 1, 8, 12), .2)
        pred, hw = adapter.predict(np.zeros((8, 12)))
        self.assertFalse(pred.any())  # A second sigmoid would turn .2 into >.5.
        self.assertEqual(hw, (8, 12))

    def test_versa_features_pool_encoder7_to_512_dimensions(self):
        adapter = object.__new__(VersaMammoAdapter)
        adapter.preprocess = lambda image: torch.zeros(1, 3, 8, 12)
        adapter.model = torch.nn.Module()
        for index in range(7):
            setattr(adapter.model, f"encoder{index}", torch.nn.Identity())
        adapter.model.encoder7 = torch.nn.Conv2d(3, 512, kernel_size=1)
        features = adapter.features(np.zeros((8, 12)))
        self.assertEqual(tuple(features.shape), (1, 512))

    def test_dataset_labels_and_explicit_plot_colors_are_preserved(self):
        rows = [{"dataset": dataset, "case_id": dataset, "dice": .5,
                 "ood_score": .3, "ood_score_raw": .6}
                for dataset in ("mamamia", "inbreast")]
        fig, summary = save_analysis(rows, self.root, "Dataset colors", {})
        self.assertEqual(set(summary), {"mamamia", "inbreast"})
        collections = {c.get_label(): c for c in fig.axes[0].collections}
        from matplotlib.colors import to_rgba
        for dataset, color in [("mamamia", "tab:blue"), ("inbreast", "tab:orange")]:
            np.testing.assert_allclose(collections[dataset].get_facecolor()[0], to_rgba(color, alpha=.6))
        with (self.root / "per_image_metrics.csv").open() as handle:
            saved = list(csv.DictReader(handle))
        self.assertEqual({r["dataset"] for r in saved}, {"mamamia", "inbreast"})
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_nnunet_native_prediction_layout_and_label_conversion(self):
        adapter = object.__new__(NNUNetAdapter)
        class Predictor:
            def predict_single_npy_array(self, image, properties, save_or_return_probabilities):
                self.image, self.properties = image, properties
                self.probabilities = save_or_return_probabilities
                output = np.zeros((1, 8, 12), dtype=np.uint8)
                output[0, 2:6, 3:9] = 1
                return output
        adapter.predictor = Predictor()
        pred, hw = adapter.predict(np.zeros((8, 12)))
        self.assertEqual(adapter.predictor.image.shape, (1, 1, 8, 12))
        self.assertEqual(adapter.predictor.properties["spacing"], (999, 1, 1))
        self.assertFalse(adapter.predictor.probabilities)
        self.assertEqual(pred.sum(), 24)
        self.assertEqual(pred.shape, hw)


if __name__ == "__main__":
    unittest.main()
