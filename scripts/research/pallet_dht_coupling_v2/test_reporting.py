"""Generated CPU contract tests; fixtures cannot pass actual21-run visual QA."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import aggregate as A
from . import evaluate as E
from . import report as R
from . import visual_qa as Q


def fake_stores(offset):
    return [{f"frame{i}": dict(session_id=f"session{i//2}", errors=np.array([i+seed+2+offset, i+seed+4+offset], float))
             for i in range(8)} for seed in range(3)]


def fake_summary(family_support=True):
    arms, runs, comparisons = {}, [], {}
    for arm in A.ARMS:
        newer = arm in A.NEW_ARMS
        arms[arm] = dict(metrics={m: dict(mean=(1 if newer else 2) if m in A.S.LOWER else (2 if newer else 1)) for m in A.METRICS})
        for seed in A.SEEDS:
            runs.append(dict(arm=arm, seed=seed, coverage=1., two_d=dict(keypoint_matched_frame_count_iou50=319)))
    for arm in A.NEW_ARMS:
        for reference in A.PRIMARY_REFERENCES:
            comparisons[f"{arm}_vs_{reference}"] = dict(metrics={m: dict(session_cluster=dict(confirmed_benefit=True),
                session_simultaneous=dict(confirmed_benefit=family_support)) for m in A.METRICS})
    return dict(arms=arms, runs=runs, comparisons=comparisons)


def gallery_fixture():
    image = np.full((48, 64, 3), 125, dtype=np.uint8)
    points = [[8, 28], [48, 25], [48, 35], [8, 38], [15, 20], [50, 18], [50, 24], [15, 26], [30, 28]]
    spatial = R.quantize_maps(np.arange(12*34*40, dtype=np.float32).reshape(12, 34, 40))
    hough = R.quantize_maps(np.arange(12*90*113, dtype=np.float32).reshape(12, 90, 113)/1000000)
    evidence = dict(spatial=spatial, hough=hough, affine=[[1, 0, 100], [0, 1, 100]], input_shape_hw=[544, 640],
                    theta_degrees=(np.arange(90)*2).tolist(), rho_values=(np.arange(-56, 57)*.5).tolist(),
                    npz="GENERATED_FIXTURE_ONLY", meaning="Generated fixture, not model output")
    frame = dict(id=R.REQUIRED_CASE, session="GENERATED_FIXTURE_ONLY", domain="TEST", object_type="TEST", width=64,
        height=48, image=R.H.data_uri(image), gt=points, r0=points, r0_error=1., difficulty="easy_le10", runs={})
    for arm in R.ARMS:
        for seed in (1, 2, 3):
            frame["runs"][f"{arm}_seed{seed}"] = dict(points=points, box=[5, 15, 54, 40], score=.9,
                n_candidates=1, error=1., evidence=None if arm=="point_only" else evidence)
    return dict(summary={}, verdict={}, protocol={"stage":"GENERATED_FIXTURE_ONLY"}, runtime={}, progress=[], runs=[],
                frames=[frame], complete=False, gradients=[], views={})


class ReportingContracts(unittest.TestCase):
    def test_canonical_predict_math_unchanged(self):
        old = ast.parse((E.V1 / "evaluate.py").read_text())
        new = ast.parse(Path(E.__file__).read_text())
        def methods(tree):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=="CanonicalPredictor")
            return {n.name:ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        a,b = methods(old),methods(new)
        for name in ("predict", "_capture_hough", "save_evidence"):
            self.assertEqual(a[name],b[name])

    def test_smoke_checkpoint_cannot_enter_real_eval(self):
        with self.assertRaisesRegex(ValueError,"completed main"):
            E.completed_training_binding(Path("unused"), dict(complete=True,stage="smoke"), Path("unused"), {})

    def test_paired_seed_cluster_constant_difference(self):
        row=A.paired_metric(fake_stores(1),fake_stores(0),"keypoint_location_median_px",session_resamples=256,frame_resamples=128)
        self.assertEqual(row["paired_frames"],8)
        self.assertEqual(row["paired_sessions"],4)
        self.assertAlmostEqual(row["difference"],1)
        for field in ("session_cluster","session_simultaneous","frame_level"):
            self.assertAlmostEqual(row[field]["low"],1)
            self.assertAlmostEqual(row[field]["high"],1)
            self.assertFalse(row[field]["confirmed_benefit"])
        self.assertEqual(row["session_simultaneous"]["family_size"],48)

    def test_bonferroni_is_wider_and_no_p_values(self):
        draws=np.linspace(-1,2,100000)
        plain=A.interval(draws,True)
        family=A.interval(draws,True,.05/48)
        self.assertLess(family["low"],plain["low"])
        self.assertGreater(family["high"],plain["high"])
        self.assertNotIn("p_value",family)

    def test_95_ci_does_not_replace_family_or_coverage(self):
        verdict=A.make_verdict(fake_summary(False))
        self.assertFalse(verdict["overall_accuracy_improved"])
        self.assertIsNone(verdict["selected_winning_arm"])
        good=fake_summary(True)
        self.assertTrue(A.make_verdict(good)["overall_accuracy_improved"])
        for row in good["runs"]:
            if row["arm"] in A.NEW_ARMS and row["seed"]==2:
                row["two_d"]["keypoint_matched_frame_count_iou50"]=318
        self.assertFalse(A.make_verdict(good)["overall_accuracy_improved"])

    def test_runtime_failure_is_not_converted_to_pass(self):
        data=gallery_fixture()
        data["runtime"] = dict(complete=True,timing_collection_complete=True,PASS=False,parity_PASS=False,
            status="COMPLETE_WITH_STRICT_PARITY_FAILURE",parity_policy=dict(atol=1e-4,rtol=0,criterion_changed=False),
            strict_failures=[dict(error="synthetic mismatch")],runs=[dict(arm="balanced",seed=1,n=1,
                median_ms=12.,p90_ms=12.,mean_ms=12.,observations=[dict(max_abs_delta_by_field=dict(keypoints_xy=.002))])])
        output=R.sections(data)["runtime"]
        self.assertIn("PASS=false / parity_PASS=false",output)
        self.assertIn("0.002000000px",output)
        self.assertNotIn("Accuracy 예측과의 parity를 검증했습니다.",output)
        data["runtime"]["parity_policy"]["criterion_changed"]=True
        with self.assertRaises(ValueError):R.sections(data)

    def test_fixture_report_never_claims_actual_completion(self):
        with tempfile.TemporaryDirectory(prefix="coupling-report-fixture-") as name:
            root=Path(name);(root/"PURPOSE.md").write_text("Generated fixture only; no experiment output")
            with patch.object(R,"collect",return_value=gallery_fixture()):
                receipt=R.render(root,screenshot=False)
            self.assertFalse(receipt["experiment_complete"])
            self.assertEqual(receipt["n_completed_evaluations"],0)
            document=(root/"index.html").read_text()
            self.assertIn("실험 미완료",document)
            self.assertNotIn("@@",document)
            self.assertIn("99.8958%",document)
            with self.assertRaisesRegex(ValueError,"completed new12"):
                Q.audit_saved_data(root,Q.Inputs())
            # Exercise native JS and palette on generated pixels only.
            browser=Q.Browser.__new__(Q.Browser)
            try:
                browser.__init__(root)
                browser.call("Page.navigate",dict(url=(root/"index.html").as_uri()))
                browser.ready()
                self.assertEqual(browser.js("document.getElementById('arm').value"),"balanced")
                self.assertEqual(browser.js("document.getElementById('role').value"),"7")
                for arm in R.ARMS:
                    browser.select("arm",arm)
                    for seed in (1,2,3):
                        browser.select("seed",seed)
                        self.assertEqual(browser.js("REPORT_CURRENT_FRAME"),R.REQUIRED_CASE)
                browser.select("arm","incidence")
                for role in (0,7,11):browser.select("role",role)
                self.assertFalse(browser.errors)
                self.assertFalse(browser.console_errors)
                self.assertFalse(browser.external_requests)
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
