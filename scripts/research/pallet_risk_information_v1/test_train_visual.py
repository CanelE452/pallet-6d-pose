"""Constructed gate checks; no real payloads, CUDA or optimizer execution."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import train_visual as V


class VisualLauncherTests(unittest.TestCase):
    def test_explicit_GO_before_any_input_IO(self):
        with patch.object(V,'read',side_effect=AssertionError('input decoded')):
            with self.assertRaises(PermissionError):V.run(SimpleNamespace(training_GO=False))

    def test_fresh_CUBLAS_environment_before_any_input_IO(self):
        with patch.dict(V.os.environ,{},clear=True),patch.object(V,'read',side_effect=AssertionError('input decoded')):
            with self.assertRaises(RuntimeError):V.run(SimpleNamespace(training_GO=True))

    def test_no_eager_Torch_training_or_visual_provider_import(self):
        tree=ast.parse(Path(V.__file__).read_text())
        imports=[node.module for node in tree.body if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any(name and any(word in name for word in('torch','risk_models','train','visual_patches'))for name in imports))

    def gate_fixture(self,root,missed=1):
        for name in('EXPERIMENT_PLAN.json','INPUT_MANIFEST.json','extra.json','normalizer.json'):
            (root/name).write_text('{}\n')
        cache=root/'cache';cache.mkdir();(cache/'CACHE_SEAL.json').write_text('{}\n')
        rep=root/'training/R_ORDER_F1_s20261018/RUN_RECEIPT.json';rep.parent.mkdir(parents=True)
        rep.write_text(json.dumps(dict(complete=True,arm='R_ORDER_F1',seed=20261018,
            best_epoch_diagnostics=dict(validation=dict(counts=dict(missed_good_action=missed))))))
        gate=dict(run_visual=True,plan_sha256=V.sha(root/'EXPERIMENT_PLAN.json'),input_manifest_sha256=V.sha(root/'INPUT_MANIFEST.json'),
            extra_feature_seal_sha256=V.sha(root/'extra.json'),extra_normalizer_sha256=V.sha(root/'normalizer.json'),
            synthetic_cache_seal_sha256=V.sha(cache/'CACHE_SEAL.json'),F1_representative_run_receipt_sha256=V.sha(rep))
        path=root/'V_GATE.json';path.write_text(json.dumps(gate))
        return path,cache

    def test_actual_gate_bindings_then_changed_input_rejected(self):
        with TemporaryDirectory()as directory:
            root=Path(directory);gate,cache=self.gate_fixture(root)
            V.gate_check(root,gate,root/'extra.json',root/'normalizer.json',cache)
            (root/'EXPERIMENT_PLAN.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):V.gate_check(root,gate,root/'extra.json',root/'normalizer.json',cache)

    def test_gate_not_triggered_without_residual_selection_failure(self):
        with TemporaryDirectory()as directory:
            root=Path(directory);gate,cache=self.gate_fixture(root,missed=0)
            with self.assertRaises(ValueError):V.gate_check(root,gate,root/'extra.json',root/'normalizer.json',cache)


if __name__=='__main__':unittest.main()
