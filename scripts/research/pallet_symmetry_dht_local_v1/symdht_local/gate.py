"""Execute and record the pre-training scientific implementation gates."""

from __future__ import annotations

import argparse
import inspect
import subprocess
from pathlib import Path

import torch

from .constants import R0_SHA256
from .geometry import local_wls_fusion
from .hough import Lattice, decode_modes
from .model import LocalLineFusion
from .runner import resolve_device
from .util import immutable_json, read_json, sha256


def fixture():
    points = torch.tensor([[[20.,20.],[80.,20.],[80.,80.],[20.,80.],
                            [30.,30.],[70.,30.],[70.,70.],[30.,70.],[50.,50.]]])
    return {"features": torch.randn(1,192,24,24), "content": torch.ones(1,1,24,24),
            "box": torch.tensor([[0.,0.,100.,100.]]), "image_hw": torch.tensor([[100.,100.]]),
            "dims": torch.tensor([[1.2,.8,.15]]), "base_points": points,
            "point_valid": torch.ones(1,9,dtype=torch.bool), "point_sigma": torch.full((1,9),.5)}


def run(export: Path, integrity: Path, output: Path) -> dict:
    checks, evidence = {}, {}
    audit = read_json(integrity); provenance = read_json(export / "EXPORT_PROVENANCE.json")
    affine = read_json(export / "AFFINE_AUDIT.json")
    repo = Path(__file__).resolve().parents[4]
    branch = subprocess.run(("git","branch","--show-current"),cwd=repo,text=True,
                            stdout=subprocess.PIPE,check=True).stdout.strip()
    origin_ancestor = subprocess.run(("git","merge-base","--is-ancestor","origin/main","HEAD"),
                                     cwd=repo).returncode == 0
    checks["G1"] = branch == "main" and origin_ancestor
    checks["G2"] = provenance["stock_checkpoint_sha256"] == R0_SHA256
    checks["G3"] = affine["raw_input_raw_roundtrip_max_abs_px"] == 0
    checks["G4"] = affine["PASS"] and affine["ramp_max_abs_cells"] < 1e-5 and affine["impulse_peak"] > .95
    checks["G5"] = "same frozen R0 forward" in provenance["feature_population"]
    checks["G6"] = all(record["dims_deployable"] for split in ("train","calibration","synth_val")
                       for record in read_json(export / f"{split}.json")["records"])
    checks["G7"] = audit["PASS"] and audit["c4_status"] == "C4_NOT_EVALUATED"
    lattice = Lattice(); logits = torch.zeros(1,12,lattice.bins); valid = torch.ones_like(logits,dtype=torch.bool)
    uniform = decode_modes(logits, valid, torch.zeros(1,12), lattice, torch.tensor([[0.,0.,100.,100.]]))
    checks["G8"] = torch.equal(uniform["absolute_weight"], torch.zeros_like(uniform["absolute_weight"]))
    obs = fixture(); lines = torch.zeros(1,12,1,3); weights = torch.zeros(1,12,1)
    lines[:,0,0] = torch.tensor([1.,0.,-10.]); weights[:,0,0] = 1
    corrected, delta = local_wls_fusion(obs["base_points"], obs["point_valid"], torch.ones(1,9),
                                        lines, weights, torch.ones(1), torch.tensor([[1000.,1000.]]))
    checks["G9"] = torch.equal(delta[0,:2,1], torch.zeros(2))
    checks["G10"] = torch.equal(corrected[:,8], obs["base_points"][:,8])
    far = lines.clone(); far[:,0,0,2] = -10000
    _, limited = local_wls_fusion(obs["base_points"], obs["point_valid"], torch.full((1,9),100.),
                                  far, weights, torch.ones(1), obs["image_hw"])
    checks["G11"] = float(torch.linalg.vector_norm(limited[0,0])) <= .01 * 100 * 2**.5 + 1e-5
    from . import runner
    score_source = inspect.getsource(runner.score) + inspect.getsource(runner.score_point)
    checks["G12"] = "targets=False" in score_source and "supervision" not in score_source
    gradients, finite_updates = {}, {}
    for arm in ("direct", "hough"):
        torch.manual_seed(9); model = LocalLineFusion(arm=arm); sample = fixture()
        before = [p.detach().clone() for p in model.line_head.parameters()]
        result = model(sample); target = sample["base_points"].clone(); target[:,:8,0] += 2
        loss = torch.nn.functional.smooth_l1_loss(result["points"][:,:8], target[:,:8]); loss.backward()
        grad = sum(float(p.grad.abs().sum()) for p in model.line_head.parameters() if p.grad is not None)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); optimizer.step()
        gradients[arm] = grad
        finite_updates[arm] = bool(torch.isfinite(loss) and grad > 0 and
                                   all(torch.isfinite(p).all() for p in model.parameters()) and
                                   any(not torch.equal(a,b) for a,b in zip(before,model.line_head.parameters())))
    checks["G13"] = all(value > 0 for value in gradients.values())
    checks["G14"] = all(finite_updates.values())
    device_source = inspect.getsource(resolve_device)
    checks["G15"] = ("refusing silent CPU fallback" in device_source
                     and "DHTLineHead" in inspect.getsource(LocalLineFusion)
                     and "DirectLineHead" in inspect.getsource(LocalLineFusion))
    cuda_available = torch.cuda.is_available(); cuda_error = None
    if not cuda_available:
        try: resolve_device("cuda:0")
        except RuntimeError as error: cuda_error = str(error)
    result = {
        "schema": "symdht_local_pretraining_gate_v1", "integrity_gate_PASS": all(checks.values()),
        "git": {"branch": branch, "origin_main_is_ancestor": origin_ancestor},
        "checks": checks, "evidence": {"corrected_point_to_line_head_gradient_l1": gradients,
                                         "finite_update": finite_updates,
                                         "affine_audit_sha256": sha256(export / "AFFINE_AUDIT.json"),
                                         "integrity_audit_sha256": sha256(integrity)},
        "cuda_available": cuda_available, "cuda_error": cuda_error,
        "main_2000_step_training_ready": all(checks.values()) and cuda_available,
        "training_blocker": None if cuda_available else "requested cuda:0 unavailable; no CPU fallback",
        "pytest_contract": "tests/test_contracts.py; run separately and bind its result in RUN_STATUS.json",
    }
    immutable_json(output, result); return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export",type=Path,required=True);parser.add_argument("--integrity",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    value=run(args.export.resolve(),args.integrity.resolve(),args.output.resolve())
    print("PASS" if value["integrity_gate_PASS"] else "FAIL",
          "GPU_READY" if value["main_2000_step_training_ready"] else "GPU_BLOCKED")
    if not value["integrity_gate_PASS"]: raise SystemExit(1)


if __name__ == "__main__": main()
