"""PHASE 5 -- pose invariance, measured rather than argued.

The detached gate is supposed to leave the pose network untouched.  That is a
claim about parameters and outputs, so it is checked on both, not inferred from
the fact that `torch.no_grad` appears in the source.

The output half of that check runs on **CPU**.  The first version of this file
compared two GPU forwards and reported 1.853e-03 for seed1 with a parameter
diff of exactly zero.  That difference is not the network changing: allocating
a dummy GPU block between two forwards -- no model, no presence code, no seed
touched -- reproduces 1.853e-03 exactly, and a larger block moves it to
2.139e-03.  cuDNN picks its convolution algorithm from the workspace that
happens to be free, and `benchmark=False` / `deterministic=True` do not pin
that choice.  So the old check measured allocator state, not weight state.
CPU convolution is bitwise reproducible, so on CPU a non-zero diff really does
mean the network moved.  The GPU figure is still recorded, as a noise floor,
under GPU_NOISE_FLOOR -- it is evidence about the runtime, not a gate.

The model is moved to CPU and kept as **one object** throughout: a copy would
hide exactly the mutation this phase exists to catch.
"""
from __future__ import annotations

import json, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_curriculum as CU  # noqa: E402
import mh_data as MD        # noqa: E402
import mh_presence as PR    # noqa: E402
import mh_poseaware as PA   # noqa: E402
import mh_screen as MS      # noqa: E402

OUT = MD.OUT
N_FRAMES = 8          # CPU VGG forwards; 8 is enough to catch a weight change
FLOOR_FRAMES = 64     # the batch the original 1.853e-03 was observed on


def _outputs(model, images, features):
    with torch.no_grad():
        corner = CU.corner_forward(model, images)[-1].clone()
        line = CU.line_forward(model, images, features).clone()
    return corner, line


def gpu_noise_floor(model_seed, stems, features):
    """Control: the same forward, twice, with only a dummy allocation between.

    Nothing here touches the model or the presence code, so whatever this
    returns is pure runtime numerics.
    """
    model, _ = PA.build_model(model_seed)
    model.eval()
    pack = CU.load_pack_items([(MD.DATA, s) for s in stems])
    with torch.no_grad():
        first = CU.corner_forward(model, pack["images"])[-1].clone()
    block = torch.empty(int(2**30 // 4), dtype=torch.float32, device=MD.DEV)
    with torch.no_grad():
        second = CU.corner_forward(model, pack["images"])[-1]
    floor = float((first - second).abs().max())
    del block, model, pack, first, second
    torch.cuda.empty_cache()
    return floor


def main():
    MS.deterministic()
    _, _, _, features = MS.lattice()
    rows = MD.load_split()
    stems = sorted(r["stem"] for r in rows if r["split"] == "MH_DEV")[:N_FRAMES]

    floor_stems = sorted(r["stem"] for r in rows
                         if r["split"] == "MH_DEV")[:FLOOR_FRAMES]
    floor = gpu_noise_floor(1, floor_stems, features)
    print(f"  GPU noise floor @{FLOOR_FRAMES} frames (dummy alloc only) "
          f"= {floor:.3e}", flush=True)

    features_cpu = features.cpu()
    report = {"n_frames": len(stems),
              "output_diff_device": "cpu",
              "GPU_NOISE_FLOOR": floor,
              "GPU_NOISE_FLOOR_frames": FLOOR_FRAMES,
              "GPU_NOISE_FLOOR_note":
                  "control only, not a gate: same model, same input, one dummy"
                  " GPU allocation between two forwards. The floor is batch"
                  " dependent -- at 8 frames it is 0.0 because the workspace is"
                  " never tight enough to switch algorithm.",
              "seeds": {}}
    for seed in (1, 2):
        model, _ = PA.build_model(seed)
        model.eval()
        model.to("cpu")                      # same object, not a copy
        pack = CU.load_pack_items([(MD.DATA, s) for s in stems])
        images = pack["images"].cpu()
        snapshot = {k: v.detach().clone()
                    for k, v in model.state_dict().items()}

        corner_a, line_a = _outputs(model, images, features_cpu)

        # the whole presence pipeline: cache read + linear fit + scoring
        cache = np.load(OUT / f"presence_z_cache_seed{seed}.npz",
                        allow_pickle=True)
        layer, _ = PR.fit_linear(cache["pos_train"], cache["neg_train"], seed)
        _ = PR.apply_linear(layer, cache["pos_dev"])

        corner_b, line_b = _outputs(model, images, features_cpu)

        param = max(float((model.state_dict()[k] - v).abs().max())
                    for k, v in snapshot.items())
        corner_out = float((corner_a - corner_b).abs().max())
        line_out = float((line_a - line_b).abs().max())
        grads = sum(1 for p in model.parameters() if p.grad is not None)
        requires = sum(1 for p in model.parameters() if p.requires_grad)
        entry = {"max_abs_param_diff": param,
                 "max_abs_corner_output_diff": corner_out,
                 "max_abs_line_output_diff": line_out,
                 "pose_params_with_grad": grads,
                 "pose_params_requiring_grad": requires,
                 "presence_trainable": int(sum(
                     p.numel() for p in layer.parameters()))}
        entry["POSE_INVARIANT"] = bool(param == 0.0 and corner_out == 0.0
                                       and line_out == 0.0 and grads == 0)
        report["seeds"][f"seed{seed}"] = entry
        print(f"  seed{seed} param {param:.3e}  corner_out {corner_out:.3e}  "
              f"line_out {line_out:.3e}  grads {grads}  "
              f"presence params {entry['presence_trainable']}  "
              f"INVARIANT={entry['POSE_INVARIANT']}", flush=True)
        del model, pack, images, snapshot, corner_a, corner_b, line_a, line_b
    report["POSE_INVARIANT_ALL"] = all(
        v["POSE_INVARIANT"] for v in report["seeds"].values())
    (OUT / "presence_pose_invariance.json").write_text(json.dumps(report, indent=1))
    print(f"POSE_INVARIANT_ALL = {report['POSE_INVARIANT_ALL']}", flush=True)
    if not report["POSE_INVARIANT_ALL"]:
        raise SystemExit("pose invariance broken -- experiment invalid")


if __name__ == "__main__":
    main()
