"""Can a low-rank delta inside late A1 recover the F1 signal without unfreezing it?

Three arms now sit around one axis and the gap on it is specific.

```
F1   full parameter adaptation inside net.vgg[19:27]
     2.070244 / 1.077348, the strongest accuracy so far, D0/D2 gap 42.5%
F2   residual adaptation outside the feature extractor, after F50
     3.0850 / 1.5541, no specialization, 17-21% against a 40% threshold
R1   F2 plus a deeper role decoder
     2.9097 / 1.2309, no specialization, 5.68% / 20.80%
```

The untested cell is adaptation *inside* the late convolutions -- where F1's
effect came from -- but constrained instead of free.  Base weights go back to
frozen and each late convolution gets an additive low-rank branch:

```
y = Conv_base(x) + B(A(x))
A   1x1, in -> rank, no bias, standard init
B   base kernel, rank -> out, no bias, initialised to exactly zero
```

No nonlinearity between A and B, so the branch is a linear low-rank update and
`merge` checks it can be folded into the base kernel.  No extra scalar gate:
B at zero already makes L1 the L0 function exactly, and a gate would add a
second factor.

```
L0   DIRECT_HOUGH_TOKEN_XY_V0 with a fully frozen A1 -- Phase A's F0 exactly
     no post-F50 adapter, no extra role block
L1   L0 plus the low-rank branches, rank 8, fixed
```

Decision at 25,545 on `D2_LINE_DEV512` against F0 at full precision.  F1, F2 and
R1 are context and select nothing.  Scope is fixed in
`ROLE_ENCODER_DEPTH_SCOPE_ADDENDUM.md` before this runs.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, pathlib, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LONG = _load("DH_LONG_D", "scripts/stage0/direct_hough_full_step_extension.py")
DH, CAP, V2, SCALE = LONG.DH, LONG.CAP, LONG.V2, LONG.SCALE
OUT, DEV = LONG.OUT, LONG.DEV

RANK = 8                                   # fixed, not swept
FIRST_LATE_INDEX = 19                      # net.vgg[18] is the last MaxPool
EXPECTED_LATE_CONVS = 4                    # audited, HARD_BLOCK if different
EXPECTED_LATE_BASE_PARAMS = 5014912        # the count F1 unfroze
MARKS = LONG.LONG_MARKS
DECISION_STEP = LONG.DECISION_STEP
PER_ROLE_MARKS = LONG.PER_ROLE_MARKS
DIAGNOSTIC_MARKS = LONG.DIAGNOSTIC_MARKS
PROBE_FRAMES = 32
STEP0_TOLERANCE = 1e-6
MERGE_TOLERANCE = 1e-5
PARITY_STEPS = LONG.PARITY_STEPS
PARITY_FRAMES = LONG.PARITY_FRAMES
DETERMINISTIC_WORKSPACE = LONG.DETERMINISTIC_WORKSPACE
REDUCTION = DH.REDUCTION
PHASE_A_RESULT = "direct_hough_long.json"          # F0, the selection baseline
CONTEXT_RESULTS = {"F1_LATE_A1": ("late_a1_adaptation.json", "histories",
                                  "F1_LATE_A1_TRAINABLE"),
                   "F2_ADAPTER": ("f50_adapter.json", "history", None),
                   "R1_ROLE_DEPTH": ("role_depth.json", "history", None)}
ARMS = ("L0_FROZEN_A1", "L1_LOW_RANK_LATE_A1")
TAG = "l1"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class LowRankDelta(nn.Module):
    """A linear low-rank update to one convolution.  Zero at initialisation."""

    def __init__(self, base, rank=RANK):
        super().__init__()
        self.down = nn.Conv2d(base.in_channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, base.kernel_size,
                            padding=base.padding, bias=False)
        nn.init.zeros_(self.up.weight)
        self.padding = base.padding
        self.rank = rank

    def forward(self, x):
        return self.up(self.down(x))

    def effective_weight(self):
        """The single kernel this branch is equivalent to.

        A 1x1 followed by a k x k with nothing in between is a k x k, so the
        branch could be folded into the frozen kernel at inference time.
        """
        return torch.einsum("orhw,ri->oihw", self.up.weight,
                            self.down.weight[:, :, 0, 0])

    def report(self):
        return {"rank": self.rank,
                "down_params": self.down.weight.numel(),
                "up_params": self.up.weight.numel(),
                "params": sum(p.numel() for p in self.parameters()),
                "delta_norm": float(self.effective_weight().norm())}


class LowRankLateA1(nn.Module):
    """Frozen A1 with an optional low-rank branch on each late convolution.

    The trunk below `FIRST_LATE_INDEX` runs under `no_grad` and is detached: no
    branch lives there, so nothing downstream needs its graph.  From that index
    on the graph is kept, because a delta at one convolution changes the input
    to the next -- the base weights stay frozen but their ops must still pass
    gradient to their inputs.
    """

    def __init__(self, rank=None):
        super().__init__()
        self.inner = V2.load_a1()
        self.vgg = self.inner.model.net.vgg
        self.inner.model.eval()
        for parameter in self.inner.model.parameters():
            parameter.requires_grad_(False)
        self.late_indices = [int(i) for i, m in self.vgg.named_children()
                             if int(i) >= FIRST_LATE_INDEX
                             and isinstance(m, nn.Conv2d)]
        self.deltas = None
        if rank is not None:
            self.deltas = nn.ModuleDict(
                {str(i): LowRankDelta(self.vgg[i], rank)
                 for i in self.late_indices})

    def audit(self):
        rows = []
        for i in self.late_indices:
            conv = self.vgg[i]
            row = {"index": i, "in_channels": conv.in_channels,
                   "out_channels": conv.out_channels,
                   "kernel": list(conv.kernel_size),
                   "padding": list(conv.padding),
                   "bias": conv.bias is not None,
                   "base_params": sum(p.numel() for p in conv.parameters())}
            if self.deltas is not None:
                row["low_rank_params"] = self.deltas[str(i)].report()["params"]
            rows.append(row)
        return rows

    def low_rank_parameters(self):
        return [] if self.deltas is None else list(self.deltas.parameters())

    def forward(self, images):
        with torch.no_grad():
            x = images
            for i in range(FIRST_LATE_INDEX):
                x = self.vgg[i](x)
        x = x.detach()
        for i in range(FIRST_LATE_INDEX, len(self.vgg)):
            module = self.vgg[i]
            if self.deltas is not None and str(i) in self.deltas:
                x = module(x) + self.deltas[str(i)](x)
            else:
                x = module(x)
        if self.deltas is None:
            x = x.detach()
        return x


def trainable_a1_origin(model):
    return sum(p.numel() for p in model.inner.model.parameters()
               if p.requires_grad)


def check_audit(model):
    rows = model.audit()
    if len(rows) != EXPECTED_LATE_CONVS:
        raise RuntimeError(f"HARD_BLOCK late conv count {len(rows)} != "
                           f"{EXPECTED_LATE_CONVS}")
    total = sum(r["base_params"] for r in rows)
    if total != EXPECTED_LATE_BASE_PARAMS:
        raise RuntimeError(f"HARD_BLOCK late base params {total} != "
                           f"{EXPECTED_LATE_BASE_PARAMS}")
    smallest = min(min(r["in_channels"], r["out_channels"]) for r in rows)
    if RANK >= smallest:
        raise RuntimeError(f"HARD_BLOCK rank {RANK} not below the smallest "
                           f"channel dimension {smallest}")
    return rows, total, smallest


def build_pair(rank=RANK):
    """L0 and L1 with bit-identical shared weights.

    `DirectHoughModel` seeds itself, so two constructions agree only if nothing
    consumes RNG in between.  Both decoders are built and copied first; the
    low-rank modules are created last, so their initialisation cannot move
    anything L0 has.
    """
    decoder_l0 = DH.DirectHoughModel().to(DEV)
    decoder_l1 = DH.DirectHoughModel().to(DEV)
    decoder_l1.load_state_dict(decoder_l0.state_dict())
    frozen = LowRankLateA1(None).to(DEV)
    adapted = LowRankLateA1(rank).to(DEV)
    return decoder_l0, frozen, decoder_l1, adapted


def shared_checksum(decoder):
    return sum(float(v.double().abs().sum())
               for _, v in sorted(decoder.state_dict().items()))


def parameter_audit(decoder, adapted):
    rows = adapted.audit()
    low_rank = sum(p.numel() for p in adapted.low_rank_parameters())
    encoder = sum(p.numel() for p in decoder.encoder.parameters())
    head = sum(p.numel() for p in decoder.head.parameters())
    return {"late_convs": rows,
            "late_base_params": sum(r["base_params"] for r in rows),
            "LOW_RANK_FEATURE_PARAMS": low_rank,
            "role_encoder_params": encoder,
            "direct_hough_head_params": head,
            "a1_origin_trainable_params": trainable_a1_origin(adapted),
            "total_trainable": low_rank + encoder + head,
            "f1_full_late_trainable": EXPECTED_LATE_BASE_PARAMS,
            "low_rank_fraction_of_f1": low_rank / EXPECTED_LATE_BASE_PARAMS}


def probe_indices():
    return V2.split_indices()[0][:PROBE_FRAMES]


def run_merge():
    """Is the branch foldable into the frozen kernel?

    Diagnostic about a possible inference-time simplification.  It is asked with
    a non-zero B, because the zero-initialised branch would pass trivially.  The
    modules here are throwaway so the canonical pair is never touched.
    """
    generator = torch.Generator(device="cpu").manual_seed(CAP.SEED)
    reference = LowRankLateA1(RANK)
    worst, rows = 0.0, []
    for index in reference.late_indices:
        branch = reference.deltas[str(index)]
        with torch.no_grad():
            branch.up.weight.copy_(torch.randn(
                branch.up.weight.shape, generator=generator) * 0.05)
        conv = reference.vgg[index]
        x = torch.randn(2, conv.in_channels, 25, 25, generator=generator)
        with torch.no_grad():
            unmerged = branch(x)
            merged = F.conv2d(x, branch.effective_weight(),
                              padding=branch.padding)
        gap = float((unmerged - merged).abs().max())
        scale = float(unmerged.abs().max())
        rows.append({"index": index, "max_abs_delta": gap,
                     "output_scale": scale})
        worst = max(worst, gap)
    del reference
    return {"per_conv": rows, "max_abs_delta": worst,
            "tolerance": MERGE_TOLERANCE,
            "LOW_RANK_CONV_MERGEABLE": bool(worst <= MERGE_TOLERANCE)}


def run_step0(edges):
    """Is L1 the L0 function before any step, and is L0 the frozen A1 path?"""
    decoder_l0, frozen, decoder_l1, adapted = build_pair()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    indices = probe_indices()
    gaps = {"late_f50": 0.0, "descriptor": 0.0, "logits": 0.0, "loss": 0.0,
            "frozen_path_against_a1": 0.0}
    reference_a1 = V2.load_a1()
    with torch.no_grad():
        for start in range(0, len(indices), CAP.BATCH):
            pack = V2.load_pack(indices[start:start + CAP.BATCH])
            theta_c, rho_c, support = DH.batch_rows(pack, edges)
            target = DH.target_distribution(
                theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho,
                valid).reshape(*theta_c.shape, -1)
            left, right = frozen(pack["images"]), adapted(pack["images"])
            gaps["late_f50"] = max(gaps["late_f50"],
                                   float((left - right).abs().max()))
            official, _, _ = reference_a1(pack["images"])
            gaps["frozen_path_against_a1"] = max(
                gaps["frozen_path_against_a1"],
                float((left - official).abs().max()))
            gaps["descriptor"] = max(gaps["descriptor"], float(
                (decoder_l0.descriptors(left)
                 - decoder_l1.descriptors(right)).abs().max()))
            logits_l0 = decoder_l0(left, features)
            logits_l1 = decoder_l1(right, features)
            gaps["logits"] = max(gaps["logits"],
                                 float((logits_l0 - logits_l1).abs().max()))
            gaps["loss"] = max(gaps["loss"], abs(
                float(DH.cross_entropy(logits_l0, target, support, valid))
                - float(DH.cross_entropy(logits_l1, target, support, valid))))
    rows, total, smallest = check_audit(adapted)
    report = {"frames": PROBE_FRAMES, "tolerance": STEP0_TOLERANCE, "gaps": gaps,
              "up_weight_zero": bool(all(
                  float(adapted.deltas[str(i)].up.weight.abs().max()) == 0.0
                  for i in adapted.late_indices)),
              "shared_checksum_equal": bool(
                  shared_checksum(decoder_l0) == shared_checksum(decoder_l1)),
              "late_conv_audit": rows, "late_base_params": total,
              "smallest_channel_dim": smallest,
              "audit": parameter_audit(decoder_l1, adapted)}
    report["LOW_RANK_A1_STEP0_EQUIVALENT"] = bool(
        all(v <= STEP0_TOLERANCE for v in gaps.values())
        and report["up_weight_zero"] and report["shared_checksum_equal"])
    del reference_a1
    return report


def trainable_groups(decoder, backbone):
    groups = [{"params": list(decoder.parameters()), "lr": CAP.LR}]
    low_rank = backbone.low_rank_parameters()
    if low_rank:
        groups.append({"params": low_rank, "lr": CAP.LR})
    return groups


def run_wiring(edges):
    """B before A, base never, decoder always."""
    _, _, decoder, backbone = build_pair()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = torch.optim.AdamW(trainable_groups(decoder, backbone),
                                  lr=CAP.LR, weight_decay=CAP.WD)
    pack = V2.load_pack(V2.split_indices()[0][:CAP.BATCH])
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)

    def step():
        loss = DH.cross_entropy(decoder(backbone(pack["images"]), features),
                                target, support, valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        return loss

    before = {str(i): backbone.deltas[str(i)].up.weight.detach().clone()
              for i in backbone.late_indices}
    step()
    first = {str(i): {"up_grad_norm": float(
        backbone.deltas[str(i)].up.weight.grad.norm()),
        "down_grad_norm": float(backbone.deltas[str(i)].down.weight.grad.norm())}
        for i in backbone.late_indices}
    optimiser.step()
    moved = {str(i): float(
        (backbone.deltas[str(i)].up.weight.detach() - before[str(i)]).norm())
        for i in backbone.late_indices}
    step()
    second = {str(i): {"up_grad_norm": float(
        backbone.deltas[str(i)].up.weight.grad.norm()),
        "down_grad_norm": float(backbone.deltas[str(i)].down.weight.grad.norm())}
        for i in backbone.late_indices}
    base_with_grad = sum(
        1 for p in backbone.inner.model.parameters()
        if p.grad is not None and float(p.grad.abs().sum()) != 0.0)
    report = {"step0": first, "up_moved_after_one_step": moved, "step2": second,
              "base_params_with_grad": base_with_grad,
              "a1_origin_trainable_params": trainable_a1_origin(backbone),
              "role_encoder_grad_norm": float(
                  decoder.encoder.attention.in_proj_weight.grad.norm()),
              "head_grad_norm": float(decoder.head.project.weight.grad.norm())}
    report["LOW_RANK_A1_GRADIENT_WIRING"] = bool(
        all(v["up_grad_norm"] > 0.0 for v in first.values())
        and all(v > 0.0 for v in moved.values())
        and all(v["down_grad_norm"] > 0.0 for v in second.values())
        and base_with_grad == 0
        and report["a1_origin_trainable_params"] == 0
        and report["role_encoder_grad_norm"] > 0.0
        and report["head_grad_norm"] > 0.0)
    return report


def run_memory(edges):
    torch.cuda.reset_peak_memory_stats(DEV)
    _, _, decoder, backbone = build_pair()
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    optimiser = torch.optim.AdamW(trainable_groups(decoder, backbone),
                                  lr=CAP.LR, weight_decay=CAP.WD)
    pack = V2.load_pack(V2.split_indices()[0][:CAP.BATCH])
    theta_c, rho_c, support = DH.batch_rows(pack, edges)
    target = DH.target_distribution(
        theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
    ).reshape(*theta_c.shape, -1)
    loss = DH.cross_entropy(decoder(backbone(pack["images"]), features), target,
                            support, valid)
    optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
    peak = torch.cuda.max_memory_allocated(DEV)
    total = torch.cuda.get_device_properties(DEV).total_memory
    return {"batch": CAP.BATCH, "peak_bytes": int(peak),
            "peak_mib": peak / 2 ** 20, "device_total_mib": total / 2 ** 20,
            "loss": float(loss.detach()),
            "LOW_RANK_A1_BATCH8_MEMORY_OK": bool(peak < total)}


def run_parity(edges):
    """With rank off, is this trainer the locked one?

    F0 is reused from Phase A rather than retrained, so the two arms are only
    comparable if this loop with no low-rank branch is the loop that produced
    it.  Asked under deterministic kernels because the default path is not
    bit-reproducible, as `direct_hough_long_parity.json` records.
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != DETERMINISTIC_WORKSPACE:
        raise RuntimeError("parity needs CUBLAS_WORKSPACE_CONFIG="
                           f"{DETERMINISTIC_WORKSPACE} in the environment")
    pool = V2.split_indices()[0][:PARITY_FRAMES]
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    populations = {"PARITY": pool[:CAP.BATCH]}
    torch.use_deterministic_algorithms(True)
    try:
        a1 = V2.load_a1()
        locked = DH.train_network(pool, (PARITY_STEPS,), edges, a1, populations,
                                  "lowrank_locked")[1]
        control = DH.train_network(pool, (PARITY_STEPS,), edges, a1, populations,
                                   "lowrank_control")[1]
        candidate = train_arm(pool, (PARITY_STEPS,), edges, populations,
                              per_pass, populations["PARITY"], rank=None,
                              tag="lowrank_probe")[1]
    finally:
        torch.use_deterministic_algorithms(False)
    report = {"steps": PARITY_STEPS, "frames": PARITY_FRAMES,
              "deterministic_control": LONG.parameter_distance(locked, control),
              "l0_against_locked": LONG.parameter_distance(locked, candidate)}
    report["DETERMINISTIC_MODE_VERIFIED"] = bool(
        report["deterministic_control"]["max_abs_delta"] == 0.0)
    report["L0_CODE_PATH_PARITY"] = bool(
        report["DETERMINISTIC_MODE_VERIFIED"]
        and report["l0_against_locked"]["max_abs_delta"] == 0.0)
    return report


@torch.no_grad()
def low_rank_use(indices, backbone, reference):
    """Is the branch used, and how far has F50 drifted from the frozen one?"""
    per_conv = {}
    for i in backbone.late_indices:
        branch = backbone.deltas[str(i)]
        base = backbone.vgg[i]
        per_conv[str(i)] = {
            "delta_weight_norm_ratio": float(
                branch.effective_weight().norm() / base.weight.norm()),
            "rank": branch.rank}
    drift, cosine, output_ratio = [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        pack = V2.load_pack(indices[start:start + CAP.BATCH])
        adapted = backbone(pack["images"])
        base = reference(pack["images"])
        delta = adapted - base
        drift.append(float(delta.flatten(1).norm(dim=1).mean()
                           / base.flatten(1).norm(dim=1).mean().clamp_min(1e-12)))
        cosine.append(float(nn.functional.cosine_similarity(
            adapted.flatten(1), base.flatten(1), dim=1).mean()))
        output_ratio.append(drift[-1])
    return {"per_conv": per_conv,
            "f50_relative_l2_drift": float(np.mean(drift)),
            "f50_cosine_against_frozen": float(np.mean(cosine)),
            "delta_output_norm_ratio": float(np.mean(output_ratio))}


@torch.no_grad()
def descriptor_drift(indices, decoder, backbone, reference):
    values = []
    for start in range(0, len(indices), CAP.BATCH):
        pack = V2.load_pack(indices[start:start + CAP.BATCH])
        adapted = decoder.descriptors(backbone(pack["images"]))
        base = decoder.descriptors(reference(pack["images"]))
        values.append(float((adapted - base).flatten(1).norm(dim=1).mean()
                            / base.flatten(1).norm(dim=1).mean().clamp_min(1e-12)))
    return float(np.mean(values))


@torch.no_grad()
def evaluate(indices, decoder, backbone, edges, features, grid_theta, grid_rho,
             valid, per_role=False):
    decoder.eval()
    angle, offset, roles = [], [], []
    for start in range(0, len(indices), CAP.BATCH):
        chunk = indices[start:start + CAP.BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        scores = decoder(backbone(pack["images"]), features)
        for frame in range(scores.shape[0]):
            live = torch.nonzero(support[frame]).flatten()
            if live.numel() == 0:
                continue
            theta_p, rho_p = DH.decode(scores[frame][live], grid_theta, grid_rho,
                                       valid)
            a, o = DH.measure(theta_p, rho_p, theta_c[frame][live],
                              rho_c[frame][live])
            angle.append(a); offset.append(o); roles.append(live.cpu().numpy())
    angle = np.concatenate(angle) if angle else np.zeros(1)
    offset = np.concatenate(offset) if offset else np.zeros(1)
    report = DH.summarise(angle, offset)
    if per_role:
        index = np.concatenate(roles)
        report["per_role"] = {}
        for r in range(DH.ROLES):
            keep = index == r
            if keep.sum():
                report["per_role"][str(r)] = {
                    "n": int(keep.sum()),
                    "angle_median": float(np.median(angle[keep])),
                    "angle_p90": float(np.percentile(angle[keep], 90)),
                    "offset_median": float(np.median(offset[keep])),
                    "offset_p90": float(np.percentile(offset[keep], 90))}
    return report


def train_arm(pool, marks, edges, populations, per_pass, probe, rank=RANK,
              tag=TAG):
    grid_theta, grid_rho, valid = DH.lattice()
    features = DH.hypothesis_features(grid_theta, grid_rho)
    _, frozen, decoder, backbone = build_pair(rank)
    if trainable_a1_origin(backbone) != 0:
        raise RuntimeError("HARD_BLOCK: an A1 parameter is trainable")
    optimiser = torch.optim.AdamW(trainable_groups(decoder, backbone),
                                  lr=CAP.LR, weight_decay=CAP.WD)
    history, losses, done = {}, [], 0
    for chunk, _ in V2.step_schedule(pool, max(marks), CAP.BATCH):
        decoder.train()
        pack = V2.load_pack(chunk)
        theta_c, rho_c, support = DH.batch_rows(pack, edges)
        target = DH.target_distribution(
            theta_c.reshape(-1), rho_c.reshape(-1), grid_theta, grid_rho, valid
        ).reshape(*theta_c.shape, -1)
        loss = DH.cross_entropy(decoder(backbone(pack["images"]), features),
                                target, support, valid)
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
        losses.append(float(loss.detach()))
        done += 1
        if done in marks:
            decoder.eval()
            entry = {"step": done,
                     "train_loss_mean_last250": float(np.mean(losses[-250:])),
                     "train_loss_slope_last250": LONG.slope(losses[-250:]),
                     "train_loss_mean_last_pass": float(np.mean(losses[-per_pass:])),
                     "train_loss_slope_last_pass": LONG.slope(losses[-per_pass:]),
                     "diagnostic_only": done in DIAGNOSTIC_MARKS,
                     "finite": bool(np.isfinite(losses[-1]))}
            if backbone.deltas is not None:
                entry["low_rank_use"] = low_rank_use(probe, backbone, frozen)
                entry["low_rank_use"]["descriptor_relative_drift"] = \
                    descriptor_drift(probe, decoder, backbone, frozen)
            for label, indices in populations.items():
                entry[label] = evaluate(indices, decoder, backbone, edges,
                                        features, grid_theta, grid_rho, valid,
                                        per_role=(label == "D2_LINE_DEV512"
                                                  and done in PER_ROLE_MARKS))
                log(f"  {tag} @{done:6d} {label:<16} angle med "
                    f"{entry[label]['angle_median']:7.4f} p90 "
                    f"{entry[label]['angle_p90']:7.3f} | offset med "
                    f"{entry[label]['offset_median']:7.4f} p90 "
                    f"{entry[label]['offset_p90']:7.3f}")
            if "D0_SEEN512" in entry and "D2_LINE_DEV512" in entry:
                d0, d2 = entry["D0_SEEN512"], entry["D2_LINE_DEV512"]
                entry["generalization"] = {
                    "angle_ratio": d2["angle_median"] / d0["angle_median"],
                    "offset_ratio": d2["offset_median"] / d0["offset_median"]}
            use = entry.get("low_rank_use")
            if use:
                log(f"  {tag} @{done:6d} CE "
                    f"{entry['train_loss_mean_last250']:.6f} slope "
                    f"{entry['train_loss_slope_last_pass']:+.3e} | F50 drift "
                    f"{use['f50_relative_l2_drift']:.5f} cos "
                    f"{use['f50_cosine_against_frozen']:.6f} desc "
                    f"{use['descriptor_relative_drift']:.5f} | dW/W "
                    + " ".join(f"{k}:{v['delta_weight_norm_ratio']:.4f}"
                               for k, v in use["per_conv"].items())
                    + (f" | D2/D0 {entry['generalization']['angle_ratio']:.3f}/"
                       f"{entry['generalization']['offset_ratio']:.3f}"
                       if "generalization" in entry else ""))
            torch.save({"tag": tag, "step": done,
                        "decoder": decoder.state_dict(),
                        "low_rank": (backbone.deltas.state_dict()
                                     if backbone.deltas is not None else None),
                        **CAP.provenance()},
                       CAP.checkpoint_path(f"DH_{tag}", f"step_{done:05d}"))
            history[str(done)] = entry
    return history, decoder, backbone


def baseline_f0():
    return json.loads((OUT / PHASE_A_RESULT).read_text())["history"][str(DECISION_STEP)]


def context_arms():
    out = {}
    for name, (source, key, arm) in CONTEXT_RESULTS.items():
        blob = json.loads((OUT / source).read_text())[key]
        entry = blob[arm][str(DECISION_STEP)] if arm else blob[str(DECISION_STEP)]
        d0, d2 = entry["D0_SEEN512"], entry["D2_LINE_DEV512"]
        out[name] = {k: d2[k] for k in ("angle_median", "offset_median",
                                        "angle_p90", "offset_p90")}
        out[name]["d2_over_d0_angle"] = d2["angle_median"] / d0["angle_median"]
        out[name]["d2_over_d0_offset"] = d2["offset_median"] / d0["offset_median"]
    return out


def judge(history):
    f0_entry = baseline_f0()
    f0 = f0_entry["D2_LINE_DEV512"]
    l1 = history[str(DECISION_STEP)]["D2_LINE_DEV512"]
    keys = ("angle_median", "offset_median", "angle_p90", "offset_p90")
    context = context_arms()
    out = {"decision_step": DECISION_STEP, "population": "D2_LINE_DEV512",
           "F0": {k: f0[k] for k in keys}, "L1": {k: l1[k] for k in keys},
           "vs_F0": {k: 1.0 - l1[k] / f0[k] for k in keys},
           "context_only": context,
           "vs_F1_context_only": {
               k: 1.0 - l1[k] / context["F1_LATE_A1"][k] for k in keys},
           "ABSOLUTE_PASS": bool(l1["PASS"] and l1["SAFETY"]),
           "finite": bool(history[str(DECISION_STEP)]["finite"])}
    out["REDUCTION_40"] = bool(
        out["vs_F0"]["angle_median"] >= REDUCTION
        and out["vs_F0"]["offset_median"] >= REDUCTION)
    f1 = context["F1_LATE_A1"]
    out["PARETO_BETTER_THAN_FULL_UNFREEZE"] = bool(
        all(l1[k] <= f1[k] for k in keys) and any(l1[k] < f1[k] for k in keys))
    final = history[str(DECISION_STEP)]
    out["generalization"] = {"L1": final["generalization"],
                             "F0_context_only": {
                                 "angle_ratio": f0["angle_median"]
                                 / f0_entry["D0_SEEN512"]["angle_median"],
                                 "offset_ratio": f0["offset_median"]
                                 / f0_entry["D0_SEEN512"]["offset_median"]}}
    d0 = final["D0_SEEN512"]
    out["SPECIALIZES"] = bool(
        1.0 - d0["angle_median"] / f0_entry["D0_SEEN512"]["angle_median"] >= REDUCTION
        and out["vs_F0"]["angle_median"] < REDUCTION)
    if not out["finite"]:
        out["DECISION"] = "LOW_RANK_A1_UNSTABLE"
        out["RETRY_WITH_NEW_LR_OR_RANK"] = "FORBIDDEN"
    elif out["ABSOLUTE_PASS"]:
        out["DECISION"] = "LATE_A1_LOW_RANK_VALID_CANDIDATE"
        out["STATUS"] = "LINE_STAGE_CANDIDATE"
        out["NEXT"] = "execution_stability_replicate"
    elif out["REDUCTION_40"]:
        out["DECISION"] = "LATE_A1_LOW_RANK_SIGNAL"
        out["NEXT"] = "decide_from_the_failure_shape"
    else:
        out["DECISION"] = "LATE_A1_LOW_RANK_INSUFFICIENT"
        out["SCOPE"] = "rank 8, one formulation, one learning rate"
        out["LOW_RANK_ADAPTATION_REFUTED"] = False
        out["FULL_UNFREEZE_PROVEN_REQUIRED"] = False
        out["F1_SIGNAL"] = "unchanged"
        out["NEXT"] = "consider_REGULARIZED_LATE_A1_FULL_ADAPTATION_in_a_separate_lock"
    out.setdefault("STATUS", "NOT_LOCKED")
    out["CIGM"] = "BLOCKED"
    return out


def build_plan(pool):
    decoder, _, _, adapted = build_pair()
    rows, total, smallest = check_audit(adapted)
    plan = {"arms": list(ARMS), "factor": "CONSTRAINED_IN_BLOCK_LOW_RANK_LATE_A1",
            "rank": RANK, "rank_sweep": False, "lr_sweep": False,
            "marks": list(MARKS), "decision_step": DECISION_STEP,
            "decision_population": "D2_LINE_DEV512",
            "diagnostic_population": "D0_SEEN512",
            "per_role_marks": list(PER_ROLE_MARKS),
            "frames": len(pool), "batch": CAP.BATCH,
            "late_conv_audit": rows, "late_base_params": total,
            "smallest_channel_dim": smallest,
            "audit": parameter_audit(decoder, adapted),
            "lr": CAP.LR, "weight_decay": CAP.WD, "scheduler": None,
            "gradient_clipping": None, "extra_scalar_gate": False,
            "post_f50_adapter": False, "extra_role_block": False,
            "gate": {"angle_median": CAP.ANGLE_BUDGET_DEG,
                     "offset_median": CAP.OFFSET_BUDGET_CELL,
                     "angle_p90": CAP.SAFETY_ANGLE,
                     "offset_p90": CAP.SAFETY_OFFSET},
            "reduction": REDUCTION,
            "baseline_F0_full_precision": baseline_f0()["D2_LINE_DEV512"],
            "context_only": context_arms(), **CAP.provenance()}
    del decoder, adapted
    torch.cuda.empty_cache()
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "merge", "step0", "wiring",
                                            "memory", "parity", "run"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    for name in [PHASE_A_RESULT] + [v[0] for v in CONTEXT_RESULTS.values()]:
        if not (OUT / name).exists():
            raise RuntimeError(f"HARD_BLOCK: {name} is missing")
    pool = V2.split_indices()[0]

    if arguments.command == "plan":
        plan = build_plan(pool)
        (OUT / "low_rank_a1_plan.json").write_text(json.dumps(plan, indent=2))
        audit = plan["audit"]
        log(f"[plan] late convs {len(plan['late_conv_audit'])} base "
            f"{plan['late_base_params']:,} | rank {RANK} low-rank "
            f"{audit['LOW_RANK_FEATURE_PARAMS']:,} = "
            f"{audit['low_rank_fraction_of_f1']:.4%} of the F1 unfreeze")
        log(f"[plan] trainable {audit['total_trainable']:,} (encoder "
            f"{audit['role_encoder_params']:,} + head "
            f"{audit['direct_hough_head_params']:,} + low-rank "
            f"{audit['LOW_RANK_FEATURE_PARAMS']:,}) | A1-origin trainable "
            f"{audit['a1_origin_trainable_params']}")
        log(f"[plan] decision {DECISION_STEP} on D2 only  gate "
            f"{CAP.ANGLE_BUDGET_DEG}/{CAP.OFFSET_BUDGET_CELL} safety "
            f"{CAP.SAFETY_ANGLE}/{CAP.SAFETY_OFFSET}  reduction {REDUCTION}")
        return

    if arguments.command == "merge":
        report = run_merge()
        (OUT / "low_rank_a1_merge.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[merge] max abs delta {report['max_abs_delta']:.3e} (tol "
            f"{MERGE_TOLERANCE})  MERGEABLE="
            f"{report['LOW_RANK_CONV_MERGEABLE']}")
        return

    if arguments.command == "step0":
        report = run_step0(edges)
        (OUT / "low_rank_a1_step0.json").write_text(
            json.dumps(report, indent=2, default=float))
        g = report["gaps"]
        log(f"[step0] late F50 {g['late_f50']:.3e} descriptor "
            f"{g['descriptor']:.3e} logits {g['logits']:.3e} loss "
            f"{g['loss']:.3e} | frozen path vs A1 "
            f"{g['frozen_path_against_a1']:.3e} (tol {STEP0_TOLERANCE})  "
            f"EQUIVALENT={report['LOW_RANK_A1_STEP0_EQUIVALENT']}")
        if not report["LOW_RANK_A1_STEP0_EQUIVALENT"]:
            raise RuntimeError("LOW_RANK_A1_STEP0_MISMATCH")
        return

    if arguments.command == "wiring":
        report = run_wiring(edges)
        (OUT / "low_rank_a1_wiring.json").write_text(
            json.dumps(report, indent=2, default=float))
        log("[wiring] step0 up grads " + " ".join(
            f"{k}:{v['up_grad_norm']:.2e}" for k, v in report["step0"].items()))
        log("[wiring] step2 down grads " + " ".join(
            f"{k}:{v['down_grad_norm']:.2e}" for k, v in report["step2"].items()))
        log(f"[wiring] base with grad {report['base_params_with_grad']} | "
            f"encoder {report['role_encoder_grad_norm']:.3e} head "
            f"{report['head_grad_norm']:.3e}  OK="
            f"{report['LOW_RANK_A1_GRADIENT_WIRING']}")
        if not report["LOW_RANK_A1_GRADIENT_WIRING"]:
            raise RuntimeError("LOW_RANK_A1_GRADIENT_WIRING_FAIL")
        return

    if arguments.command == "memory":
        report = run_memory(edges)
        (OUT / "low_rank_a1_memory.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[memory] batch {report['batch']} peak {report['peak_mib']:.1f} MiB "
            f"of {report['device_total_mib']:.0f} MiB  OK="
            f"{report['LOW_RANK_A1_BATCH8_MEMORY_OK']}")
        if not report["LOW_RANK_A1_BATCH8_MEMORY_OK"]:
            raise RuntimeError("LOW_RANK_A1_BATCH8_MEMORY_FAIL")
        return

    if arguments.command == "parity":
        report = run_parity(edges)
        (OUT / "low_rank_a1_parity.json").write_text(
            json.dumps(report, indent=2, default=float))
        log(f"[parity] deterministic control "
            f"{report['deterministic_control']['max_abs_delta']:.3e}  L0 vs "
            f"locked {report['l0_against_locked']['max_abs_delta']:.3e}  "
            f"PARITY={report['L0_CODE_PATH_PARITY']}")
        if not report["L0_CODE_PATH_PARITY"]:
            raise RuntimeError("L0_CODE_PATH_PARITY_FAIL")
        return

    for name, key, label in (
            ("low_rank_a1_step0.json", "LOW_RANK_A1_STEP0_EQUIVALENT",
             "LOW_RANK_A1_STEP0_MISMATCH"),
            ("low_rank_a1_wiring.json", "LOW_RANK_A1_GRADIENT_WIRING",
             "LOW_RANK_A1_GRADIENT_WIRING_FAIL"),
            ("low_rank_a1_memory.json", "LOW_RANK_A1_BATCH8_MEMORY_OK",
             "LOW_RANK_A1_BATCH8_MEMORY_FAIL"),
            ("low_rank_a1_parity.json", "L0_CODE_PATH_PARITY",
             "L0_CODE_PATH_PARITY_FAIL")):
        path = OUT / name
        if not path.exists() or not json.loads(path.read_text())[key]:
            raise RuntimeError(f"{label}: preflight must pass first")
    plan = build_plan(pool)
    per_pass = V2.steps_per_pass(pool, CAP.BATCH)
    history, _, backbone = train_arm(pool, MARKS, edges, SCALE.populations(),
                                     per_pass, probe_indices())
    merge = OUT / "low_rank_a1_merge.json"
    report = {"plan": plan, "history": history, "verdict": judge(history),
              "merge": json.loads(merge.read_text()) if merge.exists() else None,
              "low_rank_final": {str(i): backbone.deltas[str(i)].report()
                                 for i in backbone.late_indices},
              **CAP.provenance()}
    (OUT / "low_rank_a1.json").write_text(
        json.dumps(report, indent=2, default=float))
    v = report["verdict"]
    log(f"[run] {v['DECISION']}  L1 {v['L1']['angle_median']:.6f}/"
        f"{v['L1']['offset_median']:.6f} p90 {v['L1']['angle_p90']:.6f}/"
        f"{v['L1']['offset_p90']:.6f}")
    log(f"[run] vs F0 angle {v['vs_F0']['angle_median']:+.2%} offset "
        f"{v['vs_F0']['offset_median']:+.2%} | pareto vs F1 "
        f"{v['PARETO_BETTER_THAN_FULL_UNFREEZE']} | SPECIALIZES "
        f"{v['SPECIALIZES']}")


if __name__ == "__main__":
    main()
