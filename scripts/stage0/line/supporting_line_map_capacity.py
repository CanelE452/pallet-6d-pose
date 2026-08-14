"""Can RGB or a frozen A1 feature predict twelve structural supporting-line maps?

Everything before this measured a decoder.  `f5ac650` established that a *perfect*
supporting-line probability map decodes to 0.0066 degree and 0.0061 canonical50
cell through the locked H2 Hough readout, so for the first time the readout is
not what limits the answer.  This screen asks the network question.

The target comes from the ground-truth `(theta, rho)` alone -- no endpoints, no
segment length -- so target and decoder describe one geometric object.  The loss
is map-only: the Hough argmax is not differentiable and carries no gradient, and
nothing here optimises the decoded line.  An unsupported role is masked out
rather than supervised towards an empty map, because "not measured here" is not
"no structural line".

No PnP, no CIGM, no dimensions, no intrinsics, no pose.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, pathlib, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
for _e in ("scripts/stage0", "Deep_Object_Pose/common", "challenge/scripts"):
    if str(ROOT / _e) not in sys.path:
        sys.path.insert(0, str(ROOT / _e))


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SEM = _load("SEM_CAP", "scripts/stage0/line/structural_line_target_semantics.py")
H, SLM, V2 = SEM.H, SEM.SLM, SEM.V2
CANON, MAP, SIGMA_CELLS = SEM.CANON, SEM.MAP, SEM.SIGMA_CELLS
DEV, OUT, PRIMARY = SEM.DEV, SEM.OUT, SEM.PRIMARY

SEED, LR, WD, BATCH = 1, 1e-3, 1e-4, 8
EPOCH_LADDER = (1, 3, 5)
OVERFIT_FRAMES, OVERFIT_STEPS, OLOSS_STEPS = 32, 1500, 500
POSITIVE_THRESHOLD = 1e-3
LINE_DEV_POPULATION_SHA = "00c605b9116e214b"       # locked by f5ac650
OLOSS_GATE = {"angle_median": 0.05, "offset_median": 0.05,
              "angle_p90": 0.10, "offset_p90": 0.10}
OVERFIT_GATE = {"angle_median": 0.10, "offset_median": 0.05,
                "angle_p90": 0.25, "offset_p90": 0.15}
ANGLE_BUDGET_DEG, OFFSET_BUDGET_CELL = 1.0, 0.5
SAFETY_ANGLE, SAFETY_OFFSET = 2.0, 1.0
APPROACH_ANGLE, APPROACH_OFFSET = 1.5, 0.75         # reused from 2def93c
SHUFFLE_ANGLE_MARGIN, SHUFFLE_OFFSET_MARGIN = 5.0, 2.0
DERANGEMENT = SLM.DERANGEMENT
ARMS = {"M0_F50_SLINE": False, "M1_F50_RGB_SLINE": True}
FINITE_GATE = 1.0


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------ pieces
class SupportingLineHead(nn.Module):
    """Feature -> twelve role-fixed logits.  No support head, no coordinate head.

    The old support head predicted whether a *finite visible segment* existed,
    which is the semantics this screen just removed; reusing it would smuggle
    segment extent back in.
    """

    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.to_map = nn.Conv2d(64, 12, 1)

    def forward(self, feature):
        return self.to_map(self.body(feature))


def build_arm(name):
    torch.manual_seed(SEED)
    stem = SLM.RgbLineStem().to(DEV) if ARMS[name] else None
    channels = 128 + (stem.out_channels if stem is not None else 0)
    head = SupportingLineHead(channels).to(DEV)
    parameters = list(head.parameters())
    if stem is not None:
        parameters += list(stem.parameters())
    return head, stem, parameters


def features(pack, a1, stem):
    with torch.no_grad():
        f50, _, _ = a1(pack["images"])
    up = F.interpolate(f50.detach(), size=(MAP, MAP), mode="bilinear",
                       align_corners=False)
    return up if stem is None else torch.cat([up, stem(pack["images"])], 1)


def geometry(pack, edges):
    theta, rho, p0, p1, length = V2.gt_lines(pack["grid"], edges)
    seg = V2.visible_segments(p0, p1, length)
    target = SEM.raster_supporting_line(theta, rho, seg["hit"])
    return theta, rho, seg, target


def map_loss(logit, target, supported):
    """Balanced positive/negative MSE on the probability, masked to supported.

    An unsupported role contributes nothing at all -- it is UNKNOWN, not a
    negative example of a line that does not exist.
    """
    probability = torch.sigmoid(logit)
    error = (probability - target) ** 2
    mask = supported[..., None, None].float()
    positive = (target > POSITIVE_THRESHOLD).float() * mask
    negative = (target <= POSITIVE_THRESHOLD).float() * mask
    return 0.5 * (error * positive).sum() / positive.sum().clamp_min(1.0) \
        + 0.5 * (error * negative).sum() / negative.sum().clamp_min(1.0)


@torch.no_grad()
def decode_maps(probability, supported, theta, rho, coarse, xx, yy):
    """Locked f5ac650 path: sigmoid probability -> H2 coarse+fine Hough."""
    angle, offset = [], []
    for frame in range(probability.shape[0]):
        live = torch.nonzero(supported[frame]).flatten()
        if live.numel() == 0:
            continue
        maps = probability[frame][live].reshape(live.numel(), -1).T.contiguous()
        decoded = H.decode(maps, coarse, xx, yy)[PRIMARY]
        a, o = H.measure(decoded["normal"], decoded["rho"],
                         theta[frame][live], rho[frame][live])
        angle.append(a); offset.append(o)
    return (np.concatenate(angle) if angle else np.zeros(0),
            np.concatenate(offset) if offset else np.zeros(0))


def summarise(angle, offset, extra=None):
    if angle.size == 0:
        angle = offset = np.zeros(1)
    report = {"angle_median": float(np.median(angle)),
              "angle_p90": float(np.percentile(angle, 90)),
              "offset_median": float(np.median(offset)),
              "offset_p90": float(np.percentile(offset, 90)), "n": int(angle.size)}
    report["PASS"] = bool(report["angle_median"] <= ANGLE_BUDGET_DEG
                          and report["offset_median"] <= OFFSET_BUDGET_CELL)
    report["SAFETY"] = bool(report["angle_p90"] <= SAFETY_ANGLE
                            and report["offset_p90"] <= SAFETY_OFFSET)
    report["APPROACH"] = bool(report["angle_median"] <= APPROACH_ANGLE
                              and report["offset_median"] <= APPROACH_OFFSET)
    report.update(extra or {})
    return report


# ------------------------------------------------------------------ oracles
def run_oloss(indices, edges, coarse, xx, yy, steps=OLOSS_STEPS):
    """Free logits: no image, no network.  Does the map loss alone produce a
    probability the locked decoder can read?"""
    packs = [V2.load_pack(indices[start:start + BATCH])
             for start in range(0, OVERFIT_FRAMES, BATCH)]
    logits, targets, supports, thetas, rhos = [], [], [], [], []
    for pack in packs:
        theta, rho, seg, target = geometry(pack, edges)
        logits.append(torch.zeros_like(target, requires_grad=True))
        targets.append(target)
        supports.append(torch.tensor(seg["hit"], device=DEV))
        thetas.append(torch.tensor(theta, dtype=torch.float32, device=DEV))
        rhos.append(torch.tensor(rho, dtype=torch.float32, device=DEV))
    optimiser = torch.optim.AdamW(logits, lr=1e-1, weight_decay=0.0)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        total = sum(map_loss(l, t, s) for l, t, s in zip(logits, targets, supports))
        total.backward(); optimiser.step()
    angle, offset = [], []
    for logit, support, theta, rho in zip(logits, supports, thetas, rhos):
        a, o = decode_maps(torch.sigmoid(logit.detach()), support, theta, rho,
                           coarse, xx, yy)
        angle.append(a); offset.append(o)
    angle, offset = np.concatenate(angle), np.concatenate(offset)
    report = summarise(angle, offset, {"steps": steps, "final_loss": float(total)})
    report["gates"] = {k: bool(report[k] <= v) for k, v in OLOSS_GATE.items()}
    report["OLOSS_PASS"] = all(report["gates"].values())
    return report


@torch.no_grad()
def evaluate(indices, head, stem, a1, edges, coarse, xx, yy, permute=None):
    head.eval()
    angle, offset, full = [], [], []
    diagnostics = {"map_positive_mse": [], "map_negative_mse": [],
                   "map_ncc": [], "probability_mass": [], "peak_probability": []}
    finite = supported_total = 0
    for start in range(0, len(indices), BATCH):
        chunk = indices[start:start + BATCH]
        if len(chunk) < 2:
            continue
        pack = V2.load_pack(chunk)
        theta, rho, seg, target = geometry(pack, edges)
        support = torch.tensor(seg["hit"], device=DEV)
        logit = head(features(pack, a1, stem))
        if permute is not None:
            logit = logit[:, list(DERANGEMENT)]
        probability = torch.sigmoid(logit)
        theta_t = torch.tensor(theta, dtype=torch.float32, device=DEV)
        rho_t = torch.tensor(rho, dtype=torch.float32, device=DEV)
        a, o = decode_maps(probability, support, theta_t, rho_t, coarse, xx, yy)
        angle.append(a); offset.append(o)
        full.append(seg["in_frame_full"][seg["hit"]])
        mask = support[..., None, None].float()
        error = (probability - target) ** 2
        positive = (target > POSITIVE_THRESHOLD).float() * mask
        negative = (target <= POSITIVE_THRESHOLD).float() * mask
        diagnostics["map_positive_mse"].append(
            float((error * positive).sum() / positive.sum().clamp_min(1)))
        diagnostics["map_negative_mse"].append(
            float((error * negative).sum() / negative.sum().clamp_min(1)))
        flat_p = (probability * mask).flatten(2)
        flat_t = (target * mask).flatten(2)
        centred_p = flat_p - flat_p.mean(-1, keepdim=True)
        centred_t = flat_t - flat_t.mean(-1, keepdim=True)
        ncc = (centred_p * centred_t).sum(-1) / (
            centred_p.norm(dim=-1) * centred_t.norm(dim=-1) + 1e-12)
        diagnostics["map_ncc"].append(float(ncc[support].mean()))
        diagnostics["probability_mass"].append(float(flat_p[support].sum(-1).mean()))
        diagnostics["peak_probability"].append(float(flat_p[support].max(-1).values.mean()))
        finite += int(np.isfinite(a).sum()); supported_total += int(a.size)
    angle = np.concatenate(angle) if angle else np.zeros(0)
    offset = np.concatenate(offset) if offset else np.zeros(0)
    full = np.concatenate(full).astype(bool) if full else np.zeros(0, bool)
    report = summarise(angle, offset, {
        "finite_fraction": finite / max(supported_total, 1),
        "diagnostics": {k: float(np.mean(v)) for k, v in diagnostics.items()}})
    for label, keep in (("in_frame_full", full), ("in_frame_partial", ~full)):
        if keep.size and keep.sum():
            report[label] = {"n": int(keep.sum()),
                             "angle_median": float(np.median(angle[keep])),
                             "angle_p90": float(np.percentile(angle[keep], 90)),
                             "offset_median": float(np.median(offset[keep])),
                             "offset_p90": float(np.percentile(offset[keep], 90))}
    return report


def train_steps(head, stem, parameters, indices, steps, a1, edges):
    optimiser = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)
    for chunk, visit in V2.step_schedule(indices, steps, BATCH):
        head.train()
        pack = V2.load_pack(chunk)
        _, _, seg, target = geometry(pack, edges)
        logit = head(features(pack, a1, stem))
        loss = map_loss(logit, target, torch.tensor(seg["hit"], device=DEV))
        optimiser.zero_grad(set_to_none=True)
        loss.backward(); optimiser.step()
    return optimiser


def checkpoint_path(name, tag):
    directory = OUT / "supporting_line_map" / "checkpoints" / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{tag}.pth"


def provenance():
    return {"runner_sha": V2.sha_file(pathlib.Path(__file__)),
            "target_semantics_sha": V2.sha_file(
                ROOT / "scripts/stage0/line/structural_line_target_semantics.py"),
            "hough_decoder_sha": V2.sha_file(
                ROOT / "scripts/stage0/line/structural_line_hough_decoder.py"),
            "split_sha": V2.sha_file(OUT / "line_internal_split.csv"),
            "population_sha": LINE_DEV_POPULATION_SHA,
            "sigma_map100_pixel": SIGMA_CELLS, "seed": SEED}


def population_sha(indices, edges):
    roles = []
    for start in range(0, len(indices), 12):
        chunk = indices[start:start + 12]
        corners = np.stack([V2.load_geometry(i) for i in chunk])
        _, _, p0, p1, length = V2.gt_lines(corners, edges)
        seg = V2.visible_segments(p0, p1, length)
        for frame, index in enumerate(chunk):
            roles.extend((index, int(r)) for r in np.flatnonzero(seg["hit"][frame]))
    return hashlib.sha256(repr(roles).encode()).hexdigest()[:16], len(roles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "oloss", "overfit",
                                            "search2k", "search2k-budget",
                                            "confirm6k"])
    arguments = parser.parse_args()
    import instance_edge_topology as IET
    edges = [tuple(e) for e in IET.build_topology()["edges"]]
    if not V2.sha_file(OUT / "line_internal_split.csv").startswith(V2.LINE_SPLIT_SHA):
        raise RuntimeError("HARD_BLOCKED: LINE split changed")
    _, full_dev = V2.split_indices()
    dev = V2.manifest("line_dev512")
    train_ids = V2.manifest("line_search2k")
    results_file = OUT / "supporting_line_map_arms.json"
    coarse, (xx, yy) = H.CoarseRadon(), H.pixel_coordinates()

    if arguments.command == "verify":
        sha, count = population_sha(full_dev, edges)
        record = {"line_dev_population_sha": sha, "supported_roles": count,
                  "expected_sha": LINE_DEV_POPULATION_SHA, **provenance()}
        (OUT / "supporting_line_map_population.json").write_text(
            json.dumps(record, indent=2))
        log(f"[verify] LINE_DEV supported roles {count}  sha {sha}")
        if sha != LINE_DEV_POPULATION_SHA or count != 27684:
            raise RuntimeError(f"POPULATION_CHANGED: {sha} / {count}")
        return

    results = json.loads(results_file.read_text()) if results_file.exists() else {}

    if arguments.command == "oloss":
        report = run_oloss(train_ids, edges, coarse, xx, yy)
        report.update(provenance())
        (OUT / "supporting_line_map_oloss.json").write_text(json.dumps(report, indent=2))
        log(f"[O_LOSS] angle med {report['angle_median']:.4f} p90 "
            f"{report['angle_p90']:.4f} | offset med {report['offset_median']:.4f} "
            f"p90 {report['offset_p90']:.4f}  n={report['n']}  "
            f"PASS={report['OLOSS_PASS']}")
        if not report["OLOSS_PASS"]:
            raise RuntimeError("SUPPORTING_LINE_MAP_LOSS_NOT_IDENTIFIABLE")
        return

    oloss = OUT / "supporting_line_map_oloss.json"
    if not oloss.exists() or not json.loads(oloss.read_text())["OLOSS_PASS"]:
        raise RuntimeError("SUPPORTING_LINE_MAP_LOSS_NOT_IDENTIFIABLE: training "
                           "is blocked until the loss oracle passes")
    a1 = V2.load_a1()

    if arguments.command == "overfit":
        for name in ARMS:
            head, stem, parameters = build_arm(name)
            optimiser = train_steps(head, stem, parameters,
                                    train_ids[:OVERFIT_FRAMES], OVERFIT_STEPS,
                                    a1, edges)
            report = evaluate(train_ids[:OVERFIT_FRAMES], head, stem, a1, edges,
                              coarse, xx, yy)
            report["gates"] = {k: bool(report[k] <= v) for k, v in OVERFIT_GATE.items()}
            report["OVERFIT_PASS"] = bool(all(report["gates"].values())
                                          and report["finite_fraction"] >= FINITE_GATE)
            torch.save({"arm": name, "stage": "overfit32",
                        "model": head.state_dict(),
                        "stem": None if stem is None else stem.state_dict(),
                        "optimizer": optimiser.state_dict(), **provenance()},
                       checkpoint_path(name, "overfit32"))
            results.setdefault(name, {})["overfit32"] = report
            results_file.write_text(json.dumps(results, indent=2, default=float))
            log(f"  {name} overfit32: angle med {report['angle_median']:.4f} p90 "
                f"{report['angle_p90']:.4f} | offset med {report['offset_median']:.4f}"
                f" p90 {report['offset_p90']:.4f}  PASS={report['OVERFIT_PASS']}")
        if not any(results[n]["overfit32"]["OVERFIT_PASS"] for n in ARMS):
            raise RuntimeError("SUPPORTING_LINE_MAP_OPTIMIZATION_FAIL: both arms")
        return

    # search2k keeps its original eligibility -- the overfit verdict in 7c6602a
    # stands untouched.  search2k-budget is a different question: both arms
    # already land inside the primary budget on the overfit frames, so whether
    # they generalise to it was never asked.  Same architecture, target, loss,
    # decoder, ladder and gates; only the entry condition differs.
    stage = "search2k" if arguments.command == "search2k-budget" else arguments.command
    pool = train_ids if stage == "search2k" else V2.manifest("line_confirm6k")
    per_pass = V2.steps_per_pass(pool, BATCH)
    if arguments.command == "search2k-budget":
        eligible = list(ARMS)          # gated by O_LOSS above, not by overfit32
    else:
        eligible = [n for n in ARMS
                    if results.get(n, {}).get("overfit32", {}).get("OVERFIT_PASS")]
    if stage == "confirm6k":
        eligible = [n for n in eligible
                    if results[n].get("search2k_epoch5", {}).get("APPROACH")]
    if not eligible:
        raise RuntimeError(f"NO_ELIGIBLE_ARM for {stage}")
    for name in eligible:
        head, stem, parameters = build_arm(name)
        optimiser = torch.optim.AdamW(parameters, lr=LR, weight_decay=WD)
        entry = results.setdefault(name, {})
        done, running = 0, []
        for chunk, visit in V2.step_schedule(pool, per_pass * max(EPOCH_LADDER), BATCH):
            head.train()
            pack = V2.load_pack(chunk)
            _, _, seg, target = geometry(pack, edges)
            logit = head(features(pack, a1, stem))
            loss = map_loss(logit, target, torch.tensor(seg["hit"], device=DEV))
            optimiser.zero_grad(set_to_none=True)
            loss.backward(); optimiser.step()
            running.append(float(loss.detach()))
            done += 1
            if done % per_pass == 0 and done // per_pass in EPOCH_LADDER:
                epoch = done // per_pass
                key = f"{stage}_epoch{epoch}"
                entry[key] = evaluate(dev, head, stem, a1, edges, coarse, xx, yy)
                # the training signal itself, per epoch; diagnostic only
                entry[key]["train_map_loss"] = float(np.mean(running[-per_pass:]))
                torch.save({"arm": name, "stage": key, "model": head.state_dict(),
                            "stem": None if stem is None else stem.state_dict(),
                            "optimizer": optimiser.state_dict(), **provenance()},
                           checkpoint_path(name, key))
                results_file.write_text(json.dumps(results, indent=2, default=float))
                log(f"  {name} {key}: angle med {entry[key]['angle_median']:.4f} "
                    f"p90 {entry[key]['angle_p90']:.4f} | offset med "
                    f"{entry[key]['offset_median']:.4f}  n={entry[key]['n']}  "
                    f"PASS={entry[key]['PASS']}")
        last = f"{stage}_epoch{max(EPOCH_LADDER)}"
        shuffled = evaluate(dev, head, stem, a1, edges, coarse, xx, yy,
                            permute=DERANGEMENT)
        entry[f"{stage}_shuffle"] = shuffled
        entry[f"{stage}_ROLE_SEMANTICS_LEARNED"] = bool(
            shuffled["angle_median"] >= entry[last]["angle_median"] + SHUFFLE_ANGLE_MARGIN
            or shuffled["offset_median"] >= entry[last]["offset_median"] + SHUFFLE_OFFSET_MARGIN)
        state = torch.load(checkpoint_path(name, last), map_location=DEV,
                           weights_only=False)
        reload_head, reload_stem, _ = build_arm(name)
        reload_head.load_state_dict(state["model"])
        if reload_stem is not None:
            reload_stem.load_state_dict(state["stem"])
        reloaded = evaluate(dev, reload_head, reload_stem, a1, edges, coarse, xx, yy)
        entry[f"{stage}_reload_parity"] = {
            "max_delta": max(abs(reloaded[k] - entry[last][k]) for k in
                             ("angle_median", "angle_p90", "offset_median",
                              "offset_p90"))}
        results_file.write_text(json.dumps(results, indent=2, default=float))
        log(f"  {name} shuffle angle med {shuffled['angle_median']:.4f}  "
            f"role_semantics={entry[f'{stage}_ROLE_SEMANTICS_LEARNED']}  "
            f"reload delta {entry[f'{stage}_reload_parity']['max_delta']:.2e}")
    log(f"[{stage}] done")


if __name__ == "__main__":
    main()
