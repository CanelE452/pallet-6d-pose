"""PHASE 1-2 -- one pass over the labels, then every later phase reads the index.

The point of this file is to stop calling four different quantities "V".  The
generator already distinguishes them and the project's own loader does not:

    mh_data.frame_row["v"]        corners whose *projection* lands in the image
                                  rectangle.  No occlusion test at all.
    v2_labels.V_actual            the generator's own in-frame count
    v2_labels.V_vis_actual        in-frame *and* not occluded
    pnp_conditioning.visible_kp_count
                                  what the generator counted as usable for PnP
    belief_valid (mh_data)        channels the corner loss actually supervises,
                                  which is the in-grid test at 50-grid resolution

`_stratum`, every risk map in this study, and the phrase "V<8" all come from the
first one.  Whether the others agree with it is measured here rather than assumed.

The generator also already stores what PHASE 4 was going to recompute --
`pnp_conditioning` carries coplanarity and 2D collinearity of the *visible* set
plus a `degeneracy` verdict -- so the routing question is read from the labels,
not re-derived.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import mh_data as MD                                             # noqa: E402

OUT = MD.OUT
INDEX = OUT / "regime_index.npz"

ELEV_EDGES = (0.0, 8.0, 15.0, 30.0, 1e9)
ELEV_NAMES = ("E0 <8", "E1 8-15", "E2 15-30", "E3 >=30")
YAW_EDGES = (0.0, 5.0, 15.0, 30.0, 1e9)
YAW_NAMES = ("Y0 0-5", "Y1 5-15", "Y2 15-30", "Y3 >=30")
POINT_NAMES = ("P8", "P67", "P45", "P03")


def log(message):
    print(message, flush=True)


def point_bucket(count):
    if count >= 8:
        return "P8"
    if count >= 6:
        return "P67"
    if count >= 4:
        return "P45"
    return "P03"


def bucket(value, edges, names):
    index = int(np.clip(np.searchsorted(edges, value, side="right") - 1,
                        0, len(names) - 1))
    return names[index]


def run_index(_arguments):
    rows = MD.load_split()
    store = {k: [] for k in (
        "stem", "split", "n_inframe", "V_actual", "V_vis_actual",
        "visible_kp_count", "ext_occ", "elev_actual", "elev_target",
        "azimuth_target", "front_vis_cos", "facing_margin",
        "coplanar_ratio", "coplanar_visible_set", "collinear_ratio",
        "collinear_visible_set", "degeneracy", "loss_cause",
        "gates_all_pass", "g1_vvis4", "g2_extocc", "g3_half_unocc",
        "g4_center", "g5_luma", "proj_size", "distance_m", "pallet_type",
        "kp12_valid", "front_face_type", "width", "height", "n_supervised")}
    occlusion = []
    for count, row in enumerate(rows):
        payload = MD.read_label(row["stem"])
        camera = payload["camera_data"]
        obj = payload["objects"][0]
        # these can be present-but-null, not just absent
        labels = obj.get("v2_labels") or {}
        gates = obj.get("safety_gates") or {}
        cond = obj.get("pnp_conditioning") or {}
        front = obj.get("efront_kp12") or {}
        width, height = float(camera["width"]), float(camera["height"])
        cuboid = np.asarray(obj["projected_cuboid"], float)
        inframe = int(sum((0 <= x < width and 0 <= y < height)
                          for x, y in cuboid))
        # what the corner loss actually supervises: the 50-grid in-bounds test
        grid = np.stack([cuboid[:, 0] * MD.GRID / width,
                         cuboid[:, 1] * MD.GRID / height], 1)
        supervised = int(sum((0 <= x < MD.GRID and 0 <= y < MD.GRID)
                             for x, y in grid))
        store["stem"].append(row["stem"])
        store["split"].append(row["split"])
        store["n_inframe"].append(inframe)
        store["n_supervised"].append(supervised)
        store["V_actual"].append(int(labels.get("V_actual", -1)))
        store["V_vis_actual"].append(int(labels.get("V_vis_actual", -1)))
        store["visible_kp_count"].append(int(cond.get("visible_kp_count", -1)))
        store["ext_occ"].append(int(labels.get("ext_occ_corners_actual", -1)))
        store["elev_actual"].append(float(labels.get("elevation_deg_actual",
                                                     np.nan)))
        store["elev_target"].append(float(labels.get("elevation_deg_target",
                                                     np.nan)))
        store["azimuth_target"].append(float(labels.get("azimuth_deg_target",
                                                        np.nan)))
        store["front_vis_cos"].append(float(obj.get("front_visibility_cos",
                                                    np.nan)))
        store["facing_margin"].append(float(obj.get("facing_margin", np.nan)))
        store["coplanar_ratio"].append(float(cond.get("coplanar_ratio",
                                                      np.nan)))
        store["coplanar_visible_set"].append(
            bool(cond.get("coplanar_visible_set", False)))
        store["collinear_ratio"].append(float(cond.get("collinear_2d_ratio",
                                                       np.nan)))
        store["collinear_visible_set"].append(
            bool(cond.get("collinear_2d_visible_set", False)))
        store["degeneracy"].append(str(cond.get("degeneracy", "unknown")))
        store["loss_cause"].append(str(cond.get("loss_cause", "none")))
        store["gates_all_pass"].append(bool(gates.get("all_pass", False)))
        store["g1_vvis4"].append(bool(gates.get("G1_Vvis>=4", False)))
        store["g2_extocc"].append(bool(gates.get("G2_extocc_1to4", False)))
        store["g3_half_unocc"].append(bool(gates.get("G3_visible>=0.5unocc",
                                                     False)))
        store["g4_center"].append(bool(gates.get("G4_center_inframe", False)))
        store["g5_luma"].append(bool(gates.get("G5_luma_floor", False)))
        store["proj_size"].append(float(labels.get("projected_size_actual",
                                                   np.nan)))
        store["distance_m"].append(float(labels.get("camera_distance_actual_m",
                                                    np.nan)))
        store["pallet_type"].append(str(labels.get("pallet_type", "?")))
        store["kp12_valid"].append(
            bool(front.get("kp12_valid", False)))
        store["front_face_type"].append(
            str(front.get("front_face_type", "?")))
        store["width"].append(width)
        store["height"].append(height)
        fractions = labels.get("occlusion_fraction") or [np.nan] * 9
        occlusion.append(np.asarray(fractions, float)[:9])
        if (count + 1) % 5000 == 0:
            log(f"  {count + 1}/{len(rows)}")
    payload = {k: np.asarray(v) for k, v in store.items()}
    payload["occlusion_fraction"] = np.stack(occlusion)
    np.savez_compressed(INDEX, **payload)
    log(f"-> {INDEX}  {len(rows)} frames")


def load():
    return np.load(INDEX, allow_pickle=True)


def run_contract(_arguments):
    """PHASE 1: do the four counts actually agree?  And what does the gate do?"""
    data = load()
    n = len(data["stem"])
    inframe = data["n_inframe"]
    supervised = data["n_supervised"]
    v_actual = data["V_actual"]
    v_vis = data["V_vis_actual"]
    visible_kp = data["visible_kp_count"]
    report = {"n": int(n)}

    def agree(a, b):
        return round(float((a == b).mean()), 5)

    report["agreement"] = {
        "n_inframe == V_actual": agree(inframe, v_actual),
        "n_inframe == V_vis_actual": agree(inframe, v_vis),
        "n_inframe == n_supervised": agree(inframe, supervised),
        "V_actual == V_vis_actual": agree(v_actual, v_vis),
        "V_vis_actual == visible_kp_count(8 of 9)": agree(v_vis, visible_kp),
    }
    report["means"] = {k: round(float(np.mean(v)), 4) for k, v in
                       (("n_inframe", inframe), ("n_supervised", supervised),
                        ("V_actual", v_actual), ("V_vis_actual", v_vis),
                        ("visible_kp_count", visible_kp))}
    report["occluded_frames_pct"] = round(
        100.0 * float((data["ext_occ"] > 0).mean()), 3)
    report["elev_target_equals_actual_pct"] = round(100.0 * float(
        np.isclose(data["elev_target"], data["elev_actual"],
                   atol=0.05, equal_nan=True).mean()), 3)
    report["gates"] = {k: round(float(data[k].mean()), 5) for k in
                       ("gates_all_pass", "g1_vvis4", "g2_extocc",
                        "g3_half_unocc", "g4_center", "g5_luma")}
    values, counts = np.unique(data["degeneracy"], return_counts=True)
    report["degeneracy"] = {str(k): int(v) for k, v in zip(values, counts)}
    values, counts = np.unique(data["loss_cause"], return_counts=True)
    report["loss_cause"] = {str(k): int(v) for k, v in zip(values, counts)}
    report["coplanar_visible_set_pct"] = round(
        100.0 * float(data["coplanar_visible_set"].mean()), 3)
    report["collinear_visible_set_pct"] = round(
        100.0 * float(data["collinear_visible_set"].mean()), 3)
    for name, array in (("n_inframe", inframe), ("V_vis_actual", v_vis)):
        values, counts = np.unique(array, return_counts=True)
        report[f"hist_{name}"] = {int(k): int(v)
                                  for k, v in zip(values, counts)}
    (OUT / "point_data_acceptance_contract.json").write_text(
        json.dumps(report, indent=1))
    log(json.dumps(report, indent=1)[:3000])
    log(f"-> {OUT / 'point_data_acceptance_contract.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["index", "contract"])
    arguments = parser.parse_args()
    {"index": run_index, "contract": run_contract}[arguments.command](arguments)


if __name__ == "__main__":
    main()
