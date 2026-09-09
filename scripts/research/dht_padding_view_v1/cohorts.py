"""Build frozen, group-separated DHT padding/view cohorts without changing GT.

CPU/I/O only. Existing real DEV52 stays evaluation-only. Optional low-view data
are selected from the current clean broad40k and oblique releases, never v8.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "data/pallet/results/hough_attention_transfer_v1/manifest.json"
BROAD = ROOT / "data/pallet/training_data/paper_release/v2_prod40k_clean_merged"
OBLIQUE = ROOT / "data/pallet/training_data/paper_release/oblique/extracted"
EDGES = ((1,2),(3,0),(5,6),(7,4),(0,4),(1,5),(2,6),(3,7))
FACES = {"front": (0,3,2,1), "left": (0,4,7,3), "right": (1,2,6,5), "top": (0,1,5,4)}
BINS = {
    "elevation_bin": ["elev_lt5", "elev_5_15", "elev_15_30", "elev_ge30", "unknown"],
    "side_bin": ["side_le0p1", "side_0p1_0p3", "side_gt0p3", "unknown"],
    "top_bin": ["top_le1", "top_1_4", "top_gt4", "unknown"],
    "size_bin": ["width_lt0p25", "width_0p25_0p5", "width_0p5_1", "width_ge1", "unknown"],
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def order_key(seed, text):
    return hashlib.sha256(f"dht-padding-view-v1:{seed}:{text}".encode()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def intersects(a, b, width, height):
    delta, low, high = b-a, 0., 1.
    for p,q in ((-delta[0],a[0]),(delta[0],width-a[0]),(-delta[1],a[1]),(delta[1],height-a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        elif p < 0:
            low = max(low,q/p)
        else:
            high = min(high,q/p)
    return low <= high


def world_view(annotation):
    obj, cam = annotation["objects"][0], annotation["camera_data"]
    if "cuboid" not in obj or "location_worldframe" not in cam:
        return None, None
    w = np.asarray(obj["cuboid"], float)[:8]
    c = np.asarray(cam["location_worldframe"], float)
    if w.shape != (8,3) or not np.isfinite(w).all() or not np.isfinite(c).all():
        raise ValueError("invalid world cuboid/camera")
    up = w[[0,1,4,5]].mean(0) - w[[2,3,6,7]].mean(0)
    up /= np.linalg.norm(up)
    ray = c - w.mean(0)
    vertical = float(up @ ray)
    horizontal = ray - vertical*up
    elevation = float(np.degrees(np.arctan2(vertical,np.linalg.norm(horizontal))))
    normal = w[:4].mean(0) - w.mean(0)
    normal -= (normal @ up)*up
    cosine = float(normal @ horizontal) / (np.linalg.norm(normal)*np.linalg.norm(horizontal))
    frontal_yaw = float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    return elevation, frontal_yaw


def cohort(record, annotation):
    p = np.asarray(record["gt_points"], float)[:8]
    valid = np.asarray(record.get("gt_valid", [True]*8),bool)[:8]
    valid &= np.isfinite(p).all(-1) & ~(p == -1).all(-1)
    areas = {}
    for name, indices in FACES.items():
        q = p[list(indices)]
        areas[name] = float(.5*np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))) if valid[list(indices)].all() else None
    denom = abs(areas["front"]) if areas["front"] is not None else 0
    side = max(-areas["left"],-areas["right"],0)/denom if denom>=1 and all(areas[x] is not None for x in ("left","right")) else None
    top = max(-areas["top"],0)/denom if denom>=1 and areas["top"] is not None else None
    width = float((np.linalg.norm(p[0]-p[1])+np.linalg.norm(p[2]-p[3]))/2/record["width"]) if valid[:4].all() else None
    real = str(record.get("source_kind","")).startswith("real") or record["population"] == "real_dev"
    elevation, yaw = (None,None) if real else world_view(annotation)
    eb = "unknown" if elevation is None else "elev_lt5" if elevation<5 else "elev_5_15" if elevation<15 else "elev_15_30" if elevation<30 else "elev_ge30"
    sb = "unknown" if side is None else "side_le0p1" if side<=.1 else "side_0p1_0p3" if side<=.3 else "side_gt0p3"
    tb = "unknown" if top is None else "top_le1" if top<=1 else "top_1_4" if top<=4 else "top_gt4"
    wb = "unknown" if width is None else "width_lt0p25" if width<.25 else "width_0p25_0p5" if width<.5 else "width_0p5_1" if width<1 else "width_ge1"
    support = [bool(valid[a] and valid[b] and np.linalg.norm(p[a]-p[b])>=2
                    and intersects(p[a],p[b],record["width"],record["height"])) for a,b in EDGES]
    return {"elevation_bin":eb,"side_bin":sb,"top_bin":tb,"size_bin":wb,
            "elevation_deg_world":elevation,"frontal_yaw_deg_world":yaw,"side_exposure":side,
            "top_exposure":top,"front_width_fraction":width,"supported_side_roles":support,
            "supported_side_count":sum(support),"physical_edge_visibility":"unknown"}


def cohort_counts(records):
    out = {name: {b:sum(r["cohort"][name]==b for r in records) for b in choices} for name,choices in BINS.items()}
    cells = []
    for eb,sb in itertools.product(BINS["elevation_bin"],BINS["side_bin"]):
        rs = [r for r in records if r["cohort"]["elevation_bin"]==eb and r["cohort"]["side_bin"]==sb]
        cells.append({"elevation_bin":eb,"side_bin":sb,"n_frames":len(rs),
                      "n_supported_side_roles":sum(r["cohort"]["supported_side_count"] for r in rs),
                      "n_no_supported_frames":sum(r["cohort"]["supported_side_count"]==0 for r in rs),
                      "coverage_status":"empty" if not rs else "sparse_lt20" if len(rs)<20 else "n_ge20"})
    out["elevation_x_side"] = cells
    out["n_frames"] = len(records)
    out["n_supported_side_roles"] = sum(r["cohort"]["supported_side_count"] for r in records)
    return out


def png_size(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("source is not PNG")
    return struct.unpack(">II",header[16:24])


def candidate(record_meta, directory, stem, family, group, audit):
    label, image = directory/"labels"/f"{stem}_label.json",directory/"rgb"/f"{stem}_rgb.png"
    try:
        annotation = json.loads(label.read_text())
        if len(annotation.get("objects",[])) != 1:
            raise ValueError("not exactly one pallet")
        obj,cam = annotation["objects"][0],annotation["camera_data"]
        if obj.get("keypoint_convention") != "camera_dynamic_0123_v4":
            raise ValueError("wrong keypoint convention")
        p = np.asarray(obj["projected_cuboid"],float)[:8]
        if p.shape!=(8,2) or not np.isfinite(p).all() or ((p == -1).all(-1)).any():
            raise ValueError("invalid cuboid annotations")
        if png_size(image) != (cam["width"],cam["height"]):
            raise ValueError("image dimension mismatch")
        intr = cam["intrinsics"]
        if not (intr["fx"]>0 and intr["fy"]>0):
            raise ValueError("invalid intrinsics")
        rec = {"id":f"{family}__{directory.name}__{stem}","population":"low_pool",
               "image":str(image.resolve()),"annotation":str(label.resolve()),"width":cam["width"],"height":cam["height"],
               "gt_points":p.tolist(),"gt_valid":[True]*8,
               "gt_in_frame":((p[:,0]>=0)&(p[:,0]<cam["width"])&(p[:,1]>=0)&(p[:,1]<cam["height"])).tolist(),
               "group":group,"convention":"camera_dynamic_0123_v4","source_kind":"synthetic",
               "source_family":family,"source_asset":obj.get("source_asset"),
               "conditions":{"scene_preset":cam.get("scene_preset"),"background_asset":cam.get("background_asset"),
                             "renderer_elevation_deg":record_meta.get("elev_actual",record_meta.get("elevation_deg_actual"))},
               "source_record_evidence":{"renderer_group":group,"frame_seed":record_meta.get("frame_seed",record_meta.get("seed")),
                                         "source_bucket":record_meta.get("source_bucket"),"file_id":stem},
               "annotation_sha256":sha(label)}
        rec["cohort"] = cohort(rec,annotation)
        elevation = rec["cohort"]["elevation_deg_world"]
        expected = rec["conditions"]["renderer_elevation_deg"]
        difference = abs(elevation-expected) if expected is not None and elevation is not None else float("inf")
        if difference>.005:
            raise ValueError(f"merged frame/world elevation mismatch ({difference:g}deg)")
        audit["max_world_metadata_elevation_difference_deg"] = max(audit["max_world_metadata_elevation_difference_deg"],difference)
        if not (0<=elevation<15):
            raise ValueError("not low view below15deg")
        if rec["cohort"]["supported_side_count"] == 0:
            raise ValueError("no supported side roles")
        return rec
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        reason = str(exc)
        audit["rejected_reasons"][reason.split(" (")[0]] += 1
        if len(audit["rejection_examples"])<25:
            audit["rejection_examples"].append({"image":str(image),"reason":reason})
        return None


def load_low_pool(audit):
    candidates = []
    audit["source_record_files"] = []
    if (BROAD/"records.jsonl").is_file():
        audit["source_record_files"].append({"path":str(BROAD/"records.jsonl"),"sha256":sha(BROAD/"records.jsonl")})
        with (BROAD/"records.jsonl").open() as stream:
            for index,line in enumerate(stream):
                meta = json.loads(line)
                if meta.get("elev_actual",90)>=15:
                    continue
                shard = meta.get("_src_shard") or meta.get("_src_root")
                if not shard or not re.fullmatch(r"v2_(?:prod10k[2345]|replace)_s\d+_public",shard):
                    audit["rejected_reasons"]["missing authoritative renderer group"] += 1
                    continue
                # Current merged file order, independently verified by world vs
                # renderer elevation, not the obsolete Windows label_path.
                rec = candidate(meta,BROAD,f"f{index:04d}","broad40k",f"broad40k:{shard}",audit)
                if rec is not None:
                    candidates.append(rec)
    for name in ("corner_la_oblique_v1_y15_30","corner_la_oblique_v1_y30_plus"):
        directory = OBLIQUE/name
        if not (directory/"records.jsonl").is_file():
            continue
        audit["source_record_files"].append({"path":str(directory/"records.jsonl"),"sha256":sha(directory/"records.jsonl")})
        for line in (directory/"records.jsonl").open():
            meta = json.loads(line)
            if "shard" not in meta or meta.get("revision") != "CORNER_LA_OBLIQUE_V1":
                audit["rejected_reasons"]["missing oblique renderer shard"] += 1
                continue
            rec = candidate(meta,directory,meta["file_id"],"oblique",f"oblique_CLAO1:shard_{meta['shard']:03d}",audit)
            if rec is not None:
                candidates.append(rec)
    return candidates


def select_cells(candidates,n,seed,used_hashes,audit):
    queues = collections.defaultdict(list)
    for rec in candidates:
        c = rec["cohort"]
        queues[(c["elevation_bin"],c["side_bin"],c["size_bin"],rec["source_family"])].append(rec)
    for key in queues:
        queues[key].sort(key=lambda r:order_key(seed,r["id"]),reverse=True)
    chosen = []
    keys = sorted(queues,key=lambda x:order_key(seed,repr(x)))
    while len(chosen)<n and any(queues.values()):
        for key in keys:
            while queues[key]:
                rec = queues[key].pop()
                image_hash = sha(rec["image"])
                if image_hash in used_hashes:
                    audit["selected_candidate_image_hash_duplicates_excluded"] += 1
                    continue
                used_hashes.add(image_hash)
                rec["image_sha256"] = image_hash
                chosen.append(rec)
                break
            if len(chosen)>=n:
                break
    if len(chosen)!=n:
        raise RuntimeError(f"Eligible group-separated low-view pool supplies only{len(chosen)}/{n}")
    return chosen


def build_manifest(run_dir:Path,source_manifest:Path=DEFAULT_SOURCE,seed:int=17):
    run_dir,source_manifest = Path(run_dir).resolve(),Path(source_manifest).resolve()
    run_dir.mkdir(parents=True,exist_ok=True)
    dest,audit_path = run_dir/"manifest.json",run_dir/"DATA_AUDIT.json"
    if dest.exists() or audit_path.exists():
        if not dest.exists() or not audit_path.exists():
            raise RuntimeError("Partial frozen cohort artifacts; inspect before rebuilding")
        audit = json.loads(audit_path.read_text())
        if audit["manifest_sha256"]!=sha(dest) or audit["source_manifest_sha256"]!=sha(source_manifest) or audit["split_seed"]!=seed:
            raise RuntimeError("Frozen cohort provenance/configuration differs")
        return json.loads(dest.read_text())
    source = json.loads(source_manifest.read_text())
    records,populations = [],{}
    source_hashes = set()
    for population,old_rows in source["populations"].items():
        populations[population] = []
        for row in old_rows:
            rec = dict(row,population=population,index=len(records))
            annotation = json.loads(Path(rec["annotation"]).read_text())
            for key in ("image","annotation"):
                if sha(rec[key])!=rec[f"{key}_sha256"]:
                    raise RuntimeError(f"Source {key} hash changed: {rec['id']}")
            if rec["image_sha256"] in source_hashes:
                raise RuntimeError("Inherited populations have duplicate images")
            source_hashes.add(rec["image_sha256"])
            if population == "real_dev" and annotation["objects"][0].get("split")!="eval":
                raise RuntimeError("Non-eval real annotation in source DEV")
            rec["source_family"] = "inherited_real_dev" if population=="real_dev" else "inherited_synthetic"
            rec["cohort"] = cohort(rec,annotation)
            records.append(rec);populations[population].append(rec["index"])
    audit = {"schema":"dht_padding_view_data_audit_v1","split_seed":seed,"source_manifest":str(source_manifest),
             "source_manifest_sha256":sha(source_manifest),"inherited_record_count":len(records),
             "rejected_reasons":collections.Counter(),"rejection_examples":[],
             "max_world_metadata_elevation_difference_deg":0.,"selected_candidate_image_hash_duplicates_excluded":0}
    low = load_low_pool(audit)
    if not low:
        raise RuntimeError("No eligible current low-angle source; base-only artifacts not silently substituted")
    groups_by_family = collections.defaultdict(set)
    for r in low:
        groups_by_family[r["source_family"]].add(r["group"])
    split_of,group_split = {},{}
    for family,groups in groups_by_family.items():
        ordered = sorted(groups,key=lambda g:order_key(seed,g))
        if len(ordered)<3:
            raise RuntimeError(f"Insufficient renderer groups for {family}")
        n_eval = max(1,round(.1*len(ordered)))
        group_split[family] = {}
        for i,g in enumerate(ordered):
            partition = "low_val" if i<n_eval else "low_test" if i<2*n_eval else "low_train"
            split_of[g] = partition;group_split[family][g] = partition
    audit["pool_eligible_counts_by_family"] = dict(collections.Counter(r["source_family"] for r in low))
    audit["pool_eligible_cohort_counts"] = cohort_counts(low)
    audit["pool_eligible_true_frontal_yaw_counts"] = {
        family:{"below15deg":sum(r["cohort"]["frontal_yaw_deg_world"]<15 for r in low if r["source_family"]==family),
                "total":sum(r["source_family"]==family for r in low)} for family in groups_by_family}
    audit["renderer_group_split"] = group_split
    audit["pool_partition_counts"] = dict(collections.Counter(split_of[r["group"]] for r in low))
    used_hashes = set(source_hashes)
    for population,count in (("low_val",128),("low_test",256),("low_train",1024)):
        pool = [r for r in low if split_of[r["group"]]==population]
        chosen = select_cells(pool,count,seed,used_hashes,audit)
        populations[population] = []
        for r in chosen:
            r["population"],r["index"] = population,len(records)
            records.append(r);populations[population].append(r["index"])
    base = populations["synth_train"]
    if len(base)!=2048:
        raise RuntimeError("Matched design requires inherited2048 training images")
    half_base = sorted(base,key=lambda i:order_key(seed,records[i]["id"]))[:1024]
    regimes = {"base":list(base),"low_balanced":half_base+populations["low_train"]}
    eval_indices = {i for pop,idx in populations.items() if pop not in ("synth_train","low_train") for i in idx}
    if any(set(idx)&eval_indices for idx in regimes.values()):
        raise RuntimeError("Evaluation index entered a training regime")
    inherited_synthetic_groups = collections.defaultdict(set)
    for pop,idx in populations.items():
        if pop!="real_dev":
            for i in idx:
                inherited_synthetic_groups[records[i]["group"]].add(pop)
    collisions = {g:sorted(v) for g,v in inherited_synthetic_groups.items() if len(v)>1}
    if collisions:
        raise RuntimeError(f"Renderer group split overlap: {collisions}")
    manifest = {"schema":"dht_padding_view_manifest_v1","source_manifest_sha256":audit["source_manifest_sha256"],
                "split_seed":seed,"records":records,"populations":populations,"train_regimes":regimes}
    audit.update({"counts":{"populations":{k:len(v) for k,v in populations.items()},"train_regimes":{k:len(v) for k,v in regimes.items()}},
                  "cohort_bins":BINS,"cohort_counts":{p:cohort_counts([records[i] for i in idx]) for p,idx in populations.items()},
                  "training_regime_cohorts":{p:cohort_counts([records[i] for i in idx]) for p,idx in regimes.items()},
                  "selection":f"Split entire renderer invocation groups by family with fixed SHAseed{seed} (approximately80/10/10); cyclic equal sampling across nonempty elevation×side×size×source-family cells, fixed SHA order, then deduplicate against all inherited and selected image hashes. Validation/test chosen before train.",
                  "group_definition":"Broad current-clean source _src_shard or explicit replacement _src_root; oblique revision+shard across both yaw packages and all workers. Never an individual-frame random split.",
                  "cohort_definitions":{"elevation":"Synthetic camera ray from world cuboid center relative to pallet top normal; bins<5,5–15,15–30,>=30deg. Real unknown; no PnP fit.",
                                        "side":"max(camera-facing side projected area,0)/abs(front area):<=.1,.1–.3,>.3; 2Dproxy, not yaw degrees.",
                                        "top":"max(camera-facing top projected area,0)/abs(front area):<=1,1–4,>4; 2Dproxy, not elevation degrees.",
                                        "size":"mean front upper/lower projected width / image width:<.25,.25–.5,.5–1,>=1.",
                                        "support":"Valid cuboid segment>=2px intersecting original frame. Physical edge visibility unknown."},
                  "integrity":{"source_all_hashes_revalidated":True,"all_selected_image_hashes_unique":len(used_hashes)==len(records),
                               "renderer_group_cross_population_collisions":collisions,"no_eval_indices_in_training":True,
                               "real_scope":"Inherited canonical manual eval DEV52 only; no other real populations accessed.",
                               "gt_semantics":"Original projected cuboid copied without rewriting or clamping; index/cohort/provenance metadata added."},
                  "limitations":["DATA_MIX comparison: low-view mix also changes renderer release, assets, backgrounds, materials, resolution and truncation; cannot isolate viewpoint causality.",
                                  "Existing oblique pool was used in a previous YOLO low-angle-diversity experiment; it is not newly rendered or previously unseen project data.",
                                  "Inherited synthetic backbone pretraining manifest unavailable; exact pretraining overlap with old/new synthetic eval unverified.",
                                  "Real cohort angles are unknown; 2Dprojected proxies cannot be relabelled calibrated elevations.",
                                  "Empty and sparse view cells are reported, never claimed tested or filled by duplicated frames.",
                                  "Only eligible low-view source frames are admitted; high-view cells remain in inherited populations."]})
    write_json(dest,manifest)
    audit["manifest_sha256"] = sha(dest)
    audit["builder_sha256"] = sha(Path(__file__))
    audit["PASS"] = True
    write_json(audit_path,audit)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--source-manifest",type=Path,default=DEFAULT_SOURCE)
    parser.add_argument("--seed",type=int,default=17)
    args = parser.parse_args()
    result = build_manifest(args.run_dir,args.source_manifest,args.seed)
    print(json.dumps({"records":len(result["records"]),"populations":{k:len(v) for k,v in result["populations"].items()},
                      "train_regimes":{k:len(v) for k,v in result["train_regimes"].items()}},indent=2))
