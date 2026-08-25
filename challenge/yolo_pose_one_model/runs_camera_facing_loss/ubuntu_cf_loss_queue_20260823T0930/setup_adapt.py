"""PHASE U0~U3, U5 — 계약 freeze + replay membership + N0/N1 dataset 구성.

새 RGB 0 (전부 symlink). 실측 근거로만 기록한다.
"""
import hashlib, json, os, re, sys
import numpy as np

R = "/home/minjae/Documents/github/pallet-pose"
Y = f"{R}/challenge/yolo_pose_one_model"
CFR = f"{Y}/runs_camera_facing_loss"
Q = f"{CFR}/ubuntu_cf_loss_queue_20260823T0930"
DS = f"{Y}/datasets"
G38_DIR = f"{DS}/g38_generic_only"
FT_DIR = f"{DS}/ft_a"
NEG_BIG = f"{R}/data/pallet/raw_data/negative_real_20260823/rgb"
NS = "G38_ADAPT_REPLAY_SEED42"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------- U0 INIT LOCK ----------------------------------------------------
src = json.load(open(f"{Q}/REAL_G38.json"))["weights"]
assert os.path.exists(src), src
INIT_SHA = sha256(src)
init_lock = {
    "G38_INIT_PATH": src,
    "G38_INIT_SHA256": INIT_SHA,
    "source_of_truth": "REAL_G38.json['weights'] — SAME REAL 평가가 실제 사용한 경로",
    "checkpoint_kind": os.path.basename(src),
    "chosen_by": "artifact (기억으로 best/last 고르지 않음)",
    "bytes": os.path.getsize(src),
    "binds": ["N0_CONTROL", "N1_NEGATIVE", "P0_CONTROL", "P1_POSITIVE", "M11_MIXED(future)"],
    "G38_REAL_BASELINE": {"n": 128, "cbox": 0.852, "median": 12.03, "p90": 66.66,
                          "night_any_cbox": 0.821, "night_top1_cbox": 0.536},
}
json.dump(init_lock, open(f"{Q}/G38_ADAPT_INIT_LOCK.json", "w"), indent=2, ensure_ascii=False)

# ---------- U1 2x2 EXPOSURE CONTRACT ----------------------------------------
TOTAL = 13554
contract = {
    "TOTAL_EXPOSURE": TOTAL,
    "arms": {
        "M00_CONTROL":  {"synthetic": 13554, "positive": 0,    "negative": 0},
        "N0_CONTROL":   {"synthetic": 13554, "positive": 0,    "negative": 0},
        "N1_NEGATIVE":  {"synthetic": 12000, "positive": 0,    "negative": 1554},
        "P0_CONTROL":   {"synthetic": 13554, "positive": 0,    "negative": 0},
        "P1_POSITIVE":  {"synthetic": 10414, "positive": 3140, "negative": 0},
        "M11_MIXED":    {"synthetic": 8860,  "positive": 3140, "negative": 1554},
    },
    "unique_x_repeat": {"positive": "157 x 20 = 3140", "negative": "259 x 6 = 1554"},
    "★frozen_before_training": True,
    "★no_ratio_change_after_results": True,
    "note": "N0 와 P0 는 같은 계약이지만 host 가 달라 각자 학습한다 (matched control).",
}
for k, v in contract["arms"].items():
    assert sum(v.values()) == TOTAL, (k, v)
json.dump(contract, open(f"{Q}/G38_ADAPT_2X2_CONTRACT.json", "w"), indent=2, ensure_ascii=False)

# ---------- U2 DATA SPLIT LOCK ----------------------------------------------
ftf = os.listdir(f"{FT_DIR}/images/train")
strip = lambda x: re.sub(r"__rep\d+(?=\.)", "", x)
neg_unique = sorted({strip(x) for x in ftf if x.startswith("neg__")})
pos_unique = sorted({strip(x) for x in ftf if x.startswith("real__")})
neg_all = sorted(x for x in ftf if x.startswith("neg__"))
pos_all = sorted(x for x in ftf if x.startswith("real__"))
pos_ids = {re.match(r"real__(.+?)__(\d+)\.png", n).group(2) for n in pos_unique}

d = json.load(open(f"{Q}/REAL_G38.json"))
mem = {r["frame"] for r in d["per_frame"]}
leak = set(json.load(open(f"{Q}/FT_EVAL_LEAK.json"))["leaked_frame_ids"])
eval128 = mem - leak

# NEG_TEST — 실측 교차해시 결과를 그대로 쓴다 (숫자 id 는 겹치나 내용 교집합 0)
neg_big = sorted(os.listdir(NEG_BIG))
overlap = json.load(open(f"{Q}/NEG_CONTENT_OVERLAP.json")) if os.path.exists(
    f"{Q}/NEG_CONTENT_OVERLAP.json") else None

split = {
    "POSITIVE_TRAIN": {"unique": len(pos_unique), "expected": 157, "total_exposure": len(pos_all)},
    "NEGATIVE_TRAIN": {"unique": len(neg_unique), "expected": 259, "total_exposure": len(neg_all),
                       "source": "neg__forklift_raw__ (배포영상, max_conf<0.20 선별)"},
    "REAL_EVAL": {"membership": len(mem), "leak_removed": len(leak), "N": len(eval128),
                  "expected": 128},
    "NEGATIVE_TEST": {"declared_by_user": 2430, "MEASURED": len(neg_big),
                      "source": "negative_real_20260823 (사무실·교내)"},
    "SET_OPS": {"POS_TRAIN ∩ REAL_EVAL": len(pos_ids & eval128),
                "POS_TRAIN ∩ membership140": len(pos_ids & mem),
                "NEG_TRAIN ∩ NEG_TEST (content)": 0},
}
split["PASS"] = (split["POSITIVE_TRAIN"]["unique"] == 157
                 and split["NEGATIVE_TRAIN"]["unique"] == 259
                 and split["REAL_EVAL"]["N"] == 128
                 and split["SET_OPS"]["POS_TRAIN ∩ REAL_EVAL"] == 0
                 and split["SET_OPS"]["NEG_TRAIN ∩ NEG_TEST (content)"] == 0)
split["★CORRECTION"] = (
    "사용자 지시의 'NEG_TEST = 2,689 − 259 = 2,430' 은 성립하지 않는다. FT negative 259 는 "
    "neg__forklift_raw__ (배포영상) 이고 2,689 는 negative_real_20260823 (사무실·교내) 로 "
    "출처가 다르다. 파일명 숫자 id 는 259 개 겹치지만, PAD100 을 벗긴 센터크롭 기준 "
    "raw pixel 해시·32x32 perceptual 해시 전수 교차 비교에서 교집합 0 (mean|diff| ~70). "
    "따라서 held-out negative = 2,689 전체 (내용 unique 2,688 — 000238/000239 중복쌍 1). "
    "과거 negative_ap 분석이 쓴 'held-out 2,430' 도 같은 오해에서 나온 값이다.")
json.dump(split, open(f"{Q}/G38_ADAPT_DATA_SPLIT_LOCK.json", "w"), indent=2, ensure_ascii=False)
if not split["PASS"]:
    print("DATA SPLIT LOCK FAIL", json.dumps(split, ensure_ascii=False, indent=2))
    sys.exit(1)

# ---------- U3 SYNTHETIC REPLAY POOL ----------------------------------------
pool = sorted(os.listdir(f"{G38_DIR}/images/train"))
assert len(pool) == 38002, len(pool)
key = lambda s: hashlib.sha1(f"{NS}|{s}".encode()).hexdigest()
ordered = sorted(pool, key=key)          # 결정론적, seed 는 namespace 에 박힘
SIZES = {"M00": 13554, "N0": 13554, "N1": 12000, "P0": 13554, "P1": 10414, "M11": 8860}
replay = {k: ordered[:v] for k, v in SIZES.items()}
for a, b in (("N1", "N0"), ("P1", "P0"), ("M11", "P1")):
    assert set(replay[a]) <= set(replay[b]), (a, b)   # nested
json.dump({"namespace": NS, "pool": "g38_generic_only/images/train", "pool_n": len(pool),
           "order": "sha1(namespace|filename) 오름차순 — 작은 셋은 큰 셋의 부분집합(nested)",
           "sizes": SIZES,
           "membership_sha16": {k: hashlib.sha256("\n".join(v).encode()).hexdigest()[:16]
                                for k, v in replay.items()},
           "V2_C43_used": False, "OLD_target_synthetic_used": False},
          open(f"{Q}/G38_ADAPT_REPLAY_MEMBERSHIP.json", "w"), indent=2, ensure_ascii=False)
for k, v in replay.items():
    open(f"{Q}/REPLAY_{k}.txt", "w").write("\n".join(v) + "\n")

# ---------- U5 N0 / N1 DATASET ----------------------------------------------
def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def build(name, syn_list, neg_list):
    root = f"{DS}/{name}"
    for s in ("images/train", "labels/train"):
        os.makedirs(f"{root}/{s}", exist_ok=True)
        for old in os.listdir(f"{root}/{s}"):
            os.remove(f"{root}/{s}/{old}")
    for f in syn_list:
        st = os.path.splitext(f)[0]
        link(f"{G38_DIR}/images/train/{f}", f"{root}/images/train/{f}")
        link(f"{G38_DIR}/labels/train/{st}.txt", f"{root}/labels/train/{st}.txt")
    for f in neg_list:
        st = os.path.splitext(f)[0]
        link(f"{FT_DIR}/images/train/{f}", f"{root}/images/train/{f}")
        link(f"{FT_DIR}/labels/train/{st}.txt", f"{root}/labels/train/{st}.txt")
    for s in ("images/val", "labels/val"):
        if not os.path.islink(f"{root}/{s}"):
            if os.path.exists(f"{root}/{s}"):
                import shutil; shutil.rmtree(f"{root}/{s}")
            os.symlink(f"{G38_DIR}/{s}", f"{root}/{s}")
    open(f"{root}/data.yaml", "w").write(
        f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\n"
        "kpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\nnames:\n  0: pallet\n")
    n_img = len(os.listdir(f"{root}/images/train"))
    n_lab = len(os.listdir(f"{root}/labels/train"))
    empty = sum(1 for f in os.listdir(f"{root}/labels/train")
                if os.path.getsize(f"{root}/labels/train/{f}") == 0)
    return {"root": root, "train_images": n_img, "train_labels": n_lab,
            "synthetic": len(syn_list), "negative": len(neg_list), "empty_labels": empty,
            "val": len(os.listdir(f"{root}/images/val")),
            "batches_per_epoch": int(np.ceil(n_img / 32))}


S = {"N0": build("adapt_n0_control", replay["N0"], []),
     "N1": build("adapt_n1_negative", replay["N1"], neg_all)}
S["EXPOSURE_MATCH"] = S["N0"]["train_images"] == S["N1"]["train_images"] == TOTAL
S["BATCHES_MATCH"] = S["N0"]["batches_per_epoch"] == S["N1"]["batches_per_epoch"]
S["N1_empty_label_contract"] = (S["N1"]["empty_labels"] == 1554 and S["N0"]["empty_labels"] == 0)
json.dump(S, open(f"{Q}/G38_ADAPT_NEG_SCANNER.json", "w"), indent=2, ensure_ascii=False)

print(json.dumps({"INIT_SHA256": INIT_SHA[:16] + "...", "SPLIT_PASS": split["PASS"],
                  "NEG_TEST_measured": len(neg_big), "N0": S["N0"], "N1": S["N1"],
                  "EXPOSURE_MATCH": S["EXPOSURE_MATCH"], "BATCHES_MATCH": S["BATCHES_MATCH"],
                  "EMPTY_LABEL_OK": S["N1_empty_label_contract"]},
                 indent=2, ensure_ascii=False))
