"""PHASE U4 — Windows POSITIVE screen handoff package. CPU only, GPU 와 병렬."""
import hashlib, json, os, re, shutil, subprocess, time

R = "/home/minjae/Documents/github/pallet-pose"
Y = f"{R}/challenge/yolo_pose_one_model"
Q = f"{Y}/runs_camera_facing_loss/ubuntu_cf_loss_queue_20260823T0930"
FT = f"{Y}/datasets/ft_a"
STAMP = time.strftime("%Y%m%dT%H%M%S")
NAME = f"G38_POSITIVE_SCREEN_TO_WINDOWS_{STAMP}"
OUT = f"/home/minjae/Desktop/{NAME}"
DESK = "/home/minjae/Desktop"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for s in ("weights", "positive_train/images", "positive_train/labels", "manifests",
          "contracts", "configs"):
    os.makedirs(f"{OUT}/{s}", exist_ok=True)

# 1) init checkpoint
LOCK = json.load(open(f"{Q}/G38_ADAPT_INIT_LOCK.json"))
shutil.copy2(LOCK["G38_INIT_PATH"], f"{OUT}/weights/G38_INIT.pt")
assert sha256(f"{OUT}/weights/G38_INIT.pt") == LOCK["G38_INIT_SHA256"], "init sha 불일치"

# 2) positive 157 unique + labels
ftf = os.listdir(f"{FT}/images/train")
strip = lambda x: re.sub(r"__rep\d+(?=\.)", "", x)
pos_unique = sorted({strip(x) for x in ftf if x.startswith("real__")})
pos_all = sorted(x for x in ftf if x.startswith("real__"))
assert len(pos_unique) == 157 and len(pos_all) == 3140, (len(pos_unique), len(pos_all))
for n in pos_unique:
    st = os.path.splitext(n)[0]
    shutil.copy2(f"{FT}/images/train/{n}", f"{OUT}/positive_train/images/{n}")
    shutil.copy2(f"{FT}/labels/train/{st}.txt", f"{OUT}/positive_train/labels/{st}.txt")
open(f"{OUT}/manifests/POSITIVE_UNIQUE.txt", "w").write("\n".join(pos_unique) + "\n")

# 3) exposure manifest — 157 x 20 = 3140 (파일 복제 없이 repeat 지시)
json.dump({"unique": pos_unique, "repeat": 20, "total": 3140,
           "build_rule": "각 unique 파일을 20 회 노출. 물리 복제든 sampler 반복이든 "
                         "총 노출 3,140 이면 계약 충족.",
           "source_of_truth": "ft_a/images/train real__*__repN (원본 FT 구성과 동일)"},
          open(f"{OUT}/manifests/POSITIVE_EXPOSURE.json", "w"), indent=2, ensure_ascii=False)

# 4) replay membership (ID 만 — Windows 는 자기 G38 generic pool 에서 해석)
RM = json.load(open(f"{Q}/G38_ADAPT_REPLAY_MEMBERSHIP.json"))
G38T = f"{Y}/datasets/g38_generic_only/images/train"
for arm, n_syn, n_pos in (("P0", 13554, 0), ("P1", 10414, 3140)):
    ids = open(f"{Q}/REPLAY_{arm}.txt").read().split()
    assert len(ids) == n_syn, (arm, len(ids))
    # 신원 확인용 spot-check (전수 아님 — 결정론적 200 개)
    spot = ids[::max(1, len(ids) // 200)][:200]
    json.dump({"arm": arm, "synthetic_ids": ids, "n_synthetic": n_syn,
               "n_positive_exposure": n_pos, "total_exposure": n_syn + n_pos,
               "namespace": RM["namespace"], "order_rule": RM["order"],
               "membership_sha16": RM["membership_sha16"][arm],
               "SPOT_CHECK_NOT_CENSUS": {
                   "n": len(spot),
                   "sha256": {f: sha256(f"{G38T}/{f}") for f in spot}},
               "★pool_source": "OLD generic pool (G38). V2/C43 및 OLD target synthetic 사용 금지."},
              open(f"{OUT}/manifests/{arm}_REPLAY_MANIFEST.json", "w"), indent=2, ensure_ascii=False)

# 5) contracts
for f in ("G38_ADAPT_INIT_LOCK.json", "G38_ADAPT_2X2_CONTRACT.json",
          "G38_ADAPT_DATA_SPLIT_LOCK.json", "NEG_SCREEN_GATE_PREREG.json"):
    shutil.copy2(f"{Q}/{f}", f"{OUT}/contracts/{f}")

# 6) recipe
open(f"{OUT}/configs/ADAPT_FAST_RECIPE.yaml", "w").write("""\
# ADAPT FAST SCREEN recipe — OLD YOLO26n-ft 실제 args 를 source of truth 로 함
# (runs_ft/ft_a_real157_neg259_synth12k/args.yaml 에서 확인)
# ★ P0 와 P1 은 data 이외 단 하나도 달라선 안 된다.
model: weights/G38_INIT.pt      # exact G38 evaluation checkpoint
task: pose
epochs: 15                      # FAST SCREEN (OLD FT 는 40 이었다)
patience: 0                     # early stop 금지
batch: 32
imgsz: 640
optimizer: SGD
lr0: 0.002
lrf: 0.01
cos_lr: true
warmup_epochs: 1.0
mosaic: 0.15
close_mosaic: 10
scale: 0.25
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.35
fliplr: 0.0
flipud: 0.0
erasing: 0.4
seed: 42
deterministic: true
single_cls: true
save_period: 5
resume: false
val: true
plots: false
# loss: STANDARD PoseLoss26 only — KDM/NRL/PEVL/projective 금지
# primary checkpoint: epoch15 last.pt
""")

# 7) README
open(f"{OUT}/README_WINDOWS_POS_SCREEN.md", "w").write(f"""\
# WINDOWS POSITIVE SUPERVISION SCREEN (P0 vs P1)

Ubuntu 는 같은 시간에 NEGATIVE 축(N0/N1)을 돌린다. **평가는 Ubuntu 가 전담**한다.

## 할 일

두 arm 을 학습해서 `last.pt` 두 개를 돌려보내면 끝이다.

```
P0_CONTROL    synthetic 13,554   positive 0
P1_POSITIVE   synthetic 10,414   positive 3,140  (157 unique x 20)
                                  총 노출 둘 다 13,554
```

`configs/ADAPT_FAST_RECIPE.yaml` 을 그대로 쓴다. **data 이외에는 한 글자도 바꾸지 않는다.**
init 은 반드시 `weights/G38_INIT.pt` — sha256 `{LOCK['G38_INIT_SHA256']}`.

## synthetic replay 를 어디서 가져오나

이 패키지에는 synthetic RGB 가 들어있지 않다(용량). `manifests/P0_REPLAY_MANIFEST.json`
과 `P1_REPLAY_MANIFEST.json` 의 `synthetic_ids` 를 **Windows 가 이미 가진 OLD generic
pool (G38)** 에서 해석한다.

- pool 은 38,002 장짜리 generic-only. V2/C43 dataset, OLD target synthetic 사용 금지.
- 파일명이 그대로 키다 (`G__fXXXX.png` + 같은 stem 의 `.txt`).
- 각 manifest 의 `SPOT_CHECK_NOT_CENSUS.sha256` 200 개로 pool 동일성을 먼저 확인한다.
  하나라도 어긋나면 **학습하지 말고 알린다** — 다른 pool 이면 P0/P1 대조가 무의미하다.
- `membership_sha16` 은 id 목록 자체의 해시다. 목록을 재정렬하거나 잘라내지 않는다.

## positive 데이터

`positive_train/images` 157 장 + `positive_train/labels` 157 개.
P1 은 이걸 **20 회 노출**한다 (물리 복제든 sampler 반복이든 총 3,140 이면 된다).
라벨은 camera-facing 9kp YOLO-pose 포맷, 좌표는 PAD100 캔버스 기준으로 이미 정규화돼 있다.

## 절대 하지 말 것

- real evaluation 금지. 이 패키지에 평가셋 128 장은 **의도적으로 없다**.
  Windows 가 산출한 real metric 은 사용하지 않는다.
- threshold tuning 금지, 결과 보고 exposure 비율 변경 금지.
- self-training / pseudo-label 금지. KDM/NRL/PEVL 금지. 새 render·새 synthetic 금지.
- P0 를 건너뛰고 P1 만 돌리지 말 것 — control 이 없으면 DELTA 를 못 만든다.

## 돌려보낼 것

```
P0/  last.pt  args.yaml  results.csv  RUNTIME_AUDIT.json(있으면)
P1/  last.pt  args.yaml  results.csv  RUNTIME_AUDIT.json(있으면)
사용한 replay manifest 사본 + SHA256SUMS
```

Ubuntu 가 SAME REAL n=128 / NEG held-out 2,689 에서 직접 평가하고,
`DELTA_POS = P1 - P0` 를 `DELTA_NEG = N1 - N0` 와 합쳐 routing 한다.

★ 두 host 의 **absolute 비교는 금지**다. 각 host 의 matched control 대비 DELTA 만 본다.

생성: {time.strftime('%F %T')}  (Ubuntu)
""")

# 8) SHA256SUMS
lines = []
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        if f == "SHA256SUMS.txt":
            continue
        p = os.path.join(root, f)
        lines.append(f"{sha256(p)}  {os.path.relpath(p, OUT)}")
open(f"{OUT}/SHA256SUMS.txt", "w").write("\n".join(sorted(lines, key=lambda x: x.split('  ')[1])) + "\n")

zp = f"{DESK}/{NAME}.zip"
subprocess.run(["zip", "-rq", zp, NAME], cwd=DESK, check=True)
zsha = sha256(zp)
open(f"{zp}.sha256.txt", "w").write(f"{zsha}  {NAME}.zip\n")

info = {"NAME": NAME, "ZIP": zp, "ZIP_SHA256": zsha,
        "zip_MB": round(os.path.getsize(zp) / 1e6, 2),
        "positive_unique": len(pos_unique), "positive_exposure": 3140,
        "init_sha256": LOCK["G38_INIT_SHA256"],
        "n_files": len(lines), "READY": True}
json.dump(info, open(f"{Q}/WINDOWS_POS_PACKAGE.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(info, indent=2, ensure_ascii=False))
