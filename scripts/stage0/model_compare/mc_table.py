"""MODEL_COMPARE_AUC.json 을 사람이 읽는 표로 낸다.

헤더에 방향 화살표를 박는다 — `↑` 는 클수록 좋고 `↓` 는 작을수록 좋다.  숫자만
있는 표는 어느 열이 어느 방향인지 매번 되짚어야 하고, `corner` 처럼 작을수록 좋은
열과 `IoU3D` 처럼 클수록 좋은 열이 섞여 있으면 실제로 잘못 읽힌다.

방향은 `mc_geom` / `re_metrics` 의 정의에서 나온 것이지 취향이 아니다.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/home/minjae/Documents/github/pallet-pose"
OUT = os.path.join(ROOT, "data/pallet/results/model_compare")

# (json key, 표시명, 방향, 폭, 소수자리)   방향 True = 클수록 좋다
COLUMNS = [
    ("pnp_rate",      "pnp",     True,  8, 3),
    ("corner_px_med", "corner",  False, 8, 2),
    ("R_deg_med",     "R med",   False, 8, 2),
    ("R_deg_p90",     "R p90",   False, 8, 1),
    ("yaw_deg_med",   "yaw med", False, 9, 2),
    ("yaw_deg_p90",   "yaw p90", False, 9, 1),
    ("t_m_med",       "t med",   False, 8, 3),
    ("ADD_S_med",     "ADD-S",   False, 8, 3),
    ("ADD_AUC",       "ADDauc",  True,  8, 4),
    ("ADD_S_AUC",     "ADDSauc", True,  9, 4),
    ("IoU3D_med",     "IoU3D",   True,  7, 3),
]
DETECTION = [
    ("AP",           "AP",     True,  8, 4),
    ("AUROC",        "AUROC",  True,  8, 4),
    ("FPR_at_TPR95", "FPR95",  False, 8, 4),
]
ORDER = ["yolo26n_synth", "yolo26n_paper_generic_v1", "Y0E", "YN",
         "yolo26n_broad40k_5ep", "FINAL40K_seed1", "yolo26n_ft", "yolo26m_ft"]

# 표시명만 바꾼다 — run 이름 / JSON 키 / kps_*.json 파일명은 그대로다.  이름에
# 넣는 것은 두 가지다: **무슨 데이터를 봤나**와 **타깃 파렛트를 봤나**.  두 번째가
# 이 비교의 최대 교란요인이라(정본 161 이 전부 타깃 파렛트다) 이름에서 바로
# 보이지 않으면 표가 "어느 모델이 더 좋다" 로 잘못 읽힌다.
LABELS = {
    "yolo26n_synth":            "n-SYN74K  타깃+G38",
    "yolo26n_paper_generic_v1": "n-GEN40K  generic만",
    "Y0E":                      "n-G38+반복9K",
    "YN":                       "n-G38+neg9K",
    "yolo26n_broad40k_5ep":     "n-GEN40K  5ep만",
    "FINAL40K_seed1":           "DOPE-GEN40K",
    "yolo26n_ft":               "n-SYN74K + realFT",
    "yolo26m_ft":               "m-SYN74K + realFT",
}
LEGEND = [
    "n / m         yolo26 nano / medium        DOPE = 별 계열(벨리프맵)",
    "SYN74K        합성 73,916 = G38 38,002 + 타깃 v1/v2 35,914  ★타깃 봄",
    "GEN40K        generic 만 (broad40k 39,500).  타깃 v1/v2 의도적 제외",
    "G38           generic 38,002 (+반복9K = positive 9,000 재노출,",
    "                              +neg9K  = synthetic negative 9,000)",
    "realFT        real 157장 + negative 259장 파인튜닝  ★타깃 봄",
    "",
    "★타깃 봄 = 평가셋(정본 161)의 파렛트를 학습에서 이미 봤다는 뜻.",
    "  같은 줄에 놓여 있어도 감독량이 다르다 — 순위를 그대로 읽지 말 것.",
]


def header(columns, first="model", first_width=22):
    cells = "".join(f"{name + ('↑' if up else '↓'):>{w}} "
                    for _k, name, up, w, _d in columns)
    line = f"{first:{first_width}}{cells}"
    return line, "─" * len(line)


def row(name, block, columns, first_width=22):
    cells = ""
    for key, _n, _up, w, dec in columns:
        v = block.get(key)
        cells += (f"{'-':>{w}} " if v is None
                  else f"{v:>{w}.{dec}f} ")
    return f"{name:{first_width}}{cells}"


def main():
    report = json.load(open(os.path.join(OUT, "MODEL_COMPARE_AUC.json")))
    models = report["models"]
    names = [n for n in ORDER if n in models] + \
            [n for n in models if n not in ORDER]

    print("↑ 클수록 좋음   ↓ 작을수록 좋음\n")
    for pop in ("OPEN_56", "SEALED_105", "ALL_161"):
        if not any(models[n].get(pop) for n in names):
            continue
        print(f"### {pop}")
        head, rule = header(COLUMNS)
        print(head)
        print(rule)
        for n in names:
            block = models[n].get(pop)
            if block:
                print(row(LABELS.get(n, n), block, COLUMNS))
        print()

    per_model = report.get("detection", {}).get("per_model") or {}
    if per_model:
        print("### DETECTION  (positive 161 vs real negative 2,689)")
        head, rule = header(DETECTION)
        print(head)
        print(rule)
        for n in names:
            if n in per_model:
                print(row(LABELS.get(n, n), per_model[n], DETECTION))
        skipped = [n for n in names if n not in per_model]
        if skipped:
            print(f"\n제외: {', '.join(LABELS.get(x, x) for x in skipped)} "
                  f"— {report['detection'].get('excluded', '')}")


    print("\n--- 이름 읽는 법 " + "-" * 60)
    for line in LEGEND:
        print(line)


if __name__ == "__main__":
    sys.exit(main())
