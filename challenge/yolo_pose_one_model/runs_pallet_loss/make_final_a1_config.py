"""GLB symmetry contract 도착 시 FINAL A1 config 를 생성한다.

사용:  python make_final_a1_config.py --contract <ASSET_SYMMETRY_FINAL.json>

contract 는 asset -> {SYM180_EQUIVALENT | ASYM180_DISTINGUISHABLE} 매핑이어야 한다.
UNRESOLVED 가 하나라도 남으면 생성하지 않고 종료한다 (임의 분류 금지).

★ A1_USD_ONLY checkpoint 를 resume 하지 않는다.  FINAL A1 은 항상 clean pretrained
  init + seed42 + 60ep 으로 새로 학습한다.
★ role term 의 margin 은 여기서 임의로 정하지 않는다.  ASYM 집합이 확정된 뒤
  그 집합에서만 sep 분포를 재서 정한다 (활성 빈도로 항의 존폐를 판단하지 않는다).
"""
import argparse, json, os

R = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--contract", required=True)
ap.add_argument("--margin", type=float, default=None,
                help="ASYM 집합 확정 후 별도 캘리브레이션으로 정한 값")
ap.add_argument("--lambda-role", type=float, default=None)
A = ap.parse_args()

c = json.load(open(A.contract))
assets = c.get("assets", c)
sym = sorted(k for k, v in assets.items() if v == "SYM180_EQUIVALENT")
asym = sorted(k for k, v in assets.items() if v == "ASYM180_DISTINGUISHABLE")
unres = sorted(k for k, v in assets.items() if k not in sym and k not in asym)

if unres:
    raise SystemExit(f"UNRESOLVED 남음 {unres} — CASE 3 BLOCK. 임의 분류 금지.")

case = "CASE_ALL_SYM" if not asym else "CASE_MIXED"
if case == "CASE_MIXED" and (A.margin is None or A.lambda_role is None):
    raise SystemExit(
        "CASE_MIXED 인데 margin/lambda_role 미지정.\n"
        "ASYM 집합에서만 sep 분포를 재서 정한 뒤 --margin --lambda-role 로 넘길 것.\n"
        "활성 빈도가 낮다는 이유로 role term 을 빼지 않는다 (빈도 != 중요도).")

cfg = dict(
    enabled=True, mode="exact_min", softmin_tau=0.0,
    lambda_role=0.0 if case == "CASE_ALL_SYM" else A.lambda_role,
    margin=0.0 if case == "CASE_ALL_SYM" else A.margin,
    p180=[5, 4, 7, 6, 1, 0, 3, 2], centroid_index=8,
    sym_assets=sym, asym_assets=asym,
    stem_asset_map=f"{R}/STEM_ASSET_MAP.json",
    role_ramp=[5, 20])
json.dump(cfg, open(f"{R}/A1_FINAL_CONFIG.json", "w"), indent=2)
json.dump({"case": case, "sym": sym, "asym": asym, "contract": A.contract,
           "init": "clean pretrained yolo26n-pose.pt (resume 금지)",
           "seed": 42, "epochs": 60,
           "run_name": f"PSPC_A1_FINAL_{case}_V1MATCHED10K_60EP_SEED42"},
          open(f"{R}/A1_FINAL_PROVENANCE.json", "w"), indent=2)
print(f"{case}\n  SYM  {sym}\n  ASYM {asym}\n  -> A1_FINAL_CONFIG.json")
print("  다음: a1_driver.py 의 NAME/A1_CONFIG 를 FINAL 로 바꿔 새로 학습 (resume 금지)")
