"""Publish the failed source-only objective control; no model computation."""
import base64
import csv
import hashlib
import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

P = Path(__file__).resolve().parents[1]
V66 = P.parent / "query_visibility_controlled_real_20261022_v66"


def bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path):
    return json.loads(path.read_text())


def pct(value):
    return "—" if value is None else f"{100*value:.1f}%"


def main():
    assert not (P / "RESULTS_KO.json").exists()
    admission_path = P / "source_admission/VISIBLE_BCE1/SOURCE_ADMISSION.json"
    admission = load(admission_path)
    assert admission["integrity_PASS"] and not admission["source_eligible"]
    assert admission["status"] == "HOLD" and not admission["current_inference_allowed"]
    freeze_path = P / "runs/VISIBLE_BCE1/MODEL_FREEZE.json"
    freeze = load(freeze_path)
    sources = [
        ("v65", V66 / "source_evaluation/V65_BASELINE/SOURCE_METRICS.json"),
        ("v66 PLUS", V66 / "source_evaluation/CONTROLLED_PLUS_PALLOX/SOURCE_METRICS.json"),
        ("v67 BCE1", P / "source_evaluation/VISIBLE_BCE1/SOURCE_METRICS.json"),
    ]
    rows = []
    for model, path in sources:
        report = load(path)
        for source, metrics in report["per_source"].items():
            exact = metrics["exact"]
            visible, external = exact["visible_binary_pV_ge_0_5"], exact["class_metrics"]["E"]
            rows.append({"model": model, "source": source, "known_GT": exact["known_GT"],
                         "UNKNOWN_GT": exact["UNKNOWN_GT"], **{k: visible[k] for k in ("TP", "FP", "FN", "TN")},
                         "visible_precision": visible["precision"], "visible_recall": visible["recall"],
                         "confident_E_TP": external["TP"], "E_support": external["GT_support"],
                         "confident_E_precision": external["precision"], "confident_E_recall": external["recall"]})
    csv_path = P / "SOURCE_COMPARISON.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gallery = V66 / "source_audit/SOURCE_VISIBLE_REGRESSION_CROPS_V3"
    selection_path = gallery / "SELECTION_AND_PIXEL_CHECK.json"
    examples = load(selection_path)
    with (P / "source_evaluation/VISIBLE_BCE1/PER_QUERY.csv").open() as handle:
        predictions = {r["sample_id"]: r for r in csv.DictReader(handle)}
    example_rows = []
    for example in examples["chosen"]:
        r = predictions[example["sample_id"]]
        example_rows.append({"sample_id": example["sample_id"],
                             "source_inside_distance_px": float(example["v66"]["visible_inside_distance_px"]),
                             "rawSAM_pV": float(example["v66"]["rawSAM_exact_pV"]),
                             "v65_pV": float(example["v65"]["exact_pV"]),
                             "v66_pV": float(example["v66"]["exact_pV"]),
                             "v67_pV": float(r["exact_pV"]), "v67_confident_state": r["exact_predicted"]})
    refs = [bind(p) for _, p in sources] + [bind(admission_path), bind(freeze_path), bind(P / "FIT_PROTOCOL.json"),
            bind(P / "SOURCE_CANDIDATE_SEAL.json"), bind(P / "provenance/FIT_INPUT_LOCK.json"),
            bind(P / "source_audit/INDEPENDENT_OBJECTIVE_CONTROL_REVIEW.json"),
            bind(P / "source_audit/INDEPENDENT_SOURCE_ADMISSION_CODE_REVIEW.json"),
            bind(V66 / "source_audit/SOURCE_LOSS_TRADEOFF_DIAGNOSIS_CONCLUSION.json"),
            bind(V66 / "source_audit/SOURCE_TRAINING_PRESSURE_AUDIT.json"), bind(selection_path), bind(csv_path), bind(Path(__file__))]
    report = {"schema": "V67_FAILED_SOURCE_ONLY_OBJECTIVE_CONTROL_KOREAN_REPORT",
              "created_at_KST": datetime.now(timezone(timedelta(hours=9))).isoformat(),
              "verdict": "HOLD", "solved": False, "current223_new_inference": 0,
              "current223_selection_or_fit": False, "new_PnP": 0,
              "changed_factor": "Partial NLL plus lambda1 known exact-visible/nonvisible marginal BCE; all original source inputs, architecture, seed, batches, budget and NLL selector unchanged.",
              "interpretation": "This specific extra objective did not preserve source-visible recall. It does not prove all loss formulations fail or identify a feature-branch cause.",
              "completed_head_updates": freeze["optimizer_updates"], "selected_update": freeze["best_update"],
              "selected_checkpoint": bind(P / "runs/VISIBLE_BCE1/BEST_PARTIAL_HEAD.pt"),
              "source_comparison": rows, "source_admission": admission,
              "fixed_prefit_source_examples": example_rows,
              "full_goal_limit": "Exact native query-pixel occupancy and human reference-keypoint/crop semantic visibility remain different targets. Source pixel success alone would not prove full goal achievement.",
              "next_work": "Frozen SAM2.1 base-plus source benchmark and independent flat-pallet public-data intake; no threshold/lambda/radius search on current223.",
              "references": refs}
    (P / "RESULTS_KO.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    body_rows = "".join(f"<tr><td>{html.escape(r['model'])}</td><td>{r['source']}</td><td>{r['known_GT']} / {r['UNKNOWN_GT']}</td><td>{pct(r['visible_precision'])}</td><td>{pct(r['visible_recall'])}</td><td>{pct(r['confident_E_precision'])}</td><td>{pct(r['confident_E_recall'])}</td></tr>" for r in rows)
    example_table = "".join(f"<tr><td>{r['sample_id']}</td><td>{r['source_inside_distance_px']:.2f}</td><td>{r['rawSAM_pV']:.3f}</td><td>{r['v65_pV']:.3f}</td><td>{r['v66_pV']:.3f}</td><td>{r['v67_pV']:.3f} ({r['v67_confident_state']})</td></tr>" for r in example_rows)
    failed = "".join(f"<li>{html.escape(c['source'])} · {html.escape(c['condition'])} · {html.escape(c['metric'])}: {c['value']:.4f} &lt; {c['minimum']:.4f}</li>" for c in admission["conditions"] if not c["passed"])
    links = "".join(f"<li><a href='{html.escape(r['path'],quote=True)}'>{html.escape(Path(r['path']).name)}</a></li>" for r in refs)
    encoded = base64.b64encode((gallery / "PUBLIC_VISIBLE_REGRESSIONS.png").read_bytes()).decode()
    document = f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>v67 소스 대조 실험</title>
<style>body{{font-family:'Noto Sans CJK KR',sans-serif;background:#f3f6fa;color:#213441;margin:0}}main{{max-width:1180px;margin:30px auto}}section,header{{background:white;padding:26px;margin:18px 0;border-radius:12px}}header{{background:#293a49;color:white}}h1{{font-size:29px}}h2{{font-size:22px}}p,li{{line-height:1.7}}table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{padding:10px;border-bottom:1px solid #dce3e9;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#eaf0f5}}.hold{{color:#ffb7ae;font-weight:bold}}.note{{padding:15px;background:#fff2df}}img{{width:100%;height:auto}}a{{color:#26739f}}</style>
<main><header><p class='hold'>HOLD · 목표 미달 · 현재 평가 재추론 0</p><h1>보임/비가시 직접 감독을 추가해도<br>공개 소스의 보이는 점 손실은 해결되지 않았습니다</h1>
<p>기존 v66의 입력·구조·배치·seed·2,000 update·소스 NLL 선택은 유지하고 BCE 한 항만 추가했습니다. 고정 소스 기준 16개 중 7개가 실패했습니다. 현재 223장으로 이 후보를 재평가하지 않았습니다.</p></header>
<section><h2>무엇을 확인했는지</h2><p>이 BCE는 보이는 점과 확정 비가시 점을 모두 재가중합니다. 양성만 보호하는 항이 아니며, 알려진 사실의 분모가 원래 partial NLL과 달라 전체 양성 가중치가 언제나 정확히 2배인 것도 아닙니다.</p>
<p>소스 손실 분해에서는 v65→v66의 보이는 점 손해가 다른 상태의 이득에 상쇄됐습니다. 그러나 균등 출력에서 라벨 전체가 단순히 VV를 억누르는 구조는 발견되지 않았습니다. 특정 학습 항을 추가한 대조 실험의 실패이며, 원인을 하나의 branch로 확정하지 않습니다.</p></section>
<section><h2>같은 소스 query의 exact 비교</h2><table><tr><th>모델</th><th>소스</th><th>known / unknown GT</th><th>보임 P</th><th>보임 R</th><th>확신 E P</th><th>확신 E R</th></tr>{body_rows}</table>
<p class='note'>모든 precision은 알려진 GT 부분집합에 대한 조건부 수치입니다. UNKNOWN을 음성으로 넣지 않았습니다. 확신 E는 기존 six-joint 최대확률 ≥0.5 기준이며, binary 비가시와 다릅니다. ORIGINAL_YCB에는 확정 E 정답이 없습니다.</p></section>
<section><h2>학습 전에 고정한 기준: 9개 통과 / 7개 실패</h2><ul>{failed}</ul><p>명확한 5×5 표면 내부에서도 원본 605/714, 제어 가림 241/296만 보임으로 판단했습니다. 이전 v65의 679/714, 278/296에 못 미칩니다. 다른 checkpoint나 문턱으로 다시 고르지 않았습니다.</p></section>
<section><h2>v67 학습 전에 고정한 실제 공개 crop</h2><p>아래 네 점은 모두 원본 YCB visible-mask의 5×5 내부이며, v65 보임→v66 비가시로 바뀐 점 중 고정 hash 순서로 골랐습니다. BOP 마스크는 render/depth 파생 라벨이며 사람의 정밀 pixel 주석은 아닙니다.</p>
<table><tr><th>공개 sample</th><th>경계 거리 px</th><th>SAM pV</th><th>v65 pV</th><th>v66 pV</th><th>v67 pV / 상태</th></tr>{example_table}</table><img src='data:image/png;base64,{encoded}' alt='공개 물체의 보이는 내부 점을 head가 지운 실제 crop'/></section>
<section><h2>남은 목표와 다음 작업</h2><p>정확한 query 한 픽셀의 점유와 사람의 참조 키포인트·crop 의미 가시성은 다릅니다. 소스 pixel 점수가 좋아져도 전체 목표 달성이 되는 것은 아닙니다. 현재 목표를 이 작은 하위 문제로 바꾸지 않습니다.</p>
<p>다음은 고정 SAM2.1 base-plus의 같은 소스 box/query 기준선과, 낮은 목재 팔레트가 보이는 독립 공개 실사의 출처·라벨 점검입니다. 현재 평가 데이터로 λ·문턱·radius를 고르지 않으며 PnP를 실행하지 않습니다.</p></section>
<section><h2>원본 결과와 검산 자료</h2><ul>{links}</ul></section></main></html>"""
    (P / "RESULTS_KO.html").write_text(document)
    print(json.dumps({"report": str(P / "RESULTS_KO.html"), "verdict": "HOLD", "failed_gates": 7, "current_inference": 0}))


if __name__ == "__main__":
    main()
