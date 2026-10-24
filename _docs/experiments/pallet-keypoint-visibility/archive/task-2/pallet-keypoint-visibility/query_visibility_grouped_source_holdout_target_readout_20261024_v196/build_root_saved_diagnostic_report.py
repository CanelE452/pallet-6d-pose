"""Summarize only the generated V196 source-HOLDOUT readout; never run models."""
import collections
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

V = Path(__file__).parent
G = V / "all360_source_holdout_viewer_v1"


def checked_path(path):
    path = Path(path)
    assert "_v112" not in str(path).lower()
    assert path.is_absolute()
    for node in (path, *path.parents):
        assert not node.is_symlink()
    return path


def binding(path):
    path = checked_path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(path):
    return json.loads(checked_path(path).read_text())


def write_new(path, value):
    path = checked_path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def main():
    complete = read_json(G / "COMPLETE.json")
    assert complete["completed"] and complete["PASS"]
    assert complete["rows360"] == 360 and complete["known45"] == 45
    assert complete["UNKNOWN315"] == 315 and complete["triplets15"] == 15
    viewer_review = read_json(V / "independent_saved_viewer_review_c/REVIEW.json")
    root_visual = read_json(V / "ROOT_ACTUAL_FIXED8_SOURCE_CONTACT_VISUAL_QA.json")
    assert viewer_review["PASS"] and root_visual["PASS"]
    assert len(root_visual["viewed_contacts"]) == 8
    assert viewer_review["actual_visual_QA"]["actual_views"] == 7
    root_images = {r["image"]["path"] for r in root_visual["viewed_contacts"]}
    c_images = {r["image"]["path"] for r in viewer_review["actual_visual_QA"]["images"]}
    assert not root_images & c_images
    assert root_images | c_images == {r["path"] for r in complete["contacts"]}

    associations = read_json(G / "ALL15_TARGET_ASSOCIATIONS.json")["triplets"]
    with checked_path(G / "ALL360_HOLDOUT_TARGET_MAPPING.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    known = [r for r in rows if r["direct_weak_target"] != "-1"]
    assert len(rows) == 360 and len(known) == 45
    assert all(r["role"] == "HOLDOUT" and r["training_admission"] == "False" for r in rows)
    assert collections.Counter(r["native_slot"] for r in rows) == {str(i): 45 for i in range(8)}
    assert collections.Counter(r["direct_weak_target"] for r in known) == {"0": 30, "1": 15}
    metrics = {}
    table = []
    for model, expected in (("C56", [[28, 2, 0], [11, 4, 0]]), ("C57", [[28, 2, 0], [8, 7, 0]])):
        cm = [[sum(r["direct_weak_target"] == str(t) and r[model + "_D_E_U"] == pred for r in known)
               for pred in ("D", "E", "U")] for t in (0, 1)]
        assert cm == expected == viewer_review["independently_recounted_CM"][model]
        metrics[model] = {"CM_D_E_by_D_E_U": cm,
                          "D_recall": cm[0][0] / 30, "E_recall": cm[1][1] / 15,
                          "D_precision": cm[0][0] / (cm[0][0] + cm[1][0]),
                          "E_precision": cm[1][1] / (cm[0][1] + cm[1][1]),
                          "predicted_U_known": sum(r[model + "_D_E_U"] == "U" for r in known),
                          "predicted_U_all360": sum(r[model + "_D_E_U"] == "U" for r in rows)}
    for i, a in enumerate(associations):
        subset = sorted((r for r in known if r["attempt_id"] == a["attempt_id"]),
                        key=lambda r: ("BEFORE", "COVER", "SHAM").index(r["variant"]))
        assert len(subset) == 3 and [r["direct_weak_target"] for r in subset] == ["0", "1", "0"]
        assert {int(r["native_slot"]) for r in subset} == {a["selected_native_slot"]}
        x0, y0, x1, y1 = a["candidate_enclosures_original96_diagnostic"][0]
        changed = a["end_mask_RGB_changed_samples_vs_BEFORE"]
        assert changed["COVER"] == a["end_related_original96_samples"] > 0 and changed["SHAM"] == 0
        table.append({"contact_index": i, "attempt_id": a["attempt_id"], "base": a["source_base_index"],
                      "native": a["selected_native_slot"], "end_related96_samples": a["end_related_original96_samples"],
                      "q_to_finite_end_center96_px_DIAGNOSTIC_NOT_REFERENCE_ERROR": math.hypot((x0+x1)/2-47.5, (y0+y1)/2-47.5),
                      "continuous_end_enclosure96_wh": [x1-x0, y1-y0],
                      "C56_B_C_S": "".join(r["C56_D_E_U"] for r in subset),
                      "C57_B_C_S": "".join(r["C57_D_E_U"] for r in subset),
                      "contact": f"all360_source_holdout_viewer_v1/TRIPLET_{i:02d}/CONTACT_SELECTED_WEAK_END.png"})
    distances = [r["q_to_finite_end_center96_px_DIAGNOSTIC_NOT_REFERENCE_ERROR"] for r in table]
    patterns = {m: dict(collections.Counter(r[m + "_B_C_S"] for r in table)) for m in ("C56", "C57")}
    result = {"completed": True, "PASS": True, "analysis_only_saved_generated_source_readout": True,
              "code": binding(__file__), "viewer_complete": binding(G / "COMPLETE.json"),
              "saved_viewer_review": binding(V / "independent_saved_viewer_review_c/REVIEW.json"),
              "root_visual_QA": binding(V / "ROOT_ACTUAL_FIXED8_SOURCE_CONTACT_VISUAL_QA.json"),
              "mapping": binding(G / "ALL360_HOLDOUT_TARGET_MAPPING.csv"),
              "associations": binding(G / "ALL15_TARGET_ASSOCIATIONS.json"),
              "all_rows": 360, "known_weak": 45, "truth_UNKNOWN": 315, "input_U": 0,
              "triplets": 15, "unique_bases": len({r["base"] for r in table}),
              "known_native_triplet_counts": dict(collections.Counter(r["native"] for r in table)),
              "metrics": metrics, "triplet_patterns": patterns, "table": table,
              "diagnostic_q_to_finite_end96_distance": {"min": min(distances), "median": statistics.median(distances), "max": max(distances)},
              "actual_tool_view_contacts_union": 15, "all360_actual_tool_view": False,
              "new_labels_or_fit_or_inference_or_selection": False,
              "current223_publicheld_COMMON_oldVAL_access_in_this_report": 0,
              "training_admission": False, "selected_model": None, "fully_solved": False,
              "conclusion": "Stored weak source-HOLDOUT generalization failures exist before real transfer. Weak finite-end/native semantic alignment and input identifiability remain unproven. Tiny support, q offset, appearance and architecture are hypotheses, not isolated causes."}
    md = ["# V196: 학습에서 뺀 합성 자료의 끝부위 가림 판독", "",
          "**새 학습·추론 없이 저장된 모든 source HOLDOUT을 확인했다. 합성 내부에서도 가림을 놓치므로 실사 외관 차이만으로 실패를 설명할 수 없다.** 원 query·L·native0..7을 유지했다. 현재 실사 평가·publicheld·COMMON·기존 VAL은 이 진단에서 읽지 않았다.", "",
          "15 triplet은 6개 base에서 나온 BEFORE/COVER/SHAM 대조다. 전체 360행 중 직접 weak 라벨은 D30/E15의 45행이고 나머지315행은 UNKNOWN이다. 입력 U는0이다. 실제 도구로 root00–07, 독립 검토자08–14를 나누어 15개 contact를 모두 보았다. 360개 crop을 각각 도구로 봤다는 뜻은 아니다.", "",
          "| 모델 | CM 실제D→D/E/U | CM 실제E→D/E/U | D recall | E recall | E precision | D/E/D 완전 반응 |", "|---|---|---|---:|---:|---:|---:|"]
    for m, d in metrics.items():
        cm = d["CM_D_E_by_D_E_U"]
        md.append(f"| {m} | {cm[0]} | {cm[1]} | {d['D_recall']:.1%} | {d['E_recall']:.1%} | {d['E_precision']:.1%} | {patterns[m].get('DED',0)}/15 |")
    md += ["", "C56은11세트, C57은8세트에서 BEFORE/COVER/SHAM을 모두 D로 답했다. 두 모델 모두1세트에서는 세 변형을 모두 E로 답했다. 학습327행 완전 적합이라는 V194 기록과 구분해야 한다. 이45행은 자연 실사 native 정답이 아니라 구성한 finite-end weak 과제다.", "",
           "## 입력과 표적에서 확인한 사실", "",
           "모든15세트에서 COVER는 표적 끝부분 관련 RGB 샘플을 바꾸었고, SHAM은 그 subset을 보존했다. 따라서 입력에 가림 변화가 전혀 없었다는 설명은 이 저장 대조와 맞지 않는다. 다만 변화한 샘플이20–30개라는 것은 보간 지지 수이며, 사람이 읽을 수 있는 신호량·물리 면적·모델 특징 보존량을 측정한 수치는 아니다.", "",
           f"연속 end enclosure는96좌표에서 대략1×1–2픽셀이고, query 중심과 enclosure 중심 거리는 {min(distances):.2f}–{max(distances):.2f}px(중앙값{statistics.median(distances):.2f}px)다. **이 거리는 합성 표적 대응을 보는 진단값이며, 실사 키포인트의 참조 좌표 오차가 아니다.** enclosure·정답 위치는 표시·감사에만 쓰고 모델 입력으로 추가하지 않았다.", "",
           "실제 그림에는 목재와 비슷한 색의 작은 가림을 놓치는 경우와, 회색 패치가 뚜렷하고 query에 가까워도 놓치는 경우가 함께 있다. 색 대비 하나·query 거리 하나·SAM 유무 하나를 단독 원인으로 결론내릴 수 없다. native4/6/7은 이45개 known 정답에 없지만 원 입력 native8개는 모두 유지했다.", "",
           "| triplet | base/native | 끝 관련 샘플 | q→finite-end 거리96px | C56 B/C/S | C57 B/C/S | 그림 |", "|---|---|---:|---:|---|---|---|"]
    for r in table:
        md.append(f"| {r['attempt_id']} | {r['base']}/{r['native']} | {r['end_related96_samples']} | {r['q_to_finite_end_center96_px_DIAGNOSTIC_NOT_REFERENCE_ERROR']:.2f} | {r['C56_B_C_S']} | {r['C57_B_C_S']} | [원 crop]({r['contact']}) |")
    md += ["", "## 해결된 부분과 남은 문제", "",
           "계보를 묶은 사전 source 분할, 모든 학습 손실의 TRAIN 전용 draw, 동일 input·라벨·고정0 문턱으로 두 모델을 비교하는 절차, crop/예측/라벨의 대응은 확보했다. 저장 source 결과를 재현했고, 현재45행에 입력 누락은 없다. 이것이 모델의 실제 판별 능력을 보장하지는 않는다.", "",
           "남은 문제는 새로운 source 장면에서의 가림 관계 일반화, finite-end 감독과 사람이 판단한 native D/E의 의미 대응, 작은 부위와 문맥을 모델이 실제로 읽는지다. 물리 GT가 없다는 이유로 기존 사람 D/E 평가를 지우지 않는다. weak 끝부위를 강한 native 물리 정답으로 승격하지도 않는다.", "",
           "후속 작업은 TRAIN source만으로 native→원query→의도한 부위→감독 영역의 연결을 먼저 점검하는 것이다. 잘못된 대응과 입력에서 읽을 수 없는 감독을 구분한 뒤, 허용된 source 개발자료에서만 한 가지 가설을 정하고 설정을 사전 동결해야 한다. 이번 HOLDOUT과 반복 노출 실사는 진단 결과로 보존하며, 이를 이용한 문턱·체크포인트·모델·데이터 선택이나 실패행 재학습은 하지 않는다. 새 학습·PnP·위치 보정은 이 단계에서 실행하지 않았다.", "",
           "[전체 원 crop 갤러리](all360_source_holdout_viewer_v1/PORTABLE_GALLERY_KO.md) · [360행 대응표](all360_source_holdout_viewer_v1/ALL360_HOLDOUT_TARGET_MAPPING.csv) · [표적 근거](all360_source_holdout_viewer_v1/ALL15_TARGET_ASSOCIATIONS.json) · [수치 JSON](ROOT_RESULTS_KO.json) · [독립 저장결과/그림 검토](independent_saved_viewer_review_c/REVIEW.json) · [root 그림 검토](ROOT_ACTUAL_FIXED8_SOURCE_CONTACT_VISUAL_QA.json)", ""]
    text = "\n".join(md)
    write_new(V / "ROOT_RESULTS_KO.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_new(V / "ROOT_RESULTS_KO.md", text)
    write_new(V / "PORTABLE_REPORT_KO.md", text)
    cards = "".join(f'<article><h2>{html.escape(r["attempt_id"])}</h2><p>C56 {r["C56_B_C_S"]} / C57 {r["C57_B_C_S"]}; weak target D/E/D</p><img src="{r["contact"]}" alt="BEFORE COVER SHAM 원 crop 및 진단 표시"></article>' for r in table)
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V196 source HOLDOUT 진단</title><style>body{font-family:sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;line-height:1.6}img{max-width:100%;height:auto}article{border-top:1px solid #aaa;margin-top:32px}pre{white-space:pre-wrap;background:#f5f5f5;padding:20px}</style><h1>V196: 합성 HOLDOUT 가림 판독</h1><p>15세트 전체, 원 query/L/native 유지. 감독 끝부위는 weak이며 물리 native gold가 아닙니다. 표시 overlay는 입력에 넣지 않았습니다.</p><p><a href="PORTABLE_REPORT_KO.md">전체 보고서·표·근거</a> · <a href="all360_source_holdout_viewer_v1/index.html">360행 갤러리</a></p><pre>'+html.escape(text)+'</pre>'+cards+'</html>'
    write_new(V / "ROOT_READOUT_KO.html", page)
    receipt = {"completed": True, "PASS": True, "code": binding(__file__),
               "outputs": [binding(V / x) for x in ("ROOT_RESULTS_KO.json", "ROOT_RESULTS_KO.md", "PORTABLE_REPORT_KO.md", "ROOT_READOUT_KO.html")],
               "HTML_browser_rendered": False, "new_labels_or_models_or_fit_or_selection": False,
               "training_admission": False, "fully_solved": False}
    write_new(V / "ROOT_SAVED_DIAGNOSTIC_REPORT_COMPLETE.json", json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"completed": True, "receipt": binding(V / "ROOT_SAVED_DIAGNOSTIC_REPORT_COMPLETE.json"), "metrics": metrics, "triplet_patterns": patterns}, ensure_ascii=False))


if __name__ == "__main__":
    main()
