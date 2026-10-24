"""Finalize the failed source component experiment; never load a model/current data."""
from pathlib import Path
import datetime
import hashlib
import html
import json
import shutil

V = Path(__file__).resolve().parent
P = V.parent
OLD = P / "query_visibility_grounded_target_20261023_v81"
NEXT = P / "query_visibility_public_cue_audit_20261023_v83"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {"path": str(path), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    assert not (V / "REPORT_FINAL_SEAL.json").exists()
    rels = {
        "decisions": "EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json",
        "criteria": "SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json",
        "detector": "detector/outputs/DETECTOR_OUTPUT_SEAL.json",
        "detector_audit": "source_audit/INDEPENDENT_DETECTOR_OUTPUT_AUDIT.json",
        "SAM": "segmentation/outputs/SAM_OUTPUT_SEAL.json",
        "SAM_audit": "source_audit/INDEPENDENT_SAM_OUTPUT_AUDIT.json",
        "evaluation": "source_evaluation/FINAL_SAVED_EVALUATION_SEAL.json",
        "metrics_audit": "source_audit/metrics_reconciliation/INDEPENDENT_SAVED_METRICS_CRITERIA_TRANSITIONS_RECONCILIATION.json",
        "gallery": "FINAL_REPORT_AND_GALLERY_SEAL.json",
        "complementarity": "source_evaluation/V76_V82_SAVED_COMPLEMENTARITY/SEAL.json",
        "SAM3_docs_only": "next_source_review/SAM3_OFFICIAL_DOCUMENTS_ONLY.json",
    }
    bindings = {k: bind(V / name) for k, name in rels.items()}
    bindings["next_source_visual_scope"] = bind(NEXT / "SOURCE_VISUAL_REVIEW_SCOPE_BEFORE_RGB.json")
    for key in ["detector_audit", "SAM_audit", "metrics_audit"]:
        assert read(bindings[key]["path"])["PASS"]
    new = read(V / "source_evaluation/REQUESTED_CATEGORY_SAM21_FIXED/SOURCE_MASK_RESULTS.json")
    old = read(OLD / "source_evaluation/GROUNDED_SAM21_FIXED/SOURCE_MASK_RESULTS.json")
    comp = V / "source_evaluation/REQUESTED_CATEGORY_SAM21_FIXED_COMPARISON"
    screen = read(comp / "PROSPECTIVE_COMPONENT_SCREEN_KO.json")
    own = read(comp / "OWNERSHIP_DIAGNOSTIC_KO.json")
    pairs = read(comp / "SAME_IDS_V81_V82_COMPARISON_KO.json")
    assert not screen["component_screen_PASS"] and sum(row["PASS"] for row in screen["checks"].values()) == 0
    strict = new["benchmarks"][0]["strict535_same_frozen_IDs"]
    assert strict["CM_rows_GT_CV_cols_pred_CVU"] == [[22, 24, 30], [47, 251, 161]]
    c_transition = next(row for row in pairs["same_strict535_prediction_transitions"] if row["GT_label"] == 0)["V81_to_V82_CVU_counts"]
    assert c_transition == [[4, 5, 6], [17, 17, 22], [1, 2, 2]]
    decision = {
        "completed": True, "phase": "V82_SOURCE_COMPONENT_AND_DIAGNOSIS_COMPLETE",
        "goal_status": "active", "fully_solved": False,
        "disposition": "SOURCE_HOLD_COMPONENT_SCREEN_0_OF_4_FAIL",
        "bindings": bindings, "strict535": strict,
        "matched_strict_C_transition_rows_old_CVU_cols_new_CVU": c_transition,
        "old_false_visible56": {"corrected_C": 17, "still_false_V": 17, "held_U": 22},
        "ownership": own,
        "interpretation_KO": [
            "요청category를제공해도0/4기준실패. 문턱·동의어·체크포인트sweep이나현재223평가를하지않습니다.",
            "전체1035조건부mask평균은높아졌지만공개실사800전체IoU는0.52909→0.52866으로개선되지않았습니다. 공개실사targetU는122→218입니다.",
            "기존falseV56개중17개는C정답,22개는보류,17개는여전히falseV입니다. 최종falseV24개에는기존C/U의새falseV7개도포함됩니다.",
            "같은고정실제crop에서요청및예측phrase가bleach cleanser로바뀌어도앞mustard병box/mask를선택합니다. 출력문구일치가target소유권증거는아닙니다.",
            "Reference-local support, exactqmask, 사람코너영상단서D/E/S는별개이며535정렬부분에서도인식오류와보류가남습니다.",
            "V76×V82저장결정상보성은joint빈도진단입니다. GT선택ensemble·성능상한·배포규칙으로승격하지않습니다.",
        ],
        "root_visual_review": {"all_four_contact_sheet_viewed": True, "case2_full_panel_viewed": True,
            "case2_q_r_px": 1.166, "case2_requested_and_output_phrase": "bleach cleanser", "case2_actual_mask_target": "front yellow mustard bottle; intended white cleanser behind", "case2_intended_IoU_approx": 0.027},
        "phase_execution": {"detector_source_forwards_including_repeat": 1036, "SAM_mask_predictions": 790,
            "SAM_image_embeddings": 408, "new_fit_updates": 0, "current223_underlying_data_reads_or_inference": 0, "PnP": 0,
            "failed_actual_model_runs_in_V82": 0},
        "current_capture_confirmation": "직접 촬영했고 KU Leuven 자료와 무관합니다",
        "next_KO": "V83에서이미로컬에있는공개팔레트세후보reference의실제영상단서를두AIreviewer가독립검토합니다. Native8대응이없으면UNKNOWN으로남기며AI동의를humansemanticgold로부르지않습니다. 새teacher/합성paste/학습/모델추론은아직승인·실행하지않았습니다.",
        "fresh_independent_generalization_claim": False,
    }
    write(V / "ROOT_RESULT_AND_NEXT_DECISION.json", decision)
    esc = lambda value: html.escape(str(value), quote=True)
    strict_old = old["benchmarks"][0]["strict535_same_frozen_IDs"]
    def metrics_row(name, metric):
        return f'<tr><td>{name}</td><td>{100*metric["C_end_to_end_recall_U_is_miss"]:.1f}%</td><td>{100*metric["V_end_to_end_recall_U_is_miss"]:.1f}%</td><td>{100*metric["C_precision_known_CV_conditional"]:.1f}%</td><td>{100*metric["hold_rate"]:.1f}%</td></tr>'
    mask_rows = []
    for before, after in zip(old["benchmarks"], new["benchmarks"]):
        a, b = before["mask_IoU"], after["mask_IoU"]
        mask_rows.append(f'<tr><td>{esc(after["source"])}</td><td>{a["all_rows_N"]}</td><td>{a["all_rows_no_mask0_mean"]:.4f} → {b["all_rows_no_mask0_mean"]:.4f}</td><td>{a["usable_selected_mask_conditional_mean"]:.4f} → {b["usable_selected_mask_conditional_mean"]:.4f}</td><td>{a["nomatch_N"]} → {b["nomatch_N"]}</td></tr>')
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>공개 source 비교 V82: 아직 해결되지 않음</title><style>body{{font-family:Arial,"NanumGothic",sans-serif;max-width:1120px;margin:30px auto;padding:20px;line-height:1.7;color:#17232c}}table{{border-collapse:collapse;width:100%}}td,th{{padding:11px;border:1px solid #bbc6ce;text-align:left}}.note{{background:#fff0e4;padding:18px}}img{{max-width:100%}}a{{color:#075d99}}</style>
<h1>대상 종류를 명시해도 가림 판별 기준을 통과하지 못했습니다</h1>
<p class="note">V82의 사전 기준은0/4 통과입니다. 현재223장은 직접 촬영한 것으로 KU Leuven 자료와 무관하다는 사용자 확인을 기록했습니다. 이번 V81/V82 SAM 비교는 공개·합성 source만 사용했고, 새 학습·현재223 추론·PnP는 없습니다. 가림 판별이 해결됐다고 보고하지 않습니다.</p>
<p>V81은 모든 종류를 동시에 요청했고 V82는 각 target의 요청 종류를 제공했습니다. 실제 목표는 고정된 pallet이며 YCB에서는 공개 metadata의 종류를 추가로 제공받은 조건입니다. 같은 입력 대조나 예측 class로 해석하지 않습니다. 같은1,035개·q·L·문턱·SAM 규칙을 유지했습니다.</p>
<h2>중심 상태가 맞는 같은 공개 실사535점</h2><p>C는 source 가림/support proxy, V는 visible/support, U는 보류입니다. 사람 D/E/S 정답과 구분합니다. U는 recall의 miss로 포함했습니다.</p>
<table><tr><th>경로</th><th>C recall</th><th>V recall</th><th>C precision</th><th>보류율</th></tr>{metrics_row('V81 모든 종류 요청', strict_old)}{metrics_row('V82 종류 제공', strict)}</table>
<p>V82 혼동행렬은 행 C/V, 열 C/V/U 순서로 <strong>[[22,24,30],[47,251,161]]</strong>입니다. 같은 C76/V459 분모입니다. Precision은 이 known C/V 집합의 조건부 값입니다.</p>
<h2>마스크가 있는 경우만 보면 개선을 과장할 수 있습니다</h2>
<table><tr><th>자료</th><th>target 수</th><th>전체 IoU<br>실패0 포함</th><th>mask 있는 조건부 IoU</th><th>target 보류 수</th></tr>{''.join(mask_rows)}</table>
<p>공개 실사800개 전체 평균 IoU는 개선되지 않았고 보류가 늘었습니다. 조건부 평균 상승만으로 전체 경로가 좋아졌다고 말할 수 없습니다. source VAL은 반복 개발 자료이며 새 독립 최종 평가가 아닙니다.</p>
<h2>기존 오답56개에서 실제로 바뀐 것은</h2>
<p>기존 C를 V라고 한56점은 새 결과에서 <strong>C 정답17점·V 오답 유지17점·U 보류22점</strong>입니다. 새 전체 falseV24점에는 기존 C/U에서 새로 생긴7점도 포함됩니다. falseV 수가32개 줄었다고32개가 정답으로 바뀐 것은 아닙니다.</p>
<h2>이름을 맞혀도 다른 물체를 보는 실제 crop</h2>
<p>뒤의 흰 용기가 target입니다. 요청과 출력 phrase가 bleach cleanser여도 검출 box와 SAM mask는 앞 노란 병을 선택했습니다. q-r는1.166px이고 target M4 IoU는약0.027입니다. 좌표 오차나 출력 문구만으로 이 오류가 해결되지 않습니다.</p>
<img src="{esc(V/'fixed_natural_gallery/2_FIXED_REQUESTED_MASK.png')}" alt="종류를 제공해도 앞 가림물체를 선택하는 공개 crop">
<h2>다음은 감독 목표를 실제 공개 팔레트 단서에 연결하는 일입니다</h2>
<p>Reference 가시성과 q 한 pixel의 target mask는 다릅니다. 더 큰 mask나 더 많은 같은 라벨 학습부터 시작하지 않고, 공개 팔레트의 기존 세 reference 후보가 실제 코너·부분선 영상 단서를 가리키는지 독립 검토합니다. native8 대응이 불명확하면 UNKNOWN이고 AI 검토를 사람 semantic gold로 승격하지 않습니다. 이번 단계에서 새 source label·합성 paste·학습은 시작하지 않았습니다.</p>
<p>V76과 V82 저장 출력의 joint 진단도 보존했습니다. 서로 다른 오류가 있지만 GT로 둘 중 정답을 고르는 ensemble이나 성능 상한으로 제시하지 않았습니다. SAM3의 text/concept 입력도 <a href="https://github.com/facebookresearch/sam3/blob/main/README.md">공식 문서</a>에서 확인했으나 접근·runtime·성능은 검증하지 않았고 설치·다운로드·실행하지 않았습니다.</p>
<p><a href="{esc(V/'RESULTS_KO.html')}">전체 수치와 네 사례</a> · <a href="{esc(V/'RESULTS_INTERPRETATION_ADDENDUM_KO.html')}">실제 그림·대응 변화 판독</a> · <a href="{esc(V/'ROOT_RESULT_AND_NEXT_DECISION.json')}">감사·실행·다음 단계 근거</a></p></html>'''
    (V / "ROOT_READOUT_KO.html").write_text(body)
    (V / "STATUS.json").write_text(json.dumps({"phase": "SOURCE_COMPONENT_AND_DIAGNOSIS_COMPLETE", "goal_status": "active", "fully_solved": False, "screen_pass": 0, "screen_total": 4, "next": str(NEXT), "new_fit": 0, "current223": 0, "PnP": 0}, indent=2) + "\n")
    backup = V / "provenance/ROOT_POINTERS_BEFORE_V82_RESULTS"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ["LATEST_STATUS_KO.json", "ACTIVE_HANDOFF_KO.json"]:
        path = P / name
        shutil.copy2(path, backup / (path.stem + "_" + sha(path)[:12] + ".json"))
    files = []
    for path in sorted(V.rglob("*")):
        if not path.is_file() or any(part in {"masks", "__pycache__"} for part in path.relative_to(V).parts):
            continue
        if path.name in {"ARTIFACT_INDEX_SOURCE_RESULTS.json", "REPORT_FINAL_SEAL.json"} or path.stat().st_size > 3_000_000:
            continue
        files.append({**bind(path), "bytes": path.stat().st_size})
    write(V / "ARTIFACT_INDEX_SOURCE_RESULTS.json", {"completed": True, "scope": "Local report/code/proof/detector trace files up to3MB; masks bound by successful SAM stage seal and independent audit.", "artifact_count": len(files), "files": files, "authoritative_stage_seals": bindings})
    write(V / "REPORT_FINAL_SEAL.json", {"completed": True, "phase_complete": "V82_SOURCE_COMPONENT_AND_DIAGNOSIS", "goal_status": "active", "fully_solved": False, "index": bind(V / "ARTIFACT_INDEX_SOURCE_RESULTS.json"), "root_report": bind(V / "ROOT_READOUT_KO.html"), "root_decision": bind(V / "ROOT_RESULT_AND_NEXT_DECISION.json"), "current223_new_underlying_reads_or_inference": 0, "new_fit": 0, "PnP": 0})
    latest = read(P / "LATEST_STATUS_KO.json")
    latest.update(schema="PALLET_QUERY_VISIBILITY_V82_COMPLETE_V83_SOURCE_CUE_REVIEW", latest_report=str(V / "ROOT_READOUT_KO.html"), latest_index=str(V / "ARTIFACT_INDEX_SOURCE_RESULTS.json"), latest_source_only_result="V82 requestedcategory source screen0/4fail; strictC22/76,V251/459,U191/535; oldfalseV56→C17/V17/U22. No current223.", next_active_work=str(NEXT), next_direction="V83 independent raw-public-pallet cue/reference/initializer evidence audit only; no new labels/model/train/paste yet.", V82_active_state="SOURCE_COMPONENT_AND_DIAGNOSIS_COMPLETE_0_OF_4_FAIL", V82_report_final_seal=bind(V / "REPORT_FINAL_SEAL.json"), V82_current_reads=0, V82_fit=0, V82_PnP=0, goal_status="active", fully_solved=False, updated_at_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (P / "LATEST_STATUS_KO.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n")
    handoff = read(P / "ACTIVE_HANDOFF_KO.json")
    handoff.update(goal_status="active", fully_solved=False, latest_source_report=bind(V / "REPORT_FINAL_SEAL.json"), V82_active_state="SOURCE_COMPONENT_AND_DIAGNOSIS_COMPLETE_0_OF_4_FAIL", source_findings="V82strictC28.9/V54.7/U35.7%;public800wholeIoU.52909→.52866, targetU122→218. Conditionalmaskmean improvement doesnot solvefullpath. Samebleachcaptionstillmustardmask.", next_required=bindings["next_source_visual_scope"], next_work="V83reviewersA/B inspect existing3publicpalletreferences independently and sourceinitializer q lineage. KeepAIweakteacher tier, unknownnative8 mapping explicit; nohumansemanticgold/fit/current/PnP.", updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (P / "ACTIVE_HANDOFF_KO.json").write_text(json.dumps(handoff, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": True, "root_report": str(V / "ROOT_READOUT_KO.html"), "final_seal": bind(V / "REPORT_FINAL_SEAL.json"), "indexed_files": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
