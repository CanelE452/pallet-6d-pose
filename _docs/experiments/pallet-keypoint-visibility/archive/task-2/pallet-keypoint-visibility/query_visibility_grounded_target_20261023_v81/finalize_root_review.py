"""Saved-report finalization only. No model, RGB, or current evaluation reader."""
from pathlib import Path
import datetime
import hashlib
import html
import json
import shutil

V = Path(__file__).resolve().parent
P = V.parent
V82 = P / "query_visibility_requested_category_20261023_v82"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def read(rel):
    return json.loads((V / rel).read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    assert not (V / "REPORT_FINAL_SEAL.json").exists(), "Preserve completed finalization"
    result = read("source_evaluation/GROUNDED_SAM21_FIXED/SOURCE_MASK_RESULTS.json")
    audit = read("source_audit/metrics_reconciliation/INDEPENDENT_SAVED_METRICS_RECONCILIATION.json")
    attribution = read("saved_target_attribution/TARGET_OWNERSHIP_ATTRIBUTION_RESULTS_KO.json")
    replay = read("root_attribution_check/ROOT_FIXED56_MASK_REPLAY.json")
    assert all(x["PASS"] for x in [result, audit, attribution, replay])
    assert not audit["discrepancies"]
    assert attribution["strict_C_falsevisible56_global_IoU_rank_bins"] == {"OTHER_IOU_GREATER": 56}
    all_b, _, public_b = result["benchmarks"]
    strict = all_b["strict535_same_frozen_IDs"]
    assert strict["CM_rows_GT_CV_cols_pred_CVU"] == [[15, 56, 5], [83, 259, 117]]
    assert strict["hold_N"] == 122
    paths = {
        "protocol": V / "EXPERIMENT_DECISIONS_BEFORE_ASSETS_AND_SOURCE.json",
        "detector": V / "detector/outputs/DETECTOR_OUTPUT_SEAL.json",
        "SAM": V / "segmentation/outputs/SAM_OUTPUT_SEAL.json",
        "evaluation": V / "source_evaluation/GROUNDED_SAM21_FIXED/EVALUATION_SEAL.json",
        "detector_audit": V / "detector_independent_audit/INDEPENDENT_DETECTOR_OUTPUT_AUDIT.json",
        "SAM_audit": V / "source_audit/INDEPENDENT_SAVED_SAM_OUTPUT_AUDIT.json",
        "metrics_audit": V / "source_audit/metrics_reconciliation/INDEPENDENT_SAVED_METRICS_RECONCILIATION.json",
        "attribution": V / "saved_target_attribution/ATTRIBUTION_SEAL.json",
        "root_fixed56_replay": V / "root_attribution_check/ROOT_FIXED56_MASK_REPLAY.json",
        "gallery": V / "fixed_natural_gallery/FINAL_REPORT_AND_GALLERY_SEAL.json",
        "depth_feasibility_only": V / "next_semantic_review/REVIEW_SEAL.json",
        "next_protocol": V82 / "EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json",
        "next_independent_review": V / "next_category_review/EXACT_FROZEN_V82_PROTOCOL_REVIEW.json",
        "direct_user_and_Pro_review": V / "ROOT_DIRECT_REVIEW_KO.json",
    }
    bound = {k: bind(v) for k, v in paths.items()}
    assert bound["next_protocol"]["sha256"] == "ab89f3885627cdd319b93fb6ead6357950a10f7fa51d751798ddd7cabba5af28"
    decision = {
        "completed": True, "phase": "V81_SOURCE_PILOT_AND_DIAGNOSIS_COMPLETE",
        "goal_status": "active", "fully_solved": False, "bindings": bound,
        "result_KO": "모든 종류를 동시에 요청하고 초기 box IoU로 연결한 frozen grounding+SAM 경로는 대상 소유권 오류가 남아 채택하지 않습니다.",
        "strict535": strict,
        "ownership_diagnostic": {"strict_C_false_visible_points": 56, "unique_targets": 50,
            "all56_other_instance_mask_IoU_better": True,
            "all56_other_visible_at_q": True,
            "intended_M4_IoU_median": replay["intended_median_IoU"],
            "best_other_M4_IoU_median": replay["best_other_median_IoU"],
            "limit": "예측 mask와 box의 다른 대상 대응 증거입니다. 전체56개를 detector 단독 원인으로 확정하지 않습니다."},
        "target_definition_KO": "YCB reference-primary 2018 중669점은 정확한 q-M4 bit와 기존 reference-local label이 다릅니다. strict535는 이 중심 상태 차이가0인데도 실패하므로 정의 차이만의 문제가 아닙니다. q-M4 진단은 모델 성능 상한이나 semantic E/S 정답이 아닙니다.",
        "next_stage_KO": "V82는 requested category를 별도 입력으로 제공하는 source-only frozen pilot입니다. SYNTH는 pallet 고정, YCB는 공개 metadata의 종류를 제공받는 조건임을 명시합니다. 현재223 평가나 학습은 자동 승인되지 않습니다.",
        "depth_review_KO": "공개1000RGB에 대응하는 depth/pose는 로컬에 있습니다. 기존 one-case depth proxy의 invalid-depth 처리 한계를 보존하고 보수적 외부-ray draft를 검토했습니다. 새 depth teacher/학습은 실행하지 않았습니다.",
        "phase_execution": {"detector_forward_calls_including_repeat": 436,
            "successful_SAM_mask_predict_calls": 823, "successful_SAM_embeddings": 345,
            "preserved_failed_SAM_predict_calls": 1, "preserved_failed_SAM_embeddings": 1,
            "new_training_updates": 0, "current223_underlying_data_reads_or_inference": 0, "PnP": 0},
        "current_capture_confirmation": "직접 촬영했고 KU Leuven 자료와 무관합니다",
        "fresh_independent_generalization_claim": False,
    }
    write(V / "ROOT_RESULT_AND_NEXT_DECISION.json", decision)
    url = lambda name: html.escape(str(V / name), quote=True)
    report = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>가림 판별: V81 진단과 다음 단계</title>
<style>body{{font-family:Arial,"NanumGothic",sans-serif;max-width:1100px;margin:28px auto;padding:20px;line-height:1.75;color:#17242d}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbc6ce;padding:10px;text-align:left}}.note{{background:#fff1e6;padding:18px}}img{{max-width:100%}}a{{color:#095b9b}}</style>
<h1>대상 연결 오류가 확인됐고, 가림 판별은 아직 해결되지 않았습니다</h1>
<p class="note">사용자가 현재223장을 직접 촬영했으며 KU Leuven 자료와 무관하다고 확인했습니다. V81은 공개·합성 source만 사용했고 새 학습·현재223 추론·PnP는 없습니다. 공개 source에서 detector436회와 성공 SAM mask823회는 실제 실행했습니다. 최초 실패1회도 별도로 보존했습니다.</p>
<p>동일한 전체 source 1,035개 중 mask823개, 대상 연결 보류212개입니다. 전체 평균 mask IoU는0.512이고 mask가 있는823개 조건부 평균은0.643입니다. 이 수치만으로 특정 키포인트의 가림을 잘 구분한다고 판단할 수 없습니다.</p>
<h2>중심 좌표·라벨이 맞는 공개 실사535점</h2>
<p>q-r≤3px이고 중심 pixel 상태가 reference-local label과 같은 고정535점입니다. C는 source 가림/support proxy, V는 source visible/support이며 사람의 외부·자기 가림 정답과 구분합니다. U는 보류입니다.</p>
<table><tr><th>실제 상태</th><th>예측 C</th><th>예측 V</th><th>보류 U</th><th>Recall (U도 miss)</th></tr>
<tr><td>C76점</td><td>15</td><td>56</td><td>5</td><td>19.7%</td></tr><tr><td>V459점</td><td>83</td><td>259</td><td>117</td><td>56.4%</td></tr></table>
<p>C precision15.3%, V precision82.2%, 보류율22.8%입니다. Precision은 GT C/V로 알려진 이 집합의 조건부 값입니다.</p>
<h2>틀린 마스크는 주로 어느 물체를 보고 있었나</h2>
<p>C를 V라고 한56점 전부에서 예측 mask는 의도한 target보다 다른 annotated instance와 더 잘 겹쳤습니다. 의도한 target IoU 중앙값은0.011, 다른 instance는0.909입니다. 동일56점의 실제 저장 mask pixel·원 q·CSV 집합을 별도로 재검산했습니다. 이는 오대상 연결과 일관된 증거이며 detector만의 단독 원인이라고 확정하지는 않습니다.</p>
<p>아래 고정 사례는 뒤의 흰 용기가 target인데 앞의 노란 병이 검출·분할된 경우입니다. q-r는1.166px여서 좌표가 틀린 탓으로 설명하기 어렵습니다.</p>
<img src="{url('fixed_natural_gallery/2_FIXED_GROUNDED_MASK.png')}" alt="의도한 흰 용기 대신 앞 노란 병을 선택한 실제 공개 crop">
<h2>목표 정의 차이와 실제 오류를 함께 유지합니다</h2>
<p>YCB reference-primary2,018점에서 완벽한 GT q-M4 bit를 읽어도 기존 reference label과669점이 다릅니다. 보이는 코너 근처 q가 배경에 놓이거나 다른 표면에 놓일 수 있기 때문입니다. 반면 위 strict535는 이 중심 상태 차이가0이므로 실제 대상 소유권 인식 실패가 남습니다. 원래 라벨을 바꾸거나 q-M4 진단을 모델의 성능 상한이라고 부르지 않습니다.</p>
<h2>다음은 대상 종류를 명시한 독립 버전입니다</h2>
<p>V82에서 알려진 요청 대상 종류를 grounding 입력에 추가합니다. 실제 목표는 항상 pallet이고, 공개 YCB에서는 metadata의 종류를 제공받는 더 강한 입력 조건입니다. 이는 class-free 또는 같은 입력 대조가 아닙니다. 동일1,035개·q·L·문턱·SAM 규칙을 유지하고, GT mask나 visibility로 대상을 고르지 않습니다. 같은 종류의 다른 instance 문제와 pixel/reference 목표 차이는 여전히 남습니다. 현재223 자동 평가나 새 학습은 시작하지 않습니다.</p>
<p>별도로 공개 depth를 이용한 외부-ray proxy 감독의 가능성을 확인했습니다. 누락 depth는 U, 자기 target의 앞 표면은 별도 상태로 남기는 draft이며 사람 D/E/S 정답으로 승격하지 않았습니다.</p>
<p><a href="{url('RESULTS_KO.html')}">전체 수치·원래 그림</a> · <a href="{url('RESULTS_INTERPRETATION_ADDENDUM_KO.html')}">고정4사례 판독</a> · <a href="{url('ROOT_RESULT_AND_NEXT_DECISION.json')}">근거와 실행 기록</a></p>
<p>Pro 대화 <a href="https://chatgpt.com/c/6ad599a5-f7c4-83ee-8f28-a9146700ff0a">실험 감사와 실패 원인 진단</a>의 최근 전달10turn을 앱에서 직접 확인했습니다. 긴 항목 일부는 잘렸으므로 전체 과거 원문을 완독했다고 주장하지 않습니다.</p></html>'''
    (V / "ROOT_READOUT_KO.html").write_text(report)
    write(V / "STATUS.json", {"state": "SOURCE_PILOT_AND_DIAGNOSIS_COMPLETE", "goal_status": "active", "fully_solved": False, "next": str(V82), "current223_new_reads": 0, "new_fit": 0, "PnP": 0})
    # Index local report/code/proof material. Large model/runtime/mask arrays are
    # bound by the already audited stage seals rather than rehashed here.
    excluded_dirs = {"overlay", "__pycache__", "masks", "npz"}
    excluded = []
    files = []
    for path in sorted(V.rglob("*")):
        if not path.is_file() or any(part in excluded_dirs for part in path.relative_to(V).parts):
            continue
        if path.name in {"ARTIFACT_INDEX_SOURCE_RESULTS.json", "REPORT_FINAL_SEAL.json"}:
            continue
        if path.stat().st_size > 3_000_000 or path.suffix in {".safetensors", ".whl", ".bin"}:
            excluded.append(str(path))
            continue
        files.append(bind(path))
    write(V / "ARTIFACT_INDEX_SOURCE_RESULTS.json", {"completed": True, "scope": "Local report/code/proof files up to3MB excluding runtime overlay/cache and masks/npz folders. Authoritative detector/SAM output stage seals bind omitted arrays.", "artifact_count": len(files), "files": files, "omitted_large_files": excluded, "authoritative_stage_seals": bound})
    final = {"completed": True, "phase_complete": "V81_SOURCE_PILOT_AND_DIAGNOSIS", "goal_status": "active", "fully_solved": False,
        "index": bind(V / "ARTIFACT_INDEX_SOURCE_RESULTS.json"), "root_report": bind(V / "ROOT_READOUT_KO.html"),
        "full_report": bind(V / "RESULTS_KO.html"), "root_decision": bind(V / "ROOT_RESULT_AND_NEXT_DECISION.json"),
        "current223_new_underlying_reads_or_inference": 0, "new_fit": 0, "PnP": 0}
    write(V / "REPORT_FINAL_SEAL.json", final)
    backup = V / "provenance/ROOT_POINTERS_BEFORE_V81_RESULTS"
    backup.mkdir(exist_ok=True)
    for filename in ["LATEST_STATUS_KO.json", "ACTIVE_HANDOFF_KO.json"]:
        old = P / filename
        shutil.copy2(old, backup / (old.stem + "_" + sha(old)[:12] + ".json"))
    latest = json.loads((P / "LATEST_STATUS_KO.json").read_text())
    latest.update({"schema": "PALLET_QUERY_VISIBILITY_LATEST_V81_COMPLETE_V82_REQUESTED_CATEGORY_PREPARATION",
        "latest_report": str(V / "ROOT_READOUT_KO.html"), "latest_index": str(V / "ARTIFACT_INDEX_SOURCE_RESULTS.json"),
        "latest_source_only_result": "V81strict535 C15/76,V259/459,U122; all56CfalseV overlap anotherinstancebetter. Not solved.",
        "next_direction": "V82 requestedcategory-provided frozen grounding+SAM sourcepilot; metadata input and code preparation. No training/current/PnP.",
        "next_active_work": str(V82), "V81_state": "SOURCE_PILOT_AND_DIAGNOSIS_COMPLETE",
        "V81_report_final_seal": bind(V / "REPORT_FINAL_SEAL.json"), "V81_detector_forward_calls": 436,
        "V81_SAM_successful_predict_calls": 823, "V81_SAM_preserved_failed_predict_calls": 1,
        "V81_current_reads": 0, "V81_current223_new_inference": 0, "V81_fit": 0, "V81_PnP": 0,
        "V82_decisions": bound["next_protocol"], "V82_protocol_review": bound["next_independent_review"],
        "current_capture_family_confirmation": "Direct user: 직접 촬영했고 KU Leuven 자료와 무관합니다",
        "fully_solved": False, "goal_status": "active", "updated_at_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    write(P / "LATEST_STATUS_KO.json", latest)
    handoff = json.loads((P / "ACTIVE_HANDOFF_KO.json").read_text())
    handoff.update({"goal_status": "active", "fully_solved": False, "latest_source_report": bind(V / "REPORT_FINAL_SEAL.json"),
        "current_capture_confirmation": "직접 촬영했고 KU Leuven 자료와 무관합니다",
        "completed_this_turn": ["V81all1035sourceactualgrounding+SAMoutputsevaluatedandaudited", "same535andfixed4cropsreviewed; targetownership56falseVdiagnosed;definitionoracleseparate", "V82suppliedcategorycontractfrozenandindependentlyreviewed;input/codepreparation"],
        "source_findings": "V81strictC19.7/V56.4/U22.8%; falseV56points allothermaskIoUbetter. No GT-best outputrepair. Definitionmismatch real but notsufficientexplanation.",
        "next_required": bound["next_protocol"],
        "next_work": "V82metadata-onlyinputbuild+code/generatedtests+independententryreview;rootlockbeforeactualsourceforward. No current223/newtraining/PnP. Category is declaredprovidedsideinformation forYCB, constantpalletatdeployment.",
        "updated_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    write(P / "ACTIVE_HANDOFF_KO.json", handoff)
    print(json.dumps({"completed": True, "report": str(V / "ROOT_READOUT_KO.html"), "seal": bind(V / "REPORT_FINAL_SEAL.json"), "indexed_files": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
