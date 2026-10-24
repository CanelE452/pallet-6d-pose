"""Publish V156 diagnostic completion, preserving V155 source training."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

V = Path(__file__).resolve().parent
P = V.parent


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    review = V / "ROOT_FINAL_INDEPENDENT_REPORT_REVIEW.json"
    assert read(review)["PASS"]
    phase = read(V / "ROOT_PHASE_RESULT.json")
    assert phase["stage_complete"] and not phase["fully_solved"]
    summary = read(V / "current_result_summary_agent/CURRENT_RESULTS_SUMMARY_KO.json")
    result = read(V / "ROOT_RESULTS_KO.json")
    phase_binding = binding(V / "ROOT_PHASE_RESULT.json")
    result_binding = binding(V / "ROOT_RESULTS_KO.json")
    report = V / "ROOT_READOUT_KO.html"
    prediction = binding(V / "frozen_four_head_current_benchmark_v1/PREDICTION_SEAL.json")
    evaluation = binding(V / "saved_seven_current_evaluation_v1/EVALUATION_SEAL.json")
    current = {
        "version": "V156_FROZEN_C35_C36_C37_C38_DIAGNOSTIC",
        "report": str(report), "results": result_binding, "phase_result": phase_binding,
        "evaluation": evaluation, "prediction": prediction,
        "independent_saved_review": binding(V / "independent_saved_evaluation_review/REVIEW.json"),
        "root_actual_crop_review": binding(V / "ROOT_ACTUAL_FIXED_CROP_VISUAL_QA.json"),
        "final_report_review": binding(review),
        "stored_baseline": "V124_C16",
        "new_candidates_fixed_before_private_reads": ["C35", "C36", "C37", "C38"],
        "existing_model_results_preserved": ["v61", "legacy", "C16"],
        "primary_N": 1237, "all_original_slots": 1784, "valid": 1739,
        "factual_U": 45, "additional_U": 0,
        "aggregate_primary": [dict(model=m, **x) for m, x in summary["visibility"]["paired_same_query_RGB"]["ALL"].items()],
        "selected_model": None, "benchmark_not_adoption": True,
        "source_qualification_unchanged": True, "development_exposed_not_blind": True,
        "current_used_for_training_or_model_selection": False,
    }
    snapshots = V / "provenance/PARENT_POINTERS_BEFORE_V156_PUBLICATION"
    snapshots.mkdir(parents=True, exist_ok=False)
    before = {}
    for name in ("LATEST_STATUS_KO.json", "ACTIVE_HANDOFF_KO.json"):
        path = P / name
        shutil.copy2(path, snapshots / name)
        before[name] = read(path)
    for name, old in before.items():
        new = dict(old)
        new.update(
            schema="PALLET_QUERY_VISIBILITY_V156_CURRENT_DIAGNOSTIC_COMPLETE_UNSOLVED",
            updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            goal_status="active_unsolved", fully_solved=False,
            previous_current_evaluation_before_V156=old["latest_current_evaluation"],
            latest_current_evaluation=current, latest_actual_current=current,
            latest_report=str(report), latest_results=result_binding,
            latest_index=result_binding, latest_phase_result=phase_binding,
            latest_phase_seal=phase_binding, latest_phase_result_V156=phase_binding,
            latest_report_QA=binding(V / "ROOT_REPORT_VISUAL_QA.json"),
            latest_completed_work=str(V), active_work=str(V),
            active_work_state="V156_CURRENT_DIAGNOSTIC_COMPLETE_GOAL_UNSOLVED",
            active_prediction_seal=prediction,
            latest_report_kind="V156_FOUR_FROZEN_CURRENT_TRANSFER_DIAGNOSTIC_NOT_SOLUTION",
            selected_model=None, model_candidate=None,
            V156_state="COMPLETE_FOUR_FROZEN_HEADS_SAVED_SEVEN_EVALUATION_AND_FIXED8_ACTUAL_CROPS",
            V156_results=result_binding, V156_phase_result=phase_binding,
            V156_evaluation_seal=evaluation,
            V156_prediction_seal=prediction,
            V156_actual_saved_evaluation=binding(V / "ROOT_ACTUAL_SAVED_SEVEN_EVALUATION_EXECUTION_RECEIPT.json"),
            V156_actual_crop_review=binding(V / "ROOT_ACTUAL_FIXED_CROP_VISUAL_QA.json"),
            V156_selected_model=None, V156_adopted_model=None,
            V156_new_training_encoder_R0_PnP_or_q_L_corrections=0,
            V156_report_review=binding(review),
            latest_user_question_scope="현재 진전과 성공 가능성에 대한 상태 설명",
            current_turn_status_KO="C35/C36/C37/C38 동결 실사평가와8개 실제crop 확인 완료. 같은1237점에서C16 보임1137/1147·외부가림35/90, C35는1036·64, C36은787·76, C37은991·66, C38은992·57. 가림 검출과 보임 보존의 동시개선은 없고 새모델 미채택. 원1784/U45와원좌표 보존, 새학습·PnP·q/L수정0. 전체목표 미해결이며 다음source점검은평가전약속유지.",
            known_limit="외부가림 신호는 있으나새4모델은C16보다보이는점손상이늘었다. near376도오판이남고독립실사좌표gold는0. source기준실패·미채택·현재223은개발노출평가이며blind아님. 성공에가깝다고말할근거는아직없다.",
            next_required_actions=[
                "평가 전에 정한 V155 약속대로 공개·합성 TRAIN과 held source의 대상/context/crop 범위, frame 단위 entity 오류와 knownE 오류를 점검한다.",
                "추가 공개자료 이용 가능성은 공식 manifest 근거를 사용한다. 기존 held45를 재라벨하거나 학습에 옮기지 않는다.",
                "현재223 결과는 다음 구조·손실·문턱·checkpoint·arm 선택에 쓰지 않는다. 새 source-only 설계 전 새 fit을 시작하지 않는다.",
                "WOOD/SQUARE/native8/fullnative35 및 초기q 고정 목표를 유지하며 자기 가림은 상위 기하 범위, PnP·q/L/xy 보정은 수행하지 않는다.",
            ],
            next_safe_source_action=dict(
                **result["next_action_unchanged"]["commitment"],
                state="FROZEN_PRIVATE_BENCHMARK_COMPLETE_SOURCE_CROP_CONTEXT_AND_ENTITY_ERROR_DIAGNOSIS_PENDING",
                chosen_before_private_outcomes=True,
            ),
            latest_progress={
                "full_goal_solved": False, "diagnostic_phase_complete": True,
                "current_phase": phase_binding, "all_original_slots": 1784,
                "input_U": 45, "primary_paired_N": 1237,
                "new_candidate_head_calls_with_repeat": 8,
                "original_C16_parity_calls": 2, "actual_saved_seven_evaluation_complete": True,
                "actual_fixed_crop_views": 8, "actual_plot_views": 3,
                "new_training_encoder_R0_PnP_q_L_changes": 0,
                "selected_model": None,
                "next_action_commitment": result["next_action_unchanged"]["commitment"],
            },
            previous_parent_pointers=str(snapshots), live_processes=[],
            no_user_approval_pending=True,
        )
        for key in ("latest_source_report", "latest_source_training", "latest_source_phase_seal"):
            assert new[key] == old[key]
        path = P / name
        temporary = path.with_suffix(".V156.tmp")
        with temporary.open("x") as stream:
            json.dump(new, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    qa = {
        "completed": True, "PASS": True,
        "published_current_phase": phase_binding,
        "parent_pointers": [binding(P / name) for name in before],
        "previous_parent_snapshots": str(snapshots),
        "latest_source_report_training_and_source_phase_preserved_exact": True,
        "latest_current_evaluation": current,
        "goal_status": "active_unsolved", "selected_model": None,
        "live_processes": [], "next_action_chosen_before_private_outcomes": True,
    }
    for name, old in before.items():
        saved = read(P / name)
        assert saved["latest_current_evaluation"] == current
        assert saved["latest_source_training"] == old["latest_source_training"]
        assert saved["latest_source_report"] == old["latest_source_report"]
        assert saved["goal_status"] == "active_unsolved" and not saved["fully_solved"]
    with (V / "ROOT_PARENT_POINTER_PUBLICATION_QA.json").open("x") as stream:
        json.dump(qa, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"completed": True, "PASS": True, "QA": binding(V / "ROOT_PARENT_POINTER_PUBLICATION_QA.json")}))


if __name__ == "__main__":
    main()
