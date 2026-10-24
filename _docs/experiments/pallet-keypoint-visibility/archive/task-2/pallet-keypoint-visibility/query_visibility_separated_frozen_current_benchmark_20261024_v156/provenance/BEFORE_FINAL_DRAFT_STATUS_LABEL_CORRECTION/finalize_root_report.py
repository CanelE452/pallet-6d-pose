"""Assemble the completed saved benchmark and actually viewed fixed crops."""
from pathlib import Path
from html.parser import HTMLParser
import datetime
import hashlib
import html
import json
import re

V = Path(__file__).resolve().parent


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    draft = V / "report_draft_agent"
    crops = V / "fixed_current_crop_review_v2_alias_preserving"
    root_qa_path = V / "ROOT_ACTUAL_FIXED_CROP_VISUAL_QA.json"
    root_qa = read(root_qa_path)
    crop_receipt = read(crops / "CROP_RENDER_COMPLETE.json")
    assert root_qa["PASS"] and root_qa["all8_actually_viewed"]
    assert root_qa["selected_case_ids_unchanged"]
    assert crop_receipt["PASS"] and crop_receipt["selected_cases"] == 8
    result = read(draft / "ROOT_RESULTS_DRAFT.json")
    result.update(
        schema="V156_FROZEN_SEVEN_CURRENT_BENCHMARK_COMPLETE_GOAL_UNSOLVED",
        draft_only=False, completed=True, PASS=True, stage_complete=True,
        numeric_stage_complete=True, UTC=now, goal_status="active_unsolved",
        goal_solved=False, fully_solved=False, selected_model=None,
        adopted_model=None,
    )
    result["draft_origin"] = binding(draft / "ROOT_RESULTS_DRAFT.json")
    result["visual_status"].update(
        crop_gallery_status="COMPLETE_FIXED8_RENDER_AND_ROOT_ACTUAL_VIEW",
        crop_gallery_full_visual_complete=True,
        reason_KO="원래 봉인한8개 사례를 유지하고 alias 경로 문자열 비교만 고쳤습니다. 원본 SHA·96 crop·중앙32·원래q/L 검증을 유지했고 root가8개 패널과3개 지표 그림을 실제 확인했습니다.",
        crop_render=binding(crops / "CROP_RENDER_COMPLETE.json"),
        root_actual_crop_visual_QA=binding(root_qa_path),
        original_render_failure_preserved=binding(V / "fixed_current_crop_review_v1/FAILED_EXECUTION.json"),
        crop_gallery=binding(crops / "index.html"),
    )
    result["visual_observations"] = root_qa["reviewed_panels"]
    result["visual_interpretation_limits_KO"] = root_qa["observational_limits_KO"]
    result["actual_executions"] = {
        "head_inference": binding(V / "ROOT_ACTUAL_FROZEN_FOUR_HEAD_EXECUTION_RECEIPT.json"),
        "saved_evaluation": binding(V / "ROOT_ACTUAL_SAVED_SEVEN_EVALUATION_EXECUTION_RECEIPT.json"),
        "fixed_crop_render_terminal_exit_code": 0,
        "new_candidate_head_calls_including_repeat": 8,
        "original_C16_parity_head_calls": 2,
        "new_training_encoder_initializer_PnP_or_q_L_correction_calls": 0,
        "all_jobs_terminal": True,
    }
    result["independent_evidence"] = {
        "prediction": binding(V / "independent_saved_prediction_review/REVIEW.json"),
        "evaluation": binding(V / "independent_saved_evaluation_review/REVIEW.json"),
        "summary_and_figures": binding(V / "independent_saved_evaluation_review/SUMMARY_AND_FIGURE_REVIEW.json"),
        "crop_alias_code": binding(V / "independent_preexecution_review/CROP_ALIAS_REPAIR_CODE_REVIEW.json"),
        "legacy_training_scope": binding(V / "independent_saved_evaluation_review/LEGACY_COMPARISON_SCOPE_APPENDIX.json"),
    }
    result["source_only_additional_data_availability"] = binding(V / "source_only_next_data_availability/REVIEW.json")
    result["agent_draft_access"] = result.pop("access")
    result["access"] = {
        "report_assembly_reads_saved_aggregates_receipts_and_generated_PNGs_only": True,
        "phase_current_head_inference_from_saved_feature_packet": True,
        "phase_original_RGB_reads_only_for_fixed_case_visualization": True,
        "private_current223_training_calibration_threshold_checkpoint_arm_or_architecture_selection": False,
        "V112_or_PnP_or_q_L_correction": False,
    }
    result["live_processes"] = []
    write(V / "ROOT_RESULTS_KO.json", result)

    page = (draft / "ROOT_READOUT_DRAFT.html").read_text()
    section = [
        '<p>사전에 봉인한 목재4개·정사각형4개를 실제 확인했습니다. 왼쪽은 원래 초기 좌표를 표시한 전체 프레임, 오른쪽은 모델에 저장되어 있던96 crop과 정확한 중앙32 slice입니다. crop은 확대 표시만 했으며 원래 좌표를 옮기거나 정답 중심으로 다시 자르지 않았습니다.</p>',
        '<p>콘에 가린 위치를 새 모델이 회복한 사례(317·1044)와, 보이는 목재·플라스틱 모서리를 가림으로 오판한 사례(232·880)가 함께 있습니다. 이8개는 조건을 정해 뽑은 진단 사례로, 전체 오류 빈도나 하나의 인과적 실패 원인을 증명하지 않습니다. 야간 암부 사례의 물리 가림 여부도 RGB만으로 새로 확정하지 않았습니다.</p>',
        '<p>원 렌더는 symlink alias와 canonical 경로 문자열 비교 때문에 중단됐습니다. 파일 SHA는 같았습니다. 기존 실패와 사례 목록을 보존하고 경로 검사만 수리하여 같은8개를 렌더했습니다.</p>',
    ]
    for item in root_qa["reviewed_panels"]:
        path = Path(item["panel"]["path"])
        caption = html.escape(item["observation_KO"])
        section.append(f'<figure><img style="width:100%;max-width:1120px" src="{path.relative_to(V)}" alt="고정 사례 {item["global_index"]}"><figcaption>{caption}</figcaption></figure>')
    section.append('<p><a href="fixed_current_crop_review_v2_alias_preserving/index.html">8개 원본 crop 갤러리</a> · <a href="ROOT_ACTUAL_FIXED_CROP_VISUAL_QA.json">실제 시각 확인 기록</a></p>')
    page = re.sub(r'<!-- ROOT_FIXED_CROP_SECTION -->.*?<!-- ROOT_FIXED_CROP_SECTION_END -->', "\n".join(section), page, flags=re.S)
    page = page.replace("실제 crop 확인은 아직 완료되지 않았습니다", "고정8개 실제 crop에서 성공·실패를 함께 확인했습니다")
    page = page.replace("같은 원 q/crop에서 비교할 수 있는 paired1237개", "같은 초기 q를 공통으로 평가할 수 있는 paired1237개")
    page = page.replace("report_draft_agent/ROOT_RESULTS_DRAFT.json", "ROOT_RESULTS_KO.json")
    page = page.replace("이 초안의 구조화된 결과", "완료된 구조화 결과")
    page = page.replace("<body>", '<body><p><strong>V156 저장 평가·실제 crop 확인 완료 / 전체 목표 미해결 / 새 모델 미채택</strong></p>', 1)
    page = page.replace("</body>", '<p>v61·legacy는 같은 초기 query와 평가 집합을 사용하지만 입력 crop 연산자는 DINO 계열과 다릅니다. primary 표의 같은 query 비교를 완전히 같은 모델 입력 tensor 비교라고 해석하지 않습니다.</p></body>')
    final_page = V / "ROOT_READOUT_KO.html"
    with final_page.open("x") as stream:
        stream.write(page)

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if (tag == "a" and key == "href") or (tag == "img" and key == "src"):
                    self.links.append(value)

    parser = Links()
    parser.feed(page)
    local_links = [s for s in parser.links if not s.startswith(("https://", "http://", "#"))]
    assert all((V / s).exists() for s in local_links)
    assert len([s for s in local_links if s.endswith("_PANEL.png")]) == 8
    assert "ROOT_FIXED_CROP_SECTION" not in page
    assert "PENDING_RENDER_REPAIR" not in page
    qa = {
        "completed": True, "PASS": True, "UTC": now,
        "results": binding(V / "ROOT_RESULTS_KO.json"), "HTML": binding(final_page),
        "root_actual_crop_visual_QA": binding(root_qa_path),
        "summary_plot_QA": binding(V / "current_result_summary_agent/FINAL_SUMMARY_QA.json"),
        "root_actually_viewed_all3_scientific_PNGs_and_all8_crop_panels": True,
        "HTML_links_exist": True, "local_links_checked": len(local_links),
        "HTML_browser_preview_performed": False,
        "HTML_QA_scope": "Source/link check plus actual generated-PNG views; no claim of browser-render screenshot.",
        "private_outcomes_changed_next_source_commitment": False,
        "goal_solved": False, "selected_model": None,
    }
    write(V / "ROOT_REPORT_VISUAL_QA.json", qa)
    phase = {
        "schema": "V156_CURRENT_DIAGNOSTIC_PHASE_COMPLETE_GOAL_UNSOLVED",
        "completed": True, "stage_complete": True, "execution_QA_PASS": True,
        "UTC": now, "goal_status": "active_unsolved", "fully_solved": False,
        "source_scientific_qualification_PASS": False,
        "selected_model": None, "adopted_model": None,
        "results": binding(V / "ROOT_RESULTS_KO.json"), "report": binding(final_page),
        "report_QA": binding(V / "ROOT_REPORT_VISUAL_QA.json"),
        "prediction": binding(V / "frozen_four_head_current_benchmark_v1/PREDICTION_SEAL.json"),
        "evaluation": binding(V / "saved_seven_current_evaluation_v1/EVALUATION_SEAL.json"),
        "crop_render": binding(crops / "CROP_RENDER_COMPLETE.json"),
        "root_crop_review": binding(root_qa_path),
        "actual_candidate_models": ["C35", "C36", "C37", "C38"],
        "all_original_slots": 1784, "input_U": 45, "primary_paired_N": 1237,
        "new_training_encoder_R0_PnP_or_q_L_correction": 0,
        "next_action_unchanged": result["next_action_unchanged"], "live_processes": [],
    }
    write(V / "ROOT_PHASE_RESULT.json", phase)
    print(json.dumps({"completed": True, "stage_complete": True, "goal_solved": False, "phase": binding(V / "ROOT_PHASE_RESULT.json")}))


if __name__ == "__main__":
    main()
