"""Additive finalization of already selected and directly viewed saved source figures.

Reads only this audit's JSON/PNG bytes. No original RGB, mask, NPZ, model or P3 IO.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def binding(path):
    p = Path(path)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size}


def load(name):
    return json.loads((ROOT / name).read_text())


def write_new(name, value):
    p = ROOT / name
    with p.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return binding(p)


OBSERVATIONS = [
    {
        "actual_RGB_KO": "붉은 드릴이 목재 블록 위에 있다. q/r은 붉은 드릴의 실제 윤곽 밖 회색 영역에 놓이며, 아래쪽에 밝은 목재 블록 윗면이 보인다. 실제 모델 RGB96에도 붉은 드릴 경계와 주변 회색 영역이 함께 들어온다.",
        "target_identity_and_owner_KO": "목표는 저장 metadata의 class15 드릴, instance1이다. r의 유일한 OTHER owner는 instance2/class16 목재 블록으로 기록돼 있다. 다만 q/r의 회색 픽셀이 물리적으로 목재 표면인지 RGB만으로 확정하지 않는다.",
        "interpretation_KO": "관측 range가 가상 endpoint보다 약139mm 뒤라는 CLEAR 교사와 외부 가림 p=0.734가 어긋난 FP이다. 물리적 드릴 코너가 직접 보인다는 D 교사가 아니므로 '보이는 드릴 코너를 지웠다'고 해석할 수 없다.",
        "remaining_uncertainty_KO": "가상 cuboid endpoint와 실제 드릴 구조 단서의 대응, raster owner와 실제 픽셀 재질의 일치가 확인되지 않았다. foreground 모양만을 외부 가림으로 읽었는지는 이 사례만으로 증명되지 않는다.",
    },
    {
        "actual_RGB_KO": "비슷한 드릴/목재 장면에서 q/r이 드릴 실제 몸체 밖 회색 영역에 놓인다. 실제 모델 RGB96은 드릴의 붉은 가장자리, 회색 영역, 밝은 목재 윗면 일부를 포함한다.",
        "target_identity_and_owner_KO": "목표 class15 드릴 instance1, 저장 OTHER owner는 instance2/class16 목재 블록이다. 실제 q/r 픽셀의 물리적 표면 재질은 RGB만으로 확정하지 않는다.",
        "interpretation_KO": "관측 range가 가상 endpoint보다 약133mm 뒤인 CLEAR인데 p_external=0.770으로 FP이다. case00과 self proxy 값이 반대인 두 사례 모두 FP라는 사실은 기록하되 self proxy의 인과 효과를 뜻하지 않는다.",
        "remaining_uncertainty_KO": "실제 물리 코너의 보임 여부와 가상 reference endpoint까지의 ray-clear 여부는 다르다. 이 FP를 인간 D/E 오류와 동일시할 수 없다.",
    },
    {
        "actual_RGB_KO": "목표 목재 블록 앞/위에 붉은 드릴이 놓인다. q/r은 붉은 드릴 몸체 안에 있고, 실제 모델 RGB96에도 붉은 표면과 검은 세부가 분명하게 보인다. 초록 M4 윤곽은 목표 목재 블록을 구분하기 위한 GT 표시이다.",
        "target_identity_and_owner_KO": "목표 class16 목재 블록 instance2, OTHER owner는 instance1/class15 붉은 드릴로 기록된다. full RGB에서 드릴과 목재라는 다른 물체의 전후 배치와 양립하는 시각적 증거가 있다.",
        "interpretation_KO": "q-r=0.574px, local canvas valid=1인데 external teacher의 약159mm 전방 물체를 p_external=0.331로 놓친 FN이다. 큰 좌표 이탈이나 로컬 crop의 canvas 누락이 이 사례의 설명은 아니다.",
        "remaining_uncertainty_KO": "목표 instance와 앞 물체의 관계를 조건부로 구분하거나 depth order를 학습하는 데 실패했을 가능성을 보여준다. 목표 조건화, 표현, 손실, 데이터 분포 중 어느 하나가 단독 원인인지는 저장 사례만으로 분리하지 못한다. 인간의 물리 코너 E gold는 없다.",
    },
    {
        "actual_RGB_KO": "실제 모델 RGB96은 native 25.77px 범위의 거의 균일한 밝은 흰색/분홍색 병 표면이다. 별도의 native96 상세에는 흰 병의 라벨/가장자리와 왼쪽 노란 병이 더 보이지만, 이 상세는 실제 로컬 모델 입력보다 넓은 범위이다.",
        "target_identity_and_owner_KO": "목표는 metadata class4 instance0이며 이 로컬 흰색 패치에서 목표를 직접 식별할 수 없다. OTHER owner class12 instance4 흰 병은 full RGB의 물체와 양립한다. 목표 M4의 작고 가려진 윤곽만으로 목표의 물리적 구조를 인증하지 않는다.",
        "interpretation_KO": "q-r=0.751px, local canvas valid=1, 외부 teacher range gap 약350mm인데 p_external=0.0499인 FN이다. 로컬 RGB branch에는 목표 정체성과 경계 정보가 적다는 관찰이 가능하다.",
        "remaining_uncertainty_KO": "P3의 수용영역, targetRGB128, context42에는 더 넓은 정보가 있을 수 있다. 이 진단에서는 P3/targetRGB128을 열거나 새 forward를 하지 않았으므로 전체 모델의 관측 불가능이나 작은 L이 단독 원인이라는 결론은 내리지 않는다.",
    },
    {
        "actual_RGB_KO": "노란 병 위쪽의 금색 캔 뚜껑/테두리와 주변 배경 근처에 q/r이 있다. 실제 모델 crop은 캔 뚜껑과 노란 병 상단 일부를 함께 포함한다.",
        "target_identity_and_owner_KO": "목표 class5 노란 병 instance1, OTHER owner는 instance3/class9 캔으로 저장돼 있다. q/r은 목표 노란 병의 직접 보이는 물리 표면으로 확인되지 않는다.",
        "interpretation_KO": "OTHER owner가 있어도 그 관측 range가 가상 endpoint보다 약193mm 뒤라 CLEAR이다. p_external=0.0991의 정답은 단순 '다른 물체가 보임'과 external-positive가 같지 않다는 대조 사례이다.",
        "remaining_uncertainty_KO": "CLEAR 정답을 실제 노란 병 코너의 보임 D나 인간 semantic 정답으로 확장하지 않는다. 한 대조 사례로 학습 규칙이 인증되지는 않는다.",
    },
    {
        "actual_RGB_KO": "q/r이 흰 병 밑부분/가장자리와 천 바닥 근처에 있다. 실제 모델 crop에는 흰 병 밑면 윤곽, 회색/천 영역, 왼쪽 노란 병 가장자리가 들어온다.",
        "target_identity_and_owner_KO": "목표 class5 노란 병 instance1, OTHER owner는 instance4/class12 흰 병으로 저장돼 있다. 관찰된 흰 병 배치가 OTHER owner 증거와 양립한다.",
        "interpretation_KO": "가상 endpoint보다 관측 물체가 약28mm 앞이며 고정15mm margin을 넘는 EXTERNAL 교사이다. p_external=0.8421로 proxy 정답이다.",
        "remaining_uncertainty_KO": "FN 두 사례보다 margin이 작고, 병 밑 경계의 pose/raster 오차에 대한 물리 정확도 교정은 없다. 실제 목표 코너의 인간 E 증거나 인과적인 성공 이유를 확정하지 않는다.",
    },
]


def main():
    selection = load("SELECTION_AND_COHORT_LOCK_BEFORE_RGB.json")
    owner = load("SELECTED_OWNER_METADATA_PROVENANCE.json")
    render = load("RENDER_AND_PIXEL_PROVENANCE_RECEIPT.json")
    layout = load("LAYOUT_QA_V2/LAYOUT_REPAIR_RECEIPT.json")
    assert selection["cohort_N"] == 1538 and selection["selected_case_N"] == 6
    assert render["P3_members_deserialized"] == 0
    assert layout["PASS"] and all(c["all_source_data_panels_identical_crop_bytes"] for c in layout["cases"])
    assert binding(ROOT / "ALL_FIXED_SIX_CONTACT.png")["sha256"] == "b77f35924324dd6275b72d3cf2283856f9918ebd360ea8fc79fbfa60db2a708a"
    assert binding(ROOT / "LAYOUT_QA_V2/ALL_FIXED_SIX_CONTACT.png")["sha256"] == "31b16d5a8786265159e5465bf62cff417517af7c98e054fd952cc8564881d0b9"
    cases = []
    for n, selected in enumerate(selection["selected_cases"]):
        p = ROOT / f"CASE_{n:02d}_PIXEL_AND_PROVENANCE.json"
        d = json.loads(p.read_text())
        assert d["selection"] == selected and d["local_canvas_valid_fraction"] == 1.0
        assert not d["actual_targetM4_at_q"] and not d["actual_targetM4_at_reference_pixel"]
        o = owner["cases"][n]
        assert o["target_id"] == selected["id"] and o["case_number"] == n
        t = d["original_teacher_numeric_metadata"]
        gap = (t["ref_range_mm8"] - t["observed_range_hi_mm8"] if selected["GT"] == 1
               else t["observed_range_lo_mm8"] - t["ref_range_mm8"])
        assert gap > 15 and selected["Euclidean_q_r_native_px"] <= 3
        cases.append({
            **selected, **OBSERVATIONS[n], "pixel_and_provenance": binding(p),
            "figure_V2": layout["cases"][n]["layoutV2"],
            "figure_V1_preserved": layout["cases"][n]["original"],
            "teacher_numeric_metadata_original_unchanged": t,
            "target_and_owner_original_metadata": o,
            "teacher_conservative_range_gap_mm": gap,
            "local_canvas_valid_fraction": d["local_canvas_valid_fraction"],
            "targetM4_at_rounded_q": d["actual_targetM4_at_q"],
            "targetM4_at_rounded_reference": d["actual_targetM4_at_reference_pixel"],
            "visual_observation_vs_proxy_distinguished": True,
            "no_case_replacement": True,
        })
    report = {
        "completed": True, "PASS": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": binding(ROOT / "SCOPE_BEFORE_INDIVIDUAL_SCORES_AND_RGB.json"),
        "selection_seal": binding(ROOT / "SELECTION_SEAL_BEFORE_RGB.json"),
        "selection": binding(ROOT / "SELECTION_AND_COHORT_LOCK_BEFORE_RGB.json"),
        "render_receipt": binding(ROOT / "RENDER_AND_PIXEL_PROVENANCE_RECEIPT.json"),
        "layout_V2_receipt": binding(ROOT / "LAYOUT_QA_V2/LAYOUT_REPAIR_RECEIPT.json"),
        "actual_contact_V2": binding(ROOT / "LAYOUT_QA_V2/ALL_FIXED_SIX_CONTACT.png"),
        "original_contact_V1_preserved": binding(ROOT / "ALL_FIXED_SIX_CONTACT.png"),
        "cohort": {
            "definition": selection["selection_rule_exact"]["cohort"], "N": 1538,
            "teacher_clear_N": 1019, "teacher_external_N": 519,
            "CM_GT_clear_external_by_pred_clear_external": [[837, 182], [150, 369]],
            "strata": [{"name": s["name"], "eligible_N": s["eligible_N"], "empty": s["empty"]} for s in selection["strata"]],
            "selected_N": 6, "all_six_local_canvas_valid_fraction": 1.0,
            "all_six_q_and_r_targetM4_zero": True,
            "emulated_source_near3_not_current_manual_click_near3": True,
            "fixed_diagnostic_cases_not_prevalence_sample_or_new_gate": True,
        },
        "conclusions_KO": [
            "작은 좌표 오차, 입력 유효성, reference의 crop 포함을 모두 충족해도 external-ray proxy FP/FN이 남는다. 선택된 여섯 경우의 로컬 canvas valid는 모두 1이다.",
            "두 FP의 CLEAR는 가상 cuboid endpoint 전까지 다른 표면이 없다는 range 교사이다. 실제 대상 코너의 직접 보임을 인증하지 않으므로 이 FP를 물리적 D 제거로 해석하지 않는다.",
            "목재 목표 앞 붉은 드릴 FN은 실제 모델 crop에서도 앞 물체가 드러나는 실패이다. 정의 차이만으로 모든 proxy 오류를 설명할 수 없고 목표 관계/표현 학습 실패 가능성이 남는다.",
            "다른 FN의 실제 로컬 모델 범위는 25.77 nativepx이며 거의 흰 병 표면뿐이다. 별도 native96 그림에 추가 맥락이 보이지만 이것을 모델이 받은 로컬 입력으로 착각하면 안 된다.",
            "모든 선택 사례에서 q/r의 target M4는 0인데 teacher clear와 external이 모두 있다. M4=0은 external-ray label이나 자기 가림의 대체 정의가 아니다.",
            "관찰은 목표 조건화/전후 관계/로컬 단서 부족 가설과 양립하나 단독 원인을 확정하지 않는다. frozen source 성능 gate는 변경하지 않으며 인간 semantic D/E/S 해결이나 현재 데이터 전이를 주장하지 않는다.",
        ],
        "meaning_limits_KO": [
            "CLEAR != 인간 D; EXTERNAL_RAY_PROXY != 인간 E; CONVEX_SELF_PROXY != 인간 S.",
            "r은 고정된 가상 cuboid reference이며 실제 물리 표면점이나 visible cue일 필요가 없다. 물리 cue/native8 대응은 이 검사에서 새로 부여하지 않았다.",
            "원래 range, pose, 보이는 instance mask owner는 보존된 약한 source 교사이다. 고정15mm margin을 쓰지만 pose/raster/깊이 오차의 실제 물리 정확도는 교정하지 않았다.",
            "VAL은 연구 중 이미 사용된 source이며 독립적인 새 배포 성능 검증이 아니다. error minSHA 사례는 분포를 대표하는 무작위 표본이 아니다.",
            "이 보고서는 모델 선택/학습/문턱 변경 근거로 사용하지 않았으며 현재223 RGB/GT/score를 읽지 않았다.",
        ],
        "cases": cases,
        "actual_views": {"selected_raw_full_RGB_N": 6, "individual_V1_figures_N": 6,
                         "individual_V2_figures_N": 6, "V2_all_six_contact_directly_viewed": True},
        "model_RGB96_geometry": "q+((j+.5)/96-.5)*L; q at index47.5; original L unchanged; separate native96 is diagnostic only",
        "new_model_forward": 0, "new_training": 0, "current_data_reads": 0, "PnP_calls": 0,
        "P3_members_deserialized": 0, "checkpoint_parameter_loads": 0,
        "NPZ_hashing_scope_KO": "선택된 source NPZ 컨테이너의 바이트 SHA에는 압축된 P3 멤버 바이트도 포함되지만, 열고 역직렬화한 멤버는 RGB96/valid96뿐이다.",
        "coords_L_source_order_teacher_threshold_model_gates_unchanged": True,
    }
    report_binding = write_new("SOURCE_FAILURE_ACTUAL_OBSERVATIONS_KO.json", report)
    qa = {
        "completed": True, "PASS": True,
        "observation_report": report_binding,
        "direct_visual_QA": report["actual_views"],
        "V1_original_files_preserved": True,
        "V1_contact_SHA256": binding(ROOT / "ALL_FIXED_SIX_CONTACT.png")["sha256"],
        "V2_contact": binding(ROOT / "LAYOUT_QA_V2/ALL_FIXED_SIX_CONTACT.png"),
        "all_six_V2_source_data_panels_identical_to_V1": True,
        "V2_change": "Legend overlap corrected in new paths only; original model/native/scene/valid pixel panels copied unchanged.",
        "selection_ID_q_r_L_score_teacher_parity": True,
        "new_forward_training_current_PnP": 0,
    }
    write_new("VISUAL_QA.json", qa)
    files = [binding(p) for p in sorted(ROOT.rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.name != "FINAL_DIAGNOSTIC_SEAL.json"]
    seal = {"completed": True, "PASS": True, "files": files,
            "scope": binding(ROOT / "SCOPE_BEFORE_INDIVIDUAL_SCORES_AND_RGB.json"),
            "selection_seal": binding(ROOT / "SELECTION_SEAL_BEFORE_RGB.json"),
            "observation_report": report_binding,
            "visual_QA": binding(ROOT / "VISUAL_QA.json"),
            "contact_V2": binding(ROOT / "LAYOUT_QA_V2/ALL_FIXED_SIX_CONTACT.png"),
            "old_contact_V1_preserved": binding(ROOT / "ALL_FIXED_SIX_CONTACT.png"),
            "unchanged_coords_teacher_weights_threshold_gates": True,
            "new_forward_training_current_PnP": 0,
            "human_semantic_gold_claim": False, "solved": False}
    print(json.dumps({"report": report_binding,
                      "seal": write_new("FINAL_DIAGNOSTIC_SEAL.json", seal),
                      "contact_V2": seal["contact_V2"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
