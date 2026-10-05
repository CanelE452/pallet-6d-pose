"""Add verified receiver-linkage information to a separate manuscript copy."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import shutil

DOC = Path("_docs/experiments/pallet_remaining_evidence_connection_20261006_v1")
DATA = Path("data/pallet/results/pallet_remaining_evidence_connection_20261006_v1")
OLD_PAPER = Path("_docs/experiments/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/paper_updated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(); root = args.root
    docs, source = root / DOC, root / OLD_PAPER
    paper = docs / "paper_updated"
    cells = []

    def read(rel):
        return json.loads((root / rel).read_text())

    def value(rel, pointer, unit="count", denominator="audit population", seed="not applicable"):
        obj = read(rel)
        for token in pointer.strip("/").split("/"):
            obj = obj[int(token)] if isinstance(obj, list) else obj[token]
        cells.append({"source": str(rel), "json_pointer": pointer, "value": obj, "unit": unit,
                      "denominator": denominator, "seed": seed})
        return obj

    a = DATA / "annotations/CONNECTION.json"
    t = DATA / "time_physical/VALIDATION.json"
    n = DATA / "neutral_intervals/NEUTRAL_COMMAND_RESULT.json"
    sensitivity = DATA / "annotations/LATER_DRAFT72_SENSITIVITY.json"
    frames = value(t, "/timeline/frames", "frame", "all stored frames")
    duplicates = value(t, "/timeline/original_duplicate_observations_retained", "row", "all stored frames")
    square = value(t, "/square119/original_exact_annotations_reused", "image", "square119")
    declared = value(t, "/square119/source_corner_counts/manual_declared", "point", "square119 manual_declared")
    in_frame = value(t, "/square119/source_corner_counts/manual_in_frame", "point", "square119 manual_in_frame")
    sixd = value(t, "/timeline/raw_prediction_exports/0/nested_full_6D_valid_exports/Base", "pose", "all stored frames", "Base42")
    sixd_n3 = value(t, "/timeline/raw_prediction_exports/0/nested_full_6D_valid_exports/N3", "pose", "all stored frames", "N3=1")
    assert sixd == sixd_n3
    primary = value(a, "/denominators/frozen_primary", "task", "original frozen primary")
    repeat = value(a, "/denominators/frozen_repeat_tasks", "task", "original frozen repeat")
    reused = value(a, "/disjoint_task_status_counts/REUSE_COMPLETED_EXPLORATORY_GEOMETRY", "task", "original144task ledger")
    pending_primary = value(a, "/disjoint_task_status_counts/PENDING_REFERENCE", "task", "original144task ledger")
    pending_repeat = value(a, "/disjoint_task_status_counts/PENDING_REAL_REPEAT_RECORD", "task", "original144task ledger")
    excluded_primary = value(a, "/task_status_by_pass/primary/EXCLUDED_BY_USER", "task", "original120primary")
    excluded_repeat = value(a, "/task_status_by_pass/repeat/EXCLUDED_BY_USER", "task", "original24repeat")
    assert primary + repeat == reused + pending_primary + pending_repeat + excluded_primary + excluded_repeat
    intervals = value(n, "/candidate_count", "interval", "all handed neutral command candidates")
    neutral_frames = value(n, "/neutral_command_unique_stored_frames", "frame", "all handed neutral command candidates")
    neutral_fresh = value(n, "/methods/Base/fresh", "frame", "8263 stored neutral command frames", "Base42")
    neutral_fresh_n3 = value(n, "/methods/N3/fresh", "frame", "8263 stored neutral command frames", "N3=1")
    neutral_missing = value(n, "/methods/Base/no_pose", "frame", "8263 stored neutral command frames", "Base42")
    neutral_missing_n3 = value(n, "/methods/N3/no_pose", "frame", "8263 stored neutral command frames", "N3=1")
    assert neutral_fresh == neutral_fresh_n3 and neutral_missing == neutral_missing_n3
    joins = value(t, "/physical_reference/fixed_reference_join_rows", "target observation", "square119+lifter8910")
    accepted = value(t, "/physical_reference/accepted_independent_reference_rows", "target observation", "square119+lifter8910")
    assert accepted == 0
    old_files = []
    for p in sorted(source.rglob("*")):
        if p.is_file():
            rel = p.relative_to(source); target = paper / rel
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, target)
            old_files.append({"path": str(rel), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    main_note = (f"추가 자료 인계 후 원영상 {frames:,}개 저장 픽셀과 센서·host 시각의 대응을 대조했고 중복 관측 {duplicates}개를 보존하였다. "
                 f"정사각형 {square}장의 원사진과 원주석 해시도 일치하여 {declared}/{in_frame}점 모드를 그대로 재사용하였다. "
                 f"리프터 요약 출력과 달리 원시 내부 기록에는 각 방법의 유효한 전체 3D 이동·회전 예측 {sixd:,}개가 이미 존재한다. "
                 f"그러나 정사각형과 리프터의 {joins:,}개 대상 관측에 독립 실측으로 연결된 참조는 {accepted}개이므로 물리 T/R의 x를 교체하지 않았다. "
                 f"중립 CAN 명령 {intervals}구간의 {neutral_frames:,}개 저장 출력은 별도 사후 기술 자료로 대조했으며 실제 상대 정지가 확인되지 않아 정지 잡음으로 해석하지 않는다. "
                 "공식 직접 가시 참조·반복 검수의 미완료 상태와 완료된 기하 보조 참조를 구분한 연결 검산은 보충자료에 제시한다.")
    p = paper / "sections/06_case_study.tex"
    text = p.read_text(); marker = "\\begin{table}[t]"
    assert marker in text
    p.write_text(text.replace(marker, main_note + "\n\n" + marker, 1))
    p = paper / "manuscript_ko.md"
    text = p.read_text(); marker = "\n# 논의와 한계"
    assert marker in text
    p.write_text(text.replace(marker, "\n## 추가 인계 자료와의 연결 검산\n\n" + main_note + "\n" + marker, 1))
    linkage_note = (
        f"원래 검수 계획은 주 {primary}건과 같은 사진의 반복 {repeat}건으로, 고유 사진 {primary}장이다. "
        f"실제 사용자 제외 기록은 주 {excluded_primary}건·반복 {excluded_repeat}건이며 제외 사진을 재투입하지 않았다. "
        f"기존 {reused}장의 탐색적 기하 참조를 주 기록에 재사용할 수 있고, 미확보 주 참조 {pending_primary}건·실제 반복 기록 {pending_repeat}건은 따로 남긴다. "
        "이 원장 집계는 공식 전체 가시 코너 정확도나 반복 품질의 완료가 아니다. 원계획의 분모를 소급 변경하거나 주 좌표를 반복 주석으로 복사하지 않았다. "
        "추가 클릭을 새로 요구하지 않았으며, 승인 미제출인 후속 좌표 저장본의 버전 민감도는 아래에 별도로 제시한다.")
    table = ("\\begin{table}[t]\\centering\\footnotesize\n"
             "\\caption{추가 인계 자료와 수신 측 기존 결과의 연결 검산. 정확도 표가 아니다.}\\label{sup:remaining_connection}\n"
             "\\begin{tabularx}{\\columnwidth}{Yrr}\\toprule\n항목 & 분모 & 확인 수 \\\\\\midrule\n"
             f"원영상 픽셀·시각 연결 & {frames:,} & {frames:,} \\\\\n"
             f"보존한 중복 관측 & {frames:,} & {duplicates} \\\\\n"
             f"정사각형 원사진·주석 & {square} & {square} \\\\\n"
             f"기존 full 6D 예측 / 방법 & {frames:,} & {sixd:,} \\\\\n"
             f"독립 물리 참조 연결 & {joins:,} & {accepted} \\\\\n"
             "\\bottomrule\\end{tabularx}\n\\end{table}\n")
    command_note = (f"모델 출력을 보지 않고 명령 기록에서 정의한 모든 {intervals}개 중립 CAN 구간을 재사용하였다. "
                    f"센서 시각으로 연결한 저장 프레임은 {neutral_frames:,}개이며 Base와 N3 각각 fresh {neutral_fresh:,}개, no-pose {neutral_missing}개, held 0개이다. "
                    "동일 구간 안에서 x/z의 모집단 표준편차와 360도 wrapping만 제거한 yaw 표준편차를 계산했고 90/180도 가지 변화와 결측을 남겼다. "
                    "이는 명령 조건의 출력 분산이며 물리 정지·정착·잡음·정확도를 증명하지 않는다. 구간을 낮은 분산 순으로 고르거나 이동 구간을 제거하지 않았다. "
                    "실제 상대 정지 승인과 사전 선정 이력의 근거가 없어 원래 정지 evaluator의 x는 유지한다.")
    neutral_table = ("\\begin{table}[t]\\centering\\footnotesize\n"
                     "\\caption{중립 명령 32구간의 출력 가용성. 상대 정지 미확인 사후 기술 패널.}\\label{sup:command_intervals}\n"
                     "\\begin{tabularx}{\\columnwidth}{Yrr}\\toprule\n항목 & Base & N3 \\\\\\midrule\n"
                     f"저장 프레임 & {neutral_frames:,} & {neutral_frames:,} \\\\\n"
                     f"fresh & {neutral_fresh:,} & {neutral_fresh_n3:,} \\\\\n"
                     f"no-pose & {neutral_missing} & {neutral_missing_n3} \\\\\n"
                     "held & 0 & 0 \\\\\n정지 잡음 & \\notrun & \\notrun \\\\\n"
                     "\\bottomrule\\end{tabularx}\n\\end{table}\n")
    point_denominator = value(sensitivity, "/later_fixed_point_denominator", "point", "fixed later saved coordinates")
    preserved_clicks = value(sensitivity, "/original_manual_points_preserved_exactly", "point", "fixed later saved coordinates")
    changed_pnp_slots = value(sensitivity, "/later_saved_coordinates_replacing_original_pnp_coordinates", "point", "fixed later saved coordinates")
    self_slots = value(sensitivity, "/known_no_coordinate_self_occlusion_slots", "slot", "fixed12frame96slots")
    sensitivity_frames = value(sensitivity, "/later_fixed72_saved_draft_coordinates_metrics/frame_count", "frame", "fixed later saved coordinates")
    sensitivity_note = (f"후속 저장본의 같은 {sensitivity_frames}장·{point_denominator}좌표만을 고정하여 참조 버전 민감도를 확인하였다. 기존 직접 클릭 {preserved_clicks}좌표는 정확히 같고 원래 PnP였던 {changed_pnp_slots}자리에는 실제 나중 저장 좌표가 존재한다. "
                        f"나머지 자체 가림 {self_slots}자리는 좌표를 새로 생성하지 않았다. 저장본은 승인 미제출 초안이며 evaluation\\_use=false이므로 이 결과를 공식 직접 가시 참조 정확도라고 주장하지 않는다. "
                        "원래 96점 결과를 교체하거나 오차를 보고 참조 버전을 선택하지 않았으며 각 표의 분모를 구분하였다.")
    sensitivity_md = "| 방법 | 중앙값 px | P90 px | PCK≤10px | 유효/전체 | 실패 |\n| --- | --- | --- | --- | --- | --- |\n"
    sensitivity_table = ("\\begin{table}[t]\\centering\\footnotesize\n"
                         "\\caption{미제출 후속 저장본의 참조 버전 민감도(12장·72좌표). 공식 정확도와 별도.}\\label{sup:reference_version_sensitivity}\n"
                         "\\begin{tabularx}{\\columnwidth}{Yrrrr}\\toprule\n방법 & 중앙 px & P90 px & PCK(\\%) & 유효/전체 \\\\\\midrule\n")
    for method in ("Base", "N3"):
        base = f"/later_fixed72_saved_draft_coordinates_metrics/methods/{method}"
        median = value(sensitivity, base + "/conditional_error/median_px", "px", "fixed72 saved draft coordinates", "Base42,N3=1")
        p90 = value(sensitivity, base + "/conditional_error/p90_px", "px", "fixed72 saved draft coordinates", "Base42,N3=1")
        pck = value(sensitivity, base + "/pck10_full_reference_percent", "percent", "fixed72 saved draft coordinates", "Base42,N3=1")
        hits = value(sensitivity, base + "/pck10_hit_count", "point", "fixed72 saved draft coordinates", "Base42,N3=1")
        valid = value(sensitivity, base + "/valid_matching_prediction_points", "point", "fixed72 saved draft coordinates", "Base42,N3=1")
        failures = value(sensitivity, base + "/failed_reference_points", "point", "fixed72 saved draft coordinates", "Base42,N3=1")
        sensitivity_table += f"{method} & {median:.3f} & {p90:.3f} & {pck:.3f} & {valid}/{point_denominator} \\\\\n"
        sensitivity_md += f"| {method} | {median:.3f} | {p90:.3f} | {hits}/{point_denominator} ({pck:.3f}%) | {valid}/{point_denominator} | {failures} |\n"
    sensitivity_table += "\\bottomrule\\end{tabularx}\n\\end{table}\n"
    for name, content in [("remaining_connection.tex", table), ("neutral_command_coverage.tex", neutral_table), ("later_reference_sensitivity.tex", sensitivity_table)]:
        (paper / "supplement_tables" / name).write_text(content)
    figure = "all_32_command_spreads.png"
    shutil.copyfile(docs / "neutral_intervals/images" / figure, paper / "figures" / figure)
    latex = ("\n\\section{추가 인계 자료와 기존 결과의 연결 검산}\n" + linkage_note + "\n"
             "\\input{supplement_tables/remaining_connection}\n"
             "\\subsection{중립 명령 조건의 저장 출력}\n" + command_note + "\n\\input{supplement_tables/neutral_command_coverage}\n"
             "\\begin{figure}[t]\\centering\n\\includegraphics[width=\\columnwidth]{figures/" + figure + "}\n"
             "\\caption{모든 중립 명령 후보의 구간별 출력 분산. 실제 상대 정지 미확인으로 정지 잡음이 아니다.}\\label{sup:neutral_command_spread}\n\\end{figure}\n"
             "\\subsection{후속 좌표 저장본의 참조 버전 민감도}\n" + sensitivity_note + "\n\\input{supplement_tables/later_reference_sensitivity}\n")
    p = paper / "supplement.tex"; text = p.read_text(); assert text.count("\\end{document}") == 1
    p.write_text(text.replace("\\end{document}", latex + "\\FloatBarrier\n\\end{document}"))
    markdown = ("\n# 추가 인계 자료와 기존 결과의 연결 검산\n\n" + linkage_note + "\n\n"
                "| 항목 | 분모 | 확인 수 |\n| --- | --- | --- |\n"
                f"| 원영상 픽셀·시각 연결 | {frames:,} | {frames:,} |\n| 보존한 중복 관측 | {frames:,} | {duplicates} |\n"
                f"| 정사각형 원사진·주석 | {square} | {square} |\n| 기존 full 6D 예측/방법 | {frames:,} | {sixd:,} |\n| 독립 물리 참조 연결 | {joins:,} | {accepted} |\n\n"
                "## 중립 명령 조건의 저장 출력\n\n" + command_note + "\n\n"
                f"| 항목 | Base | N3 |\n| --- | --- | --- |\n| 저장 프레임 | {neutral_frames:,} | {neutral_frames:,} |\n"
                f"| fresh | {neutral_fresh:,} | {neutral_fresh_n3:,} |\n| no-pose | {neutral_missing} | {neutral_missing_n3} |\n| held | 0 | 0 |\n| 정지 잡음 | x | x |\n\n"
                f"![모든 중립 명령 후보 출력 분산](figures/{figure})\n\n"
                "## 후속 좌표 저장본의 참조 버전 민감도\n\n" + sensitivity_note.replace("evaluation\\_use", "evaluation_use") + "\n\n" + sensitivity_md + "\n")
    p = paper / "supplement_ko.md"; p.write_text(p.read_text() + markdown)
    patch = []; changed = []
    for p in sorted(paper.rglob("*")):
        if not p.is_file(): continue
        rel = p.relative_to(paper); old = source / rel
        if not old.exists() or old.read_bytes() != p.read_bytes():
            changed.append({"path": str(rel), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
            if p.suffix in (".tex", ".md"):
                before = old.read_text().splitlines(keepends=True) if old.exists() else []
                patch.extend(difflib.unified_diff(before, p.read_text().splitlines(keepends=True),
                             fromfile="a/" + str(rel) if old.exists() else "/dev/null", tofile="b/" + str(rel)))
    (docs / "RECEIVER_EVIDENCE_PAPER.patch").write_text("".join(patch))
    assert all(hashlib.sha256((source / item["path"]).read_bytes()).hexdigest() == item["sha256"] for item in old_files)
    (docs / "PAPER_INSERTION_CELL_MAP.json").write_text(json.dumps({"cells": cells, "count": len(cells), "rounding": "integers exact; px/percent tables use3decimals", "insertion_status": "INSERTED"}, ensure_ascii=False, indent=2) + "\n")
    (docs / "PAPER_INTEGRATION_RECEIPT.json").write_text(json.dumps({"status": "COMPLETE_RECEIVER_CONNECTION_INSERTION_WITH_REFERENCE_GAPS",
               "original_paper_source": str(OLD_PAPER), "output_paper": str(DOC / "paper_updated"),
               "previous_paper_original_files_preserved": len(old_files), "modified_or_added_files": changed,
               "inserted_source_cells": len(cells), "source_files": old_files,
               "original96_point_results_unchanged": True, "official_reference_stationary_and_physical_x_retained": True,
               "new_reference_approval": False, "new_training": 0, "new_model_inference": 0, "new_PnP": 0, "PDF_generated": False}, ensure_ascii=False, indent=2) + "\n")
    matrix = read(DOC / "PAPER_GAP_MATRIX.json")
    for entry in matrix["entries"]:
        if entry["status"] == "COMPUTED_AND_INSERT_PENDING": entry["status"] = "COMPUTED_AND_INSERTED"
    (docs / "PAPER_GAP_MATRIX.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "source_cells_inserted": len(cells), "previous_paper_files_preserved": len(old_files),
                      "changed_or_added_files": len(changed), "official_x_unchanged": True}))


if __name__ == "__main__":
    main()
