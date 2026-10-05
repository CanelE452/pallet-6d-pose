#!/usr/bin/env python3
"""Insert verified, separate 12-frame assisted geometry results into a copy only."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "_docs/experiments/pallet_combined_closeout_20261003_v1"
SOURCE = DOCS / "closeout_20261006_v1/paper_updated"
OUT = DOCS / "pnp_assisted_lifter_20261006_v1"
COPY = OUT / "paper_updated"
PATCH = OUT / "paper_patch"
RESULT_PATH = ROOT / "data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/evaluations/20261005T182244122312Z/RESULT.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise ValueError(f"Expected one replacement in {path}: {text.count(old)}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    start = time.monotonic()
    if COPY.exists() or PATCH.exists():
        raise RuntimeError("New manuscript destination already exists; preserve it and review its receipt.")
    d = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert d["status"] == "EVALUATED_EXPLORATORY_ASSISTED_GEOMETRY_12_FRAMES"
    assert d["metrics"]["frame_count"] == 12
    assert d["metrics"]["reference_point_count"] == 96
    assert d["metrics"]["decision_counts"] == {"same": 12}
    assert d["metrics"]["reference_source_counts"] == {"manual_click": 66, "pnp_projected": 30}
    assert set(d["metrics"]["sessions"]) == {"173507", "174126", "174342", "174925"}
    assert d["canonical_corner_ids"] == list(range(8))
    assert d["full_120_24_completed"] is False
    assert d["independent_physical_TR"] == d["official_visible_corner_accuracy"] == d["stationary_noise"] == "x"
    assert d["training_runs"] == d["optimizer_updates"] == d["new_model_forward_frames"] == d["hardware_control_calls"] == 0
    sources = dict(d["sources"])
    sources["result"] = {"path": str(RESULT_PATH), "sha256": sha(RESULT_PATH)}
    recovery_path = ROOT / "data/pallet/results/pallet_lifter_case_review_20261003_v1/review/VIEWER_DRAFT_RECOVERY.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovered_coordinates = recovered_self_occluded = 0
    for frame in d["frame_results"]:
        record = recovery["drafts"][frame["frame_id"]+"|primary"]["record"]
        for corner in record["corners"]:
            recovered_coordinates += corner.get("x") is not None and corner.get("y") is not None
            recovered_self_occluded += corner.get("self_occlusion") is True
    assert (recovered_coordinates, recovered_self_occluded) == (72, 24)
    sources["later_visibility_recovery_not_evaluation_reference"] = {"path": str(recovery_path), "sha256": sha(recovery_path), "coordinate_count": 72, "self_occlusion_count": 24, "use": "version description only; never metric coordinates"}
    for name, item in sources.items():
        assert sha(Path(item["path"])) == item["sha256"], name
    for item in d["overlays"]:
        assert sha(Path(item["path"])) == item["sha256"]
    source_before = snapshot(SOURCE)
    shutil.copytree(SOURCE, COPY)
    PATCH.mkdir(parents=True)
    figroot = COPY / "figures/pnp_assisted_lifter_12"
    figroot.mkdir()
    for item in d["overlays"]:
        shutil.copy2(item["path"], figroot / Path(item["path"]).name)

    cells: list[dict] = []

    def value(pointer: str):
        node = d
        for key in pointer.strip("/").split("/"):
            node = node[int(key)] if isinstance(node, list) else node[key]
        return node

    def cell(pointer: str, file: str, label: str, row: str, unit: str, decimals: int | None = 3) -> str:
        raw = value(pointer)
        text = str(raw) if decimals is None else f"{raw:.{decimals}f}"
        cells.append({"source_path": str(RESULT_PATH), "source_sha256": sha(RESULT_PATH), "source_json_pointer": pointer,
                      "source_value": raw, "display": text, "unit": unit, "denominator": 24 if pointer.startswith("/metrics/sessions/") else 96,
                      "seed": 42 if "/Base/" in pointer else 1,
                      "seed_scope": "base_training" if "/Base/" in pointer else "N3_refiner_training",
                      "base_training_seed": 42, "refiner_training_seed": None if "/Base/" in pointer else 1,
                      "file": str(COPY / file), "label": label, "row": row})
        return text

    def total_rows(file: str, label: str, md: bool) -> list[str]:
        rows = []
        for name in ("Base", "N3"):
            ptr = f"/metrics/methods/{name}"
            vals = [name, cell(ptr+"/conditional_error/median_px", file, label, name, "px"),
                    cell(ptr+"/conditional_error/p90_px", file, label, name, "px"),
                    cell(ptr+"/pck10_full_reference_percent", file, label, name, "%"),
                    cell(ptr+"/pck10_hit_count", file, label, name, "points", None)+"/96",
                    cell(ptr+"/valid_matching_prediction_points", file, label, name, "points", None)+"/"+
                    cell(ptr+"/failed_reference_points", file, label, name, "points", None)]
            rows.append("|"+"|".join(vals)+"|" if md else " & ".join(vals)+r" \\")
        return rows

    main_table = r"""\begin{table}[t]\centering\footnotesize
\caption{사람이 확인한 PnP 보조 기하 참조의 별도 12장 탐색 패널. YOLO Base와 N3(seed 1).}\label{tab:lifter_assisted_geometry}
\begin{tabularx}{\columnwidth}{Yrrrrr}\toprule
방법 & 중앙(px) & P90(px) & PCK(\%) & 적중 & 유효/실패 \\\midrule
""" + "\n".join(total_rows("tables/lifter_assisted_results.tex", "tab:lifter_assisted_geometry", False)) + r"""
\bottomrule\end{tabularx}
\tabnote{4세션·12장·96개 기하 코너의 고정 분모다. 원래 직접 클릭 66점과 G로 채운 PnP 투영 30점을 같은 규칙으로 사용한다. PCK는 오차 $\leq10$px이며 잘못된 대상·결측은 실패로 유지한다. 중앙값/P90은 유효 일치점 조건부 통계다. 공식 120장/24장 가시 코너 계약이나 독립 물리 T/R 정확도에 해당하지 않는다.}
\end{table}
"""
    (COPY / "tables/lifter_assisted_results.tex").write_text(main_table, encoding="utf-8")

    versions_tex = "별도 12장의 주석은 버전별로 보존하였다. 원래 G 저장본의 기하 참조는 직접 클릭 66점과 PnP 투영 30점이며, 이후 가시성 재검수본은 직접 클릭 좌표 72점과 자체 가림 상태 24개다. 두 버전의 좌표를 섞지 않았다. 이번 별도 기하 참조 패널에는 원래 G 저장본의 96좌표만 사용했고, 사람이 고정 선택 박스를 확인하여 12장 모두 같은 대상으로 판정하였다. 공식 120장·24장 반복 검수의 직접 가시 코너 계약은 이 별도 패널로 완료 처리하지 않았다. 확인된 정지 구간과 독립 물리 참조도 없어 관련 값은 \\notrun 으로 유지한다."
    mixed_tex = "원래 참조와 고정 원시 예측을 대조한 표~\\ref{tab:lifter_assisted_geometry}에서는 중앙 오차가 3.974→4.184px로 악화되었지만 P90은 9.643→8.977px, PCK는 88/96→91/96(91.667→94.792\\%)으로 개선되어 결과가 혼합되었다. 모든 점과 네 세션을 유지했으며 오차를 보고 표본·seed를 고르지 않았다. 기하 보조 참조의 생성 시점에 예측 노출을 독립 통제했는지는 확인되지 않았고, 이번 대상 대응 단계에서 선택 박스를 보았다는 사실은 기록했다. 따라서 이 사례로 독립 블라인드 코너 복원, 물리 자세 정확도, 추적 정확도, 포크 삽입 성공률 또는 정렬시간 개선을 주장하지 않는다."
    case = COPY / "sections/06_case_study.tex"
    text = case.read_text(encoding="utf-8")
    old_start = "별도 12장의 가시성 입력 96개는 보존되었으며,"
    old_end = "따라서 이 사례로 추적 정확도, 포크 삽입 성공률 또는 정렬시간 개선을 주장하지 않는다."
    a, b = text.index(old_start), text.index(old_end) + len(old_end)
    text = text[:a] + versions_tex + "\n\n" + mixed_tex + "\n\\input{tables/lifter_assisted_results}\n" + text[b:]
    text = text.replace("독립 거리·각도 또는 직접 보이는 코너의 참조가 확보된 항목만 정확도로 평가한다.",
                        "독립 거리·각도 정확도와 직접 가시 코너 정확도는 해당 참조가 확보된 항목만 평가한다. 사람이 확인한 PnP 보조 기하 참조의 영상 오차는 별도 탐색 패널로 구분한다.")
    text = text.replace("가시 코너 및 물리 정확도도 사람 참조와 객체 대응 전까지 x로 둔다.",
                        "이 표의 가시 코너 정확도는 공식 120장/24장 참조 계약에 해당하며 x를 유지한다. 별도 12장 PnP 보조 기하 참조 오차는 표~\\ref{tab:lifter_assisted_geometry}에 구분한다. 독립 물리 정확도는 x다.")
    case.write_text(text, encoding="utf-8")

    md_table = "**사람이 확인한 PnP 보조 기하 참조의 별도 12장 탐색 패널. YOLO Base와 N3(seed 1).** (`tab:lifter_assisted_geometry`)\n\n|방법|중앙(px)|P90(px)|PCK≤10px(%)|적중/전체|유효/실패|\n|---|---:|---:|---:|---:|---:|\n" + "\n".join(total_rows("manuscript_ko.md", "tab:lifter_assisted_geometry", True)) + "\n\n4세션·12장·96점의 고정 분모이며 직접 클릭 66점·원래 G의 PnP 투영 30점을 함께 사용한다. 공식 120/24장 가시 코너 평가와 독립 물리 T/R 정확도를 대신하지 않는다.\n"
    md = COPY / "manuscript_ko.md"
    text = md.read_text(encoding="utf-8")
    a, b = text.index(old_start), text.index(old_end)+len(old_end)
    text = text[:a] + versions_tex.replace(r"\notrun", "x") + "\n\n" + mixed_tex.replace(r"\ref{tab:lifter_assisted_geometry}", "`tab:lifter_assisted_geometry`").replace(r"\%", "%") + "\n\n" + md_table + text[b:]
    text = text.replace("독립 거리·각도 또는 직접 보이는 코너의 참조가 확보된 항목만 정확도로 평가한다.",
                        "독립 거리·각도 정확도와 직접 가시 코너 정확도는 해당 참조가 확보된 항목만 평가한다. 사람이 확인한 PnP 보조 기하 참조의 영상 오차는 별도 탐색 패널로 구분한다.")
    text = text.replace("가시 코너 및 물리 정확도도 사람 참조와 객체 대응 전까지 x로 둔다.",
                        "이 표의 가시 코너 정확도는 공식 120/24장 참조 계약이며 x를 유지한다. 별도 12장 PnP 보조 기하 오차는 `tab:lifter_assisted_geometry`에 구분한다. 독립 물리 정확도는 x다.")
    md.write_text(text, encoding="utf-8")

    rows = []
    mdrows = []
    for session in d["metrics"]["sessions"]:
        for name in ("Base", "N3"):
            ptr = f"/metrics/sessions/{session}/methods/{name}"
            vals = [session, name]
            mdvals = [session, name]
            for key, unit in (("conditional_error/median_px", "px"), ("conditional_error/p90_px", "px"), ("pck10_full_reference_percent", "%")):
                vals.append(cell(ptr+"/"+key, "supplement_tables/lifter_assisted_sessions.tex", "sup:lifter_assisted_sessions", session+" "+name, unit))
                mdvals.append(cell(ptr+"/"+key, "supplement_ko.md", "sup:lifter_assisted_sessions", session+" "+name, unit))
            vals.append("24/24")
            mdvals.append("24/24")
            rows.append(" & ".join(vals)+r" \\")
            mdrows.append("|"+"|".join(mdvals)+"|")
    session_table = r"""\begin{table}[t]\centering\footnotesize
\caption{같은 12장 기하 참조 패널의 네 세션별 결과. 각 세션은 3장·24점이다.}\label{sup:lifter_assisted_sessions}
\begin{tabularx}{\columnwidth}{Ylrrrr}\toprule
세션 & 방법 & 중앙(px) & P90(px) & PCK(\%) & 유효/전체 \\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabularx}
\tabnote{고정 YOLO Base와 N3(seed 1)의 원시 예측을 재사용하였다. 중앙값/P90은 조건부 통계이고 PCK 분모에는 결측과 대상 불일치를 유지한다. 이 표에서는 두 방법 모두 96점이 유효하며 실패는 0점이다. 네 세션의 모든 고정 표본을 보존했다. 3 seed 평균이나 8,910프레임의 참조 정확도가 아니다.}
\end{table}
"""
    (COPY / "supplement_tables/lifter_assisted_sessions.tex").write_text(session_table, encoding="utf-8")

    paired = d["metrics"]["paired"]
    paired_tex = "같은 코너 96점에서 통계량 차이 median(N3)−median(Base)는 0.210px, P90(N3)−P90(Base)는 −0.666px이고, 점별 차이의 중앙값 median(N3−Base)는 0.090px다. 서로 다른 통계량을 혼동하지 않았다. 점별 개선/악화/동일은 46/50/0점이며, 프레임별 코너 중앙값의 변화는 개선 5장·악화 7장이다. PCK 적중은 88→91점으로 3.125\\%p 증가했다. 원래 G 참조 생성 시점의 독립 블라인드 이력은 확인되지 않았으며, 현재 선택 박스 확인 이력은 실제 사람 검수 기록에 구분해 저장했다. 직접 클릭과 투영은 참조의 출처 정보이며 그 구분만으로 투영점을 배제하거나 수동 클릭의 우월성을 가정하지 않았다."
    sup_intro_tex = "별도 탐색 패널은 네 세션에서 사전에 고정한 12장의 원래 G 저장본 96좌표를 사용한다. 직접 클릭 66점과 알려진 기하·클릭을 이용한 PnP 투영 30점을 동일한 0–7 코너 대응으로 평가한다. 이후 가시성 재검수의 직접 클릭 72점·자체 가림 24상태는 별도 버전으로 보존하고 혼합하지 않는다. 사람이 고정 선택 박스를 검수한 12장 모두 실제 주석 대상과 같았으며, 두 방법에 같은 frame ID·코너·선택 인덱스·결측 마스크를 적용했다. 참조 좌표를 모델 출력으로 교체하지 않고 재-PnP·점별 최근접 대응·튜닝·추가 학습 없이 원시 예측만 재집계했다. 이 패널은 공식 120장/24장 반복 가시 코너 검수나 독립 물리 T/R 측정을 완료한 결과가 아니다."
    sup_new_tex = "\n\\section{PnP 보조 기하 참조의 별도 12장 탐색 패널}\n" + sup_intro_tex + "\n\\input{supplement_tables/lifter_assisted_sessions}\n\n" + paired_tex + "\n\\input{figures/lifter_assisted_all_frames}\n"
    sup = COPY / "supplement.tex"
    text = sup.read_text(encoding="utf-8")
    old = "운용 모델의 출력이나 제어 명령을 정답으로 사용하지 않으며, 12장 96개 상태 중 실제 클릭 좌표 67점·좌표 없는 가시 판정 5점·자체 가림 24점을 보존했지만 최종 평가 승인과 예측 객체 대응은 아직 0건이다. 사람 입력 확보와 평가 게이트 성립을 구분하며, 코너 오차·정지·독립 물리 참조가 필요한 값은 x로 보존한다."
    new = "운용 모델의 출력이나 제어 명령을 정답으로 사용하지 않는다. 별도 12장의 원래 G 기하 참조 96좌표(직접 클릭 66점·PnP 투영 30점)와 이후 직접 클릭 72점·자체 가림 24상태의 버전을 구분하고, 실제 대상 대응 12건을 확보하였다. 원래 공식 120/24장 직접 가시 참조 계약과 정지·독립 물리 참조가 필요한 값은 x로 보존하며, 별도 기하 참조 오차는 다음 절에 제시한다."
    assert text.count(old) == 1
    text = text.replace(old, new)
    text = text.replace("\\FloatBarrier\n\\end{document}", sup_new_tex + "\\FloatBarrier\n\\end{document}")
    text = text.replace("다른 평가 집합이나 새 실험을 추가한 문서가 아니다.", "본문의 고정 자료를 사용하며, 별도 12장 PnP 보조 기하 참조의 탐색 패널은 공식 가시 코너 평가와 분리해 제시한다.")
    sup.write_text(text, encoding="utf-8")
    supmd = COPY / "supplement_ko.md"
    text = supmd.read_text(encoding="utf-8")
    assert text.count(old) == 1
    text = text.replace(old, new)
    text += "\n\n# PnP 보조 기하 참조의 별도 12장 탐색 패널\n\n" + sup_intro_tex + "\n\n**고정 12장 탐색 패널의 세션별 결과** (`sup:lifter_assisted_sessions`)\n\n|세션|방법|중앙(px)|P90(px)|PCK≤10px(%)|유효/전체|\n|---|---|---:|---:|---:|---:|\n" + "\n".join(mdrows) + "\n\n" + paired_tex.replace(r"\%p", "%p") + "\n\n## 고정 12장 전체 이미지\n\n왼쪽은 원래 참조(청록 직접 클릭·보라 PnP), 가운데는 Base, 오른쪽은 N3이며 성능으로 표본을 고르지 않았습니다.\n\n"
    for o in d["overlays"]:
        text += f"### {o['frame_id']}\n\n![{o['frame_id']}](figures/pnp_assisted_lifter_12/{Path(o['path']).name})\n\n"
    supmd.write_text(text, encoding="utf-8")

    figures = []
    for session in d["metrics"]["sessions"]:
        figures.append(r"\begin{figure*}[t]\centering")
        for o in d["overlays"]:
            if o["frame_id"].split(":")[0] == session:
                figures.append(r"\includegraphics[width=.97\textwidth]{figures/pnp_assisted_lifter_12/"+Path(o["path"]).name+r"}\par\smallskip")
        figures.append("\\caption{"+session+"세션의 사전 고정 3장 전체. 왼쪽은 원래 기하 참조(청록 직접 클릭·보라 PnP), 가운데 Base, 오른쪽 N3이다. 생성 경로를 표시하며 독립 물리 정답이나 모든 점의 직접 가시성을 뜻하지 않는다.}\\label{sup:lifter_assisted_"+session+"}")
        figures.append(r"\end{figure*}")
    (COPY / "figures/lifter_assisted_all_frames.tex").write_text("\n".join(figures)+"\n", encoding="utf-8")

    conclusion_old = "남은 확인은 등급 판정 기준과 참조 품질이며, 리프터 정확도에는 평가용 사람 참조·객체 대응과 독립 물리 참조가 필요하다."
    conclusion_new = "별도 12장 PnP 보조 기하 참조에서는 P90과 PCK가 개선되었지만 중앙값은 악화되어 결과가 혼합되었다. 이 탐색 패널의 실제 대상 대응 12건은 확보했으나 공식 120/24장 가시 코너 계약과 독립 물리 자세 정확도를 대신하지 않는다. 남은 확인은 등급 판정 기준·참조 품질과 독립 물리 참조이며, 기록 영상의 정지 잡음도 확인된 정지 구간이 필요하다."
    replace_once(COPY / "sections/08_conclusion.tex", conclusion_old, conclusion_new)
    replace_once(COPY / "manuscript_ko.md", conclusion_old, conclusion_new)

    readme = """# 2026-10-06 PnP 보조 부분 평가를 실제 반영한 원고 복사본

`main.tex`와 `supplement.tex`가 LaTeX 소스이며, [본문 Markdown](manuscript_ko.md)과 [보충 Markdown](supplement_ko.md)에도 같은 결과를 반영했습니다. 원본 `closeout_20261006_v1/paper_updated`의 모든 파일은 그대로 보존했습니다.

추가한 결과는 사전에 고정한 4세션·12장·96개 코너의 **사람이 확인한 PnP 보조 기하 참조**에 대한 YOLO Base 대 N3(seed 1)의 탐색 비교입니다. 원래 G 저장본의 직접 클릭66점·PnP 투영30점만 사용하고 이후72점 클릭본과 혼합하지 않았습니다. 실제 대상 대응은12건 모두 같은 대상으로 확인됐습니다. 중앙 오차3.974→4.184px는 악화, P90 9.643→8.977px와 PCK88/96→91/96는 개선입니다.

공식120/24장 직접 가시 코너 평가·리프터 정지 잡음·정사각형/리프터 독립 물리T/R의 x는 유지했습니다. 실제 생성 시점의 독립 블라인드 이력은 미확인이며 현재 대응 검수에서 선택 박스를 본 이력과 구분했습니다. 재어노테이션을 요구하는 문서가 아닙니다.

[새 셀 출처](evidence/ASSISTED_PAPER_CELL_MAP.json), [새 출처 해시](evidence/ASSISTED_SOURCE_MANIFEST.json), [통합 patch](../paper_patch/ASSISTED_PAPER.patch), [반영·검증 영수증](../PAPER_INTEGRATION_RECEIPT.json)을 확인하세요. 복사된 기존 audit/evidence와 과거 PDF·페이지 수 기록은 이전 수정본의 역사 기록이며 이번 소스의 검증 영수증이 아닙니다. 새 학습·새 모델 추론·optimizer update·장비 제어·PDF 생성·컴파일·push는 모두0회입니다.
"""
    for name in ("README_FIRST_KO.md", "COMBINED_UPDATE_README_KO.md"):
        (COPY / name).write_text(readme, encoding="utf-8")

    # Reuse maps remain immutable history; new entries bind new source/result and targets.
    dump(COPY / "evidence/ASSISTED_PAPER_CELL_MAP.json", {"schema": "assisted_paper_cell_map_v1", "source_result_sha256": sha(RESULT_PATH), "numeric_cells": cells, "scope": "separate_12frame_96corner_seed1_geometry_panel_only"})
    dump(COPY / "evidence/ASSISTED_SOURCE_MANIFEST.json", {"schema": "assisted_paper_source_manifest_v1", "sources": sources, "overlays": d["overlays"], "original_manuscript_hashes": source_before, "scope": "new separate copy; prior audit/evidence is historical"})

    validation = []
    for c in cells:
        target = Path(c["file"]).read_text(encoding="utf-8")
        validation.append({"file": c["file"], "label": c["label"], "pointer": c["source_json_pointer"], "pass": c["display"] in target and value(c["source_json_pointer"]) == c["source_value"]})
    assert all(x["pass"] for x in validation)
    # Check actual included sources and figure assets without compiling anything.
    input_checks = []
    for path in [COPY/"main.tex", COPY/"supplement.tex", COPY/"sections/06_case_study.tex", COPY/"figures/lifter_assisted_all_frames.tex"]:
        for included in re.findall(r"\\input\{([^}]+)\}", path.read_text(encoding="utf-8")):
            resolved = COPY/(included if included.endswith(".tex") else included+".tex")
            input_checks.append({"path": str(resolved), "pass": resolved.is_file()})
    assert all(x["pass"] for x in input_checks)
    assert len(d["overlays"]) == len(list(figroot.glob("*.png"))) == 12
    for item in d["overlays"]:
        assert sha(figroot/Path(item["path"]).name) == item["sha256"]
    for file in (COPY/"sections/06_case_study.tex", COPY/"manuscript_ko.md", COPY/"supplement.tex", COPY/"supplement_ko.md"):
        text = file.read_text(encoding="utf-8")
        assert "좌표 없는" not in text and "대응은 아직 0건" not in text
    assert "가시 코너 중앙값 / P90 (px) & \\notrun/\\notrun" in case.read_text(encoding="utf-8")
    assert source_before == snapshot(SOURCE), "Original manuscript changed"

    new_snapshot = snapshot(COPY)
    change_paths = sorted(set(source_before)|set(new_snapshot))
    parts = []
    changed = []
    for rel in change_paths:
        oldp, newp = SOURCE/rel, COPY/rel
        if oldp.is_file() and newp.is_file() and sha(oldp) == sha(newp):
            continue
        if newp.suffix.lower() not in {".md", ".tex", ".json"}:
            continue
        oldlines = oldp.read_text(encoding="utf-8").splitlines(keepends=True) if oldp.is_file() else []
        newlines = newp.read_text(encoding="utf-8").splitlines(keepends=True) if newp.is_file() else []
        parts.extend(difflib.unified_diff(oldlines, newlines, fromfile="a/"+rel, tofile="b/"+rel))
        changed.append(rel)
    patch_path = PATCH/"ASSISTED_PAPER.patch"
    patch_path.write_text("".join(parts), encoding="utf-8")
    dump(PATCH/"ORIGINAL_PAPER_HASHES.json", source_before)
    dump(PATCH/"COPIED_PAPER_HASHES.json", new_snapshot)
    dump(PATCH/"VALIDATION.json", {"status": "PASS", "cell_checks": validation, "input_checks": input_checks,
                                    "all_12_images_hash_match": True, "original_files_unchanged": len(source_before),
                                    "official_visible_x_preserved": True, "PDF_compiled": False})
    (PATCH/"PAPER_GAP_MATRIX.md").write_text("""# 이번 반영과 남은 근거

|항목|결과|원고 반영|이유·범위|
|---|---|---|---|
|12장96점 PnP 보조 기하 참조|계산·검산 완료|본문/보충 LaTeX·Markdown 반영|YOLO Base/N3 seed1, 공식 가시성 평가와 별도|
|실제 대상 대응|12장 모두 same|버전·방법 설명 반영|사람이 실제 선택, 자동 승인 없음|
|원래 G 저장본 출처|66직접 클릭+30PnP|본문/보충 반영|나중72수동 좌표본과 혼합 없음|
|공식120/24장 직접 가시 코너|x|x 유지|별도 계약의 최종 참조 승인 미성립|
|리프터 정지 잡음|x|x 유지|사람 확인 정지 구간 없음|
|정사각형/리프터 독립 물리 T/R|x|x 유지|독립 물리 참조 없음|
|실제 생성 시점 독립 블라인드 이력|미확인|제한 설명 반영|현재 박스 검수 이력과 구분|
|리프터 제어·새 학습·PDF·push|범위 제외|미실행|새 모델 추론 포함0회|
""", encoding="utf-8")
    elapsed = time.monotonic()-start
    receipt = {"schema": "assisted_paper_integration_receipt_v1", "status": "COMPLETE",
               "source_result_path": str(RESULT_PATH), "source_result_sha256": sha(RESULT_PATH),
               "numeric_cells": len(cells), "source_cell_map_path": str(COPY/"evidence/ASSISTED_PAPER_CELL_MAP.json"),
               "source_manifest_path": str(COPY/"evidence/ASSISTED_SOURCE_MANIFEST.json"),
               "manuscript_copy_path": str(COPY), "main_latex_path": str(COPY/"main.tex"),
               "supplement_latex_path": str(COPY/"supplement.tex"), "markdown_path": str(COPY/"manuscript_ko.md"),
               "supplement_markdown_path": str(COPY/"supplement_ko.md"), "patch_path": str(patch_path),
               "patch_sha256": sha(patch_path), "validation_path": str(PATCH/"VALIDATION.json"),
               "original_hash_verification": {"status": "PASS", "files_checked": len(source_before), "manifest_path": str(PATCH/"ORIGINAL_PAPER_HASHES.json")},
               "changed_text_files": changed, "overlay_image_count": 12, "new_numeric_panel_scope": "12frames_4sessions_96corners_YOLO_N3_seed1_assisted_geometry",
               "official_120_24_visible_x_preserved": True, "independent_physical_TR_x_preserved": True,
               "training_runs": 0, "optimizer_updates": 0, "new_model_forward_frames": 0,
               "hardware_control_calls": 0, "pdf_generated": False, "latex_compiled": False, "push_calls": 0,
               "cpu_wall_seconds": elapsed}
    dump(OUT/"PAPER_INTEGRATION_RECEIPT.json", receipt)
    print(json.dumps({"status": "COMPLETE", "numeric_cells": len(cells), "original_files_preserved": len(source_before), "images": 12, "cpu_wall_seconds": elapsed, "receipt": str(OUT/"PAPER_INTEGRATION_RECEIPT.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
