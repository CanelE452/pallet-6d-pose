"""Validate and package the combined static/lifter closeout evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
NAME = Path(__file__).parent.name
DOC = ROOT / "_docs/experiments" / NAME
STATIC_DOC = ROOT / "_docs/experiments/pallet_static_registry_review_20261003_v1"
STATIC_RAW = ROOT / "data/pallet/results/pallet_static_registry_review_20261003_v1"

EXPECTED_BASE = (6_552_807, "970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7")
EXPECTED_N3 = (305_470, "ceea743b7b8467cf43eef66582b923dd980e5b2fb27f5575af5df185109f22dd")
EXPECTED_PAPER_ZIP = "f2661ce4b965d5883d57f748d04afc04aa8a881d3ff9e0104e98b8d062f06474"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind(path: Path, *, shown: str | None = None, rows: int | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {
        "path": shown or str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if rows is not None:
        result["rows"] = rows
    return result


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def verify(path: Path, expected: tuple[int, str]) -> None:
    size, digest = expected
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"binding mismatch: {path}")


def replace_tail(base: Path, updated: Path, destination: Path, *, take_updated_tail: bool) -> None:
    marker = "\\section{기록 영상"
    old = base.read_text()
    new = updated.read_text()
    if old.count(marker) != 1 or new.count(marker) != 1:
        raise ValueError("supplement record-video section marker changed")
    old_head, old_tail = old.split(marker, 1)
    new_head, new_tail = new.split(marker, 1)
    destination.write_text((old_head + marker + new_tail) if take_updated_tail
                           else (new_head + marker + old_tail))


def stage_paper(reference: Path, updated: Path, work: Path) -> tuple[Path, Path]:
    static = work / "paper_static_updated"
    lifter = work / "paper_lifter_updated"
    for target in (static, lifter):
        if not target.exists():
            shutil.copytree(reference, target)

    static_files = [
        "sections/04_setup.tex", "sections/05_results.tex", "sections/07_discussion.tex",
        "tables/composition.tex", "tables/data.tex", "tables/occlusion_results.tex",
        "supplement_tables/occlusion_results.tex",
    ]
    for relative in static_files:
        copy(updated / relative, static / relative)
    replace_tail(reference / "supplement.tex", updated / "supplement.tex",
                 static / "supplement.tex", take_updated_tail=False)

    copy(updated / "sections/06_case_study.tex", lifter / "sections/06_case_study.tex")
    copy(updated / "figures/lifter_prediction_timeseries.png",
         lifter / "figures/lifter_prediction_timeseries.png")
    replace_tail(reference / "supplement.tex", updated / "supplement.tex",
                 lifter / "supplement.tex", take_updated_tail=True)
    return static, lifter


def text_patch(reference: Path, target: Path) -> str:
    extensions = {".tex", ".md", ".txt", ".csv", ".json"}
    paths = sorted({path.relative_to(reference) for path in reference.rglob("*") if path.is_file()} |
                   {path.relative_to(target) for path in target.rglob("*") if path.is_file()})
    output: list[str] = []
    for relative in paths:
        if relative.suffix.lower() not in extensions:
            continue
        before_path, after_path = reference / relative, target / relative
        before = before_path.read_text().splitlines(keepends=True) if before_path.is_file() else []
        after = after_path.read_text().splitlines(keepends=True) if after_path.is_file() else []
        if before == after:
            continue
        output.extend(difflib.unified_diff(before, after, fromfile=f"a/{relative}",
                                           tofile=f"b/{relative}"))
    return "".join(output)


def changed_files(reference: Path, updated: Path) -> list[str]:
    paths = sorted({path.relative_to(reference) for path in reference.rglob("*") if path.is_file()} |
                   {path.relative_to(updated) for path in updated.rglob("*") if path.is_file()})
    changed = []
    for relative in paths:
        old, new = reference / relative, updated / relative
        if not old.is_file() or not new.is_file() or sha256(old) != sha256(new):
            changed.append(str(relative))
    return changed


def write_task_status(path: Path) -> None:
    rows = [
        ("combined archives and contracts", "VERIFIED_INPUT", "outer ZIP, inner contract, readiness and paper ZIP hashes verified"),
        ("exact Base/N3 checkpoints", "VERIFIED_INPUT", "size and SHA-256 exact"),
        ("fixed 8,910-frame inference", "VERIFIED_COMPLETE", "one pass; 8,910 newly computed; training/update 0"),
        ("all-frame continuity", "INSERTED_IN_COPY", "8,772 fresh, 138 no_pose per method; table and figure inserted"),
        ("lifter visible-corner reference", "WAITING_HUMAN_CORNERS", "new review-120 submitted export absent"),
        ("lifter predicted-object correspondence", "WAITING_OBJECT_MATCH", "runs after corner export; separate UI on port 8766"),
        ("lifter stop variation", "WAITING_HUMAN_CORNERS", "no human-confirmed stationary intervals"),
        ("lifter physical pose accuracy", "BLOCKED_REFERENCE", "no independent synchronized reference"),
        ("existing 67 lifter annotations", "PARTIAL", "pixel match verified; provenance/axis and review120 overlap insufficient"),
        ("DEV319 frame severity", "INSERTED_IN_COPY", "319 direct human labels hash-joined; 151/87/81"),
        ("GREEN0918 frame severity", "WAITING_STATIC_REVIEW", "0/119 new human decisions; UI ready on port 8767"),
        ("static corner visibility", "WAITING_STATIC_REVIEW", "71 locked prior statuses; 3,030 referenced points pending"),
        ("static subgroup reaggregation", "INSERTED_IN_COPY", "three backbones; overall results invariant"),
        ("paper static patch", "READY_TO_INSERT", "applied in preserved copy; patch emitted"),
        ("paper lifter patch", "READY_TO_INSERT", "continuity only applied in preserved copy; x retained"),
        ("PDF/PPT compilation", "OUT_OF_SCOPE_USER", "explicitly prohibited"),
        ("push/external upload/control", "OUT_OF_SCOPE_USER", "explicitly prohibited"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "status", "evidence_or_reason"])
        writer.writerows(rows)


def user_actions() -> str:
    return """# 남은 사람 행동과 실제 재개 명령

모델 추론과 정적 319장 재집계는 끝났다. 남은 작업은 사람 판정이 필요한 세 갈래이며 서로 다른 로컬 화면과 저장 파일을 사용한다. CLI는 빈 응답을 사람 완료로 만들지 않는다.

## 1. 리프터 직접 가시 코너 검수: 포트 8765

저장소 루트에서 다음 명령을 실행하고 <http://127.0.0.1:8765>를 연다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 \
  scripts/research/pallet_lifter_case_review_20261003_v1/start_review.py
```

고정 120장과 반복 24장을 실제 물리 코너 정의로 검수한다. 직접 보이는 코너만 클릭하고 숨은 가상 코너를 추정하지 않는다. 로컬 저장은 `data/pallet/results/pallet_lifter_case_review_20261003_v1/review/annotations_in_progress.json`이며, 화면에서 `LIFTER_REFERENCE_REVIEWED.json`을 export한다. 다른 컴퓨터의 export가 이미 있으면 새 클릭을 반복하지 말고 다음 명령으로 검증·재개한다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 \
  scripts/research/pallet_lifter_case_review_20261003_v1/resume.py \
  /path/to/LIFTER_REFERENCE_REVIEWED.json
```

## 2. 리프터 객체 대응: 포트 8766

1번 export를 만든 뒤 queue를 만들고 <http://127.0.0.1:8766>을 연다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 \
  scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/bridge.py prepare-object-review \
  --predictions data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl \
  --reviewed data/pallet/results/pallet_lifter_case_review_20261003_v1/review/LIFTER_REFERENCE_REVIEWED.json \
  --manifest data/pallet/results/pallet_lifter_case_review_20261003_v1/review/MANIFEST.json \
  --output data/pallet/results/pallet_lifter_case_review_20261003_v1/review/OBJECT_MATCH_QUEUE.json

/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10 \
  scripts/research/pallet_lifter_case_review_20261003_v1/combined_integration/object_match_review.py serve \
  --queue data/pallet/results/pallet_lifter_case_review_20261003_v1/review/OBJECT_MATCH_QUEUE.json \
  --manifest data/pallet/results/pallet_lifter_case_review_20261003_v1/review/MANIFEST.json \
  --predictions data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl \
  --reviewed data/pallet/results/pallet_lifter_case_review_20261003_v1/review/LIFTER_REFERENCE_REVIEWED.json \
  --store data/pallet/results/pallet_lifter_case_review_20261003_v1/review/object_match_in_progress.json \
  --port 8766
```

각 프레임에서 frozen 선택 객체가 사람 참조 대상과 `same / different / undetermined`인지 판정한다. 모델 이름·코너 오차는 화면에 나오지 않는다. export한 `LIFTER_OBJECT_MATCH_REVIEWED.json`은 `combined_integration/README_KO.md`의 import/apply 명령으로 evaluator 파생본에 연결한다.

## 3. 정적 정사각형 가림과 코너 가시성: 포트 8767

저장소 루트에서 manifest를 검산한 뒤 <http://127.0.0.1:8767>을 연다.

```bash
PYTHONPATH=. /home/minjae/anaconda3/envs/lifter/bin/python \
  -m scripts.research.pallet_static_registry_review_20261003_v1.build_review \
  --source-root /home/minjae/Documents/github/pallet-pose

PYTHONPATH=. /home/minjae/anaconda3/envs/lifter/bin/python \
  -m scripts.research.pallet_static_registry_review_20261003_v1.serve_review \
  --source-root /home/minjae/Documents/github/pallet-pose --port 8767
```

초안은 `data/pallet/results/pallet_static_registry_review_20261003_v1/STATIC_REVIEW_IN_PROGRESS.json`에 저장된다. 319장 직사각형 가림 등급과 기존 71개 코너 상태는 잠겨 있다. 정사각형 119장 가림 등급은 제출 전에 모두 필요하지만, 3,030개 참조 코너 가시성은 부분 완료 상태로도 별도 기록할 수 있다. JSON export/import와 서버 재시작 재개가 검증되어 있다.

실제 사람 export가 생기기 전에는 가시 코너 정확도·정지 잡음·물리 정확도의 `x`를 숫자로 바꾸지 않는다.
"""


def gap_matrix() -> str:
    return """# 원고 셀 연결과 남은 x

| 원고 위치 | 결과 | 상태 | 원시 근거 | 분모·단위 |
|---|---|---|---|---|
| `tab:composition` | 직사각형 151/87/81 | INSERTED_IN_COPY | `LABEL_PROVENANCE_AUDIT.json` | 영상 319장 |
| `tab:occlusion_results` | YOLO Base/N3 전체 가림 집단 | INSERTED_IN_COPY | `STATIC_REAGGREGATION.json` | 참조 2,499, 유효 2,445 코너; px/cm/deg |
| `sup:occlusion_results` | Base/P/N2/N3 가림 집단 | INSERTED_IN_COPY | 같은 JSON | N3·N2·P 각 3 seed 통계 평균 |
| `tab:case` 영상/프레임 | 4 / 8,910 | INSERTED_IN_COPY | `LIFTER_CONTINUITY_SUMMARY.json` | 전체 저장 프레임 |
| `tab:case` 자세 산출률 | 98.451 / 98.451% | INSERTED_IN_COPY | 같은 JSON | fresh 8,772 / 8,910; held 0 |
| `tab:case` 최장 완료 결측 | 2.535 / 2.535 s | INSERTED_IN_COPY | 같은 JSON | 첫 no-pose부터 다음 fresh까지 센서 시간 |
| `tab:case` 인접 변화 | x/z/yaw 중앙값·P90 | INSERTED_IN_COPY | 같은 JSON | 양쪽 유효 같은 8,737 인접쌍; 정확도 아님 |
| `tab:case` 가시 코너 오차 | x | WAITING_HUMAN_CORNERS | 제출된 review-120 없음 | 승인 직접 가시 코너 전체 분모 필요 |
| `tab:case` 정지 구간 변동 | x | WAITING_HUMAN_CORNERS | 확인된 정지 구간 없음 | 이동 중 인접 변화를 대신 쓰지 않음 |
| `tab:case` 물리 위치·방향 오차 | x | BLOCKED_REFERENCE | 독립 동기 참조 없음 | 운용 CSV·같은 PnP는 정답 아님 |
| 정사각형 6D | x | BLOCKED_REFERENCE | 독립 6D 참조 없음 | 600/602 코너 2D와 분리 |
| 정사각형 가림 집단 | x | WAITING_STATIC_REVIEW | 119장 UI 준비 | 사람 frame severity 필요 |
| 전체 코너 가시성 분석 | 부분본만 유지 | WAITING_STATIC_REVIEW | 기존 66 direct + 5 external | 나머지 직사각형 2,428점과 정사각형 602점 대기 |

`x`는 측정되지 않았거나 계약이 성립하지 않은 값이고, `NA`는 정의되지 않는 값이다. 실제 0은 0으로 적는다.
"""


def final_report(static: Mapping[str, Any], lifter: Mapping[str, Any]) -> str:
    yolo = static["backbones"]["yolo"]
    overall = lifter["overall"]
    base, n3 = overall["methods"]["Base"], overall["methods"]["N3"]
    lines = [
        "# 팔레트 정적·리프터 통합 마감 보고", "",
        "## 직접 실행하고 검산한 항목", "",
        "- 정확한 Base/N3 가중치의 크기와 SHA-256을 확인하고 네 리프터 영상의 저장 프레임 8,910장을 고정 한 순회로 추론했다. 새 학습 0회, optimizer update 0회다.",
        f"- Base와 N3는 각각 fresh {base['fresh_count']:,}, held {base['held_count']}, no-pose {base['no_pose_count']}이며 산출률은 {base['fresh_output_rate_pct']:.3f}%로 같다.",
        f"- 같은 인접 유효 쌍 {base['adjacent_output_change']['pair_count']:,}개에서 변화량을 계산했다. x·z P90은 낮아졌고 x·z 중앙값과 yaw 중앙/P90은 높아져 혼합 결과다. 정확도 또는 정지 잡음 개선으로 부르지 않는다.",
        "- 완료 제출된 직사각형 319장 사람 가림 분류를 encoded RGB SHA-256과 frame ID로 전수 연결했다. clean 151 / moderate 87 / severe 81이며 미분류는 0이다.",
        "- 기존 원시 예측만 새 집단으로 재집계했고 세 기반의 전체 결과는 전부 정확히 불변이었다. 정적 새 추론·PnP·학습은 0회다.",
        "- 보존된 원고 복사본에 검산된 정적 표와 리프터 연속성만 실제 삽입했다. PDF는 생성하지 않았다.", "",
        "## 정적 핵심 결과", "",
        "| 집단 | n | Base 코너 중앙/P90(px) | N3 코너 중앙/P90(px) | Base→N3 T 중앙(cm) | Base→N3 R 중앙(deg) |", "|---|---:|---:|---:|---:|---:|",
    ]
    for group in ("clean", "moderate", "severe", "all"):
        b, a = yolo["Base"]["result"][group], yolo["N3"]["result"][group]
        lines.append(f"| {group} | {b['frames']} | {b['corner_median_px']:.3f}/{b['corner_P90_px']:.3f} | {a['corner_median_px']:.3f}/{a['corner_P90_px']:.3f} | {b['translation_median_cm']:.3f}→{a['translation_median_cm']:.3f} | {b['rotation_median_deg']:.3f}→{a['rotation_median_deg']:.3f} |")
    lines += ["", "severe에서는 코너와 회전 중앙값이 낮아졌지만 위치 중앙값은 13.484→15.706cm로 악화했다. DOPE와 ResNet도 일부 집단·P90에서 악화가 남는다. 가림 분석은 반복 개발자료의 사후 분석이다.", "",
              "![완료 319장 가림별 세 기반 결과](figures/completed319_backbone_occlusion.png)", "",
              "![완료 319장 사람 레이블 구성](figures/completed319_label_composition.png)", "",
              "## 리프터 실제 출력", "",
              "| 방법 | fresh/held/no-pose | 산출률 | abs Δx 중앙/P90(cm) | abs Δz 중앙/P90(cm) | abs Δyaw 중앙/P90(deg) |", "|---|---:|---:|---:|---:|---:|",
              f"| Base | {base['fresh_count']:,}/{base['held_count']}/{base['no_pose_count']} | {base['fresh_output_rate_pct']:.3f}% | {100*base['adjacent_output_change']['abs_x_delta_m']['median']:.3f}/{100*base['adjacent_output_change']['abs_x_delta_m']['p90']:.3f} | {100*base['adjacent_output_change']['abs_z_delta_m']['median']:.3f}/{100*base['adjacent_output_change']['abs_z_delta_m']['p90']:.3f} | {base['adjacent_output_change']['abs_yaw_delta_360_wrapped_deg']['median']:.3f}/{base['adjacent_output_change']['abs_yaw_delta_360_wrapped_deg']['p90']:.3f} |",
              f"| N3 | {n3['fresh_count']:,}/{n3['held_count']}/{n3['no_pose_count']} | {n3['fresh_output_rate_pct']:.3f}% | {100*n3['adjacent_output_change']['abs_x_delta_m']['median']:.3f}/{100*n3['adjacent_output_change']['abs_x_delta_m']['p90']:.3f} | {100*n3['adjacent_output_change']['abs_z_delta_m']['median']:.3f}/{100*n3['adjacent_output_change']['abs_z_delta_m']['p90']:.3f} | {n3['adjacent_output_change']['abs_yaw_delta_360_wrapped_deg']['median']:.3f}/{n3['adjacent_output_change']['abs_yaw_delta_360_wrapped_deg']['p90']:.3f} |", "",
              "![네 세션의 Base/N3 실제 예측 시계열](figures/lifter_prediction_timeseries.png)", "", "![전체·세션별 출력 coverage](figures/lifter_output_coverage.png)", "", "![고정 review-120 전부의 prediction-only overlay](figures/review120_prediction_overlay_contact_sheet.png)", "",
              "contact sheet는 결과를 보고 고른 예가 아니라 고정 120장 전부다. 주황색은 Base, 청록색은 N3이며 사람 정답과 오차는 표시하지 않았다.", "",
              "## 사람이 해야 하는 항목", "",
              "- 리프터 120장 직접 가시 코너와 반복 24장 검수", "- 같은 표본에서 frozen 예측 객체와 사람 대상의 대응 판정", "- 실제 정지 구간 확인과 별도 독립 물리 참조 확보", "- 정사각형 119장 가림 등급 및 남은 정적 코너 가시성", "",
              "기존 리프터 주석은 같은 네 세션에 67개가 있었고 픽셀 해시는 모두 일치했다. 그러나 새 120장과 겹친 것은 2장이고, reviewer/time·signed axis 확인이 없으며 312점은 수동 클릭이 아니어서 자동 승인하지 않았다.", "",
              "## 실행 비용과 검증", "",
              f"- GPU 추론 시간: {lifter['execution']['gpu_inference_seconds']:.3f}s", f"- 추론 호출 wall 시간: {lifter['execution']['invocation_wall_seconds']:.3f}s", "- 고정 forward 프레임: 8,910; cache 재사용: 0", "- 리프터 신규/통합 테스트 13 + 기존 metrics 16 = 29 PASS", "- evaluator 수치 독립 대조 197/197 PASS", "- 정적 UI localhost 저장/import/export/제출 차단 테스트 3 PASS", "- 외부 업로드·push·PDF·실제 제어: 0", "",
              "상세 상태는 `TASK_STATUS.csv`, 남은 행동은 `USER_ACTIONS_KO.md`, 원고 셀 출처는 `paper_patch/PAPER_CELL_MAP.json`과 `paper_patch/PAPER_GAP_MATRIX.md`에 있다.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifter-root", type=Path, required=True)
    parser.add_argument("--paper-reference", type=Path, required=True)
    parser.add_argument("--paper-updated", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args(argv)
    lifter_root = args.lifter_root.resolve()
    paper_reference = args.paper_reference.resolve()
    paper_updated = args.paper_updated.resolve()
    handoff = args.handoff_root.resolve()
    work = args.work_root.resolve()

    lifter_code = lifter_root / "scripts/research/pallet_lifter_case_review_20261003_v1"
    lifter_data = lifter_root / "data/pallet/results/pallet_lifter_case_review_20261003_v1"
    integration = lifter_code / "combined_integration"
    lifter_output = integration / "output"
    base_weight = lifter_code / "evaluation_checkout/challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt"
    n3_weight = lifter_code / "evaluation_checkout/data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed1/last.pt"
    verify(base_weight, EXPECTED_BASE)
    verify(n3_weight, EXPECTED_N3)
    paper_zip = handoff / "source_context/current_paper_source.zip"
    if sha256(paper_zip) != EXPECTED_PAPER_ZIP:
        raise ValueError("paper source archive mismatch")
    static = read(STATIC_DOC / "STATIC_REAGGREGATION.json")
    invariance = read(STATIC_DOC / "STATIC_INVARIANCE_CHECK.json")
    labels = read(STATIC_DOC / "LABEL_PROVENANCE_AUDIT.json")
    if invariance.get("status") != "PASS" or not invariance.get("all_exact"):
        raise ValueError("static invariant check failed")
    lifter = read(lifter_output / "LIFTER_CONTINUITY_SUMMARY.json")
    validation = read(lifter_output / "VALIDATION_RECEIPT.json")
    if lifter.get("status") != "VERIFIED_COMPLETE" or validation.get("unit_test_count") != 29:
        raise ValueError("lifter validation incomplete")

    static_paper, lifter_paper = stage_paper(paper_reference, paper_updated, work)
    paper_patch = DOC / "paper_patch"
    paper_patch.mkdir(parents=True, exist_ok=True)
    (paper_patch / "static.patch").write_text(text_patch(paper_reference, static_paper))
    (paper_patch / "lifter.patch").write_text(text_patch(paper_reference, lifter_paper))
    (paper_patch / "integrated.patch").write_text(text_patch(paper_reference, paper_updated))
    modified = changed_files(paper_reference, paper_updated)
    for relative in modified:
        source = paper_updated / relative
        if source.is_file():
            copy(source, paper_patch / "updated_files" / relative)

    figure_sources = {
        "completed319_backbone_occlusion.png": STATIC_DOC / "figures/completed319_backbone_occlusion.png",
        "completed319_label_composition.png": STATIC_DOC / "figures/completed319_label_composition.png",
        "lifter_prediction_timeseries.png": lifter_output / "figures/lifter_prediction_timeseries.png",
        "lifter_output_coverage.png": lifter_output / "figures/lifter_output_coverage.png",
        "review120_prediction_overlay_contact_sheet.png": lifter_output / "figures/review120_prediction_overlay_contact_sheet.png",
    }
    for name, source in figure_sources.items():
        copy(source, DOC / "figures" / name)
    for name in ("ACTUAL_INFERENCE_REPORT_KO.md", "LIFTER_CONTINUITY_SUMMARY.json",
                 "LIFTER_CONTINUITY_SUMMARY.csv", "EXECUTION_FILE_RECEIPT.json",
                 "EXECUTION_FILE_RECEIPT.sha256", "VALIDATION_RECEIPT.json",
                 "VALIDATION_RECEIPT.sha256"):
        copy(lifter_output / name, DOC / "lifter" / name)
    copy(integration / "EXISTING_ANNOTATION_AUDIT.json",
         DOC / "lifter/EXISTING_ANNOTATION_AUDIT.json")
    for name in ("LABEL_PROVENANCE_AUDIT.json", "REGISTRY.json", "STATIC_REAGGREGATION.json",
                 "STATIC_INVARIANCE_CHECK.json", "COHORT_OVERLAP_AUDIT.json",
                 "COMPARATOR_AND_COST_AUDIT.json", "FINAL_REPORT_KO.md"):
        copy(STATIC_DOC / name, DOC / "static" / name)
    copy(STATIC_DOC / "review/VALIDATION.json", DOC / "static/STATIC_REVIEW_VALIDATION.json")
    copy(STATIC_DOC / "generated_tables/tab_completed319_occlusion.csv",
         DOC / "static/tab_completed319_occlusion.csv")

    shutil.copytree(paper_updated, DOC / "paper_updated", dirs_exist_ok=True)
    (DOC / "paper_updated/COMBINED_UPDATE_README_KO.md").write_text(
        "# 통합 수정 원고 복사본\n\n"
        "이 디렉터리는 SHA-256이 고정된 `current_paper_source.zip`을 별도 복사한 뒤, "
        "검산된 정적 completed319 표와 리프터 8,910프레임 연속성 수치만 적용한 검토용 원고다. "
        "컴파일하거나 PDF를 생성하지 않았다. 실제 컴파일 입력은 `main.tex`, `supplement.tex`, "
        "`sections/`, `tables/`, `supplement_tables/`이다. `manuscript_ko.md`는 전달 ZIP의 과거 "
        "편의용 스냅샷이라 이번 통합 수치의 기준으로 사용하지 않는다. 변경 목록과 셀 출처는 "
        "상위 `paper_patch/PAPER_CELL_MAP.json`을 확인한다.\n")

    map_payload = {
        "schema": "pallet_combined_paper_cell_map_v1",
        "paper_source_archive_sha256": EXPECTED_PAPER_ZIP,
        "work_updated_copy": str(paper_updated),
        "reviewable_updated_copy": "../paper_updated/",
        "changed_files": modified,
        "cells": [
            {"label": "tab:composition", "source": "static/LABEL_PROVENANCE_AUDIT.json",
             "values": labels["counts"], "denominator": "319 rectangular frames",
             "status": "INSERTED_IN_COPY"},
            {"label": "tab:occlusion_results", "source": "static/STATIC_REAGGREGATION.json",
             "json_pointer": "/backbones/yolo/{Base,N3}/result/{all,clean,moderate,severe}",
             "units": "px, %, cm, degree", "seed": "N3 mean of three statistic-level seeds",
             "status": "INSERTED_IN_COPY"},
            {"label": "sup:occlusion_results", "source": "static/STATIC_REAGGREGATION.json",
             "json_pointer": "/backbones/yolo/{Base,P,N2,N3}/result/{all,clean,moderate,severe}",
             "status": "INSERTED_IN_COPY"},
            {"label": "tab:case", "source": "lifter/LIFTER_CONTINUITY_SUMMARY.json",
             "json_pointer": "/overall", "denominator": "8910 stored frames; 8737 paired valid adjacent pairs",
             "units": "%, second, cm, degree", "status": "PARTIAL_INSERTED_IN_COPY",
             "remaining_x": ["visible corner error", "reviewed stop variation", "independent physical accuracy"]},
            {"label": "sup:lifter_sessions", "source": "lifter/LIFTER_CONTINUITY_SUMMARY.json",
             "json_pointer": "/sessions", "status": "INSERTED_IN_COPY"},
        ],
        "binary_addition": {"path": "figures/lifter_prediction_timeseries.png",
                            "note": "included under updated_files; unified text patches do not embed binary data"},
    }
    write_json(paper_patch / "PAPER_CELL_MAP.json", map_payload)
    (paper_patch / "PAPER_GAP_MATRIX.md").write_text(gap_matrix())
    (DOC / "REMAINING_X.md").write_text(gap_matrix())
    (DOC / "USER_ACTIONS_KO.md").write_text(user_actions())
    write_task_status(DOC / "TASK_STATUS.csv")
    (DOC / "FINAL_REPORT_KO.md").write_text(final_report(static, lifter))

    protected = read(paper_reference / "audit/PROTECTED_FILES.json")
    protected_checks = {}
    for relative, record in protected.items():
        observed = sha256(paper_reference / relative)
        protected_checks[relative] = {"expected": record["source_sha256"],
                                      "observed": observed,
                                      "exact": observed == record["source_sha256"]}
    if not all(item["exact"] for item in protected_checks.values()):
        raise ValueError("paper protected source mismatch")
    run_status = read(lifter_data / "INFERENCE_RUN_STATUS.json")
    raw = lifter_data / "raw_predictions/ALL_STORED_FRAMES.jsonl"
    l4 = lifter_data / "raw_predictions/ALL_STORED_FRAMES_L4.jsonl"
    manifest = {
        "schema": "pallet_combined_closeout_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PARTIAL_HUMAN_REVIEW_PENDING",
        "repository_basis": {"detached_worktree_head": "7e136fca834d97b52f63be696e4fcc9cbb8bd77e",
                             "owner_repository_preserved": True},
        "inputs": {
            "outer_archive": bind(Path("/home/minjae/Downloads/pallet_cli_combined_handoff_20261003.zip")),
            "wrapper_instruction": bind(Path("/home/minjae/Downloads/EXECUTE_PALLET_COMBINED_CLI_20261003.txt")),
            "readiness": bind(Path("/home/minjae/Downloads/pallet_combined_handoff_readiness_20261003.md")),
            "inner_contract": bind(handoff / "EXECUTE_COMBINED_CLI_KO.txt"),
            "handoff_readiness": bind(handoff / "HANDOFF_READINESS_KO.md"),
            "portability_gates": bind(handoff / "PORTABILITY_AND_EVALUATOR_GATES_KO.md"),
            "paper_source_zip": bind(paper_zip),
            "base_checkpoint": bind(base_weight, shown="scripts/research/pallet_lifter_case_review_20261003_v1/assets/yolo_r0.pt"),
            "n3_checkpoint": bind(n3_weight, shown="scripts/research/pallet_lifter_case_review_20261003_v1/assets/n3_seed1.pt"),
            "lifter_plan": bind(lifter_data / "LIFTER_EVALUATION_PLAN.json", shown="data/pallet/results/pallet_lifter_case_review_20261003_v1/LIFTER_EVALUATION_PLAN.json"),
            "lifter_raw": bind(raw, shown="data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES.jsonl", rows=8910),
            "lifter_l4": bind(l4, shown="data/pallet/results/pallet_lifter_case_review_20261003_v1/raw_predictions/ALL_STORED_FRAMES_L4.jsonl", rows=8910),
            "static_labels": bind(STATIC_DOC / "LABEL_PROVENANCE_AUDIT.json", shown="_docs/experiments/pallet_static_registry_review_20261003_v1/LABEL_PROVENANCE_AUDIT.json"),
        },
        "execution": {**run_status, "forward_frame_count": 8910, "cache_reused_rows": 0,
                      "static_gpu_runs": 0, "new_training": 0, "optimizer_updates": 0,
                      "actual_control": 0, "pdf_or_ppt": 0, "external_upload_or_push": 0},
        "static": {"labels": labels["counts"], "overall_invariance": "PASS",
                   "new_training": 0, "new_inference": 0, "new_PnP": 0,
                   "square_severity": "WAITING_STATIC_REVIEW",
                   "corner_visibility": "PARTIAL_71_LOCKED_3030_PENDING"},
        "lifter": {"stored_frames": 8910, "sessions": 4,
                   "methods": lifter["overall"]["methods"],
                   "visible_corner_accuracy": "x_WAITING_HUMAN",
                   "object_match": "x_WAITING_HUMAN",
                   "stop_variation": "x_WAITING_HUMAN",
                   "physical_accuracy": "x_BLOCKED_REFERENCE",
                   "validation": {"unit_tests": 29, "evaluator_numeric_checks": 197,
                                  "status": "PASS"}},
        "paper": {"reference_path": str(paper_reference), "work_updated_copy_path": str(paper_updated),
                  "reviewable_updated_copy": str((DOC / "paper_updated").relative_to(ROOT)),
                  "changed_files": modified, "pdf_compiled": False,
                  "static_patch": "paper_patch/static.patch",
                  "lifter_patch": "paper_patch/lifter.patch",
                  "integrated_patch": "paper_patch/integrated.patch"},
        "protected_paper_sources": protected_checks,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "cwd": os.getcwd()},
    }
    write_json(DOC / "INTEGRATION_MANIFEST.json", manifest)
    receipt_sources = [DOC / "INTEGRATION_MANIFEST.json", DOC / "TASK_STATUS.csv",
                       DOC / "USER_ACTIONS_KO.md", DOC / "FINAL_REPORT_KO.md",
                       paper_patch / "static.patch", paper_patch / "lifter.patch",
                       paper_patch / "integrated.patch", paper_patch / "PAPER_CELL_MAP.json"]
    receipt = {"schema": "pallet_combined_closeout_receipt_v1", "status": "PASS",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "files": [bind(path, shown=str(path.relative_to(ROOT))) for path in receipt_sources],
               "claims": {"human_values_created": False, "training_runs": 0,
                          "optimizer_updates": 0, "fixed_lifter_passes": 1,
                          "static_gpu_runs": 0, "paper_compiled": False,
                          "git_push": False}}
    write_json(DOC / "EXECUTION_RECEIPT.json", receipt)
    base_overall = lifter["overall"]["methods"]["Base"]
    print(json.dumps({"status": "PASS", "output": str(DOC), "paper_changed": modified,
                      "static_counts": labels["counts"],
                      "lifter_fresh": base_overall["fresh_count"],
                      "lifter_no_pose": base_overall["no_pose_count"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
