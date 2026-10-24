"""V197 report from generated TRAIN metadata and completed saved/visual QA only."""
from collections import Counter
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

V = Path(__file__).parent
G = V / "fixed_train_target_contract_viewer_v1"


def safe(path):
    path = Path(path)
    assert path.is_absolute() and "v112" not in str(path).casefold()
    for node in (path, *path.parents):
        assert not node.is_symlink()
    return path


def binding(path):
    path = safe(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(safe(path).read_text())


def new(path, text):
    with safe(path).open("x", encoding="utf-8") as stream:
        stream.write(text)


def summary(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def main():
    complete = read(G / "COMPLETE.json")
    qa = read(V / "independent_saved_train_viewer_review_b/REVIEW.json")
    interpret = read(V / "independent_train_target_interpretation_a/REVIEW.json")
    root_visual = read(V / "ROOT_ACTUAL_FIXED8_TRAIN_CONTACT_AND_CONTEXT_VISUAL_QA.json")
    later_visual = read(V / "independent_saved_train_viewer_review_b/ACTUAL_FIXED_LATER_CONTACT_CONTEXT_VISUAL_QA.json")
    assert all(r["completed"] and r["PASS"] for r in (complete, qa, interpret, root_visual, later_visual))
    all109 = read(G / "ALL109_TRAIN_TRIPLET_CONTRACTS_METADATA_BEFORE_PIXELS.json")["triplets109"]
    selected = read(G / "SELECTION_BEFORE_ANY_TARGET_NPZ_HEADER_OR_PIXEL_READ.json")["selected"]
    with safe(G / "SELECTED_TRAIN_ALL8_TARGET_MAPPING.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(all109) == 109 and len(selected) == 12 and len(rows) == 288
    counts = Counter(r["direct_weak_target"] for r in rows)
    assert counts == {"-1": 252, "0": 24, "1": 12}
    assert sum(r["original_query_available"] != "True" for r in rows) == 24
    assert all(r["role"] == "TRAIN" and r["training_admission"] == "False" for r in rows)
    assert Counter(r["native_slot"] for r in rows) == {str(i): 36 for i in range(8)}
    assert complete["fullcontext_available"] == 36 and complete["fullcontext_missing"] == 0
    root_contact = {r["contact"]["path"] for r in root_visual["records"]}
    other_contact = {r["image"]["path"] for r in later_visual["images"] if r["kind"] == "selected_contact"}
    expected_contacts = {str(G / f"TRIPLET_{i:02d}/CONTACT_SELECTED_WEAK_END.png") for i in range(12)}
    assert len(root_contact) == 8 and len(other_contact) == 4
    assert not root_contact & other_contact and root_contact | other_contact == expected_contacts
    assert root_visual["actual_BEFORE_fullcontext_display_views"] == 8
    assert later_visual["actual_viewed_BEFORE_fullcontexts"] == 4
    domain_stats = []
    for domain in ("SYNTH_FINITE_END", "REAL_SEMANTIC_END_WEAK"):
        items = [r for r in all109 if r["domain"] == domain]
        distances, widths, heights = [], [], []
        for item in items:
            a = item["association"]
            assert a["training_admission"] is False and a["association_only_not_nativephysicalgold"] is True
            assert len(a["candidate_enclosures_original96_diagnostic"]) == 1
            x0, y0, x1, y1 = a["candidate_enclosures_original96_diagnostic"][0]
            distances.append(math.hypot((x0+x1)/2-47.5, (y0+y1)/2-47.5))
            widths.append(x1-x0)
            heights.append(y1-y0)
        bases = sorted({r["association"]["source_base_index"] for r in items if r["association"]["source_base_index"] is not None})
        frames = sorted({r["association"]["source_frame_index"] for r in items if r["association"]["source_frame_index"] is not None})
        canonical = sum(r["association"]["canonical_index_metadata"] is not None for r in items)
        stat = {"domain": domain, "triplets": len(items), "known_D": 2*len(items), "known_E": len(items),
                "source_bases": bases, "source_frames": frames,
                "stored_component_count": len({r["component_id"] for r in items}),
                "canonical_metadata_provided": canonical, "canonical_metadata_UNKNOWN": len(items)-canonical,
                "qcenter_to_enclosure_center_distance96": summary(distances),
                "enclosure_width96": summary(widths), "enclosure_height96": summary(heights)}
        independent = next(r for r in interpret["all109_domain_statistics"] if r["domain"] == domain)
        for key in stat:
            assert stat[key] == independent[key], key
        domain_stats.append(stat)
    native_counts = [sum(r["selected_native_slot"] == n for r in all109) for n in range(8)]
    assert native_counts == [35, 26, 13, 22, 6, 7, 0, 0]
    evidence_names = ["ROOT_ACTUAL_TRAIN_ONLY_TARGET_CONTRACT_VIEWER_SCOPE.json",
                      "ROOT_ACTUAL_TRAIN_ONLY_TARGET_CONTRACT_VIEWER_ENTRY.json",
                      "ROOT_ACTUAL_TRAIN_ONLY_TARGET_CONTRACT_VIEWER_TERMINAL.json",
                      "ROOT_ACTUAL_FIXED8_TRAIN_CONTACT_AND_CONTEXT_VISUAL_QA.json",
                      "independent_saved_train_viewer_review_b/REVIEW.json",
                      "independent_train_target_interpretation_a/REVIEW.json"]
    result = {"completed": True, "PASS": True, "code": binding(__file__),
              "viewer_complete": binding(G / "COMPLETE.json"), "evidence": [binding(V / n) for n in evidence_names],
              "all109_TRAIN_contracts": 109, "known_TRAIN_D218_E109": True,
              "domain_statistics": domain_stats, "all109_native_known_triplets0to7": native_counts,
              "selected12": {"synthetic": 8, "publicreal": 4, "rows288": 288,
                             "D": 24, "E": 12, "truth_UNKNOWN": 252, "input_U": 24,
                             "fullcontext_available": 36, "fullcontext_missing": 0},
              "actual_view_union": {"contacts": 12, "BEFORE_fullcontext_displays": 12,
                                    "root_contact8_context8": True, "B_contact4_context4": True,
                                    "all288_individual_crops_viewed": False, "all36_contexts_viewed": False,
                                    "HTML_browser_rendered": False},
              "confirmed": ["The weak target can be away from literal query center, which stays unchanged.",
                            "Constructive synthetic and assistant-real end enclosures have different definitions and scales.",
                            "Some viewed synthetic interventions are tiny or visually similar to pallet/background; others are distinct.",
                            "Real direct supervision has12sourceframes in1stored connected component; native6/7 direct supervision0."],
              "not_proven": ["Physical/native gold or human visibility identity from canonical metadata alone.",
                             "Which factor causes model failures, including pooling, contrast, target offset or SAM.",
                             "Visibility improvement, model adoption or whole-goal success."],
              "new_fit_or_predictions_or_model_or_selection_in_phase": False,
              "protected_evaluation_or_HOLDOUT_pixels_or_scores_access": 0,
              "fullsource_target_container_equality_disclosure": "Existing full3144 sourceMETA/targets equality includesHOLDOUT metadata only; selectedpixel bodies are TRAIN only.",
              "training_admission": False, "model_adopted": False, "fully_solved": False}
    md = ["# V197: 원 query와 학습 부위의 대응 진단", "",
          "**학습용 자료에서 감독 부위와 원 입력의 관계를 확인했다. 새 모델·학습·예측·PnP는 실행하지 않았다.** 현재 head는 RGB96·valid·고정 crop 상대XY와 선택적 frozenSAM을 받으며, native 번호·전체8점 좌표·감독 끝점 좌표를 직접 받지 않는다. 이 구조 사실은 기존 코드 검토에 근거한다.", "",
          "원 query·L·native0..7과 weak 라벨을 그대로 두고, 점수를 열지 않은 상태에서 전체 TRAIN109개 대응 metadata를 확인했다. 고정 hash로 선택한12세트의 원 crop과 같은 영상의 전체문맥을 함께 보았다. 이 기록은 분류 성능 검증이나 독립 human gold 작성이 아니다.", "",
          "| 직접 감독 출처 | triplet / D·E | 원본 범위 / component | canonical 제공·UNKNOWN | q→후보영역 중심96 최소·중앙·최대 |", "|---|---|---|---|---|"]
    for s in domain_stats:
        label = "구성적 합성 finite-end" if s["domain"] == "SYNTH_FINITE_END" else "공개실사 RGB-weak + YCB 차폐"
        origin = f"{len(s['source_bases'])} bases" if s["source_bases"] else f"{len(s['source_frames'])} frames"
        d = s["qcenter_to_enclosure_center_distance96"]
        md.append(f"| {label} | {s['triplets']} / {s['known_D']}·{s['known_E']} | {origin} / {s['stored_component_count']} | {s['canonical_metadata_provided']}·{s['canonical_metadata_UNKNOWN']} | {d['min']:.3f}·{d['median']:.3f}·{d['max']:.3f} |")
    md += ["", "이 거리는 crop96의 q중심과 저장된 후보 enclosure 중심 사이의 설명값이다. **물리 키포인트 좌표 오차나 좌표 보정 정답이 아니다.** 구성적 합성의 enclosure 폭/높이는 약0.918–1.567 /1.024–2.036이고, 공개실사는6.651–19.170 /8.539–23.093이다. 후보 영역의 정의·불확실성이 다르므로 물리 면적이나 난이도의 직접 비교로 해석하지 않는다.", "",
           "| native | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
           "| known triplet | " + " | ".join(map(str, native_counts)) + " |", "",
           "native6/7에는 direct known 감독이 없지만 원native8 입력을 유지했다. 공개실사의 canonical27개 UNKNOWN도 유지한다. 없는 대응을 permutation이나 GT 끝점으로 채우지 않고, UNKNOWN만을 이유로 모든 가시성 학습을 차단하지 않는다.", "",
           "## 실제 본 화면과 읽을 수 있는 범위", "",
           "선택12=합성8+공개실사4이며 288행(D24/E12/UNKNOWN252)이다. 입력 U24(8.33%)를 유지했고 모두 선택된 공개실사96행 안에 있다. 이는 전체 TRAIN109의 가용성 비율이 아니다. 같은 exact TRAINpacket의 전체문맥36개가 모두 존재했고 다른 영상으로 대체하지 않았다.", "",
           "Root는 고정00–07의 contact8장+BEFORE문맥8장, 독립 검토자는08–11의 contact4장+BEFORE문맥4장을 실제 도구로 보았다. 12개 contact와12개 BEFORE문맥을 본 범위이며, 288개 crop이나36개 문맥 전부를 눈으로 검사한 것은 아니다. PNG 원문 대응과 crop ROI는 별도 독립 검산으로 확인했다.", "",
           "공개실사 Q0041_D3·Q0033_D2·Q0381_D2에서는 원 query 중심과 표시된 끝 후보 영역이 떨어져 보인다. 이는 q나 라벨이 틀렸다는 판정이 아니다. literal 중심 픽셀의 소속과, 의도한 팔레트 끝부분의 차폐 여부가 서로 다른 질문임을 보여 준다. 전체문맥은 해당 팔레트 부위를 이해하는 데 도움을 주지만 physical/native 정체성을 인증하지는 않는다.", "",
           "합성 BASE020·BASE082에서는 작은 가림이 목재와 비슷하게 보이고, BASE095에서는 배경과 비슷한 작은 윤곽 변화로 보인다. BASE118·BASE057·BASE089처럼 색이 구분되는 사례도 있다. 픽셀 변화가 있다는 사실과 사람이 부위를 식별할 충분한 단서가 있다는 주장을 구분한다. 이번 화면에는 예측이 없으므로 이 요인이나3×3 표현 압축을 모델 실패 원인으로 확정하지 않는다.", "",
           "## 현재 판단과 다음 확인", "",
           "직접 감독의 출처·부위·입력 대응을 명시했고, 원좌표·누락·UNKNOWN을 보존한 채 비교할 화면을 확보했다. 공개실사 direct27쌍은12개 프레임·1개 저장 component에 속하며 자연가림 정답27개나 독립27영상이 아니다. 가림 관계 일반화와 사람 native D/E에 대한 전이는 여전히 미해결이다.", "",
           "다음은 학습용 근거만으로, 배포 때 얻는 native 번호와 초기8점 문맥을 head에 전달할 수 있는지 별도 계약부터 확인한다. 원 crop·query·L·키포인트 순서·기존 라벨은 유지하며 감독 끝점·GT pose를 입력으로 추가하지 않는다. 번호 조건과8점 조건의 가용성을 분리해 검토하고, 증강된 한 점과 다른 점을 임의로 같은 추론 결과처럼 합치지 않는다. 이 보고서는 새 모델·자료·문턱을 채택하거나 새 학습을 실행한 기록이 아니다.", "",
           "[고정12 원 crop·전체문맥](fixed_train_target_contract_viewer_v1/PORTABLE_GALLERY_KO.md) · [HTML 갤러리](fixed_train_target_contract_viewer_v1/index.html) · [전체109 대응](fixed_train_target_contract_viewer_v1/ALL109_TRAIN_TRIPLET_CONTRACTS_METADATA_BEFORE_PIXELS.json) · [선택288행](fixed_train_target_contract_viewer_v1/SELECTED_TRAIN_ALL8_TARGET_MAPPING.csv) · [결과 JSON](ROOT_RESULTS_KO.json)", "",
           "[독립 저장·PNG·실제뷰 검토](independent_saved_train_viewer_review_b/REVIEW.json) · [독립 해석](independent_train_target_interpretation_a/KO.md) · [root 실제뷰](ROOT_ACTUAL_FIXED8_TRAIN_CONTACT_AND_CONTEXT_VISUAL_QA.json) · [기존 head 코드 검토](../query_visibility_grouped_source_holdout_target_readout_20261024_v196/architecture_code_review_from_b/KO.md)", "",
           "[기존 V194 source 출처·학습 계약](../query_visibility_source_group_split_direct_end_training_20261024_v194/ROOT_REPORT_KO.md) · [기존 V193 공개 source 제어·권한](../query_visibility_source_end_pairs_opposite_sham_20261024_v193/source_pair_visual_readout_v2_caption_fit/PORTABLE_GALLERY_KO.md). 기존 source credit을 유지하며 새 라이선스 검사를 했다고 주장하지 않는다.", ""]
    report = "\n".join(md)
    new(V / "ROOT_RESULTS_KO.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    new(V / "ROOT_REPORT_KO.md", report)
    new(V / "PORTABLE_REPORT_KO.md", report)
    cards = "".join(f'<article><h2>{html.escape(s["attempt_id"])}</h2><img src="fixed_train_target_contract_viewer_v1/TRIPLET_{i:02d}/CONTACT_SELECTED_WEAK_END.png"><img src="fixed_train_target_contract_viewer_v1/TRIPLET_{i:02d}/BEFORE_FULL_CONTEXT_DISPLAY.png"></article>' for i, s in enumerate(selected))
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V197 TRAIN 부위 대응</title><style>body{font:18px sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;line-height:1.6}img{max-width:100%;height:auto}article{border-top:1px solid #bbb;margin-top:30px}pre{white-space:pre-wrap;background:#f4f4f4;padding:20px}</style><h1>V197 학습용 부위 대응</h1><p>새 모델·점수·학습 없음. 원 query/L/native8을 유지한 고정12세트이며 weak 감독입니다.</p><p><a href="PORTABLE_REPORT_KO.md">보고서·표·근거</a> · <a href="fixed_train_target_contract_viewer_v1/index.html">전체 고정 갤러리</a></p><pre>' + html.escape(report) + '</pre>' + cards + '</html>'
    new(V / "ROOT_READOUT_KO.html", page)
    receipt = {"completed": True, "PASS": True, "code": binding(__file__),
               "outputs": [binding(V / n) for n in ("ROOT_RESULTS_KO.json", "ROOT_REPORT_KO.md", "PORTABLE_REPORT_KO.md", "ROOT_READOUT_KO.html")],
               "HTML_browser_rendered": False, "new_model_fit_predictions_selection": False,
               "model_adopted": False, "fully_solved": False}
    out = V / "ROOT_TRAIN_TARGET_REPORT_COMPLETE.json"
    new(out, json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": True, "PASS": True, "receipt": binding(out), "native_counts": native_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
