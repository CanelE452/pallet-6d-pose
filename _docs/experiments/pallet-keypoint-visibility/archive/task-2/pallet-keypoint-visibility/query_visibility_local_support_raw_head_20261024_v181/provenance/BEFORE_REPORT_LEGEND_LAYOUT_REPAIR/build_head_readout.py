"""Render the fixed source experiment from sealed saved metrics, never fit data."""
from pathlib import Path
import argparse
import hashlib
import html
import json
from datetime import datetime, timezone

V = Path(__file__).resolve().parent
ARMS = ["C50_RAW96_DUAL_SUPPORT_OWNER_AUX", "C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX"]
LABELS = ["RGB", "RGB + frozen SAM"]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    p = Path(path)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def read_bound(binding):
    assert bind(binding["path"]) == binding
    return read(binding["path"])


def write(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def ratio(a, b):
    return None if not b else a / b


def pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(s))}</th>" for s in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>"


def run(args):
    complete = read(args.evaluation)
    assert complete["completed"] and complete["PASS"]
    metrics = read_bound(complete["summary"])["results"]
    fit = read_bound(complete["fit_complete"])
    assert fit["PASS"] and fit["total_updates"] == 16000
    assert fit["all2_frozen_before_outputs_or_VAL_access"]
    gallery_complete = read(args.gallery)
    assert gallery_complete["completed"] and gallery_complete["PASS"] and gallery_complete["cases"] == 288
    assert bind(gallery_complete["index"]["path"]) == gallery_complete["index"]
    plotdir = V / "scientific_readout_v1"
    plotdir.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.5), constrained_layout=True)
    colors = ["#3974c7", "#db7e29"]
    for row, domain in enumerate(["SYNTH_TRAIN4096", "SYNTH_VAL2048"]):
        for col, task in enumerate(["visible_end_support", "foreign_front_support"]):
            ax = axes[row, col]
            for j, arm in enumerate(ARMS):
                m = metrics[arm][domain]["ALL"][task]["primary_known_available"]
                val = [m["precision"][1], m["recall"][1]]
                x = np.arange(2) + (j - .5) * .34
                bars = ax.bar(x, [0 if v is None else v for v in val], .32, color=colors[j], label=LABELS[j])
                for bar, value in zip(bars, val):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025, pct(value), ha="center", fontsize=10)
            ax.set_xticks([0, 1], ["YES precision", "YES recall"])
            ax.set_ylim(0, 1.18)
            ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
            m = metrics[ARMS[0]][domain]["ALL"][task]["primary_known_available"]
            split = "TRAIN (apparent fit)" if row == 0 else "VALIDATION (evaluation only)"
            title = "Visible end support" if col == 0 else "Foreign in front of end"
            ax.set_title(f"{split}\n{title}; YES={m['class_support'][1]}, NO={m['class_support'][0]}", fontsize=11)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].legend(loc="upper right", frameon=False)
    fig.suptitle("Fixed 8,000 updates per arm · independent nonexclusive supports\nSource evidence; not private-real native visibility accuracy", fontsize=14)
    plot = plotdir / "FIXED_RAW_SAM_SOURCE_SUPPORT.png"
    fig.savefig(plot, dpi=170)
    fig.savefig(plot.with_suffix(".pdf"))
    plt.close(fig)
    summary = {}
    rows = []
    for arm, name in zip(ARMS, LABELS):
        summary[arm] = {}
        for domain in ["SYNTH_TRAIN4096", "PUBLIC_TRAIN1480", "CF3816", "SYNTH_VAL2048"]:
            m = metrics[arm][domain]["ALL"]
            s = {"rows": m["rows_all"], "original_inputU": m["original_inputU"], "original_inputU_rate": ratio(m["original_inputU"], m["rows_all"]), "visible": m["visible_end_support"], "foreign": m["foreign_front_support"], "regional_proxy": m["regional_proxy_D0_E1"], "owner_soft": m["owner_soft"], "owner_hard_weak_only": m["owner_hard_weak_only"], "original_native_association": m["original_native_semantic_proxy_association"]}
            summary[arm][domain] = s
            if domain in ["SYNTH_TRAIN4096", "SYNTH_VAL2048"]:
                vv = s["visible"]["primary_known_available"]
                ff = s["foreign"]["primary_known_available"]
                rows.append([name, domain, pct(vv["precision"][1]), pct(vv["recall"][1]), pct(ff["precision"][1]), pct(ff["recall"][1]), pct(s["original_inputU_rate"])])
    baseline = metrics["historical_v61_separate_fixed_diagnostic"]
    stratum_rows = []
    for arm, name in zip(ARMS, LABELS):
        for group, sm in metrics[arm]["SYNTH_VAL2048"].items():
            if not group.startswith(("material:", "aspect:")):
                continue
            vv = sm["visible_end_support"]["primary_known_available"]
            ff = sm["foreign_front_support"]["primary_known_available"]
            stratum_rows.append([name, group, sm["rows_all"], vv["class_support"][1], pct(vv["precision"][1]), pct(vv["recall"][1]), ff["class_support"][1], pct(ff["precision"][1]), pct(ff["recall"][1])])
    strata_table = table(["모델", "검증 층", "원 행", "보임 YES", "보임 P", "보임 R", "앞 가림 YES", "앞 가림 P", "앞 가림 R"], stratum_rows)
    confusion_rows = []
    unknown_rows = []
    for arm, name in zip(ARMS, LABELS):
        for key, label in [("visible_end_support", "부위 보임"), ("foreign_front_support", "외부 앞 가림")]:
            cm = metrics[arm]["SYNTH_VAL2048"]["ALL"][key]["primary_known_available"]
            for k, truth in enumerate(["NO", "YES"]):
                confusion_rows.append([name, label, truth] + cm["CM"][k])
            unknown_rows.append([name, label, cm["truth_known"], cm["truth_unknown"], metrics[arm]["SYNTH_VAL2048"]["ALL"]["original_inputU"]])
    confusion_table = table(["모델", "검증 감독", "정답", "예측 NO", "예측 YES", "예측 U"], confusion_rows)
    coverage_table = table(["모델", "검증 감독", "유효 입력 중 정답 알려짐", "유효 입력 중 감독 UNKNOWN", "원 입력 U"], unknown_rows)
    baseline_rows = []
    for domain in ["SYNTH_TRAIN4096", "SYNTH_VAL2048"]:
        for arm, name in zip(ARMS, LABELS):
            mm = metrics[arm][domain]["ALL"]["regional_proxy_D0_E1"]
            baseline_rows.append([domain, name, mm["class_support"][0], mm["class_support"][1], pct(mm["recall"][0]), pct(mm["precision"][1]), pct(mm["recall"][1]), pct(ratio(mm["prediction_U_among_known"], mm["truth_known"]))])
        mm = baseline["baseline"][domain]["ALL"]["regional_proxy_D0_E1"]
        baseline_rows.append([domain, "v61: pixel-presence diagnostic", mm["class_support"][0], mm["class_support"][1], pct(mm["recall"][0]), pct(mm["precision"][1]), pct(mm["recall"][1]), pct(ratio(mm["prediction_U_among_known"], mm["truth_known"]))])
    result = {"schema": "V181_ROOT_FIXED_RAW_SAM_SOURCE_READOUT_v1", "UTC": datetime.now(timezone.utc).isoformat(), "completed": True, "source_experiment_complete": True, "whole_goal_solved": False, "selected_model": None, "adopted": False, "goal_status": "active_unsolved", "arms": summary, "historic_v61_separate_diagnostic": baseline, "actual_training_updates": 16000, "failed_original_attempt_updates": 0, "same_initial_state_sha": fit["same_initial_state_sha"], "same_draw_sha": fit["same_draw_sha"], "evaluation": bind(args.evaluation), "fit": complete["fit_complete"], "plot": bind(plot), "private_or_publicheld_new_evaluation": 0, "original_qL_native_and_PnP_unchanged": True, "limits": ["TRAIN metrics are apparent fit only.", "Synthetic validation is evaluation-only but historically reused, not a blind research holdout.", "Public253 labels are assistantweak literal-owner auxiliary; CF314 labels are controlled owner auxiliary. Unknown rows remain in all-row ledgers.", "Two support labels are nonexclusive; original native D/E/S and position are separate diagnostic associations.", "The v61 comparison uses a regional-support proxy and is a task-mismatch diagnostic, not equal-task native visibility certification.", "No private223 real transfer improvement or final reliable visibility classifier is established by this source experiment."]}
    result["full_strata"] = complete["summary"]
    gallery = "fixed_source_visual288_v1/index.html"
    if args.gallery:
        gc = gallery_complete
        result["gallery_complete"] = bind(args.gallery)
        gallery = str(Path(gc.get("index", {}).get("path", V / gallery)).relative_to(V)) if isinstance(gc.get("index"), dict) else gallery
    write(V / "ROOT_RESULTS_KO.json", result)
    text = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>V181 고정 crop RGB/SAM 비교</title><style>body{{max-width:1160px;margin:32px auto;padding:0 20px;font:16px/1.7 sans-serif;color:#172c40;background:#f7f9fc}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px;margin:18px 0}}th,td{{padding:9px;border:1px solid #d8e0eb;text-align:left}}th{{background:#e9eff7}}img{{max-width:100%;background:white}}a{{color:#1759ad}}.note{{background:#fff3d3;padding:16px;border-radius:8px}}</style><h1>초기 query crop의 RGB와 SAM 입력 비교</h1><p class="note"><b>합성·공개 소스에서 고정 실험을 완료했습니다. 사용자 실사에서의 개선과 전체 문제 해결은 아직 입증하지 않았습니다.</b> 최종 모델 두 개를 모두 고정한 뒤 검증 출력을 만들었으며, 모델·문턱·checkpoint를 이 결과로 선택하지 않았습니다.</p><h2>학습과 데이터</h2><p>두 모델은 원래 RGB96·valid·고정 상대좌표를 읽습니다. SAM 추가 모델만 원래 query8 box로 얻은 고정 SAM의 마스크·가용성을 더 읽습니다. 초기 좌표·crop 크기·native 순서·PnP를 변경하지 않았습니다. 각 8,000회, 같은 초기 가중치·샘플 순서·손실·optimizer를 사용했습니다.</p><p>합성 TRAIN 4,096행, 공개 실사 1,480행, 기존 가림 합성 3,816행을 보존했습니다. 공개 실사 owner 감독은 기존 weak 253행, 가림 합성 owner 감독은 기존 known 314행에만 있습니다. 합성 검증 2,048행은 별도 평가 전용입니다. 입력 U는 각각 80/194/360/8행입니다. 파생 crop·가림 변형 수를 독립 실사 이미지 수로 해석하지 않습니다.</p><h2>해당 부위가 보이는 증거와 외부 물체의 앞 가림</h2><p>두 정답은 비배타적입니다. 부분 가림에는 둘 다 YES일 수 있습니다. 자기 가림·모호함·등록 불확실성이 남은 항목은 감독 UNKNOWN으로 보존했고, 배경을 외부 가림으로 이름 바꾸지 않았습니다. 아래 P/R 분모는 각 감독이 알려졌고 원 입력이 유효한 항목입니다.</p><img src="scientific_readout_v1/{plot.name}" alt="소스 학습 및 검증에서 두 모델의 부위 보임과 외부 앞 가림 precision recall">{table(['모델','소스','보임 P','보임 R','앞 가림 P','앞 가림 R','원 입력 U율'], rows)}<h2>v61과 같은 query에서 보는 별도 진단</h2><p>새 모델은 foreign YES → E, 아니면 visible YES → D, 나머지 U로 고정 응답합니다. 아래 정답은 두 support가 모두 알려진 경우의 지역 증거 proxy입니다. v61은 중심 픽셀 presence를 학습했고 BOTH를 출력하지 못하므로, 이 표를 같은 의미의 native 가림 정확도라고 부르지 않습니다.</p>{table(['소스','모델','D 분모','E 분모','D recall','E precision','E recall','알려진 proxy의 U율'], baseline_rows)}<h2>실제 crop과 남은 한계</h2><p><a href="{html.escape(gallery)}">고정 288개 사례: 원 RGB·SAM·합성 부위 감독·두 모델 응답</a></p><p>사례는 예측을 보기 전에 원 metadata의 geometry × aspect × material 각 첫 검증 base와 공개 원 촬영 sequence 각 첫 학습 frame으로 정했습니다. 원 RGB는 별도 clean panel로 보존했습니다. 공개 실사의 weak owner를 native 키포인트 정답으로 승격하지 않았습니다.</p><p>이 실험은 source에서의 학습 가능성과 남는 오류를 확인합니다. 현재 사용자 평가 223장은 학습·문턱·모델 선택에서 제외했으며, 이번 단계에서 새 평가 forward를 하지 않았습니다. 실사 전이 성능과 최종 채택 여부는 별도로 확인해야 합니다.</p><p><a href="ROOT_RESULTS_KO.json">수치와 한계 JSON</a> · <a href="{Path(args.evaluation).parent.name}/COMPLETE.json">저장 평가 영수증</a></p></html>"""
    with (V / "ROOT_READOUT_KO.html").open("x") as f:
        text = text.replace("<h2>v61과 같은 query에서 보는 별도 진단</h2>", "<h2>합성 검증의 재질·형상별 결과</h2><p>재질별 분모와 형상별 분모는 서로 겹칩니다. 사용자 실사의 목재·정사각형 평가군과 동일한 촬영 분포는 아닙니다.</p>" + strata_table + "<h2>v61과 같은 query에서 보는 별도 진단</h2>")
        text = text.replace("<h2>합성 검증의 재질·형상별 결과</h2>", "<h2>합성 검증의 혼동행렬과 감독 범위</h2><p>원 입력이 유효하고 해당 감독이 알려진 항목의 행렬입니다. 감독 UNKNOWN과 입력 U는 서로 다르며 아래에 따로 표시합니다. 입력이 유효한 항목에 confidence 문턱으로 추가 보류를 만들지 않았습니다.</p>" + confusion_table + coverage_table + "<h2>합성 검증의 재질·형상별 결과</h2>")
        text = text.replace("현재 사용자 평가 223장은 학습·문턱·모델 선택에서 제외했으며", "이번 V181 source 실험에서 사용자 평가 223장은 학습·문턱·모델 선택에서 제외했으며")
        f.write(text)
    print(json.dumps({"completed": True, "results": bind(V / "ROOT_RESULTS_KO.json"), "readout": bind(V / "ROOT_READOUT_KO.html"), "plot": bind(plot)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    run(parser.parse_args())
