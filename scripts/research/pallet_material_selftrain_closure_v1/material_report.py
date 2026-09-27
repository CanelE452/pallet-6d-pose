"""Material-stratified reporting from completed, frozen measurements only.

Writes only this experiment's namespace. It never fits, runs inference, opens
coordinate arrays, edits the paper, builds LaTeX, or performs Git mutations.
"""
from collections import Counter
from pathlib import Path
import json
import subprocess

from . import common as C

PLASTIC_ARMS = ("R0", "RAW_LR5", "REF_LR5")
WOOD_ARMS = ("R0", "WOOD_RAW_LR5", "WOOD_REF_LR5")
DISPLAY = {"R0": "R0", "RAW_LR5": "Raw ST", "REF_LR5": "Corrected ST",
           "WOOD_RAW_LR5": "Raw ST", "WOOD_REF_LR5": "Corrected ST", "SYN_LR5": "Source-only update", "TEACHER": "Frozen teacher (legacy only)"}


def fmt(value, precision=3, signed=False):
    if value is None:
        return "NA"
    return format(float(value), ("+" if signed else "") + f".{precision}f")


def percent(value, precision=2):
    return fmt(None if value is None else 100*value, precision)


def texescape(value):
    substitutions = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                     "#": r"\#", "{": r"\{", "}": r"\}", "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(substitutions.get(char, char) for char in str(value))


def table_text(headers, rows, caption, label, wide=False):
    """Return Markdown and complete LaTeX table; no filesystem side effects."""
    assert all(len(row) == len(headers) for row in rows)
    env = "table*" if wide else "table"
    size = r"\textwidth" if wide else r"\columnwidth"
    lines = [rf"\begin{{{env}}}[t]", r"\centering\small", r"\caption{" + texescape(caption) + "}",
             r"\label{tab:" + label + "}", r"\resizebox{" + size + r"}{!}{%",
             r"\begin{tabular}{" + "l"*len(headers) + "}", r"\toprule",
             " & ".join(map(texescape, headers)) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(texescape, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}}", rf"\end{{{env}}}"])
    return "# " + caption + "\n\n" + C.table(headers, rows), "\n".join(lines) + "\n"


def delta(group, raw, corrected):
    a, b = group[raw], group[corrected]
    assert a["twoD"]["corners"] == b["twoD"]["corners"]
    assert a["sixD"]["frames"] == b["sixD"]["frames"]
    return dict(PCK10_pp=100*(b["twoD"]["PCK"]["10"]-a["twoD"]["PCK"]["10"]),
                ADDsym_AUC=b["sixD"]["ADDsym_AUC"]-a["sixD"]["ADDsym_AUC"],
                correct10=b["twoD"]["correct"]["10"]-a["twoD"]["correct"]["10"],
                median_px=b["twoD"]["matched_pooled_corner8_median_px"]-a["twoD"]["matched_pooled_corner8_median_px"],
                P90_px=b["twoD"]["matched_pooled_corner8_P90_px"]-a["twoD"]["matched_pooled_corner8_P90_px"],
                frames=a["sixD"]["frames"], supported_corners=a["twoD"]["corners"])


def classify(plastic, wood):
    assert plastic["PCK10_pp"] > 0 and plastic["ADDsym_AUC"] > 0, "Frozen Plastic result changed"
    positives = (wood["PCK10_pp"] > 0, wood["ADDsym_AUC"] > 0)
    if all(positives):
        return dict(material_decision="MATERIAL_GENERAL_SIGNAL", material_consistency="CONSISTENT",
            wood_self_training_value="SUPPORTED", wood_6d_value="SUPPORTED",
            wording_en="Corrected-target self-training improved the predeclared pooled PCK10 and ADDsym AUC over matched raw-target self-training in both evaluated pallet material categories under the tested protocol.",
            final_sentence_ko="반복 DEV의 ordinary plastic과 Wood에서 corrected-vs-raw 자기학습의 두 주 지표가 같은 개선 방향을 보였지만, 이는 평가한 두 재료 범주에 한정된 신호다.")
    return dict(material_decision="MATERIAL_DEPENDENT_EFFECT", material_consistency="MIXED" if any(positives) else "NOT_SUPPORTED",
        wood_self_training_value="PARTIAL" if any(positives) else "NOT_SUPPORTED",
        wood_6d_value="SUPPORTED" if positives[1] else "NOT_SUPPORTED",
        wording_en="The effect was material-dependent under the tested protocol: the positive ordinary-plastic contrast did not extend to both predeclared primary metrics in Wood.",
        final_sentence_ko="Ordinary plastic의 corrected-vs-raw 이득은 Wood의 두 주 지표 모두로 이어지지 않아, 현재 반복 DEV와 고정 계약에서 효과는 material-dependent하다.")


def load_completed():
    names = ("WOOD_RESULTS.json", "WOOD_PAIRED_ANALYSIS.json", "WOOD_PREDICTIONS_LOCK.json", "WOOD_SCORING_START.json",
             "WOOD_PSEUDO_QUALITY.json", "WOOD_PROVENANCE_AUDIT.json", "WOOD_DATA_INVENTORY.json", "METHOD_LOCK.json",
             "WOOD_TRAIN_PROTOCOL.json", "WOOD_PAIR_PREFLIGHT.json", "POOL_DECISION_V2.json", "POOL_DECISION_CORRECTION.json",
             "PSEUDO_COMPLETE.json", "EVAL_POPULATION_LOCK.json", "PREFLIGHT.json",
             "FIT_WOOD_RAW_LR5.json", "FIT_WOOD_REF_LR5.json")
    missing = [name for name in names if not (C.DOC/name).is_file()]
    assert not missing, ("Completed data required; do not write placeholder results", missing)
    data = {name[:-5]: C.read(C.DOC/name) for name in names}
    result = data["WOOD_RESULTS"]
    C.verify(result["prediction_lock"])
    for binding in result["artifact_sources"]:
        C.verify(binding)
    assert result["raw_and_pose_frozen_before_reference_scoring"] and result["frames"] == 45
    assert result["recordings"] == 2 and not result["severity_available"]["SEVERE"]
    assert data["WOOD_PREDICTIONS_LOCK"]["created_at"] <= data["WOOD_SCORING_START"]["utc"]
    assert data["WOOD_SCORING_START"]["prediction_lock"] == C.bind(C.DOC/"WOOD_PREDICTIONS_LOCK.json")
    preflight = data["WOOD_PAIR_PREFLIGHT"]
    assert preflight["status"] == "PASS" and not preflight["evaluation_labels_opened"]
    for key in ("initialization_same", "source_same_as_plastic_main", "common_support", "same_boxes", "same_image_order", "same_optimizer_budget", "coordinate_values_differ"):
        assert preflight[key]
    for key in ("rgb_order", "boxes", "support", "source_replay", "slots", "synthetic_slots"):
        assert preflight["parity"]["RAW"][key] == preflight["parity"]["REF"][key]
    assert preflight["parity"]["RAW"]["coordinates"] != preflight["parity"]["REF"]["coordinates"]
    for arm in WOOD_ARMS[1:]:
        fit = data["FIT_"+arm]
        assert fit["complete"] and fit["optimizer_steps"] == 320 and fit["epochs"] == 5
        assert fit["exact_R0_initialization"] and fit["protected_state_exact"] and not fit["evaluation_labels_opened"]
        for key in ("checkpoint", "protocol", "preflight", "results_csv"):
            C.verify(fit[key])
        assert fit["protocol"] == C.bind(C.DOC/"WOOD_TRAIN_PROTOCOL.json")
        state = C.read(C.DOC/("RUN_STATE_"+arm+".json"))
        assert state["status"] == "COMPLETED" and state["fit"] == C.bind(C.DOC/("FIT_"+arm+".json"))
    for key in ("teacher_vs_main45", "eligible_pool_vs_main45", "selected_candidates_vs_main45"):
        assert all(not values for values in data["WOOD_PROVENANCE_AUDIT"]["overlaps"][key].values())
    assert data["WOOD_PSEUDO_QUALITY"]["Q1_WOOD"] == "UNRESOLVED"
    assert data["WOOD_PSEUDO_QUALITY"]["metrics"]["evaluated_trusted_points"] == 0
    correction = data["POOL_DECISION_CORRECTION"]
    assert correction["fits_before_correction"] == 0 and correction["evaluation_scores_not_opened"]
    assert correction["old_record_preserved"] and correction["no_filter_threshold_change"]
    C.verify(correction["prior_decision"])
    data["PLASTIC"] = C.read(C.P.DOC/"CORE_RESULTS.json")
    data["PLASTIC_QUALITY"] = C.read(C.P.DOC/"PSEUDO_LABEL_QUALITY.json")
    data["OCCURRENCES"] = C.read(C.ROOT/data["WOOD_TRAIN_PROTOCOL"]["training_occurrences"]["path"])
    assert sum(r["occurrences_per_epoch"] for r in data["OCCURRENCES"]) == 512
    assert len(data["OCCURRENCES"]) == preflight["train_unique"]
    data["REPORT_INPUTS"] = [C.bind(C.DOC/name) for name in names]+[C.bind(C.P.DOC/name) for name in ("CORE_RESULTS.json", "PSEUDO_LABEL_QUALITY.json")]
    return data


def make_tables(data):
    plastic, wood = data["PLASTIC"]["groups"], data["WOOD_RESULTS"]["groups"]
    datasets = (("Plastic", plastic, PLASTIC_ARMS), ("Wood", wood, WOOD_ARMS))
    tables = {}

    def add(name, headers, rows, caption, wide=False):
        tables[name] = dict(headers=headers, rows=rows, caption=caption,
                           texts=table_text(headers, rows, caption, name.lower().replace("_", "-"), wide))

    rows = []
    for material, groups, arms in datasets:
        for arm in arms:
            v = groups["ALL"][arm]; a, b = v["twoD"], v["sixD"]
            rows.append([material, DISPLAY[arm], b["frames"], a["corners"], percent(a["PCK"]["10"]), fmt(a["matched_pooled_corner8_median_px"]), fmt(b["ADDsym_AUC"], 5)])
    add("TABLE_MATERIAL_MAIN", ["Material", "Method", "Images", "Corners", "PCK10 %", "Med px", "ADDsym AUC"], rows,
        "Matched material-stratified self-training. PCK uses the full supported-corner denominator; median is conditional on matched detection. Plastic retains the original320-update main comparison, not the later640-update test.", True)
    add("TABLE_MATERIAL_DELTA", ["Material", "Correct10 delta", "PCK10 delta pp", "AUC delta", "Med delta px", "P90 delta px"],
        [[mat, d["correct10"], fmt(d["PCK10_pp"], 3, True), fmt(d["ADDsym_AUC"], 5, True), fmt(d["median_px"], 3, True), fmt(d["P90_px"], 3, True)]
         for mat, gs, arms in datasets for d in [delta(gs["ALL"], *arms[1:])]],
        "Within-material corrected-minus-raw contrasts. Absolute differences between material rows do not isolate a causal material-difficulty effect.", True)
    rows2, rows6 = [], []
    for material, gs, arms in datasets:
        for arm in (*arms, "SYN_LR5"):
            a, b = gs["ALL"][arm]["twoD"], gs["ALL"][arm]["sixD"]
            rows2.append([material, DISPLAY[arm], f'{a["detected"]}/{a["total_frames"]}', f'{a["matched"]}/{a["total_frames"]}',
                f'{a["correct"]["10"]}/{a["corners"]}', *[percent(a["PCK"][str(k)]) for k in (5,10,20)],
                fmt(a["matched_pooled_corner8_median_px"]), fmt(a["matched_pooled_corner8_P90_px"]), a["corners"]-a["correct"]["20"]])
            pair = lambda key: fmt(b[key]["median"])+" / "+fmt(b[key]["P90"])
            rows6.append([material, DISPLAY[arm], f'{b["available"]}/{b["frames"]}', f'{b["axis_correct_count"]}/{b["available"]}',
                pair("rotation_deg"), pair("yaw_deg"), pair("translation_cm"), fmt(b["IoU3D"]["median"],4), fmt(b["ADDsym_AUC"],5)])
    add("TABLE_MATERIAL_2D", ["Material","Method","Detected","Matched","Correct10","PCK5 %","PCK10 %","PCK20 %","Med px","P90 px",">20"], rows2,
        "Full2D results and the reused source-only update control. Med/P90 are matched-only; gross errors and PCK retain the full supported denominator.", True)
    add("TABLE_MATERIAL_6D", ["Material","Method","Pose coverage","Axis correct","R med/P90 deg","Yaw med/P90 deg","t med/P90 cm","IoU3D med","ADDsym AUC"], rows6,
        "Common deployable D9 final pose. References are geometry-derived legacy annotations, not independent physical6D measurements. Axis correctness tests W/D parity, not complete rotation correctness.", True)
    severity = []
    recordings = []
    for material, gs, arms in datasets:
        for key in ("CLEAN", "MODERATE", "SEVERE"):
            if key not in gs:
                continue
            for arm in arms:
                a,b = gs[key][arm]["twoD"],gs[key][arm]["sixD"]
                severity.append([material,key,b["frames"],DISPLAY[arm],f'{a["correct"]["10"]}/{a["corners"]}',percent(a["PCK"]["10"]),fmt(a["matched_pooled_corner8_P90_px"]),fmt(b["ADDsym_AUC"],5)])
        for key in sorted(k for k in gs if k.startswith("REC_") or k.startswith("SESSION_")):
            a,b = gs[key][arms[1]],gs[key][arms[2]]
            recordings.append([material,key,a["sixD"]["frames"],percent(a["twoD"]["PCK"]["10"]),percent(b["twoD"]["PCK"]["10"]),fmt(a["sixD"]["ADDsym_AUC"],5),fmt(b["sixD"]["ADDsym_AUC"],5)])
    add("TABLE_MATERIAL_SEVERITY", ["Material","Severity","N","Method","Correct10","PCK10 %","P90 px","ADDsym AUC"],severity,
        "All available severity strata. Wood has Clean38 and Moderate7; Wood severe is unavailable and is not represented as zero.",True)
    add("TABLE_MATERIAL_RECORDINGS", ["Material","Recording/session","N","Raw PCK10 %","Corr PCK10 %","Raw AUC","Corr AUC"],recordings,
        "All recording/session contrasts, including unfavorable groups. Session and recording summaries are overlapping descriptions, not additional independent samples.",True)
    q = data["PLASTIC_QUALITY"]["groups"]["ALL"]
    qrows = []
    for arm in ("R0","TEACHER"):
        a=q[arm]
        qrows.append(["Plastic", "Raw R0" if arm=="R0" else "Frozen teacher",a["n"],*[f'{a["PCK"][str(k)]["correct"]}/{a["n"]} ({percent(a["PCK"][str(k)]["fraction"])})' for k in (5,10,20)],fmt(a["median_px"]),fmt(a["p90_px"]),a["gt20"]])
    for arm in ("Raw R0","Frozen teacher"):
        qrows.append(["Wood",arm,0,"NA","NA","NA","NA","NA","NA"])
    add("TABLE_MATERIAL_PSEUDO_QUALITY",["Material","Output","Trusted points","PCK5 count (%)","PCK10 count (%)","PCK20 count (%)","Med px","P90 px",">20"],qrows,
        "Trusted visible pseudo-coordinate quality. Plastic:66 points in16 reused DEV images. Wood:zero eligible independently sourced direct-visible points, so Q1 is unresolved; NA is not0%.",True)
    p=data["WOOD_PAIR_PREFLIGHT"]
    add("TABLE_MATERIAL_CONTRACT",["Item","Plastic main","Wood extension"],[
        ["Frozen teacher","Same Replay9images/38corners","Same checkpoint; no refit"],
        ["Candidate/accepted/used unique", "1000 /249 /217", f'{data["WOOD_DATA_INVENTORY"]["stats"]["candidate_cap"]} /{p["accepted_unique"]} /{p["train_unique"]}'],
        ["Initial student","Same R0","Same R0"],["Trainable state","Pose branches+flow","Same"],
        ["Protected state","Backbone/detector/all buffers","Same; exact checks"],
        ["RAW/REF parity","RGB/order/box/support/source","Same; coordinate values differ"],
        ["Replay per epoch","512real+512synthetic slots","Same source512"],
        ["Optimizer/LR/updates","AdamW /1e-5 /320","Identical"],
        ["Epoch/batch/nbs/seed","5 /16 /16 /42","Identical"],
        ["Selection","Fixed final last.pt","Identical"],
        ["Final pose","Common D9; no teacher at deployment","Same method; Wood registered dimensions"],
        ["Material routing","Externally supplied material","Externally supplied material"],
        ["Evidence","Reused DEV128,7recordings","Reused DEV45,2recordings"]],
        "Matched coordinate intervention within each material. Different accepted pool sizes and evaluation conditions are disclosed rather than treated as a randomized material intervention.",True)
    return tables


def runtime_summary(data):
    return {arm:dict(checkpoint=data["FIT_"+arm]["checkpoint"],seconds=data["FIT_"+arm]["seconds"],
                epochs=data["FIT_"+arm]["epochs"],optimizer_updates=data["FIT_"+arm]["optimizer_steps"],
                epoch_updates=[r["optimizer_steps"] for r in data["FIT_"+arm]["history"]],
                exact_R0_initialization=True,protected_state_exact=True,results_csv=data["FIT_"+arm]["results_csv"])
            for arm in WOOD_ARMS[1:]}


def main():
    data=load_completed();plastic=data["PLASTIC"]["groups"];wood=data["WOOD_RESULTS"]["groups"]
    pd=delta(plastic["ALL"],*PLASTIC_ARMS[1:]);wd=delta(wood["ALL"],*WOOD_ARMS[1:]);decision=classify(pd,wd)
    tables=make_tables(data)
    for name,table in tables.items():
        for extension,text in zip(("md","tex"),table["texts"]):
            C.save(C.DOC/(name+"."+extension),text)
    paired=data["WOOD_PAIRED_ANALYSIS"];contrast=paired["groups"]["ALL"]["WOOD_REF_LR5-minus-WOOD_RAW_LR5"]
    pre=data["WOOD_PAIR_PREFLIGHT"];inv=data["WOOD_DATA_INVENTORY"]["stats"];prov=data["WOOD_PROVENANCE_AUDIT"]
    runtime=runtime_summary(data);q=data["WOOD_PSEUDO_QUALITY"];budget=q["teacher"]["by_material"]
    decision.update(status="MEASURED_MATERIAL_COMPARISON_CLOSED",PLASTIC_CORRECTED_ST_VALUE="SUPPORTED",WOOD_PSEUDO_QUALITY="UNRESOLVED",
        WOOD_CORRECTED_ST_VALUE=decision["wood_self_training_value"],WOOD_6D_VALUE=decision["wood_6d_value"],
        MATERIAL_CONSISTENCY=decision["material_consistency"],plastic_delta=pd,wood_delta=wd,
        evidence_status="REUSED_DEV_ONLY",new_fits=2,updates_each=320,total_new_updates=640,extra_SYN_fits=0,
        pair_integrity="PASS",runtime=runtime,teacher_budget=dict(images=9,corners=38,by_material=budget),
        Q1_wood_trusted_points=0,wood_severe_available=False,material_externally_provided=True,
        method_development_stopped=True,DOPE_next="NOT_RUN",user_action_required=False,
        unresolved_evidence=["Wood verified/direct-visible pseudo quality", "Independent confirmation", "Independent physical6D", "Wood severe and unseen materials"],
        not_supported=["Every material/pallet improves", "Every metric or severity improves", "Wood verified pseudo quality established", "Material alone causally explains absolute performance differences", "Automatic material classification", "Estimator architecture generalization"],
        historical_results_role="Supplementary only; different supervision/intervention, not the matched raw-corrected control",
        source_bindings=data["REPORT_INPUTS"],report_code=C.bind(Path(__file__)))
    C.save(C.DOC/"MATERIAL_FINAL_DECISION.json",decision)
    def body(name):return tables[name]["texts"][0].split("\n\n",1)[1]
    lead=(f'**{decision["material_decision"]}**. Plastic corrected−raw: PCK10 {fmt(pd["PCK10_pp"],3,True)}pp / ADDsym AUC {fmt(pd["ADDsym_AUC"],5,True)}. '
          f'Wood corrected−raw: PCK10 {fmt(wd["PCK10_pp"],3,True)}pp / ADDsym AUC {fmt(wd["ADDsym_AUC"],5,True)}. '
          '이 판정은 material 내부의 같은 RAW/REF 학습 계약에 대한 기술적 비교이며, 모든 지표 동시 개선을 요구하지 않는다.\n\n')
    reasons=data["PSEUDO_COMPLETE"]["reasons"]["WOOD"]
    process=(f'동일 Replay9/38 교사를 고정했다. Wood 후보 {inv["candidate_cap"]}장 중 raw confidence 탈락 {reasons.get("raw_confidence",0)}장, '
        f'raw flip/LOO 탈락 {reasons.get("raw_flip_LOO",0)}장, 보정 후 all8 LOO 탈락 {reasons.get("refined_all8_LOO",0)}장으로 '
        f'공통 승인 {pre["accepted_unique"]}장이다. 실제 real512 replacement 슬롯에 노출된 고유 영상은 {pre["train_unique"]}장이다. '
        '기존 synthetic512와 매 epoch 혼합하여 R0 복제 학생2개를 각5epoch/320update 학습했다. RAW/REF의 RGB·박스·support·순서·augmentation·optimizer·학습량은 같고 유효 pseudo 좌표만 다르다. '
        '교사는 student 평가/배포 추론에 붙이지 않았다.\n\n')
    scope=('Wood116의 교사 감독 exact-ID/SHA 중복은5장이고, 교사 recording에 속한 day20+night51장 전체를 평가에서 제외했다. '
        'Plastic night 교사의 REC_002도 Wood night와 같은 촬영이므로 함께 제외했다. 결과를 보기 전에 Wood main을 REC_039의25장 + REC_042의20장 =45장으로 고정했다. '
        'Clean38/Moderate7이며 Severe는 없음(0% 아님). 미주석 Wood 학습 후보는 REC_001/002에서만 고르고 기존319개 GT 영상의 exact-ID/SHA를 모두 제외했다. '
        '교사-학생 source recording은 겹치지만 두 source recording은 Wood45 평가와 겹치지 않는다. 과거 Wood116 결과는 삭제하거나 새45장 수치로 교체하지 않았다.\n\n')
    limits=('Wood45에는 출처가 확인되는 직접 클릭 가시점이0개이므로 **Q1_WOOD=UNRESOLVED**다. legacy teacher 정합 점수가 좋아도 verified pseudo-quality로 승격하지 않는다. '
        'Plastic66 가시점 teacher44→50/66과 학생43/66 동률은 그대로 유지하며, 후속640-update 학생의44/66 결과로 원래320-update main을 대체하지 않는다. '
        '두 재료 모두 반복 DEV이며 새로운 독립 TEST가 아니다. Wood는 단2recording이고 프레임/코너를 독립 반복으로 세지 않는다. '
        '6D는 geometry-derived annotation reference 정합도이며 독립 측정된 물리 pose 정확도가 아니다. '
        '재료별 별도 학생은 외부 material metadata로 선택한다(material type is externally provided for routed evaluation). '
        '자동 material 분류·unknown-material 대응·DOPE 등 estimator-generalization은 검증하지 않았다. '
        '재료별 절대 수치 차이는 난도·카메라·세션·학습 pool 차이도 포함하므로 material 자체의 인과효과로 단정하지 않는다.\n\n')
    correction=('학습 전 공통 support가6개 이상이어야 한다는 추가 gate가 잘못 들어가 전체 Wood pool을 거절한 기록이 있었다. '
        'Plastic 원래 export는 공통 support4/5개도 허용하므로 이 gate를 제거해 원래 계약을 복원했다. '
        '당시 새 fit0·평가 scoring 전이었고, 필터 threshold나 승인영상676개는 바꾸지 않았다. '
        '[원래 결정](POOL_DECISION.json), [정정 이유](POOL_DECISION_CORRECTION.json), [정정 결정](POOL_DECISION_V2.json)을 모두 보존했다. 이는 성능 결과 기반 재시도가 아니다.\n\n')
    paired_text=(f'Wood corrected−raw 정답 진입 {contrast["transitions"]["gained_correct10"]} / 이탈 {contrast["transitions"]["lost_correct10"]}점; '
        f'순변화 {contrast["correct10_delta"]:+d} / 분모 {contrast["corners"]}. 프레임 평균오차 개선/악화/동일: '
        f'{contrast["frame_mean_error"].get("improved",0)}/{contrast["frame_mean_error"].get("worsened",0)}/{contrast["frame_mean_error"].get("unchanged",0)}. '
        f'큰 오류 >20→≤10 복구 {contrast["transitions"]["recovery20_to10"]}점, <5→>10 손상 {contrast["transitions"]["damage5_to10"]}점이다. '
        'paired 변화는 같은 canonical GT point identity로 대응한다. 개별 후보/체크포인트/threshold를 결과로 다시 고르지 않았다.\n\n')
    runtime_rows=[[arm,rt["epochs"],rt["optimizer_updates"],fmt(rt["seconds"],1),rt["checkpoint"]["sha256"]] for arm,rt in runtime.items()]
    wood_report='# Wood matched RAW/corrected self-training 결과\n\n'+lead+process+'## 평가 population과 역할\n\n'+scope+'## Wood 결과\n\n'
    wood_report+=C.table(tables['TABLE_MATERIAL_MAIN']['headers'],[r for r in tables['TABLE_MATERIAL_MAIN']['rows'] if r[0]=='Wood'])+'\n'+paired_text
    wood_report+='## 모든2D/6D 지표\n\n'+C.table(tables['TABLE_MATERIAL_2D']['headers'],[r for r in tables['TABLE_MATERIAL_2D']['rows'] if r[0]=='Wood'])+'\n'+C.table(tables['TABLE_MATERIAL_6D']['headers'],[r for r in tables['TABLE_MATERIAL_6D']['rows'] if r[0]=='Wood'])+'\n'
    wood_report+='## Recording별 변화 및 LORO\n\n'+C.table(tables['TABLE_MATERIAL_RECORDINGS']['headers'],[r for r in tables['TABLE_MATERIAL_RECORDINGS']['rows'] if r[0]=='Wood'])+'\n'
    loro_rows=[]
    for rec,contrasts in paired['leave_one_recording_out'].items():
        row=contrasts['WOOD_REF_LR5-minus-WOOD_RAW_LR5']
        loro_rows.append([rec,row['frames'],fmt(row['PCK10_delta_pp'],3,True),fmt(row['ADDsym_AUC_delta'],5,True)])
    wood_report+=C.table(['Excluded recording','Remaining N','ΔPCK10 pp','ΔAUC'],loro_rows)+'\n두 recording의 기술적 민감도이며 독립 초기화 반복 또는 유의성 검정이 아니다.\n\n'
    wood_report+='## 감독·한계\n\n'+limits+'## 학습 전 정정\n\n'+correction
    C.save(C.DOC/'WOOD_RESULTS_REPORT_KO.md',wood_report)
    report='# Material-stratified corrected pseudo-label self-training 마감 보고서\n\n'+lead
    report+='## 무엇을 했는가\n\n'+process+scope
    report+=f'고정 교사의 실사 감독 예산은 총9장/38코너: Plastic {budget["PLASTIC"]["images"]}장/{budget["PLASTIC"]["manual_corners"]}코너, Wood {budget["WOOD"]["images"]}장/{budget["WOOD"]["manual_corners"]}코너다. Wood 학생용 새 수동 좌표·교사 재학습은0이다. '
    report+='이는 이 frozen teacher의 fitting 예산이며 평가 annotation/과거 프로젝트 수동 노동 전체가9/38뿐이라는 주장이 아니다. R0의 upstream COCO-pose pretraining도 숨기지 않는다.\n\n'
    for name in tables:
        report+='## '+name+'\n\n'+body(name)+'\n'
    report+='## 짝지은 변화와 손상\n\n'+paired_text
    report+='P90 delta는 corrected−raw에서 양수이면 악화, 음수이면 개선이다. 주 지표의 개선이 모든 tail·회전·이동·축 선택 또는 모든 난도 개선을 뜻하지 않는다.\n\n'
    report+='## 실제 실행 기록\n\n'+C.table(['Arm','Epochs','Updates','Seconds','Checkpoint SHA256'],runtime_rows)+'\n'
    report+='RGB/order/box/support/source hash는 [WOOD_PAIR_PREFLIGHT.json](WOOD_PAIR_PREFLIGHT.json), epoch별 update·protected 검사는 각 FIT/CSV binding에서 추적한다. 입력/export parity는 검증했지만 전체 augmented 학습 텐서를 전수 캐시해 비교했다고 주장하지 않는다.\n\n'
    report+='## 보존한 사전 오류와 정정\n\n'+correction+'## 해석 범위·미해결\n\n'+limits
    report+='과거 material-routed Replay 및 S0/S1/S2는 다른 감독/개입의 보조 역사이며 이번 matched causal control에 섞지 않았다. [재사용 역할표](WOOD_HISTORICAL_ROLE_MAP.md)를 참조한다. 결과를 이유로 teacher·threshold·loss·selector·epoch를 바꾸지 않고 방법 개발은 STOP한다.\n\n'
    report+='## 최종 한 문장\n\n'+decision['final_sentence_ko']+'\n\n원고 및 PDF 갱신·build·audit·push의 최종 확인은 저장소 마감 단계에서 별도로 기록한다. 이 보고서 생성 자체가 완료되지 않은 Git push/build를 완료했다고 선언하지 않는다.\n'
    figures=sorted((C.DOC/'figures').glob('*.png'))
    if figures:
        report+='\n## 실제 결과 이미지\n\n'+'\n'.join(f'![{p.stem}](figures/{p.name})\n' for p in figures)
    C.save(C.DOC/'REPORT_KO.md',report)
    sources=data['REPORT_INPUTS']+[C.bind(Path(__file__))]
    C.save(C.DOC/'MATERIAL_NUMBER_PROVENANCE.json',dict(sources=sources,
        original_plastic320_unchanged=True,wood_group_source='WOOD_RESULTS.json.groups',
        plastic_group_source='pallet_selftraining_paper_closure_v1/CORE_RESULTS.json.groups',
        tables={name:dict(rows=len(table['rows']),markdown=C.bind(C.DOC/(name+'.md')),latex=C.bind(C.DOC/(name+'.tex'))) for name,table in tables.items()},
        no_coordinate_arrays_published=True))
    cli(data,decision)
    reproduce(data)
    print('MATERIAL_REPORTS_GENERATED',decision['material_decision'],pd,wd,flush=True)


def cli(data, decision):
    plastic=data['PLASTIC']['groups']['ALL'];wood=data['WOOD_RESULTS']['groups']['ALL'];inv=data['WOOD_DATA_INVENTORY']['stats']
    pre=data['WOOD_PAIR_PREFLIGHT'];ov=data['WOOD_PROVENANCE_AUDIT']['overlaps']
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    remote=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
    steps=[('STATUS','MEASURED_RESULTS_COMPLETE; paper/build/Git finalization tracked separately'),('HEAD_START',data['PREFLIGHT']['head_start']),
        ('HEAD_AT_REPORT',head),('BRANCH',branch),('MATERIAL_GOAL','동일한 보정 타깃 자기학습 효과가 Plastic/Wood에서 같은 방향인지 검증'),
        ('PLASTIC_MAIN',dict(EVAL_N=128,RAW_PCK10=plastic['RAW_LR5']['twoD']['PCK']['10'],CORR_PCK10=plastic['REF_LR5']['twoD']['PCK']['10'],
            DELTA_PCK10_PP=decision['plastic_delta']['PCK10_pp'],RAW_AUC=plastic['RAW_LR5']['sixD']['ADDsym_AUC'],CORR_AUC=plastic['REF_LR5']['sixD']['ADDsym_AUC'],DELTA_AUC=decision['plastic_delta']['ADDsym_AUC'])),
        ('WOOD_DATA',dict(CANDIDATE_POOL=inv['candidate_cap'],ACCEPTED_SHARED=pre['accepted_unique'],TRAIN_UNIQUE=pre['train_unique'],
            TRAIN_RECORDINGS=sorted({r['recording'] for r in data['OCCURRENCES']}),EVAL_N=45,EVAL_RECORDINGS=sorted(inv['eligible_recordings']),
            TEACHER_IMAGE_OVERLAP=len(ov['teacher_vs_main45']['sha256']),TEACHER_SESSION_OVERLAP=ov['teacher_vs_main45']['session'],
            TRAIN_EVAL_IMAGE_OVERLAP=len(ov['selected_candidates_vs_main45']['sha256']),TRAIN_EVAL_RECORDING_OVERLAP=ov['selected_candidates_vs_main45']['recording'])),
        ('WOOD_PSEUDO_QUALITY',dict(STATUS='UNRESOLVED',POINTS=0,RAW_PCK10=None,CORR_PCK10=None,MEDIAN_DELTA=None,LIMITATION='No provenance-eligible direct-visible Wood45 points; legacy quality not verified quality')),
        ('WOOD_MATCHED_PAIR',dict(STATUS='COMPLETED',NEW_FITS=2,PAIR_INTEGRITY='PASS',runtime=decision['runtime'])),
        ('WOOD_2D',dict(R0_PCK10=wood['R0']['twoD']['PCK']['10'],RAW_PCK10=wood['WOOD_RAW_LR5']['twoD']['PCK']['10'],CORR_PCK10=wood['WOOD_REF_LR5']['twoD']['PCK']['10'],
            CORR_MINUS_RAW_PP=decision['wood_delta']['PCK10_pp'],P90_RAW=wood['WOOD_RAW_LR5']['twoD']['matched_pooled_corner8_P90_px'],P90_CORR=wood['WOOD_REF_LR5']['twoD']['matched_pooled_corner8_P90_px'])),
        ('WOOD_6D',{arm:wood[arm]['sixD'] for arm in WOOD_ARMS}),
        ('WOOD_SEVERITY',dict(CLEAN=38,MODERATE=7,SEVERE_AVAILABLE=False)),('MATERIAL_DECISION',decision['material_decision']),
        ('MATERIAL_INTERPRETATION',decision['final_sentence_ko']),('NOT_SUPPORTED',decision['not_supported']),
        ('HISTORICAL_TYPE_SPECIFIC_RESULTS','SUPPLEMENTARY_ONLY / NOT_COMPARABLE_TO_MAIN'),('DOPE_NEXT','NOT_RUN'),
        ('MANUSCRIPT','_docs/paper/selftraining_submission_v1/manuscript.tex'),('PDF','_docs/paper/selftraining_submission_v1/manuscript.pdf'),
        ('USER_ACTION_REQUIRED','NO'),('REMAINING_EVIDENCE_LIMITS',decision['unresolved_evidence']),
        ('COMMIT_AT_REPORT',head),('TRACKING_REMOTE_HEAD_AT_REPORT',remote),
        ('GIT_STATUS_AT_REPORT',subprocess.check_output(['git','status','--short','--branch','--untracked-files=no'],text=True).strip()),
        ('FINAL_SENTENCE',decision['final_sentence_ko'])]
    text='# Material closure CLI 측정 결과 보고\n\n이 파일은 보고서 생성 시점 snapshot이다. 최종 audit/tests/build/push는 실제 수행 뒤 별도로 확인한다.\n\n'
    for key,value in steps:
        text+='## '+key+'\n\n'+(value if isinstance(value,str) else '```json\n'+json.dumps(C.clean(value),ensure_ascii=False,indent=2)+'\n```')+'\n\n'
    C.save(C.DOC/'CLI_REPORT_KO.md',text)


def reproduce(data):
    text='''# Material closure 재현

기존 Plastic320-update main과 checkpoint를 덮어쓰지 않는다. 환경은 기존 `pallet-yolo26`이며 패키지/드라이버 변경이나 재부팅은 필요하지 않다. 아래 순서는 저장된 완료 단계를 검증·재사용한다. 학습 명령은 FIT가 있으면 checkpoint를 검증하고 종료하며 미완료 run은 자동 덮어쓰지 않는다. 완전히 새 복제 환경에서 다시 학습하려면 별도 결과 root를 사전 고정해야 한다.

## 입력·계약

`METHOD_LOCK.json`, `INPUT_BINDINGS.json`, `EVAL_POPULATION_LOCK.json`, `WOOD_DATA_INVENTORY.json`, `WOOD_PROVENANCE_AUDIT.json`을 먼저 검증한다. 사적 RGB/타깃/checkpoint는 공개 저장소에 포함하지 않으므로 해당 local artifacts와 SHA가 필요하다. teacher는 Replay9/38 단 하나이며 source-only control은 기존 SYN_LR5를 재사용한다.

## 실행 순서

```bash
python -m scripts.research.pallet_material_selftrain_closure_v1.lock_method
python -m scripts.research.pallet_material_selftrain_closure_v1.wood_inventory_audit
python -m scripts.research.pallet_material_selftrain_closure_v1.pseudo_pool
python -m scripts.research.pallet_material_selftrain_closure_v1.train_pair prepare
python -m scripts.research.pallet_material_selftrain_closure_v1.train_pair WOOD_RAW_LR5
python -m scripts.research.pallet_material_selftrain_closure_v1.train_pair WOOD_REF_LR5
python -m scripts.research.pallet_material_selftrain_closure_v1.infer_eval
python -m scripts.research.pallet_material_selftrain_closure_v1.score_eval
python -m scripts.research.pallet_material_selftrain_closure_v1.material_report
python -m pytest scripts/research/pallet_material_selftrain_closure_v1 -q
```

`POOL_DECISION.json`의 최초 추가 support gate 오류는 삭제하지 않았다. 실제 학습은 `POOL_DECISION_CORRECTION.json`이 설명하는 `POOL_DECISION_V2.json`을 사용했다. 빈 디렉터리에서 최초 오류 단계를 재현해 새 오류를 만드는 것이 목적은 아니다. 기존 namespace에서는 보존된 원본·정정 결정과 hash를 검증한다. `wood_inventory_audit`의 원 inventory 입력은 `WOOD_INVENTORY_AGENT.json`으로 고정되어 있어야 한다.

추론은 모든 arm과 D9 pose를 먼저 hash lock한 뒤 scoring을 수행한다. `WOOD_PREDICTIONS_LOCK.json.created_at <= WOOD_SCORING_START.json.utc` 및 source bindings를 검사한다. 보고서 생성은 checkpoint/FIT/CSV/실행 상태를 검증하지만 새로운 추론·학습을 하지 않는다.

## 수치 출처

- Plastic: 과거 `pallet_selftraining_paper_closure_v1/CORE_RESULTS.json`의 R0/RAW_LR5/REF_LR5. 후속640-update 모델로 교체하지 않는다.
- Wood: `WOOD_RESULTS.json`, `WOOD_PAIRED_ANALYSIS.json`의 사전 고정45장. 모든 recording/session/사용 가능한 severity를 유지한다.
- 검수 품질: `WOOD_PSEUDO_QUALITY.json`은 trusted0으로 UNRESOLVED. legacy 점수를 verified로 승격하지 않는다.
- 표: `MATERIAL_NUMBER_PROVENANCE.json`의 JSON→MD/TEX hash 매핑.
- 모델: 각 FIT의 checkpoint/initialization/protected-state/optimizer320 증거와 epoch CSV.

## 원고·공개 범위·종료

원고는 `_docs/paper/selftraining_submission_v1/manuscript.tex`이고 기존 빌드 절차를 그대로 사용한다. 본 모듈은 원고/PDF/build를 수정하지 않는다. 최종 paper audit/build/시각 확인은 저장소 마감 단계가 담당한다. 숫자·허용된 사례 이미지·코드만 공개하며 private 좌표 배열·원본 RGB·큰 checkpoint를 stage하지 않는다.

새 teacher/threshold/loss/selector/epoch 탐색·추가 수동 레이블·DOPE는 실행하지 않는다. 결과가 불리해도 원고에서 범위를 제한하고 종료한다.
'''
    text+='\n## 실제 학습 완료 trace\n\n```json\n'+json.dumps(runtime_summary(data),ensure_ascii=False,indent=2)+'\n```\n'
    C.save(C.DOC/'REPRODUCE.md',text)


if __name__=='__main__':
    main()
