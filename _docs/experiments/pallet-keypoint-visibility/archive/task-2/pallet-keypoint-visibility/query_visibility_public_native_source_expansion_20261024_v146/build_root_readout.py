"""Summarize completed source data work, without implying classifier success."""
from pathlib import Path
from html.parser import HTMLParser
import datetime, hashlib, html, json

V = Path(__file__).resolve().parent
def bind(p): return {"path":str(p), "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p): return json.loads(p.read_text())
def write(p, a):
    with p.open("x") as f:
        json.dump(a, f, ensure_ascii=False, indent=2, allow_nan=False);f.write("\n")

adjudication = V / "ROOT_SOURCE64_WEAK_LABEL_ADJUDICATION_COMPLETE.json"
independent = V / "independent_adjudication_review/REVIEW.json"
assert read(adjudication)["PASS"] and read(independent)["PASS"]
boards = [V / "source_report/BOARD_01_VISIBLE_EXTERNAL_PAIRS.png",
          V / "source_report/BOARD_02_ENTITY_INPUT_AND_ROLE_CASES.png"]
a = read(adjudication)
results = dict(completed=True, goal_status="active", fully_solved=False,
    created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    previous_goal_turn_classification="PROGRESS: actual public64 initializer/GT references/factual crop rendering completed EXIT0.",
    current_phase_outcome_KO="공개 실사 원64장/512입력을 보존한 실제 선택 개체·하부 역할의 부분 약한 라벨84개를 완성했습니다. 새 가림 분류기 학습·성능 개선 결과는 아닙니다.",
    original_frames=64, original_slots=512, candidate_detections=236,
    original_input_valid=463, original_input_OOB=49, original_noquery=0,
    provided_T_K_direct_projection_camera_PASS=63, provided_T_K_direct_projection_camera_UNRESOLVED=1,
    labels=a["counts"], by_split=a["by_split"], forwardable_known_by_split=a["forwardable_known_by_split"],
    known_teacher_D_but_original_input_U=[402,434], conditional_D_but_primary_entity_U=459,
    all8_slots_and_original_firstargmax_q_L_RGB96_valid_hashes_unchanged=True,
    independent_annotation_and_all512_crop_hash_review_PASS=True,
    root_views=dict(all64_scenes_in8_overviews=True, individual_sheets=a["root_individual_all8crop_sheets_viewed"], both_diagnostic_boards_actually_viewed=True),
    source_weak_not_human_gold=True, source_native_scope="Partial LOWER2/3 only, original whole-assembly XZ footprint and bottom plane; sourceUP=-OBJY provisional; scanAABB outlier unchanged.",
    external_E_diversity="All14 external examples are native3 behind separate foreground cardboard; shared assembly and occluder type limit generalization.",
    upper_rear384_labels_U=True, source_original_folds_unchanged=True, public_source_development_exposed_not_blind=True,
    full_goal_redefined_to_lower2_3=False, current223_V112_access=0, new_classifier_fit_or_forward=0, PnP_q_L_pose_optimization=0,
    latest_real_benchmark_unchanged="V143: C16 visible1137/1147 external35/90; C32 visible1070/1147 external38/90. No new private evaluation or model adoption in this phase.",
    next_KO="원 q8 minmax context와 원 q/L crop을 사용하는 동일 frozen DINO1928 특징을 모든512입력에서 추출합니다. 이후 별도 고정 source 실험으로 공개 TRAIN43의 효과를 검증하며 HALL은 최적화에 넣지 않습니다.",
    bindings=dict(initializer=bind(V / "actual_public_initializer_v1/PREDICTION_SEAL.json"),
                  projection=bind(V / "provided_pose_references_v1/PROJECTION_COMPLETE.json"),
                  contacts=bind(V / "saved_source64_contact_v1/CONTACT_RENDER_COMPLETE.json"),
                  adjudication=bind(adjudication), labels=a["labels"], independent_review=bind(independent),
                  board_QA=bind(V / "source_report/QA_COMPLETE.json"), boards=[bind(p) for p in boards]))
write(V / "ROOT_RESULTS_KO.json", results)
rows = [("전체",70,14,428,82), ("LAB · 학습 분할",35,8,213,43), ("HALL · 검증 분할",35,6,215,39)]
table = "".join(f"<tr><td>{n}</td><td>{d}</td><td>{e}</td><td>{u}</td><td>{usable}</td></tr>" for n,d,e,u,usable in rows)
body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V146 공개 실사 원 키포인트 보임·가림 라벨 검토</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1250px;margin:28px auto;padding:0 20px;line-height:1.7;color:#17243b;background:#f7f9fc}}h1{{font-size:29px}}.box{{background:white;padding:22px;border:1px solid #dae1ed;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #dce3ed;text-align:left}}img{{width:100%;height:auto}}a{{color:#1359a6}}.note{{color:#754321}}</style>
<h1>공개 실사 64장 · 초기 키포인트 512개 검토 완료</h1>
<div class="box"><strong>새 가림 모델의 성공 결과가 아니라, 실제 선택 개체와 키포인트 역할이 맞는 공개 실사 학습 자료를 정리한 단계입니다.</strong><p>원64장과512입력·모든236검출 후보를 보존했습니다. 첫 선택과 q/L을 바꾸지 않았으며 원 입력은463개 유효·49개 영상 밖입니다. 제공 T/K는 직접 투영했고63장 하부 역할 검사 통과,1장은 미정으로 남겼습니다.</p></div>
<div class="box"><h2>라벨과 입력 유효성</h2><table><thead><tr><th>분할</th><th>보임 D</th><th>외부 가림 E</th><th>미정 U</th><th>입력 유효 D/E</th></tr></thead><tbody>{table}</tbody></table>
<p>D 두 개는 대상 역할이 보이지만 초기 q가 영상 밖입니다. 의미 D를 E로 바꾸지 않고, 입력 보류를 별도로 보존했습니다. HALL의 애매한 개체 대응 한 점은 조건부 D 제안을 남기고 주 라벨 U로 판정했습니다.</p>
<p class="note">하부2/3에 한정된 구조적 assistant weak 라벨입니다. 사람 gold·정밀 물리 꼭짓점·전체8점 정합 인증이 아닙니다. 상부/뒤384점은 U이며 U를 negative로 학습하지 않습니다. 현재 목표의 전체 키포인트 범위는 유지합니다.</p></div>
<div class="box"><h2>같은 포즈 계열의 보임·외부 가림</h2><img src="source_report/BOARD_01_VISIBLE_EXTERNAL_PAIRS.png" alt="원 crop을 이용한 공개 실사 보임·외부 가림 대조"><p>LAB은 학습 분할, HALL은 검증 분할을 그대로 유지합니다. 외부 가림14개가 모두 종이상자에 가린 왼쪽 하부라서 다양성이 제한됩니다. 개발에 노출된 공개 자료군이며 새 blind 검증은 아닙니다.</p></div>
<div class="box"><h2>개체 오류·좌표 오류·가림을 구분</h2><img src="source_report/BOARD_02_ENTITY_INPUT_AND_ROLE_CASES.png" alt="다른 개체 선택과 OOB·큰 좌표 오차·보류를 구분한 원 crop"><p>다른 개체를 선택한 경우 뒤 팔레트의 라벨을 옮기지 않았습니다. 대상 역할이 보이는지와 초기 q 위치가 맞는지는 따로 기록했습니다. q 오차 문턱으로 좋은 사례만 남기지 않았습니다.</p></div>
<div class="box"><h2>남은 작업</h2><p>모든512입력의 원 q8 minmax context와 원 q/L crop에서 동일 frozen DINO1928 특징을 추출합니다. 이어 공개 TRAIN43의 효과를 기존 합성 native8과 함께 고정 비교할 준비를 합니다. 이 단계의 가림 분류기 학습·추론, 비공개 평가 데이터 접근, PnP·좌표 수정은 모두0회입니다.</p>
<p><a href="saved_source64_contact_v1/INDEX.html">전체64장과 원 crop</a> · <a href="ROOT_ADJUDICATED_SOURCE_NATIVE512_WEAK_LABELS.jsonl">512행 최종 약한 라벨</a> · <a href="ROOT_SOURCE64_WEAK_LABEL_ADJUDICATION_COMPLETE.json">판정 근거</a> · <a href="independent_adjudication_review/REVIEW.json">독립 전수 검산</a> · <a href="ROOT_RESULTS_KO.json">결과 JSON</a></p></div></html>'''
report = V / "ROOT_READOUT_KO.html"
with report.open("x") as f: f.write(body)

class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[];self.images=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=="a" and "href" in a:self.paths.append(a["href"])
        if tag=="img":self.paths.append(a["src"]);self.images.append(a["src"])
links=Links();links.feed(body)
assert len(links.images)==2 and all((V / p).is_file() for p in links.paths)
qa=dict(completed=True,PASS=True,report=bind(report),local_references=len(links.paths),all_local_references_exist=True,
        both_embedded_PNGs_root_actually_viewed=True,source_label_review=bind(independent),browser_workaround_used=False)
write(V / "ROOT_READOUT_QA.json",qa)
write(V / "ROOT_PHASE_RESULT.json",dict(completed=True,goal_status="active",fully_solved=False,
    phase="V146_PUBLIC64_NATIVE_SOURCE_WEAK_LABEL_EXPANSION",outcome="PROGRESS: completed actual source inputs and partial native-role weak supervision; no new classifier success claim.",
    results=bind(V / "ROOT_RESULTS_KO.json"),readout=bind(report),QA=bind(V / "ROOT_READOUT_QA.json"),
    adjudication=bind(adjudication),independent_review=bind(independent),new_classifier_fit=0,new_private_evaluation=0,PnP_or_q_L_correction=0))
print(json.dumps(bind(V / "ROOT_PHASE_RESULT.json")))
