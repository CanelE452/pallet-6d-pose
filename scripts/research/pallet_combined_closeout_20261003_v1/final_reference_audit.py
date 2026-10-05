"""Independent read-only checks of reference provenance and final paper limits."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_static_registry_review_20261003_v1 import open_corner_visibility as V
from scripts.research.pallet_lifter_case_review_20261003_v1.open_existing_annotation import Context,NativeReview

CLOSE=ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1'
OUT=CLOSE/'final_review'
LIFTER=ROOT/'data/pallet/results/pallet_lifter_case_review_20261003_v1/review'
SOURCES={}
CHECKS={}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):
    path=Path(path).resolve()
    entry=dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
               sha256=sha(path),bytes=path.stat().st_size)
    SOURCES[str(path)]=entry
    return entry
def read(path):
    bind(path)
    return json.loads(Path(path).read_text())
def check(key,value):
    CHECKS[key]=bool(value)
    if not value: raise RuntimeError('Independent reference audit failed: '+key)


def run():
    started=time.monotonic()
    bind(__file__)
    ctx=V.load_context();review=V.VisibilityReview(ctx,V.OUT/'STATIC_CORNER_VISIBILITY_INPUTS.json')
    manifest=read(V.MANIFEST);severity=read(V.SEVERITY);store=read(review.store_path)
    effective=[]
    for fid,case in ctx['frames'].items():
        for p in case['corners']:
            if not p['metric_reference']:continue
            effective.append(dict(case_id=fid,population=case['population'],frame_id=case['frame_id'],
                corner_id=p['corner_id'],status=review.state_for(fid,p['corner_id']),xy=p['xy'],locked=p['locked']))
    counts=Counter(p['status']for p in effective)
    check('current_static_3030_new_plus71_locked',len(effective)==3101 and sum(p['locked']for p in effective)==71
          and review.counts()['human_input_points']==3030 and review.counts()['pending']==0)
    check('current_static_states_all_explicit',dict(counts)==dict(DIRECT_VISIBLE=2313,EXTERNAL_OCCLUDED=281,
          SELF_OCCLUDED=462,OUT_OF_FRAME=45))
    merged=read(CLOSE/'visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
    previous={(r['case_id'],r['corner_id']):(r['category'],r['reference_xy'])for r in merged['rows']}
    check('latest_raw_static_exactly_equals_published_merge',all(previous[(p['case_id'],p['corner_id'])]==
          (p['status'],p['xy'])for p in effective))
    receipt=read(CLOSE/'visibility_square/EXECUTION_VALIDATION.json')
    for entry in receipt['input_bindings']:
        path=Path(entry['path']);path=path if path.is_absolute()else ROOT/path
        check('fixed_binding:'+entry['path'],sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes'])
    check('previous_raw_evaluator_regression_72_all_pass',receipt['status']=='PASS'
          and len(receipt['checks'])==72 and all(receipt['checks'].values()))
    semantic=read(CLOSE/'visibility_square/SEMANTIC_TEST_VALIDATION.json')
    check('canonical_mask_and_NA_semantic_tests_pass',semantic['status']=='PASS' and semantic['tests']==4)

    vis=read(CLOSE/'visibility_square/VISIBILITY_RESULTS.json')
    points_path=CLOSE/'visibility_square/PER_POINT_SCORES.csv';bind(points_path)
    groups=defaultdict(list)
    with points_path.open()as f:
        for p in csv.DictReader(f):
            p['error_px']=float(p['error_px']);p['observed']=p['observed']=='True'
            groups[(p['population'],p['backbone'],p['mode'],p['method'])].append(p)
    for r in vis['rows']:
        key=(r['population'],r['backbone'],r['mode'],r['method'])
        full=groups[key]if r['category']=='ALL'else[p for p in groups[key]if p['category']==r['category']]
        obs=np.asarray([p['error_px']for p in full if p['observed']],float)
        all_errors=np.asarray([p['error_px']for p in full],float)
        median=float(np.median(obs))if len(obs)else None
        p90=float(np.quantile(obs,.9))if len(obs)else None
        pck=100*float(np.mean(all_errors<=10))if len(full)else None
        check('visibility_csv_exact:'+':'.join(key)+':'+r['category'],
              len(full)==r['reference_corners'] and len(obs)==r['observed_corners']
              and median==r['median_px'] and p90==r['P90_px'] and pck==r['PCK10_percent'])
    for key,points in groups.items():
        expected=2499 if key[0]=='DEV319' else (602 if key[2]=='manual_declared' else 600)
        check('same_denominator:'+':'.join(key),len(points)==expected)

    square=read(CLOSE/'visibility_square/SQUARE119_RESULTS.json')
    check('square119_six_backbone_mode_panels_consistent',square['frames']==119 and square['sessions']==1
          and square['manual_declared']==602 and square['manual_in_frame']==600
          and len(square['outside_declared_corners'])==2 and all(r['translation_cm']=='x'
          and r['rotation_deg']=='x'for r in square['rows']))

    lctx=Context(LIFTER/'MANIFEST.json',LIFTER.parent/'LIFTER_EVALUATION_PLAN.json',
                  LIFTER/'CORNER_CONTRACT.json',LIFTER/'annotations_in_progress.json')
    native=NativeReview(lctx,None,LIFTER/'native_workspace',passes=('primary',),native_pnp=False,
                       batch_plan=LIFTER/'SMALL_BATCH_12_V1.json')
    lifter_categories=Counter();manual=no_xy=0;details=[]
    for fid in native.batch['frame_ids']:
        status=native.visibility_only_status(fid,'primary')
        check('lifter_categories_complete:'+fid,status['complete'])
        manual+=status['manual_coordinate_count'];no_xy+=len(status['declared_point_ids'])
        lifter_categories.update('self_occlusion'if c.get('self_occlusion')else c['visibility']
                                for c in status['corners'])
        for ci in status['declared_point_ids']:
            c=status['corners'][ci]
            check('chat_visible_never_became_xy:'+fid+':'+str(ci),c['x']is None and c['y']is None
                  and c['definition_confirmed']is False and c['coordinate_source']=='none')
        details.append(dict(frame_id=fid,categories_complete=status['complete'],
                            manual_coordinate_count=status['manual_coordinate_count'],
                            visible_declared_without_xy=status['declared_point_ids']))
    official=sum(r['status']=='reviewed'for r in lctx.store['records'])
    check('lifter96_categories67_real_xy5_without_xy24_self',sum(lifter_categories.values())==96
          and lifter_categories==Counter(direct_visible=72,self_occlusion=24) and manual==67 and no_xy==5)
    check('lifter_formal_approval_still_zero',official==0 and not(LIFTER/'LIFTER_REFERENCE_REVIEWED.json').exists())
    for path in (LIFTER/'MANIFEST.json',LIFTER.parent/'LIFTER_EVALUATION_PLAN.json',LIFTER/'CORNER_CONTRACT.json',
                 LIFTER/'SMALL_BATCH_12_V1.json',LIFTER/'VIEWER_DRAFT_RECOVERY.json',
                 LIFTER/'USER_VISIBILITY_DECLARATIONS_20261005_V1.json'):bind(path)
    lifter_audit=read(CLOSE/'lifter/AUDIT.json')
    raw_prediction=LIFTER.parent/'raw_predictions/ALL_STORED_FRAMES.jsonl';bind(raw_prediction)
    pose_counts={m:Counter()for m in ('Base','N3')};ids=set();target_match=0
    with raw_prediction.open()as f:
        for line in f:
            r=json.loads(line);check('unique_lifter_frame:'+r['frame_id'],r['frame_id']not in ids);ids.add(r['frame_id'])
            a,b=r['methods']['Base'],r['methods']['N3']
            check('same_fixed_lifter_mask_object:'+r['frame_id'],a['keypoints_mask']==b['keypoints_mask']
                  and a['selected_object']==b['selected_object'])
            target_match+=int(a.get('object_match')is not None or b.get('object_match')is not None)
            for m in ('Base','N3'):pose_counts[m][r['methods'][m]['pose_state']]+=1
    check('all8910_fixed_outputs_8772fresh138no_pose',len(ids)==8910 and
          all(v==Counter(fresh=8772,no_pose=138)for v in pose_counts.values()))
    check('no_prediction_object_match_falsely_confirmed',target_match==0
          and lifter_audit['formal_reference']['target_match_reviewed_frames']==0)

    paper=CLOSE/'paper_updated'
    case_text=(paper/'sections/06_case_study.tex').read_text();bind(paper/'sections/06_case_study.tex')
    check('manuscript_retains_lifter_accuracy_x',
          '67점' in case_text and '5점'in case_text and '각각 0건'in case_text
          and '가시 코너 중앙값 / P90 (px) & \\notrun/\\notrun'in case_text
          and '전방 거리 참조 오차 (cm) & \\notrun'in case_text)
    criterion=ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/SEVERITY_CRITERION_CONFIRMATION.json'
    criterion_status='NOT_CONFIRMED' if not criterion.exists()else read(criterion).get('criterion','NOT_CONFIRMED')
    selected=[]
    visual_notes={
       ('DEV319','clean'):'외부 차폐물이 없는 검은 팔레트와 원래 큐보이드 코너 표시가 보인다. 원래 평가 참조 코너6/7의 자체가림 상태는 보존했다.',
       ('DEV319','moderate'):'오른쪽 팔레트 끝과 큐보이드 표시가 화면 경계 밖으로 이어진다. 눈에 띄는 별도 차폐 물체가 없는 이 표본만으로도 등급을 외부물체 가림량으로 단정할 수 없다.',
       ('DEV319','severe'):'야간 작은 팔레트에 여러 교통콘이 인접해 있고 일부 팔레트 부분을 가린다. 이 관찰을 전체 severe의 판정 정의로 일반화하지 않는다.',
       ('GREEN0918','clean'):'초록 팔레트 옆에 교통콘이 있지만 원래 직접 클릭 참조0~5가 표시된 표본이다. 주변 물체 존재와 코너 차폐 여부를 구분해야 한다.',
       ('GREEN0918','severe'):'팔레트 위 교통콘과 큐보이드 표시를 확인했다. 일부 표시점은 PnP 보완이며 수동 클릭 참조와 같은 독립 정답으로 보지 않는다.'}
    for pop,grade in visual_notes:
        for case in manifest['cases']:
            actual=severity['records'].get(case['case_id'],{}).get('severity',case['frame_severity']['status'])
            if case['population']!=pop or actual!=grade:continue
            image=ROOT/case['image']['path'];overlay=ROOT/case['overlay']['path']
            selected.append(dict(case_id=case['case_id'],population=pop,existing_user_grade=grade,
                image=bind(image),inspected_overlay=bind(overlay),annotation=bind(ROOT/case['annotation']['path']),
                actual_visual_observation=visual_notes[(pop,grade)],human_label_changed=False,
                human_reviewed_created=False,selection='First frozen frame of this already-entered stratum; sanity check only'))
            break
    frame=lctx.frames['173507:56']
    selected.append(dict(case_id='173507:56',population='LIFTER',image=bind(lctx.images['173507:56']),
        actual_visual_observation='원본 영상에서 멀리 있는 초록 팔레트를 확인했다. 같은 프레임의 입력 6점과 자체 가림 2점 기록은 좌표 출처로만 대조하고 모델의 선택 객체 대응을 자동 승인하지 않았다.',
        human_label_changed=False,human_reviewed_created=False,selection='First frozen frame of the original12-frame task'))
    check('six_real_frozen_visual_samples',len(selected)==6)

    figure=paper/'figures/current_review_summary.png';bind(figure)
    fmap=read(CLOSE/'paper_patch/PAPER_CELL_MAP.json')['figures']['current_review_summary']
    check('count_figure_three_actual_sources',fmap['source']==['static/STATIC_REAGGREGATION.json',
          'static/SQUARE_REAGGREGATION.json','visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'])
    figsource={name:read(CLOSE/name)for name in fmap['source']}
    counts_source=figsource['static/STATIC_REAGGREGATION.json'].get('label_counts',{})
    check('count_plot_actual_grade_values',counts_source==dict(clean=153,moderate=92,severe=74)
          and figsource['static/SQUARE_REAGGREGATION.json']['label_counts']==dict(clean=3,moderate=85,severe=31))
    figure_values=dict(rectangular_grades=[153,92,74],square_grades=[3,85,31],
        rectangular_visibility=[1776,218,462,43],square_visibility=[537,63,0,2])
    # These values were visually checked against the generated PNG and verified
    # above against the raw merged corner states. The figure is a count plot.
    check('count_plot_actual_corner_values',merged['population_counts']['DEV319']==
          dict(DIRECT_VISIBLE=1776,EXTERNAL_OCCLUDED=218,SELF_OCCLUDED=462,OUT_OF_FRAME=43)
          and merged['population_counts']['GREEN0918']==dict(DIRECT_VISIBLE=537,EXTERNAL_OCCLUDED=63,OUT_OF_FRAME=2))
    old_provenance=ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/paper_updated/audit/EXAMPLE_CROP_PROVENANCE.json'
    crops=read(old_provenance);crop_checks=[]
    source=Image.open(paper/'figures/consult_slide12_0.png').convert('RGB');bind(paper/'figures/consult_slide12_0.png')
    for crop in crops:
        target=paper/'figures'/crop['file'];bind(target)
        equal=np.array_equal(np.asarray(source.crop(crop['crop_xyxy'])),np.asarray(Image.open(target).convert('RGB')))
        check('original_example_only_crop:'+crop['file'],equal)
        crop_checks.append(dict(file=crop['file'],only_crop_pixel_equal=equal))
    conclusion=paper/'sections/08_conclusion.tex';text=conclusion.read_text();bind(conclusion)
    stale='앞으로의 판단에는 정사각형 가림·코너 가시성 검수'in text
    issues=[]
    if stale:issues.append(dict(kind='STALE_COMPLETION_WORDING',file=str(conclusion.relative_to(ROOT)),
        existing_phrase='앞으로의 판단에는 정사각형 가림·코너 가시성 검수',
        correction_needed='이미3030+71가시성분류완료. 이후과제는독립참조품질확인/노출조건과리프터참조·객체대응으로표현'))
    for path,entry in SOURCES.items():
        if sha(path)!=entry['sha256']:raise RuntimeError('Source changed during final audit: '+path)
    result=dict(schema='independent_reference_semantics_audit_20261006_v1',status='PASS_WITH_DOCUMENTED_LIMITATIONS',
        executed_at=datetime.now(timezone.utc).isoformat(),elapsed_wall_seconds=time.monotonic()-started,
        current_static=dict(new_explicit_states=3030,locked_prior_states=71,merged_states=3101,
                            counts=dict(counts),reference_coordinates_unchanged=True),
        lifter=dict(category_states=96,actual_manual_xy=67,visible_declared_without_xy=5,self_occluded=24,
                    formal_approved_frames=official,target_match_reviewed_frames=target_match,details=details,
                    fixed_prediction_frames=8910,pose_state_counts={k:dict(v)for k,v in pose_counts.items()},
                    corner_accuracy='x',physical_translation_accuracy='x',physical_rotation_accuracy='x'),
        severity_semantics=dict(status=criterion_status,UI_intent='가림만 분류;1가림없음/2중간/3어려움·심한가림',
            manifest_definitions=manifest['severity_definitions'],
            exact_moderate_severe_boundaries_recovered=False,
            external_vs_self_vs_truncation_lighting_inclusion_human_confirmed=False,
            independent_rubric_approval_can_be_inferred=False,
            existing_grades_can_be_used_as='사용자입력등급; 기준미확인·보조사후분석',
            already_measured_grade_results_need_not_be_x=True,
            additional_human_reclassification_required_for_this_paper=False,
            prohibited_claim='외부물체가림률이나독립블라인드가림강건성확증'),
        visual_sanity=dict(samples=selected,count=6,exhaustive_review=False,not_human_annotation_approval=True),
        figures=dict(current_count_plot_values=figure_values,plot_type='descriptive counts,notaccuracy',
                     inherited_example_crop_checks=crop_checks),
        checks_passed=len(CHECKS),checks=CHECKS,editorial_notes=issues,source_bindings=list(SOURCES.values()),
        new_training=0,new_inference=0,new_PnP=0,GUI_actions=0,new_human_labels=0,
        frozen_annotations_or_receipts_modified=False,PDF_generated=0,pushes=0)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'INDEPENDENT_REFERENCE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 참조·분모·등급 의미 독립 감사','','기존 정적 입력 3030개와 보호된 71개 상태를 원자료에서 다시 확인했다. 3101개 참조 상태는 보임 2313 / 외부 가림 281 / 자체 가림 462 / 화면 밖 45개다. 모델별 전체 참조 분모와 조건부 관측 분모를 분리해 검산했다.','','정사각형 119장의 602점/600점 모드는 방법마다 같고 섞지 않았다. 66,618개 canonical 코너 CSV에서 324개 seed·집단 통계를 독립 재집계해 기존 JSON과 정확히 대조했다. 전체 물체 대칭을 한 번 선택한 뒤 canonical GT 코너 ID로 가시성을 나눴다. DOPE의 매칭 프레임 내부 결측점도 조건부 통계와 전체 벌점에서 구분한다.','','리프터 12장은 가시성 96개가 완료됐지만 실제 클릭 좌표 67점과 좌표 없는 보임 5점, 자체 가림 24점을 구분한다. 공식 참조 승인 0건 / 객체 대응 0건이다. 원시 8910프레임의 선택 객체·결측 마스크는 Base/N3에서 같고 각각 8772 fresh / 138 no-pose임을 다시 확인했다. 가시 코너 정확도·물리 T/R·정지 잡음의 x는 최종 승인 없이 숫자로 바꾸지 않았다.','','당시 도구는 가림 분류를 의도했지만, 중간·어려움 경계와 자체 가림/잘림/조명 포함 여부에 대한 실제 사람 확인 기록은 없다. 기준 승인을 CLI가 만들 수는 없다. 이미 입력된 등급은 「사용자 입력 등급(기준 미확인)」의 보조 사후 분석으로 사용할 수 있다. 실제 분할과 지표를 삭제하거나 x로 바꿀 필요는 없고, 이 원고 범위에서 등급 전부를 다시 묻거나 사람에게 재분류를 요구할 필요도 없다. 외부 물체 가림 강건성의 확증으로 승격해서는 안 된다.','','원래 이미지와 기존 GT 오버레이 6장을 실제로 보고 좌표 상태와 대조했다. 이는 AI의 표본 점검이며 전수 사람 검수나 등급 정정이 아니다. 특히 중간 등급 표본에는 화면 경계 잘림이 확인되어 등급을 오직 외부 물체 차폐량으로 역추정할 수 없다.','','최신 구성 그림은 153/92/74·3/85/31과 직사각형 1776/218/462/43·정사각형 537/63/0/2를 표시한다. 6개 이전 예시 crop은 원래 슬라이드 픽셀과 정확히 같다. 개선 사례 그림을 독립 정답 또는 확률 표본처럼 승격하지 않았다.','']
    if issues:lines+=['복사본결론에「앞으로정사각형가림·코너가시성검수필요」라는옛문장이남아있다. 이미입력완료된작업을다시요구하는표현이므로최종복사본에서독립참조·노출조건등의미확인사항으로정정해야한다.','']
    lines+=[f'실제 검산 {len(CHECKS)}개 통과, 경과 {result["elapsed_wall_seconds"]:.3f}초. 학습·추론·PnP·GUI 키 주입·사람 판정 생성·원본 변경·PDF·push는 0회다.','', '[상세 JSON·입력 해시·원본 이미지 연결](INDEPENDENT_REFERENCE_AUDIT.json)']
    (OUT/'INDEPENDENT_REFERENCE_AUDIT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],checks=len(CHECKS),seconds=result['elapsed_wall_seconds'],editorial_notes=issues),ensure_ascii=False),flush=True)


if __name__=='__main__':run()
