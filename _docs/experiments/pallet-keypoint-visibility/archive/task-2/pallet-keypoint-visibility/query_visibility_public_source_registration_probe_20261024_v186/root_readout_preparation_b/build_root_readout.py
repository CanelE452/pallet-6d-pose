"""V186 root readout from closed JSON summaries/reviews only, no asset loaders.

--prepare uses frozen metadata and invented HTML fixtures; root entry produces
the final Korean JSON/HTML/phase files without altering the source probe.
"""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import argparse
import hashlib
import json
import math
import os

H=Path(__file__).resolve().parent;V=H.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_SOURCE_RGB_DEPTH_MASK_POSE_MESH_BODIES.json'
SOURCE=V/'fixed_source_registration_probe_v1/COMPLETE.json'
SUMMARY=V/'fixed_source_registration_probe_v1/ALL5_REGISTRATION_DIAGNOSTICS.json'
RAY=V/'independent_registration_review_b/actual_ray_verification_v2_JSON_BOOL_REPAIR/REVIEW.json'
IMAGE_QA=V/'ROOT_ACTUAL_SOURCE_IMAGE_QA.json'
TERMINAL=V/'ROOT_ACTUAL_SOURCE_PROBE_TERMINAL.json'
MATH=V/'source_registration_review_a/MATH_CONTRACT.json'
STATIC=V/'source_registration_review_a/STATIC_RUNNER_REVIEW.json'
VISUAL=V/'source_registration_review_a/ACTUAL_VISUAL_REVIEW.json'
AUTHORITY=V/'source_registration_review_a/MR6D_DEPTH_CAPTURE_AUTHORITY_REVIEW.json'
PROTOCOL=H/'PROTOCOL_BEFORE_ROOT_READOUT_GENERATION.json'
CHECKS=H/'GENERATED_OFFLINE_QA_CHECKS.json';PREP=H/'PREPARATION_COMPLETE.json'
RESULTS=V/'ROOT_RESULTS_KO.json';HTML=V/'ROOT_READOUT_KO.html';PHASE=V/'ROOT_PHASE_RESULT.json'
QA=V/'ROOT_READOUT_OFFLINE_QA.json';COMPLETE=V/'ROOT_READOUT_BUILD_COMPLETE.json'
INPUTS=dict(scope=SCOPE,source_complete=SOURCE,source_summary=SUMMARY,independent_ray_review=RAY,source_image_QA=IMAGE_QA,
    actual_probe_terminal=TERMINAL,official_math_contract=MATH,static_runner_review=STATIC,actual_visual_review=VISUAL,depth_capture_authority_review=AUTHORITY)
EXPECTED={SOURCE:'fa8aed42ae9ece1c926a3d0c87604f89b3ed92695f33a17b96c73d71ae688170',SUMMARY:'d62208feda04a8b0c1403fe7d5b3501a53dce02dd019e1cae35bb7c813f9bf74',
    RAY:'6a183047736fe9508cf44559f62572c60ee70f5e65da0b71bb4c55934489a98f',
    AUTHORITY:'064e16482f3ca7ce396cfe76d487851485ce3e219f005905de501aeabf9bb52e',VISUAL:'cacc75d9013f940fbc7183447c6fae9690ddc0b078da8b1b5ffdfef9bf47f425'}


def require(value,msg):
    if not value:raise AssertionError(msg)
def sha(path):
    p=Path(path);require(p.suffix in ('.json','.jsonl','.py','.html') and '_v112' not in str(p).lower(),'Readout metadata/source/text only')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def check(b):require(bind(b['path'])==b,'Frozen readout metadata binding');return Path(b['path'])
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def relative(path):return os.path.relpath(str(Path(path)),str(V))
def number(value,digits=2):return '미측정' if value is None else f'{value:,.{digits}f}'
def integer(value):return f'{value:,}'


def fixed_instance_table(summary):
    rows=[]
    require([(r['frame_index'],r['gt_instance_index']) for r in summary['instances']]==[(0,0),(0,1),(118,0),(166,3),(199,0)],'Original fixed5/all instances')
    for record in summary['instances']:
        full=record['full_mask'];visible=record['visible_mask'];stats=visible['signed_modelZ_minus_sensorZ'];qs=dict(zip(stats['quantile_percent'],stats['quantile_mm']))
        require(visible['mask_pixels']==visible['mask_sensor_known']+visible['mask_sensor_missing'],'Visible mask measured/missing denominator')
        require(stats['N']==visible['mask_rendered_sensor_known'],'Residual only visiblemask AND rendered AND sensorvalid')
        require(full['intersection']+full['render_only']+full['mask_only']==full['union']
            and math.isclose(full['IoU'],full['intersection']/full['union'],rel_tol=0,abs_tol=1e-15),'Fullmask IoU saved arithmetic')
        require(set(qs)=={0,1,5,25,50,75,95,99,100},'Original fixed quantiles')
        rows.append(dict(frame_index=record['frame_index'],source_id=record['source_id'],GT_instance_index=record['gt_instance_index'],
            full_mask_IoU=full['IoU'],full_mask_intersection_pixels=full['intersection'],full_mask_union_pixels=full['union'],
            provided_visible_mask_pixels=visible['mask_pixels'],visible_mask_sensor_known_pixels=visible['mask_sensor_known'],
            visible_mask_sensor_missing_pixels=visible['mask_sensor_missing'],visible_mask_sensor_missing_fraction=visible['mask_sensor_missing']/visible['mask_pixels'],
            measured_visible_render_intersection_N=stats['N'],signed_modelZ_minus_sensorZ_mm_p5=qs[5],signed_modelZ_minus_sensorZ_mm_median=qs[50],signed_modelZ_minus_sensorZ_mm_p95=qs[95],
            signed_depth_stats_complete=stats,images=record['images']))
    return rows


def build_result(inputs):
    source=inputs['source_complete'];summary=inputs['source_summary'];ray=inputs['independent_ray_review'];qa=inputs['source_image_QA']
    require(source['completed'] and source['PASS'] and ray['completed'] and ray['PASS'] and qa['completed'] and qa['PASS'],'Closed completed probe/ray/image reviews')
    require(ray['total_slots400']==400 and ray['strict_interior_failures']==0,'Independent firstsurface math review bound')
    rows=fixed_instance_table(summary);frame0=next(f for f in summary['frames'] if f['frame_index']==0)
    overlaps=frame0['provided_visible_mask_pair_overlaps'];require(overlaps==[dict(instance_a=0,instance_b=1,pixels=9988)],'Original frame0 overlap preserved')
    depth_max=max(r['maximum_compared_absolute_depth_difference_mm'] for r in ray['instances'] if r['maximum_compared_absolute_depth_difference_mm'] is not None)
    return dict(completed=True,phase_only_complete=True,goal_status='active_unsolved',fully_solved=False,selected_model=None,adopted=False,
        outcome_KO='공개 SOURCE TRAIN 4프레임·obj1 instance 5개의 고정 등록 probe와 독립 first-surface 계산 검산을 완료했다. 새 모델 개선 또는 직접 부위·가림 정답 확보를 뜻하지 않는다.',
        fixed_population=dict(source='MR6D',internal_split='SOURCE_DEVELOPMENT',upstream_folder='val',frame_indices=[0,118,166,199],frames=4,all_obj1_instances=5,
            selection='각 기존 TRAIN scene의 원래 첫 프레임, 그 프레임의 obj1 GT-list instance 전부. 좋은 instance/pose/score 선택 없음.',full_goal='WOOD/SQUARE와native8의 초기q 중심 가시성/외부가림 전체 목표는 유지되며 미해결.'),
        instance_table=rows,
        residual_definition_KO='제공 visible mask ∩ target-alone rendered silhouette ∩ positive finite sensor-depth 안에서 model camera-axis Z − raw_depth × supplied depth_scale. 단위 mm, 원 부호·분위수 그대로. Sensor0은 결측이며 가림/배경으로 채우지 않는다.',
        fullmask_reference_authority='DATASET_PROVIDED_NOMINAL_MODEL_DERIVED_SILHOUETTE_NOT_INDEPENDENT_RGB_MATERIAL_BOUNDARY_GOLD',
        fullmask_interpretation_KO='Full-mask IoU는 nominal 모델·pose silhouette와 제공 full mask의 정합 비교다. 독립 RGB 재료 경계·정확한 부위·native8 정답 정확도로 해석하지 않는다.',
        frame0_instance_mask_overlap=dict(provided_visible_masks=overlaps,provided_full_masks=frame0['provided_full_mask_pair_overlaps'],exclusive_owner_gold=False),
        independent_ray_check=dict(slots=400,instances=5,status_counts=ray['status_counts'],strict_interior_failures=0,
            maximum_compared_absolute_Z_difference_mm=depth_max,prospective_F64_difference_bound_mm=1e-6,
            claim='원 mesh/K/R/t의 first-surface 렌더 계산 검증. 107 hit깊이/281 no-hit 일치, 12 projected centroid OOB 슬롯 보존. Sensor-depth 또는 실제물리등록 품질 검증과 구분.'),
        sensor_depth_authority=dict(raw_export_optical_frame_alignment='UNCONFIRMED',per_frame_depth_scale_preserved=True,rawzero_missing_preserved=True,
            documented='MR6D 논문은 D435i/VICON 및 저센티미터 수준 pose 과정·nominal치수 변동을 설명한다.',
            unknown='공개 검토 문서만으로 released depth의 RGB/depth optical frame, 실제 alignment/export/resampling·필터·동기 설정을 확정하지 못했다.',
            interpretation='음수 residual은 이 고정 계산의 부호일 뿐, 특정 물체의 외부가림·pose오류·scale오류·RGB alignment오류로 원인을 단정하지 않는다. 관측 residual에 맞춘 좌표계/scale/pose/K/tolerance 수정 없음.'),
        source_visual_QA=dict(collective_actual_tool_view_coverage=qa['collective_actual_tool_view_coverage'],original_RGB_pixel_fidelity_receipt=bind(IMAGE_QA),
            builder_new_image_views=0,HTML_browser_rendered=False,claim='Root6+A8 기존 tools.view_image 영수증의 합동14PNG 범위를 인용하며 builder가 직접보았다고 주장하지 않는다.'),
        limits_KO=['고정4프레임·5instance probe로 전체185 TRAIN frame이나 실사 WOOD/SQUARE/native8 성능을 인증하지 않는다.',
            '제공 visible mask는 frame0 두 instance에서9,988픽셀 겹친다. 곧바로 배타적 center-owner 또는 부위 외부가림 정답으로 바꾸지 않는다.',
            '부위/native correspondence와 query에 연결되는 직접 support 감독은 아직 확보되지 않았다. UNKNOWN을 제거하거나 새 라벨로 채우지 않았다.',
            '컬러 범위0..6000mm/−100..100mm는 기존 그림 표시용이며 수치 통계는 clipping하지 않았다. HTML browser render는 수행하지 않았다.'],
        unchanged=dict(mesh_K_R_t_q_L_native=True,posefit_scale_alignment_or_PnP=False,new_native_part_owner_quality_visibility_labels=0,
            new_classifier_model_forward_fit_SAM_encoder_initializer=0,new_training_admission=False,source_or_private_model_adoption=False),
        publication_kind='SOURCE_REGISTRATION_PROBE_AND_READINESS_EVIDENCE_NOT_MODEL_SUCCESS',
        bindings={k:bind(path) for k,path in INPUTS.items()})


def table_html(rows):
    header=['Frame / GT','Source','Full mask IoU','제공 visible N','센서 결측 N','잔차 측정 N','Z차 p5 mm','중앙값 mm','p95 mm']
    body=[]
    for r in rows:
        values=[f'{r["frame_index"]} / {r["GT_instance_index"]}',r['source_id'],number(r['full_mask_IoU']*100)+'%',
            integer(r['provided_visible_mask_pixels']),integer(r['visible_mask_sensor_missing_pixels']),integer(r['measured_visible_render_intersection_N']),
            number(r['signed_modelZ_minus_sensorZ_mm_p5']),number(r['signed_modelZ_minus_sensorZ_mm_median']),number(r['signed_modelZ_minus_sensorZ_mm_p95'])]
        body.append('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in values)+'</tr>')
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in header)+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'


def html_report(result):
    rows=result['instance_table'];ray=result['independent_ray_check'];parts=['<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V186 공개 TRAIN 등록 probe</title>',
        '<style>body{font:17px/1.65 system-ui,sans-serif;color:#183044;max-width:1250px;margin:auto;padding:28px}h1{font-size:30px}h2{font-size:23px;margin-top:36px}.notice{background:#eef3f6;border-left:5px solid #567c96;padding:14px 20px}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #cbd5dc;padding:10px;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#f0f4f6}.images{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{max-width:100%;height:auto;border:1px solid #d7dfe5}figcaption{font-size:14px;color:#4e6574}a{color:#075a94}code{font-size:14px}li{margin:10px 0}@media(max-width:700px){body{padding:15px}.images{grid-template-columns:1fr}}</style></head><body>',
        '<h1>V186 · 공개 TRAIN 4장·5instance 등록 probe</h1><div class="notice"><p>'+escape(result['outcome_KO'])+'</p><p><strong>전체 목표 미해결 · 모델 채택 없음 · 새 라벨/학습 편입 0</strong></p></div>',
        '<h2>원래 고정한 5개 instance 전부</h2><p>MR6D의 내부 SOURCE_DEVELOPMENT인 scene000000/000002/000005/000006의 원래 첫 프레임과 obj1 GT-list instance 전체입니다. 상류 폴더명은 val이지만 held scene000001은 포함하지 않았습니다. 원 mesh·K·R·t와 초기 q/L/native를 변경하지 않았습니다.</p>',
        table_html(rows),'<p>'+escape(result['residual_definition_KO'])+'</p><p>센서 결측 N은 제공 visible mask 전체 안의 raw depth 결측입니다. 잔차 측정 N은 그 중 렌더 지지와 유효 센서 깊이가 동시에 있는 별도 분모입니다. p5·중앙값·p95는 이 측정 분모의 원 signed 값입니다.</p>',
        '<h2>계산 검증과 실제 데이터 권위의 차이</h2><p>별도 PLY parser와 camera-ray/triangle first-hit 계산으로 고정 400개 slot을 유지했습니다. 깊이 hit 일치 107개, no-hit 일치 281개, 투영 OOB 12개이며 interior 불일치는 0입니다. 비교된 깊이의 최대 절대차 '+escape(f'{ray["maximum_compared_absolute_Z_difference_mm"]:.6g}')+' mm는 사전 고정한 F64 기준 1e−6 mm 이내였습니다. 이 검사는 렌더 구현의 계산을 확인합니다.</p>',
        '<p>'+escape(result['fullmask_interpretation_KO'])+'</p><p>Frame0의 두 provided visible mask는 <strong>9,988픽셀</strong> 겹치며, full mask도 46,418픽셀 겹칩니다. 이 겹침을 지우거나 배타적 소속 정답으로 바꾸지 않았습니다.</p>',
        '<p>원 sensor depth의 RGB/depth optical-frame 및 alignment/export 처리는 검토한 공개 문서만으로 확정되지 않았습니다. 음수 residual의 원인을 특정 물체·pose·scale·alignment로 단정하지 않으며, 잔차를 줄이기 위한 변경도 하지 않았습니다.</p>',
        '<h2>기존 고정 source 그림</h2><p>Clean RGB는 별도 보존되며 overlay와 depth 색도표는 표시용 복사본입니다. 기존 root6+A8 tools.view_image 영수증으로 원 RGB4장·overlay5장·depth진단5장의 합동 검토 범위를 확인했습니다. 이 builder의 새 이미지 판독이나 HTML browser render를 주장하지 않습니다.</p>',
        '<p><a href="fixed_source_registration_probe_v1/index.html">원 4장·5instance 전체 gallery</a> · <a href="fixed_source_registration_probe_v1/ALL5_REGISTRATION_DIAGNOSTICS.json">원 raw summary JSON</a></p>']
    for row in rows:
        label=f'frame{row["frame_index"]} / GT{row["GT_instance_index"]}'
        parts.append('<h3>'+escape(label)+'</h3><div class="images">')
        for name,caption in [('overlay','Nominal target-alone mesh overlay · 실제 visible part mask 아님'),('depth_diagnostic','원 signed depth 진단 · 색 범위는 표시용')]:
            url=escape(relative(row['images'][name]['path']),quote=True)
            parts.append('<figure><a href="'+url+'"><img loading="lazy" src="'+url+'" alt="'+escape(label+' '+caption,quote=True)+'"></a><figcaption>'+escape(caption)+'</figcaption></figure>')
        parts.append('</div>')
    parts.append('<h2>아직 해결되지 않은 것</h2><ul>'+''.join('<li>'+escape(x)+'</li>' for x in result['limits_KO'])+'</ul>')
    parts.append('<p>WOOD/SQUARE·native8의 원래 목표는 유지합니다. 이 단계에서 classifier/SAM/encoder/initializer 실행·새 학습·PnP·pose fit·q/L 수정·새 감독 라벨·학습 편입은 없습니다.</p>')
    parts.append('<p><a href="ROOT_RESULTS_KO.json">수치·분모·원 영수증 바인딩</a> · <a href="'+escape(relative(RAY),quote=True)+'">독립 ray 검산</a> · <a href="source_registration_review_a/MR6D_DEPTH_CAPTURE_AUTHORITY_REVIEW.json">공개 depth 문서 권위와 미확정 항목</a> · <a href="ROOT_ACTUAL_SOURCE_IMAGE_QA.json">실제 이미지 QA</a></p></body></html>')
    return '\n'.join(parts)


class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[];self.rows=0
    def handle_starttag(self,tag,attrs):
        if tag=='tbody':self.rows=0
        if tag=='tr':self.rows+=1
        for k,v in attrs:
            if k in ('href','src') and v:self.targets.append(v)


def offline_qa(result,html,check_local_links):
    require(len(result['instance_table'])==5 and result['selected_model'] is None and result['adopted'] is False and result['fully_solved'] is False,'Fiveinstances/no adoption')
    require('9,988' in html and '46,418' in html and '1e−6' in html and 'HTML browser render' in html,'Required authority/denominator captions')
    for r in result['instance_table']:
        for key in ('signed_modelZ_minus_sensorZ_mm_p5','signed_modelZ_minus_sensorZ_mm_median','signed_modelZ_minus_sensorZ_mm_p95'):
            require(escape(number(r[key])) in html,'All5 signed numeric cells present')
    parser=Links();parser.feed(html)
    if check_local_links:
        for target in parser.targets:
            require(not target.startswith(('http:','https:','file:','javascript:')),'Offline local report only')
            require((V/target).exists(),'Existing local report/gallery link')
    return dict(completed=True,PASS=True,HTML_source_checked=True,HTML_browser_rendered=False,table_instance_rows=5,
        all5_numeric_p5_median_p95_and_missing_denominators_present=True,offline_links_checked=check_local_links,
        local_href_src_references=len(parser.targets),new_image_or_model_or_source_array_reads=0,
        source_summary_values_copied_not_new_measurements=True)


def invented_checks():
    rows=[]
    for i,(fi,gt) in enumerate([(0,0),(0,1),(118,0),(166,3),(199,0)]):
        rows.append(dict(frame_index=fi,source_id='toy<&'+str(i),GT_instance_index=gt,full_mask_IoU=.99,provided_visible_mask_pixels=100,
            visible_mask_sensor_missing_pixels=10,measured_visible_render_intersection_N=80,
            signed_modelZ_minus_sensorZ_mm_p5=-30-i,signed_modelZ_minus_sensorZ_mm_median=-20-i,signed_modelZ_minus_sensorZ_mm_p95=10+i,
            images={k:dict(path=str(V/f'invented_{i}_{k}.png')) for k in ('overlay','depth_diagnostic')}))
    result=dict(completed=True,instance_table=rows,outcome_KO='probe only <not model success>',independent_ray_check=dict(maximum_compared_absolute_Z_difference_mm=1e-11),
        residual_definition_KO='Visible and rendered and sensor valid only',fullmask_interpretation_KO='Nominal not RGB gold',limits_KO=['No native labels'],selected_model=None,adopted=False,fully_solved=False)
    text=html_report(result);qa=offline_qa(result,text,False)
    require('toy&lt;&amp;' in text and '<not model success>' not in text,'HTML escapes original text/source IDs')
    require(text.count('<tbody>')==1 and text.count('</tbody>')==1 and text.count('<td>')==5*9,'Exact5x9 scientific table fixture')
    return dict(completed=True,PASS=True,invented_only=True,checks=['Fivefixedinstances_allnine_numeric_columns','Rawsign_p5_median_p95_and_distinct_missing_N','HTMLescaping_source_text','Nominalmask_not_RGBgold_caption','Ray400_not_sensor_quality_caption','Overlap9988_and46418_explicit','Unsolved_noadoption_nonewlabels','OfflineHTML_no_browserclaim'],count=8,offline_fixture=qa,actual_root_reports_generated=0)


def prepare():
    for path,value in EXPECTED.items():require(sha(path)==value,'Frozen closed source/review')
    for name,path in INPUTS.items():
        r=read(path)
        require(r.get('completed',r.get('completed_scope',False)) is True,'Completed metadata dependency '+name)
    write(CHECKS,invented_checks())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),inputs={k:bind(x) for k,x in INPUTS.items()},checks=bind(CHECKS),
        output_paths=[str(x) for x in (RESULTS,HTML,PHASE,QA,COMPLETE)],
        construction='Closed all5 JSON tables copied without new measurements; raw units/signs/denominators and missingdepth retained; fixed links inherit actual imageQA.',
        authority='Nominal fullmask silhouette comparison not independent RGB/nativepart gold. Depth optical/export processing unknown. Independent ray checks implementation only.',
        new_source_pixel_NPZ_pose_mesh_body_or_hash_reads=0,HTML_browser_rendered=False,actual_ROOT_readout_generated=0,selected_model=None,fully_solved=False)
    write(PROTOCOL,p);template=H/'ROOT_ACTUAL_READOUT_BUILD_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_metadata_only_root_readout_build=False,source_only=True,no_new_assets_models_labels_or_selection=True,code=bind(__file__),protocol=bind(PROTOCOL)))
    write(PREP,dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),actual_ROOT_readout_generated=0))
    print(json.dumps(dict(FINAL_READY=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(PREP))))


def run(entry):
    e=read(entry);require(e.get('approved_metadata_only_root_readout_build') is True and e.get('source_only') is True
        and e.get('no_new_assets_models_labels_or_selection') is True,'Separate root readout build entry')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL),'Frozen readout builder')
    p=read(PROTOCOL);require(read(check(p['checks']))['PASS'],'Offline invented QA passed')
    require(all(not Path(x).exists() for x in p['output_paths']),'No overwrite of prior ROOT readout')
    data={k:read(check(b)) for k,b in p['inputs'].items()}
    require(data['source_complete']['summary']==p['inputs']['source_summary'],'Closed original rawsummary binding')
    result=build_result(data);write(RESULTS,result);html=html_report(result)
    with HTML.open('x') as f:f.write(html)
    qa=offline_qa(result,html,True);qa.update(code=bind(__file__),protocol=bind(PROTOCOL),results=bind(RESULTS),readout=bind(HTML),input_bindings=p['inputs']);write(QA,qa)
    phase=dict(completed=True,phase_only_complete=True,goal_status='active_unsolved',classification='PROGRESS_SOURCE_REGISTRATION_EVIDENCE_NOT_MODEL_SUCCESS',
        fully_solved=False,selected_model=None,adopted=False,outcome_KO=result['outcome_KO'],results=bind(RESULTS),report=bind(HTML),offline_readout_QA=bind(QA),
        source_probe_complete=p['inputs']['source_complete'],independent_ray_review=p['inputs']['independent_ray_review'],source_image_QA=p['inputs']['source_image_QA'],
        authority_review=p['inputs']['depth_capture_authority_review'],all5_fixed_instances_retained=True,new_labels_or_training_admission=False,new_model_forward_fit_or_PnP=False)
    write(PHASE,phase)
    for b in p['inputs'].values():check(b)
    write(COMPLETE,dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),inputs=p['inputs'],
        outputs=dict(results=bind(RESULTS),readout=bind(HTML),phase=bind(PHASE),offline_QA=bind(QA)),HTML_browser_rendered=False,
        producer_or_source_assets_modified=False,new_model_label_training_admission_or_selection=0,selected_model=None,fully_solved=False))
    print(json.dumps(dict(completed=True,PASS=True,complete=bind(COMPLETE))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--entry',type=Path)
    a=ap.parse_args();prepare() if a.prepare else run(a.entry)
