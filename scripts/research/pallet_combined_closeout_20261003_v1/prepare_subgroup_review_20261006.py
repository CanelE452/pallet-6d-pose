"""Render a focused subgroup review from already computed immutable metrics."""
from pathlib import Path
import hashlib
import json
import time

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1'
OUT=BASE/'final_review'
GRADES={'clean':'없음','moderate':'중간','severe':'어려움'}
BACKBONES={'yolo':'YOLO','dope':'DOPE','resnet18':'ResNet-18'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def number(value,digits=3):
    if value is None:return 'x'
    return f'{value:.{digits}f}'
def pair(before,after,key,scale=1.):
    return number(before[key]*scale if before[key] is not None else None)+' → '+number(after[key]*scale if after[key] is not None else None)

def main():
    started=time.perf_counter()
    inputs={name:BASE/'static'/name for name in ('STATIC_REAGGREGATION.json','SQUARE_REAGGREGATION.json')}
    hashes={name:sha(path) for name,path in inputs.items()}
    data=json.loads(inputs['STATIC_REAGGREGATION.json'].read_text())
    square=json.loads(inputs['SQUARE_REAGGREGATION.json'].read_text())
    rows=[]
    def add(panel,backbone,group,before,after,bp,ap,source,*,mode=None,seed=None):
        row=dict(panel=panel,backbone=backbone,backbone_label=BACKBONES[backbone],group=group,
            group_label=GRADES.get(group,group),mode=mode,seed=seed,frames=before['frames'],
            corner_median=pair(before,after,'corner_median_px'),corner_P90=pair(before,after,'corner_P90_px'),
            PCK10_percent=pair(before,after,'PCK10_fraction',100),
            translation_median_cm=pair(before,after,'translation_median_cm'),
            rotation_median_deg=pair(before,after,'rotation_median_deg'),
            reference_corners=before['full_supervised_corners'],
            observed_corners_before=before['observed_corners'],observed_corners_after=after['observed_corners'],
            pose_before=before['pose_available_frames'],pose_after=after['pose_available_frames'],
            source=str(source.relative_to(ROOT)),source_sha256=sha(source),
            before_json_pointer=bp,after_json_pointer=ap,
            raw_before=before,raw_after=after)
        rows.append(row)
    source=inputs['STATIC_REAGGREGATION.json']
    for backbone in BACKBONES:
        b=data['backbones'][backbone]['Base']['result'];a=data['backbones'][backbone]['N3']['result']
        for group in GRADES:
            add('three_backbone_grades',backbone,group,b[group],a[group],
                f'/backbones/{backbone}/Base/result/{group}',f'/backbones/{backbone}/N3/result/{group}',source)
        for group,before in b['material_x_severity'].items():
            add('material_by_grade',backbone,group,before,a['material_x_severity'][group],
                f'/backbones/{backbone}/Base/result/material_x_severity/{group}',
                f'/backbones/{backbone}/N3/result/material_x_severity/{group}',source)
        for seed_name,seed_result in data['backbones'][backbone]['N3']['per_seed'].items():
            add('seed_all_319',backbone,'all',b['all'],seed_result['all'],
                f'/backbones/{backbone}/Base/result/all',f'/backbones/{backbone}/N3/per_seed/{seed_name}/all',
                source,seed=seed_name)
    source=inputs['SQUARE_REAGGREGATION.json']
    for mode,mode_result in square['modes'].items():
        for backbone in BACKBONES:
            b=mode_result['backbones'][backbone]['Base']['result'];a=mode_result['backbones'][backbone]['N3']['result']
            for group in GRADES:
                add('square119_by_grade',backbone,group,b[group],a[group],
                    f'/modes/{mode}/backbones/{backbone}/Base/result/{group}',
                    f'/modes/{mode}/backbones/{backbone}/N3/result/{group}',source,mode=mode)
    checks=0
    for row in rows:
        tree=json.loads((ROOT/row['source']).read_text())
        for pointer,raw in ((row['before_json_pointer'],row['raw_before']),(row['after_json_pointer'],row['raw_after'])):
            v=tree
            for part in pointer.lstrip('/').split('/'):
                v=v[part.replace('~1','/').replace('~0','~')]
            assert v==raw
            checks+=1
    for name,path in inputs.items():assert sha(path)==hashes[name]
    OUT.mkdir(exist_ok=True)
    payload=dict(schema='focused_subgroup_review_20261006_v1',
        status='DERIVED_PRESENTATION_OF_ALREADY_COMPUTED_RESULTS',
        scope='Human-input grades; criterion NOT_CONFIRMED. Descriptive review, not new experiments.',
        definitions=dict(pair='Base statistic → N3 statistic; not median(frame-wise differences)',
            mean='N3 headline is mean of each seed statistic; seeds are shown separately, not selected',
            denominators='conditional median/P90, full-reference PCK; pose failures remain in image denominator',
            square='119 images, one session, 602/600 modes kept separate, independent T/R x',
            paper_placement='three_backbone_grades is inserted into existing sup:occlusion_results; other focused panels are attached review detail'),
        rows=rows,input_hashes=hashes,actual_json_pointer_checks=checks,
        metric_recomputation=0,new_training=0,new_inference=0,optimizer_updates=0,
        pdf_generation=0,tex_compile=0,push=0,wall_seconds=time.perf_counter()-started)
    (OUT/'SUBGROUP_REVIEW.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    sections=['# 최종 하위집단 확인자료\n',
        '이미 계산된 숫자만 보기 쉽게 연결했습니다. 등급은 **사람 입력 등급(판정 기준 미확인)**이며 새로운 외부 가림 승인으로 바꾸지 않았습니다. '
        '모든 화살표는 **Base 통계 → N3 통계**입니다. N3 전체 행은 세 seed 통계의 평균이고 seed 선택이나 앙상블을 하지 않았습니다.\n']
    headings={'three_backbone_grades':'원고 보충표: 세 기반 × 세 등급',
        'material_by_grade':'별도 확인: 재질 × 등급', 'seed_all_319':'별도 확인: 전체319장의 각 seed',
        'square119_by_grade':'별도 확인: 정사각형119의 등급,602/600 분리'}
    for panel,title in headings.items():
        sections+=['\n## '+title+'\n',
            '\n|기반|집단 / 모드 / seed|영상|코너 중앙값(px)|코너 P90(px)|PCK10(%)|T 중앙값(cm)|R 중앙값(도)|참조 / 유효코너|자세산출|\n',
            '|---|---|---:|---|---|---|---|---|---|---|\n']
        for row in (r for r in rows if r['panel']==panel):
            label=' / '.join(str(v) for v in (row['group_label'],row['mode'],row['seed']) if v is not None)
            observed=f'{row["reference_corners"]} / {row["observed_corners_before"]}→{row["observed_corners_after"]}'
            pose='x' if row['pose_before'] is None else f'{row["pose_before"]}→{row["pose_after"]} / {row["frames"]}'
            sections.append('|'+ '|'.join(map(str,[row['backbone_label'],label,row['frames'],row['corner_median'],
                row['corner_P90'],row['PCK10_percent'],row['translation_median_cm'],row['rotation_median_deg'],observed,pose]))+'|\n')
    sections+=['\n각 행의 원시 JSON 위치·SHA-256·분모는 [SUBGROUP_REVIEW.json](SUBGROUP_REVIEW.json)에 연결됩니다. '
        '재질×등급·각seed·정사각형등급의 모든 행을 PDF 표로 중복삽입하지 않았습니다. '
        '조건부 오차와 결측을 유지한 전체 분모를 구분하고, 악화도 그대로 표시합니다.\n']
    (OUT/'SUBGROUP_REVIEW_KO.md').write_text(''.join(sections))
    print(json.dumps(dict(rows=len(rows),pointer_checks=checks,wall_seconds=payload['wall_seconds'],output=str(OUT))))

if __name__=='__main__':main()
