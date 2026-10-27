"""Prepare reviewed public artifacts; no publishing/model/raw input access.

manuscript-inputs emits a replacement Chapter4 result module for a NEW document.
package runs only after final numerical, figure and rendered-manuscript gates.
Original18page manuscript/source and historical MAIN results stay immutable.
"""
from pathlib import Path
import argparse
import json
import shutil
import publication_common as P


def numerical_ready(io):
    path=P.OUT/'preflight/FINAL_NUMERICAL_QA.json';qa=json.loads(path.read_text())
    P.require(qa['passed'] is True and qa['complete'] is True and qa['numerical_inputs_path']==str(P.NUMERICAL_INPUTS)
        and P.sha(P.NUMERICAL_INPUTS)==qa['numerical_inputs_sha256'],'Final numerical QA input contract differs')
    P.require(qa['code_sha256']==P.sha(P.OUT/'code/final_numerical_qa.py') and qa['helper_sha256']==P.sha(Path(P.__file__).resolve()),'Numerical QA code changed')
    for p,digest in qa['source_sha256'].items():P.require(p in io.files and P.sha(io.path(p))==digest or p==str(P.NUMERICAL_INPUTS) and P.sha(p)==digest,'Numerical QA source changed')
    return path,qa


def fmt(value,scale=1):
    value=P.number(value);return 'NA' if value is None else f'{value*scale:.2f}'


def tables(io):
    points=io.csv(io.one('table_ablation_2d'));pose=io.csv(io.one('table_supplemental_pose'));ratio=io.csv(io.one('table_ratio'))
    ci=io.csv(io.one('ci_ablation'));ratio_ci=io.csv(io.one('ci_ratio'));teacher=io.csv(io.one('table_future_teacher'))
    P.require(len(points)==len(pose)==51 and len(ratio)==15,'Final table counts differ')
    return points,pose,ratio,ci,ratio_ci,teacher


def manuscript_inputs(io):
    io.gates('numerical');qa_path,_=numerical_ready(io);points,pose,ratio,ci,ratio_ci,teacher=tables(io)
    block=lambda caption,headers,rows:{'type':'table','caption':caption,'headers':headers,'rows':rows}
    sections=[]
    sections.append({'anchor':'4.1','title':'평가 자료와 지표','blocks':[
        {'type':'p','text':'주평가는 직사각형 플라스틱128장, 목재125장, 녹색119장의372장과2891개 참조점으로 구성하였다. 목재는 기존1280×720 45장과 추가640×480 80장을 합쳤다. PCK는 원본 영상에서5/10/20 px 이내인 점의 비율이며 미검출은 실패로 처리하였다.'},
        block('평가 모집단과 참조점',['종류','영상 수','참조점','수동 보조점'],[['PLASTIC',128,985,'없음'],['WOOD',125,963,330],['GREEN',119,943,600]])]})
    sections.append({'anchor':'4.2','title':'종류별 자기학습 및 합성 추가학습 비교','blocks':[
        block('모든 고정 평가조합의2D 기준점 성능',['종류','조건','차수','PCK5 %','PCK10 %','PCK20 %','Coverage %','관측 오차 평균±SD px','오차 n'],
            [[r['kind'],r['condition'],r['round'],*[fmt(r[k],100) for k in ('PCK5','PCK10','PCK20','coverage')],fmt(r['conditional_pixel_error_mean'])+' ± '+fmt(r['conditional_pixel_error_std']),r['conditional_pixel_error_n']] for r in points]),
        {'type':'p','text':'SD는 관측된 점 오차의 모집단 표준편차(ddof=0)이다. 단일 학습seed의 서로 다른seed 간 변동을 뜻하지 않는다. 학습 실패 모델은NA이며0점으로 대체하지 않았다.'}]})
    sections.append({'anchor':'4.3','title':'필터 구성 요소 제거 실험','blocks':[
        block('full에서 제거 조건을 뺀PCK10 차이',['종류','차수','제거 조건','차이 %p','95% CI %p','판정'],
            [[r['kind'],r['round'],r['comparator'],fmt(r['point_difference_full_minus_ablation'],100),fmt(r['ci95_low'],100)+' ~ '+fmt(r['ci95_high'],100),r['decision']] for r in ci if r['metric']=='PCK10']),
        {'type':'p','text':'모델 간 차이는 같은 프레임에 짝지은 점 수 비율의 차이로 계산하였다.10000회 프레임bootstrap과seed20261026을 사용하였다. 목재의4개 녹화 단위CI는 소수 군집의 보조 분석이며 프레임CI를 주결과로 유지하였다. CI의0 포함은 차이를 확인하지 못한 결과이며 동등성 증명이 아니다.'}]})
    sections.append({'anchor':'4.4','title':'2D 참조 재구성에 따른 보조 자세 지표','blocks':[
        block('재구성 참조 보조 자세 오차',['종류','조건','차수','PnP 성공/전체','T 평균±SD cm','T n','R indexed 평균±SD °','R symmetry 평균±SD °','R n'],
            [[r['kind'],r['condition'],r['round'],str(r['n_model_PnP_success'])+'/'+str(r['n_DEV']),fmt(r['translation_error_cm_mean'])+' ± '+fmt(r['translation_error_cm_std']),r['translation_error_m_n'],
            fmt(r['rotation_indexed_deg_mean'])+' ± '+fmt(r['rotation_indexed_deg_std']),fmt(r['rotation_symmetry_deg_mean'])+' ± '+fmt(r['rotation_symmetry_deg_std']),r['rotation_indexed_deg_n']] for r in pose]),
        {'type':'p','text':'이 값은 같은 인덱스2D 기준점과 등록 치수 및K에서 재구성한참조에 대한 보조 비교이며 독립 실측6D 정답이 아니다. 목재125장 전체는PnP 성공률의 분모이고 오차는 양쪽 추정 성공 표본에 조건부이다. 서로 다른 치수 가설의 회전오차는NA다. 추가80의카메라 quality는UNKNOWN이며 일부 기준점은PnP 투영 및 외삽 출처를 포함한다.'}]})
    sections.append({'anchor':'4.5','title':'실제·합성 데이터 개수 비율과 후속teacher 진단','blocks':[
        block('R2 데이터 개수 비율 실험',['종류','조건','실제 고유 수','합성 고유 수','합성 비율 %','PCK10 %','관측 오차 평균±SD px','오차 n'],
            [[r['kind'],r['condition'],r['real_unique'],r['synthetic_unique'],r['synthetic_percent'],fmt(r['PCK10'],100),fmt(r['conditional_pixel_error_mean'])+' ± '+fmt(r['conditional_pixel_error_std']),r['conditional_pixel_error_n']] for r in ratio]),
        block('비율 실험PCK10 차이',['종류','비교 모델','기준 모델','차이 %p','95% CI %p','판정'],
            [[r['kind'],r['candidate'],r['comparator'],fmt(r['point_difference'],100),fmt(r['ci95_low'],100)+' ~ '+fmt(r['ci95_high'],100),r['decision']] for r in ratio_ci if r['metric']=='PCK10']),
        {'type':'p','text':'목재의pilot CI가0을 포함하더라도 결과를불확실로 보고하고 사용자의 최신 명시 요청에 따라15개 비율 조건을 모두 수행하였다. 결과를 보고 모델·문턱·부분집합을 선택하지 않았다. 고정된R2 학습 의사 레이블의정밀도가조건별로달랐다고해석하지않는다.'},
        block('학습 완료R2를 가상R3 teacher로 사용한 보조 진단',['종류','조건','패널','승인 영상','채점 가능한 선택점','Precision10','선택 참조 유지율'],
            [[r['kind'],r['condition'],r['panel'],r['accepted_frames'],r['scorable_accepted_points'],fmt(r['precision10'],100),fmt(r['selected_reference_coverage'],100)] for r in teacher]),
        {'type':'p','text':'이 진단은 학습을 마친R2 모델로 가상의 다음차수 필터를 적용한 값이며 실제R3 학습은 수행하지 않았다. 고정R1 오차 구간에 따른잔존오차는 기술 통계이며 confirmation bias의 인과적증명이 아니다.'}]})
    sections.append({'anchor':'4.6','title':'정성 비교 사례','blocks':[
        {'type':'p','text':'사전 고정한6개 사례의 Input/R0/R1/R2 비교를 제시한다. 이 사례들은 설명용이며 모집단 평균이나 독립6D 정확도의 근거를 대체하지 않는다. 원선정 규칙과 기존 그림출처를보존하고 통합평가 결과를보고재선정하지않았다.'},
        {'type':'qualitative_asset_slots','source_role':'qualitative_figure','expected_existing_cases':6}]})
    target=P.OUT/'manuscript_revision/SECTION4_CONTENT.json'
    P.write(target,{'schema':'new_thesis_chapter4_results_module_v1','integration_status':'ROOT integration/render review pending; original7chapter18page draft unchanged',
        'sections':sections,'primary_frames':372,'primary_reference_points':2891,'source_sha256':io.bindings,'numerical_QA_sha256':P.sha(qa_path),
        'citation_order_preserved':['DOPE2018','Tobin2017','NoisyStudent2020','FixMatch2020','EPnP2009','RANSAC1981']})
    io.unchanged()


def package(io):
    io.gates('package');qa_path,_=numerical_ready(io);points,pose,ratio,ci,ratio_ci,teacher=tables(io)
    root=P.OUT/'publication/merged372_v1';payload=root/'payload'
    P.require(not root.exists(),'Existing publication package is immutable; select a new reviewed publication version')
    payload.mkdir(parents=True);copied={}
    for source,item in io.files.items():
        if item.get('destination') is None:continue
        path=io.path(source);dest=P.safe_relative(item['destination']);target=payload/dest
        P.require(not target.exists(),'Duplicate destination')
        if path.suffix=='.json':P.no_coordinates(io.read(path))
        elif path.suffix=='.csv':io.csv(path)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        P.require(P.sha(target)==item['sha256'],'Copied artifact SHA differs');copied[source]=item['sha256']
    if not (payload/'README.md').exists():
        lines=['# DOPE 자기학습 통합372장 평가','',
            '주평가는 PLASTIC128/985점, WOOD125/963점, GREEN119/943점으로 총372장/2891점이다. '
            '목재45장(1280×720)과80장(640×480)을 합쳤다. nativepx PCK의 정규화 허용오차가 달라질 수 있으며 '
            '원45장60.12%와 새125장 수치의 차이는 모집단 변화이므로 모델 개선으로 해석하지 않는다. 모델들은 같은125장에 비교했다.','',
            '종류별17개씩51개 기본/대조/필터 평가조합과15개 비율 모델 및 가상R3 teacher 진단을 포함한다. '
            '오차의 SD는 모집단 기준(ddof=0)이고 오차 n은 조건부 표본 수다. '
            '보조6D는2D 기준점·등록치수·K에서 재구성한 참조와 비교하며 독립실측6D 정답이 아니다. '
            '가설이 다른 회전은NA, 학습 실패는NA이며0으로 대체하지 않았다.','',
            '| 종류 | 조건 | R | PCK10 % | 관측 오차 평균 ± SD px | 오차 n |','|---|---|---|---|---|---|']
        for r in points:lines.append('| '+' | '.join(map(str,[r['kind'],r['condition'],r['round'],fmt(r['PCK10'],100),fmt(r['conditional_pixel_error_mean'])+' ± '+fmt(r['conditional_pixel_error_std']),r['conditional_pixel_error_n']]))+' |')
        lines+=['','모든 원단위 평균·분산·SD·중앙값·사분위수·분모와 CI는 함께 제공한 CSV를 따른다. '
            '의사 레이블 수는 성능 지표가 아니며 승인 수·정밀도·고정참조 유지율을 구분한다. '
            '가상R3 teacher 지표는 실제R2 학습 의사 레이블 정밀도와 다르다. '
            '목재 녹화는 학습과 공유되며 녹화독립TEST가 아니다.4개 녹화 CI는 보조이고 프레임 CI가 주결과다. '
            '불확실한CI를 동등성/효과없음으로 해석하지 않는다. 단일 학습seed이며 다중비교 보정은 없다.','']
        (payload/'README.md').write_text('\n'.join(lines))
    inv=P.inventory(payload)
    P.require(any(x.endswith('.docx') for x in inv) and any(x.endswith('.pdf') for x in inv),'Reviewed new manuscript DOCX/PDF required')
    P.require(sum(x.endswith('.png') for x in inv)>=6,'Six preserved qualitative examples plus reviewed plots required')
    P.write(root/'PAYLOAD_INVENTORY.json',inv)
    io.unchanged();P.write(root/'PREPARE_QA.json',dict(passed=True,complete=True,payload_files=len(inv),primary_frames=372,primary_reference_points=2891,
        ablation_groups=51,supplemental_pose_groups=51,ratio_models=15,publication_inputs_sha256=io.manifest_sha,
        final_numerical_QA_sha256=P.sha(qa_path),payload_inventory_sha256=P.sha(root/'PAYLOAD_INVENTORY.json'),
        source_sha256=io.bindings,copied_source_sha256=copied,no_GT_xy_private_scores_predictions_checkpoints_or_pseudo_labels=True,
        code_sha256=P.sha(Path(__file__).resolve()),helper_sha256=P.sha(Path(P.__file__).resolve()),publishing_authorized=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('manuscript-inputs','package'));args=parser.parse_args()
    io=P.Inputs(P.NUMERICAL_INPUTS if args.stage=='manuscript-inputs' else P.INPUTS)
    if args.stage=='manuscript-inputs':manuscript_inputs(io)
    else:package(io)


if __name__=='__main__':main()
