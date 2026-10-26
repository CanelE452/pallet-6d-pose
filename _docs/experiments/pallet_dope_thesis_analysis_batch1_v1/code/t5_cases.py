"""T5: 기존 DEV292 점수로 고정 사례 9개를 선정하고 원본 전체 영상에 표시한다.

원본 annotation, 보호 FINAL 영상, 모델을 열지 않는다. 새 추론도 하지 않는다.
prepare는 파생 캐시만 읽고, create에서 보호 모집단 guard 통과 후 선정 RGB만 연다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import io
import math
import statistics
from pathlib import Path

import common as C

FONT_PATH='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
FONT_BOLD='/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
KOREAN={'PLASTIC':'직사각형 플라스틱','WOOD':'소형 목재','GREEN':'녹색 플라스틱'}
CATEGORIES=('max','median','min')
CATEGORY_KO={'max':'차이 최대','median':'중앙값에 가장 가까움','min':'차이 최소'}
COLORS=('#e6194b','#0072b2','#f58231','#009e73','#911eb4','#56b4e9','#a6611a','#e6ab02')
MODEL_ROLES=('R0','control_R2','ST_R2')
MODEL_KO={'R0':'R0','control_R2':'대조군 R2','ST_R2':'LOO+flip R2'}
MARKERS={'GT':'circle','R0':'plus','control_R2':'square','ST_R2':'triangle'}
EXPECTED={'PLASTIC':(128,985),'WOOD':(45,346),'GREEN':(119,943)}
SELECTION=C.OUT/'cache/T5_SELECTION.json'
DELTA_CSV=C.OUT/'tables/T5_ALL_FRAME_DELTAS.csv'
SELECTED_CSV=C.OUT/'tables/T5_SELECTED_CASES.csv'
SELECTED_MD=C.OUT/'tables/T5_SELECTED_CASES.md'


def require(condition,message):
    if not condition:raise RuntimeError('T5 중단: '+message)


def ensure_new(paths):
    for p in paths:require(not p.exists(),'기존 산출물 덮어쓰기 금지: '+str(p))


def finite_pair(v):
    return isinstance(v,list) and len(v)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in v)


def as_fraction(value):
    return Fraction(value['numerator'],value['denominator'])


def fraction_json(value):
    return dict(numerator=value.numerator,denominator=value.denominator,value=float(value))


def paths():
    return [C.FLIP/n for n in ('MANIFEST.json','SCORES.json','PREDICTIONS.json','SUMMARY.json')]+[
        C.OUT/'FIXED_PLAN.json',C.OUT/'preflight/INITIAL_GATES.json',C.OUT/'cache/METADATA_RECORDINGS.json']


def load_inputs():
    manifest=C.read(C.FLIP/'MANIFEST.json')
    scores=C.read(C.FLIP/'SCORES.json');predictions=C.read(C.FLIP/'PREDICTIONS.json')
    summary=C.read(C.FLIP/'SUMMARY.json');plan=C.read(C.OUT/'FIXED_PLAN.json')
    gate=C.read(C.OUT/'preflight/INITIAL_GATES.json')
    metadata=C.read(C.OUT/'cache/METADATA_RECORDINGS.json')
    require(gate['initial_file_count_gate_pass'] and gate['summary_count_reproduction_exact'],'초기 전제 gate 미통과')
    require(plan['bootstrap_seed']==C.SEED==20261026,'사전 고정 seed가 달라짐')
    require('T5_exception' in plan['user_clarification'],'T5 고정 사례 선정 예외 허용 기록 없음')
    require(plan['T5_even_median_rule']=='choose frame nearest population median(deltaPCK10), tie by ascending frame ID; max/min also tie by ascending ID; freeze before computation','고정 중앙값 규칙이 달라짐')
    require(predictions['complete'],'예측 캐시 미완료')
    require(scores['manifest_sha256']==predictions['manifest_sha256']==C.sha(C.FLIP/'MANIFEST.json'),'manifest SHA 연결 불일치')
    require(scores['predictions_sha256']==summary['predictions_sha256']==C.sha(C.FLIP/'PREDICTIONS.json'),'예측 SHA 연결 불일치')
    require(summary['scores_sha256']==C.sha(C.FLIP/'SCORES.json'),'점수 SHA 연결 불일치')
    records=manifest['records'];ids=[r['id'] for r in records]
    require(len(ids)==len(set(ids))==292,'DEV292 고유 ID 불일치')
    require(all(r.get('population_role')=='DEV' for r in records),'DEV 이외 역할이 주 평가에 포함됨')
    protected_ids=set(metadata['protected_final_frame_ids'])
    protected_sha=set(metadata['protected_final_image_sha256'])
    require(len(protected_ids)==229 and len(protected_sha)>=229,'보호 FINAL 메타데이터 guard 불완전')
    # 어떠한 RGB 바이트 열람보다 앞서 전체 core의 ID·메타 SHA 분리를 검증한다.
    require(not set(ids)&protected_ids,'DEV ID와 보호 FINAL ID 중첩')
    require(not {r.get('source_frame_id') for r in records if r.get('source_frame_id')}&protected_ids,'DEV source frame ID와 보호 FINAL 중첩')
    require(not {r['image_sha256'] for r in records}&protected_sha,'DEV 이미지 SHA와 보호 FINAL 중첩')
    for r in records:
        require(len(r['image_sha256'])==64,'DEV 선언 이미지 SHA 없음')
        require('/data/evaluation/pallet_eval_v1/final/' not in str(Path(r['image_path']).resolve()),'보호 FINAL 경로 접근 시도')
    frames={};preds={};model_names={}
    r0=scores['models']['R0_before_selftraining']['frames']
    require([r['id'] for r in r0]==ids,'기준점 ID 순서 불일치')
    signatures=[[(p['index'],p['xy'],p['source'],p.get('verified_manual')) for p in r['all_annotated_reference']] for r in r0]
    for kind in C.KINDS:
        model_names[kind]={'R0':'R0_before_selftraining','control_R2':'SYNTHETIC_CONTROL_R2','ST_R2':kind+'_R2'}
        frames[kind]={};preds[kind]={}
        for role,name in model_names[kind].items():
            rows=scores['models'][name]['frames'];pr=predictions['predictions'][name]
            require([r['id'] for r in rows]==ids==[r['id'] for r in pr],'모델별 ID 순서 불일치: '+name)
            require([[(p['index'],p['xy'],p['source'],p.get('verified_manual')) for p in r['all_annotated_reference']] for r in rows]==signatures,'모델별 기준점 불일치: '+name)
            for r,pred,meta in zip(rows,pr,records):
                require(pred['image_sha256']==meta['image_sha256'],'예측 이미지 SHA 연결 불일치')
                require(len(pred['points'])==len(pred['valid'])==9,'예측 semantic9 슬롯 불일치')
                refs=r['all_annotated_reference'];indices=[q['index'] for q in refs]
                require(indices and len(indices)==len(set(indices)) and all(i in range(8) for i in indices),'기준점 채널/분모 불일치')
                for q in refs:
                    require(finite_pair(q['xy']),'기준점 좌표 비유한값')
                    i=q['index'];observed=bool(pred['valid'][i]) and finite_pair(pred['points'][i])
                    require(observed==(q['error_px'] is not None),'예측 valid와 기존 오차 유무 불일치')
                    if observed:
                        err=math.hypot(*(pred['points'][i][j]-q['xy'][j] for j in (0,1)))
                        require(math.isfinite(q['error_px']) and abs(err-q['error_px'])<=1e-9,'원본 좌표 오차와 캐시 오차 불일치')
            frames[kind][role]={r['id']:r for r in rows}
            preds[kind][role]={r['id']:r for r in pr}
    return records,scores,predictions,summary,metadata,frames,preds,model_names


def metric(frame):
    refs=frame['all_annotated_reference']
    n=len(refs);observed=sum(q['error_px'] is not None for q in refs)
    correct=sum(q['error_px'] is not None and q['error_px']<=10 for q in refs)
    return dict(n_reference=n,n_observed=observed,n_missing=n-observed,correct10=correct,PCK10=correct/n)


def prepare():
    ensure_new((SELECTION,DELTA_CSV,SELECTED_CSV,SELECTED_MD))
    C.start('T5_고정사례_선정',paths())
    records,scores,predictions,summary,metadata,frames,preds,names=load_inputs()
    all_rows=[];selected=[];by_id={r['id']:r for r in records}
    for kind in C.KINDS:
        group=[]
        for r in records:
            if C.kind_of(r)!=kind:continue
            m={role:metric(frames[kind][role][r['id']]) for role in MODEL_ROLES}
            require(len({v['n_reference'] for v in m.values()})==1,'모델별 기준점 분모 불일치')
            n=m['R0']['n_reference'];delta=Fraction(m['ST_R2']['correct10']-m['control_R2']['correct10'],n)
            group.append(dict(id=r['id'],kind=kind,metrics=m,delta=fraction_json(delta)))
        require((len(group),sum(g['metrics']['R0']['n_reference'] for g in group))==EXPECTED[kind],'종류별 영상·기준점 수 불일치')
        for role in MODEL_ROLES:
            correct=sum(g['metrics'][role]['correct10'] for g in group)
            n=sum(g['metrics'][role]['n_reference'] for g in group)
            require(correct/n==summary['primary_diagonal'][kind]['models'][role]['PCK10'],'기존 SUMMARY PCK10 재현 실패')
        ordered=sorted(as_fraction(g['delta']) for g in group)
        mid=len(ordered)//2
        median=ordered[mid] if len(ordered)%2 else (ordered[mid-1]+ordered[mid])/2
        choices={
            'max':min(group,key=lambda g:(-as_fraction(g['delta']),g['id'])),
            'median':min(group,key=lambda g:(abs(as_fraction(g['delta'])-median),g['id'])),
            'min':min(group,key=lambda g:(as_fraction(g['delta']),g['id']))}
        for g in sorted(group,key=lambda g:g['id']):
            row={'종류':KOREAN[kind],'영상 ID':g['id'],'기준점 수':g['metrics']['R0']['n_reference']}
            for role in MODEL_ROLES:
                row[MODEL_KO[role]+' PCK10 정답 수']=g['metrics'][role]['correct10']
                row[MODEL_KO[role]+' PCK10']=g['metrics'][role]['PCK10']
                row[MODEL_KO[role]+' 관측점 수']=g['metrics'][role]['n_observed']
            row.update({'차이 ST-대조군 PCK10':g['delta']['value'],'차이 분자':g['delta']['numerator'],'차이 분모':g['delta']['denominator']})
            all_rows.append(row)
        for category in CATEGORIES:
            g=choices[category];r=by_id[g['id']]
            selected.append(g|dict(category=category,selection_label=CATEGORY_KO[category],population_median=fraction_json(median),
                population_n=len(group),manifest_record=r,model_names=names[kind],
                filename='T5_'+kind+'_'+category))
    require(len(all_rows)==292 and len(selected)==9,'292행·9사례 개수 불일치')
    require({s['id'] for s in selected}<=set(by_id),'선정 ID가 core에 없음')
    frozen=dict(schema='T5_fixed_cases_v1',created_utc=C.now(),seed=C.SEED,
        rule='종류별 영상당 PCK10(ST_R2-대조군R2)의 최대·모집단 중앙값에 가장 가까운 영상·최소. 모든 동률은 ID 오름차순. 유리한 재선정 없음.',
        exact_comparison='정답 수 차이/영상별 전체 기준점 수의 유리수로 비교; 짝수 모집단 중앙값은 가운데 두 값의 평균',
        inputs={str(p):C.sha(p) for p in paths()},cases=selected,
        protected_final_guard=dict(pass_=True,core_frames=292,protected_frame_ids=229,core_ID_or_SHA_overlap=0,RGB_opened=0),
        total_unique_selected_images=len({s['id'] for s in selected}),new_forward_count=0,raw_annotations_opened=0)
    C.csvwrite(DELTA_CSV,all_rows)
    table=[]
    for s in selected:
        m=s['metrics'];table.append({'종류':KOREAN[s['kind']],'선정 규칙':s['selection_label'],'영상 ID':s['id'],
            '전체 영상 수':s['population_n'],'기준점 수':m['R0']['n_reference'],
            'R0 정답/관측':str(m['R0']['correct10'])+'/'+str(m['R0']['n_observed']),
            '대조군R2 정답/관측':str(m['control_R2']['correct10'])+'/'+str(m['control_R2']['n_observed']),
            'ST_R2 정답/관측':str(m['ST_R2']['correct10'])+'/'+str(m['ST_R2']['n_observed']),
            'R0 PCK10 (%)':100*m['R0']['PCK10'],'대조군R2 PCK10 (%)':100*m['control_R2']['PCK10'],
            'ST_R2 PCK10 (%)':100*m['ST_R2']['PCK10'],'차이 (퍼센트포인트)':100*s['delta']['value'],
            '모집단 차이 중앙값 (퍼센트포인트)':100*s['population_median']['value']})
    C.csvwrite(SELECTED_CSV,table)
    text='# T5 사전 고정 규칙으로 선정한 사례\n\n[확인] 영상당 PCK10(ST_R2-대조군R2)의 최대·중앙값에 가장 가까운 값·최소를 종류별로 1개씩 선정했다. 짝수 모집단 중앙값은 가운데 두 값의 평균이다. 모든 동률은 영상 ID 오름차순으로 정했다. 정확한 유리수로 비교했고 사례를 다시 고르지 않았다.\n\n'
    text+=C.mdtable(list(table[0]),[[r[k] for k in table[0]] for r in table])
    text+='\n[확인] 정답/관측 열은 ≤10px 점 수/관측점 수다. PCK10의 분모는 관측점 수가 아닌 영상별 전체 기준점 수이며, 누락 예측은 실패다. 차이는 퍼센트포인트다. 기준점은 기존 all_annotated_reference이며 출처 미상·기하 생성점이 포함될 수 있다. 독립 물리 6D 정답이나 성공확률을 뜻하지 않는다.\n\n[확인] 전체292장의 영상별 차이는 T5_ALL_FRAME_DELTAS.csv에 보존했다. 선정된 원본 전체 영상을 비율을 유지해 패널 너비640px에 맞춘다. crop·재선정·새 추론·원본 annotation 열람은 하지 않는다.\n'
    SELECTED_MD.write_text(text)
    C.write(SELECTION,frozen)
    C.log('step_complete',task='T5_고정사례_선정',selection_sha256=C.sha(SELECTION),cases=9,all_frame_rows=292,RGB_opened=0,new_forward_count=0)
    print('T5 선정 완료: 292행, 고정9사례, 아직 RGB 열람0',flush=True)


def font(size,bold=False):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH,size)


def marker(draw,xy,color,shape,r=7,width=3):
    x,y=xy
    # 흰색 외곽선으로 밝기와 배경이 달라도 의미 채널 색을 읽을 수 있게 한다.
    for col,w,rr in [('white',width+3,r+1),(color,width,r)]:
        if shape=='circle':draw.ellipse((x-rr,y-rr,x+rr,y+rr),outline=col,width=w)
        elif shape=='square':draw.rectangle((x-rr,y-rr,x+rr,y+rr),outline=col,width=w)
        elif shape=='triangle':draw.line([(x,y-rr-2),(x-rr-2,y+rr),(x+rr+2,y+rr),(x,y-rr-2)],fill=col,width=w,joint='curve')
        else:
            draw.line((x-rr,y,x+rr,y),fill=col,width=w)
            draw.line((x,y-rr,x,y+rr),fill=col,width=w)


def figure(image,case,refs,predictions):
    from PIL import Image,ImageDraw
    scale=640/image.width;h=round(image.height*scale)
    pad=16;cellw=672;titleh=45;cellh=titleh+h+pad*2
    margin=28;gap=20;width=margin*2+cellw*2+gap
    header=192;metrics_h=178;footer=104
    height=header+cellh*2+gap+metrics_h+footer
    canvas=Image.new('RGB',(width,height),'white');d=ImageDraw.Draw(canvas)
    d.text((margin,18),KOREAN[case['kind']]+' | '+case['selection_label'],font=font(30,True),fill='#18222f')
    d.text((margin,60),'영상 ID: '+case['id'],font=font(21),fill='#334155')
    d.text((margin,91),'영상당 PCK@10 차이 (LOO+flip R2 - 대조군 R2): '+f"{100*case['delta']['value']:+.2f} %p",font=font(24,True),fill='#18222f')
    x=margin
    for role,label in [('GT','기준점'),('R0','R0'),('control_R2','대조군 R2'),('ST_R2','LOO+flip R2')]:
        marker(d,(x+9,137),'#263238',MARKERS[role],r=7)
        d.text((x+26,122),label,font=font(21),fill='#263238');x+={'GT':190,'R0':150,'control_R2':250,'ST_R2':290}[role]
    d.text((margin,160),'모서리 채널',font=font(18),fill='#475569')
    for i,color in enumerate(COLORS):
        x=margin+135+i*78
        d.ellipse((x,163,x+15,178),fill=color)
        d.text((x+23,157),str(i),font=font(19,True),fill='#263238')
    rgb=image.resize((640,h),Image.Resampling.LANCZOS)
    panels=[('전체 비교',MODEL_ROLES),('기준점 + R0',('R0',)),('기준점 + 대조군 R2',('control_R2',)),('기준점 + LOO+flip R2',('ST_R2',))]
    for pi,(title,roles) in enumerate(panels):
        left=margin+(pi%2)*(cellw+gap);top=header+(pi//2)*(cellh+gap)
        d.rounded_rectangle((left,top,left+cellw,top+cellh),radius=9,fill='#f1f5f9')
        d.text((left+pad,top+10),title,font=font(22,True),fill='#18222f')
        ix=left+pad;iy=top+titleh+pad
        canvas.paste(rgb,(ix,iy))
        # 별도 패널 레이어로 바깥 좌표가 이웃 패널을 덮지 않게 한다. 원 영상은 crop하지 않는다.
        layer=Image.new('RGBA',(cellw,h+pad*2),(0,0,0,0));ld=ImageDraw.Draw(layer)
        for q in refs:
            xy=(pad+q['xy'][0]*scale,pad+q['xy'][1]*scale)
            marker(ld,xy,COLORS[q['index']],'circle',r=8,width=3)
        for role in roles:
            pred=predictions[role]
            for i in range(8):
                if pred['valid'][i] and finite_pair(pred['points'][i]):
                    xy=(pad+pred['points'][i][0]*scale,pad+pred['points'][i][1]*scale)
                    marker(ld,xy,COLORS[i],MARKERS[role],r=7,width=3)
        # 번호는 기준점 옆에 한 번만 붙인다. 고정 색이 예측점과 동일 채널을 연결한다.
        for q in refs:
            x=pad+q['xy'][0]*scale;y=pad+q['xy'][1]*scale
            tx=max(2,min(cellw-21,x+10));ty=max(0,min(h+pad*2-23,y-23))
            ld.text((tx,ty),str(q['index']),font=font(20,True),fill=COLORS[q['index']],stroke_width=2,stroke_fill='white')
        canvas.paste(layer,(left,top+titleh),layer)
    my=header+cellh*2+gap+18
    cols=[margin,margin+355,margin+590,margin+800,margin+1050]
    headers=['모델','전체 기준점','관측점','10px 이내','PCK@10']
    d.line((margin,my-6,width-margin,my-6),fill='#94a3b8',width=2)
    for x,t in zip(cols,headers):d.text((x,my),t,font=font(21,True),fill='#263238')
    for ri,role in enumerate(MODEL_ROLES):
        m=case['metrics'][role];yy=my+36+ri*32
        vals=[MODEL_KO[role],str(m['n_reference']),str(m['n_observed']),str(m['correct10']),f"{100*m['PCK10']:.2f}%"]
        for x,t in zip(cols,vals):d.text((x,yy),t,font=font(21),fill='#263238')
    fy=my+145
    notes=[
        '누락 예측은 PCK 실패. 전체 기준점이 분모이며, 중심점(채널8)은 제외한다.',
        '원본 전체 영상 표시: '+str(image.width)+' x '+str(image.height)+'px. 패널 너비640px, 비율 유지, crop 없음.',
        '기존 2D 기준점에는 출처 미상·기하 생성점이 포함될 수 있다. 독립 물리 6D 정답이 아니다.',
        '사전 고정된 최대/중앙값 근접/최소 사례이며 전체 영상의 대표성을 보장하지 않는다.']
    for line in notes:
        d.text((margin,fy),line,font=font(18),fill='#475569');fy+=25
    require(fy<=height,'그림 footer 높이 초과')
    return canvas


def create():
    from PIL import Image
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.utils import ImageReader
    from pypdf import PdfReader
    frozen=C.read(SELECTION);require(len(frozen['cases'])==9,'고정 선정9사례 없음')
    for p,digest in frozen['inputs'].items():require(C.sha(p)==digest,'선정 이후 입력 SHA 변경: '+p)
    inputs=load_inputs();records,_,_,_,metadata,frames,preds,_=inputs
    by_id={r['id']:r for r in records};core=set(by_id)
    outputs=[C.OUT/'figures'/(s['filename']+ext) for s in frozen['cases'] for ext in ('.png','.pdf')]
    receipt=C.OUT/'cache/T5_CREATION.json';ensure_new([*outputs,receipt])
    C.start('T5_PNG_PDF_제작',[*paths(),SELECTION])
    C.OUT.joinpath('figures').mkdir(exist_ok=True)
    protected_ids=set(metadata['protected_final_frame_ids']);protected_sha=set(metadata['protected_final_image_sha256'])
    audited=[]
    # 모든 선정 경로에 대한 guard도 실제 RGB 열기 전에 한 번에 통과해야 한다.
    for case in frozen['cases']:
        r=case['manifest_record'];require(case['id'] in core and r==by_id[case['id']],'동결 사례와 core record 불일치')
        require(case['id'] not in protected_ids and r.get('source_frame_id') not in protected_ids and r['image_sha256'] not in protected_sha,'보호 FINAL ID/SHA 접근 차단')
        require('/data/evaluation/pallet_eval_v1/final/' not in str(Path(r['image_path']).resolve()),'보호 FINAL 실제 경로 차단')
    C.log('T5_RGB_guard_pass',core_ID_and_SHA_disjoint=True,selected_cases=9,protected_final_metadata_only=True)
    for case in frozen['cases']:
        r=case['manifest_record'];p=Path(r['image_path'])
        # 허용된 사례만 읽고, 디코딩 전에 바이트 SHA를 확인한다.
        content=p.read_bytes();digest=C.hashlib.sha256(content).hexdigest()
        require(digest==r['image_sha256'],'선정 원본 RGB SHA 불일치: '+str(p))
        require(digest not in protected_sha,'실제 RGB SHA가 보호 FINAL과 겹침')
        image=Image.open(io.BytesIO(content));require(image.size==(r['width'],r['height']),'실제 영상 크기와 manifest 불일치')
        rgb=image.convert('RGB');image.close()
        kind=case['kind'];fid=case['id'];refs=frames[kind]['R0'][fid]['all_annotated_reference']
        for q in refs:require(0<=q['xy'][0]<rgb.width and 0<=q['xy'][1]<rgb.height,'기존 기준점이 영상 밖임')
        pred={role:preds[kind][role][fid] for role in MODEL_ROLES}
        picture=figure(rgb,case,refs,pred)
        png=C.OUT/'figures'/(case['filename']+'.png');pdf=png.with_suffix('.pdf')
        picture.save(png,format='PNG',optimize=True)
        w,h=picture.size;doc=pdfcanvas.Canvas(str(pdf),pagesize=(w/2,h/2),pageCompression=1)
        doc.setTitle('T5 '+case['filename']);doc.setAuthor('DOPE thesis analysis Batch1')
        doc.drawImage(ImageReader(picture),0,0,width=w/2,height=h/2)
        doc.showPage();doc.save()
        require(len(PdfReader(str(pdf)).pages)==1,'PDF가 1쪽이 아님')
        audited.append(dict(id=fid,kind=kind,category=case['category'],source_image=str(p),source_image_sha256=digest,
            native_size=[rgb.width,rgb.height],figure_size=[w,h],png=str(png),png_sha256=C.sha(png),pdf=str(pdf),pdf_sha256=C.sha(pdf),
            model_metrics=case['metrics'],delta=case['delta'],protected_FINAL_guard=True))
        picture.close();rgb.close()
    require(len(list((C.OUT/'figures').glob('T5_*.png')))==9 and len(list((C.OUT/'figures').glob('T5_*.pdf')))==9,'최종 PNG9/PDF9 개수 불일치')
    C.write(receipt,dict(schema='T5_creation_v1',created_utc=C.now(),selection_sha256=C.sha(SELECTION),cases=audited,
        cases_count=9,png_count=9,pdf_count=9,RGB_files_opened=9,raw_annotations_opened=0,protected_FINAL_RGB_opened=0,
        new_forward_count=0,model_files_opened=0,rendering='PIL 전체 영상 패널 및 reportlab 단일 페이지 PDF',
        marker_contract='PDF marker operation create, expected-output-count9; 호출자 로그에 기록',
        visual_review_status='PDF9개 재렌더 후 PNG9개와 함께 전체 확인 대기'))
    C.log('step_complete',task='T5_PNG_PDF_제작',cases=9,png_count=9,pdf_count=9,RGB_files_opened=9,raw_annotations_opened=0,new_forward_count=0,receipt_sha256=C.sha(receipt))
    print('T5 PNG9/PDF9 제작 완료. 전체 visual QA 필요.',flush=True)


def render_qa():
    from PIL import Image,ImageDraw
    creation=C.read(C.OUT/'cache/T5_CREATION.json')
    qa=C.OUT/'cache/T5_pdf_renders';require(not qa.exists(),'기존 PDF 렌더 QA 덮어쓰기 금지')
    qa.mkdir();C.start('T5_PDF_렌더_QA',[C.OUT/'cache/T5_CREATION.json'])
    audit=[]
    for case in creation['cases']:
        pdf=Path(case['pdf']);require(C.sha(pdf)==case['pdf_sha256'],'PDF 제작 후 SHA 변경')
        prefix=qa/pdf.stem
        cmd=['/usr/bin/pdftoppm','-scale-to-x',str(case['figure_size'][0]),'-scale-to-y','-1','-png','-singlefile',str(pdf),str(prefix)]
        C.log('external_command',task='T5_PDF_렌더_QA',command=cmd,input_sha256=case['pdf_sha256'])
        C.subprocess.run(cmd,check=True,capture_output=True,text=True)
        raster=prefix.with_suffix('.png');require(raster.is_file(),'PDF 렌더 PNG 없음')
        audit.append(dict(id=case['id'],kind=case['kind'],category=case['category'],pdf=str(pdf),pdf_sha256=C.sha(pdf),render_png=str(raster),render_sha256=C.sha(raster)))
    # contact sheet는 전체 사례를 조망하는 보조 자료이며 개별 원해상도 검토를 대체하지 않는다.
    thumbs=[]
    for case in creation['cases']:
        im=Image.open(case['png']).convert('RGB');im.thumbnail((460,560))
        cell=Image.new('RGB',(480,610),'#e2e8f0');cell.paste(im,((480-im.width)//2,40))
        d=ImageDraw.Draw(cell);d.text((12,10),KOREAN[case['kind']]+' / '+CATEGORY_KO[case['category']],font=font(17,True),fill='#18222f')
        thumbs.append(cell);im.close()
    sheet=Image.new('RGB',(1440,1830),'white')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*480,(i//3)*610));im.close()
    contact=C.OUT/'cache/T5_CONTACT_SHEET.png';sheet.save(contact);sheet.close()
    C.write(C.OUT/'cache/T5_RENDER_QA.json',dict(schema='T5_render_qa_v1',created_utc=C.now(),renders=audit,
        pdf_count=9,renderer='/usr/bin/pdftoppm',contact_sheet=str(contact),contact_sha256=C.sha(contact),
        individual_native_visual_inspection='대기; 모든 최종 PNG와 PDF 렌더 PNG를 직접 열어야 통과'))
    print('T5 PDF9개 재렌더·contact sheet 완료. 개별18개 직접 검토 대기.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','create','render-qa'))
    args=parser.parse_args()
    try:
        {'prepare':prepare,'create':create,'render-qa':render_qa}[args.mode]()
    except Exception as exc:
        C.log('gate_failure',task='T5',mode=args.mode,error=str(exc));raise


if __name__=='__main__':
    main()
