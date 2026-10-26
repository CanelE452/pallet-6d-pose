"""Saved-output diagnosis. Portable --replay needs only included compact inputs.

No checkpoint, encoder, model forward, fitting, PnP or operating-rule selection.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = Path(__file__).absolute().parent
R = W.parent
BINS = ('REFERENCE_OUTSIDE_CROP','INSIDE_REF_NEAR_LE3','INSIDE_REF_BORDER_3_TO6','INSIDE_REF_FAR_GT6')
GROUPS = ('ALL','WOOD_PRIMARY105','SQUARE118')
TITLES = dict(zip(BINS, ('참조 좌표가 crop 밖','참조 좌표에서 3px 이내','참조 좌표와 3–6px 차이','참조 좌표와 6px 넘게 차이')))


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(x in str(p).lower() for x in ('v112','cache','.pyc'))
    for a in reversed((p,*p.parents)):
        assert not a.is_symlink()
    return p


def binding(p):
    p=safe(p)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)


def check(p, sha=None):
    p=safe(p)
    if sha:assert binding(p)['sha256']==sha
    return p


def bound(b):
    p=check(b['path'],b['sha256']);assert p.stat().st_size==b['size_bytes'];return p


def js(p,v):
    p=safe(p);assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv_write(p,rows):
    p=safe(p);assert not p.exists()
    with p.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def finite(v):
    try:x=float(v)
    except (ValueError,TypeError):return False
    return math.isfinite(x)


def prepare_nonpixel_inputs():
    inputs=safe(W/'analysis_inputs');assert not inputs.exists();inputs.mkdir()
    sources={
        'LEDGER.csv':check(R/'V263_CURRENT_EVALUATION_A/actual_evaluation_v1/ALL1784_CLASSIFICATION_LOCATION_JOIN.csv','df7b619649ede26af4e21644c22b10e84b334c6bc8708f4e5a668ec373db5175'),
        'COMPARISON.csv':check(R/'V264_HEATMAP_LOCAL_VISIBILITY/evaluation_v1/ALL1784_COMPARISON.csv','fed843738b11e21c9daa1c8c97c6c2c4e87802aa7e1ad38972b7f3a991bba4dd'),
        'POINT_DETAILS.csv':check(R/'V263_CURRENT_EVALUATION_A/actual_evaluation_v1/POINT_DETAILS.csv','7a21874b9ad85ed038c54ead0ef64e1ee1da5616b4a30ad34545db6eb42bcf52')}
    for name,p in sources.items():shutil.copyfile(p,inputs/name)
    freeze_path=check(R/'V264_HEATMAP_LOCAL_VISIBILITY/CURRENT_PREDICTIONS_FROZEN.json','b21b3fe1b6f75adf753bec8ff808f0f1edabd7a08799db30a9f75f4339cfc08a')
    freeze=json.loads(freeze_path.read_text())
    with np.load(bound(freeze['feature_packet']),allow_pickle=False) as z:uv=np.array(z['uv'],copy=True)
    old_path=bound(freeze['current_source_complete']);old=json.loads(old_path.read_text())
    h=bound(old['heatmaps']['JOINT_SELECTED']['binding'])
    with np.load(h,allow_pickle=False) as z:p=np.array(z['probability'],copy=True)
    s=bound(old['outputs']['TOKEN_SUPPORT.npy']);support=np.load(s,allow_pickle=False)
    np.savez_compressed(inputs/'FROZEN_GEOMETRY.npz',uv=uv,probability=p,support=support)
    js(inputs/'SOURCE_BINDINGS.json',{'completed':True,'source_files':{k:binding(v) for k,v in sources.items()},
        'V264_prediction_freeze':binding(freeze_path),'V264_pooled_geometry_source':freeze['feature_packet'],
        'V263_current_complete':binding(old_path),'heatmap_source':binding(h),'support_source':binding(s),
        'new_model_or_checkpoint_reads':0,'new_RGB_reads_so_far':0})
    return old


def analyze():
    inputs=safe(W/'analysis_inputs')
    ledger=list(csv.DictReader((inputs/'LEDGER.csv').open()))
    comparison=list(csv.DictReader((inputs/'COMPARISON.csv').open()))
    points=list(csv.DictReader((inputs/'POINT_DETAILS.csv').open()))
    with np.load(inputs/'FROZEN_GEOMETRY.npz',allow_pickle=False) as z:g={k:np.array(z[k],copy=True) for k in z.files}
    assert len(ledger)==len(comparison)==len(g['uv'])==1784
    details=[]; max_reuse_difference=0.
    for i,(row,c) in enumerate(zip(ledger,comparison)):
        assert int(row['global_index'])==int(c['global_index'])==i and row['row_id']==c['row_id']
        if row['JOINT_SELECTED__available']!='True':continue
        q=np.array([float(row['query_x']),float(row['query_y'])]);L=float(row['JOINT_SELECTED__crop_L'])
        proposal=q+(g['uv'][i].astype(float)-.5)*L
        old_proposal=np.array(json.loads(row['JOINT_SELECTED__LOC_proposed_xy']))
        max_reuse_difference=max(max_reuse_difference,float(np.abs(proposal-old_proposal).max()))
        if c['PRIMARY']!='True' or c['human_state']!='EXTERNAL_OCCLUDED' or c['HEATMAP_LOCAL_prediction']!='D':continue
        ref=np.array([float(row['human_reference_x']),float(row['human_reference_y'])])
        uv_ref=.5+(ref-q)/L;in_crop=bool(((uv_ref>=0)&(uv_ref<1)).all())
        d=float(np.linalg.norm(proposal-ref)); initial=float(np.linalg.norm(q-ref))
        category=BINS[0] if not in_crop else BINS[1] if d<=3 else BINS[2] if d<=6 else BINS[3]
        details.append({'global_index':i,'row_id':row['row_id'],'group':row['group'],
            'coordinate_source':row['annotation_coordinate_source'],'reference_inside_crop':in_crop,
            'initial_reference_distance_px':initial,'proposal_reference_distance_px':d,
            'local_sigma_original_px':1.5*L/16,'proposal_distance_in_sigma':d/(1.5*L/16),
            'crop_L':L,'category':category,'GLOBAL_prediction':c['GLOBAL_MATCHED_prediction'],
            'LOCAL_prediction':c['HEATMAP_LOCAL_prediction'],'LOCAL_p_E':float(c['HEATMAP_LOCAL_p_E']),
            'disputed_annotation':i==996})
    assert len(details)==67 and max_reuse_difference<1e-4
    census=[];cross=[];selection=[]
    for group in GROUPS:
        rs=[r for r in details if group=='ALL' or r['group']==group]
        census.append({'group':group,'N':len(rs),'bins':{b:sum(r['category']==b for r in rs) for b in BINS},
                       'coordinate_sources':dict(Counter(r['coordinate_source'] for r in rs)),
                       'global_E_to_local_D':sum(r['GLOBAL_prediction']=='E' for r in rs),
                       'both_models_D':sum(r['GLOBAL_prediction']=='D' for r in rs)})
        for initial_near in (True,False):
            x=[r for r in rs if (r['initial_reference_distance_px']<=3)==initial_near]
            cross.append({'group':group,'initial_reference_LE3':initial_near,'proposal_LE3':sum(r['proposal_reference_distance_px']<=3 for r in x),
                          'proposal_3to6':sum(3<r['proposal_reference_distance_px']<=6 for r in x),
                          'proposal_GT6':sum(r['proposal_reference_distance_px']>6 for r in x),
                          'of_GT6_reference_outside':sum(r['category']==BINS[0] for r in x)})
        if group=='ALL':continue
        for b in BINS:
            pool=[r for r in rs if r['category']==b and not r['disputed_annotation']]
            selection.append({'group':group,'category':b,'census_including_disputed':sum(r['category']==b for r in rs),
                              'selected_index':min((r['global_index'] for r in pool),default=None)})
    pindex={(r['model'],int(r['global_index'])):r for r in points}
    ids=[i for i,r in enumerate(ledger) if r['human_state']=='DIRECT_VISIBLE' and r['annotation_coordinate_source']=='manual_click'
         and r['original_query_valid']=='True' and ('ORIGINAL_QUERY',i) in pindex and finite(pindex[('ORIGINAL_QUERY',i)]['initial_error_px'])]
    assert len(ids)==854
    loc=[]; visible_rows=[]
    methods=[('INITIAL','ORIGINAL_QUERY','initial_error_px'),('LOC_RAW','LOC_ONLY_INITIAL_GEOMETRY','proposed_error_px'),
             ('LOC_FROZEN_ACCEPTED','LOC_ONLY_INITIAL_GEOMETRY','final_error_px'),
             ('JOINT_RAW','JOINT_SELECTED','proposed_error_px'),('JOINT_FROZEN_ACCEPTED','JOINT_SELECTED','final_error_px')]
    for i in ids:
        r={'global_index':i,'row_id':ledger[i]['row_id'],'group':ledger[i]['group']}
        for name,model,key in methods:r[name]=float(pindex[(model,i)][key])
        visible_rows.append(r)
    for group in GROUPS:
        rs=[r for r in visible_rows if group=='ALL' or r['group']==group]
        initial=np.array([r['INITIAL'] for r in rs])
        for name,model,key in methods:
            a=np.array([r[name] for r in rs]);dam=(initial<=3)&(a>3)
            loc.append({'group':group,'method':name,'N':len(rs),'LE3':int((a<=3).sum()),'LE6':int((a<=6).sum()),
                        'median_px':float(np.median(a)),'P90_px':float(np.quantile(a,.9)),
                        'recovered_GT3_to_LE3':int(((initial>3)&(a<=3)).sum()),'crossed_LE3_to_GT3':int(dam.sum()),
                        'crossing_final_3to6':int((dam&(a<=6)).sum()),'crossing_final_6to10':int((dam&(a>6)&(a<=10)).sum()),
                        'crossing_final_GT10':int((dam&(a>10)).sum()),'retrospective_human_visible_only':True})
    extra=[]
    for group in GROUPS[1:]:
        manual=[r for r in details if r['group']==group and r['category']==BINS[1] and r['coordinate_source']=='manual_click' and not r['disputed_annotation']]
        if manual:
            idx=min(r['global_index'] for r in manual)
            if idx not in [s['selected_index'] for s in selection]:extra.append({'selected_index':idx,'role':'manual_hidden_reference_ambiguity_control'})
    extra += [{'selected_index':996,'role':'user_disputed_annotation'}, {'selected_index':201,'role':'old_fixed_wood_visible_recovery'},
              {'selected_index':848,'role':'old_fixed_square_visible_recovery'}]
    result={'completed':True,'missed_external_E':67,'census':census,'initial_vs_proposal':cross,'visible_location':loc,
            'core_selection':selection,'additional_selection':extra,'max_abs_reused_proposal_difference_px':max_reuse_difference,
            'no_training_or_model_or_PnP':True,'reference_distance_is_proxy_not_visible_location_gold':True,
            'raw_allapply_is_humanD_counterfactual_not_deployable_gate':True,'new_gate_selected':False}
    return result,details,visible_rows,g,ledger,comparison


def font(size):return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size)


def render_card(index,role,census,rgb,g,row,c,out):
    image=Image.new('RGB',(1200,750),'white');d=ImageDraw.Draw(image)
    label=TITLES.get(role,role);L=float(row['JOINT_SELECTED__crop_L'])
    q=np.array([float(row['query_x']),float(row['query_y'])]);ref=np.array([float(row['human_reference_x']),float(row['human_reference_y'])])
    rp=(ref-q)*96/L+47.5;centre=g['uv'][index]*96-.5
    proposal=q+(g['uv'][index]-.5)*L
    distance=float(np.linalg.norm(proposal-ref)); initial=float(np.linalg.norm(q-ref))
    d.text((22,14),label+' · '+row['group'],font=font(28),fill='#172431')
    d.text((22,60),f"G{index} · {row['row_id']} · 기존 {row['human_state']}",font=font(22),fill='#172431')
    d.text((22,98),f"참조 거리: 초기 {initial:.2f}px → heatmap {distance:.2f}px · 좌표 출처 {row['annotation_coordinate_source']}",font=font(21),fill='#34495e')
    population=f'셀 전체 {census}점' if census is not None else '추가 검토 예시'
    d.text((22,134),f"가림 판단: 전체 평균 {c['GLOBAL_MATCHED_prediction']} / 위치 주변 {c['HEATMAP_LOCAL_prediction']} · {population}",font=font(21),fill='#34495e')
    yy,xx=np.mgrid[:16,:16];grid=np.stack((xx,yy),-1)+.5
    w=np.exp(-((grid-g['uv'][index]*16)**2).sum(-1)/(2*1.5**2))*g['support'][index]
    for j,title in enumerate(('실제 저장 crop','위치 heatmap · 상대 색','실제 국소 token 가중 영역')):
        x=26+j*396;y=214;size=360
        tile=Image.fromarray(rgb).resize((size,size),Image.Resampling.NEAREST)
        if j:
            v=g['probability'][index] if j==1 else w
            v=(v-v.min())/max(float(v.max()-v.min()),1e-12)
            z=np.asarray(Image.fromarray(v.astype('float32'),'F').resize((size,size),Image.Resampling.BILINEAR))
            color=np.stack((255*z,190*z,230*(1-z)),2).clip(0,255).astype('uint8')
            tile=Image.blend(tile,Image.fromarray(color),.48)
        draw=ImageDraw.Draw(tile)
        for p,col,kind in [(np.array([47.5,47.5]),'#00d8ff','circle'),(rp,'#ff42c4','plus'),(centre,'#edff37','plus')]:
            a,b=(p+.5)*size/96
            if 0<=a<size and 0<=b<size:
                if kind=='circle':draw.ellipse((a-6,b-6,a+6,b+6),outline=col,width=3)
                else:draw.line((a-8,b,a+8,b),fill=col,width=3);draw.line((a,b-8,a,b+8),fill=col,width=3)
        if j==2:
            a,b=(centre+.5)*size/96;rr=18*size/96
            draw.ellipse((a-rr,b-rr,a+rr,b+rr),outline='#a4ff40',width=2)
        image.paste(tile,(x,y));d.text((x,y-38),title,font=font(19),fill='#34495e')
    if index==996:note='사용자와 기존 E 주석 해석 불일치: 명백한 모델 실패로 단정하지 않음.'
    elif index in (792,1233,1009):note='팔레트 일부가 보이는 경계: 정확한 native 점의 직접 보임/주석 대응 재검토 필요.'
    elif role==BINS[0]:note='목표 참조 좌표가 원래 검색 crop 밖입니다. GT 중심으로 다시 자르지 않았습니다.'
    elif role==BINS[1]:note='좌표 근처에서도 D: 기존 E 주석이 맞다면 위치 이탈만으로 설명되지 않습니다.'
    elif role==BINS[2]:note='3–6px는 작은 경계 차이입니다. 큰 위치 실패로 단정하지 않습니다.'
    elif role==BINS[3]:note='콘 경계/바닥 등 다른 구조와의 대응을 실제 crop에서 확인합니다.'
    else:note='보임이 사후 확인된 위치 보정 예시입니다. 실제 보임 승인 규칙의 검증은 별도입니다.'
    d.text((22,606),note,font=font(20),fill='#963d16')
    d.text((22,649),'파랑○ 초기 query · 분홍+ 기존 참조 · 노랑+ heatmap 위치 · 초록 원 Gaussian2σ',font=font(19),fill='#34495e')
    d.text((22,686),'숨은 E 좌표는 직접 보이는 위치 정답이 아닙니다. 색은 보임 확률이 아닙니다.',font=font(19),fill='#34495e')
    p=safe(out/f'CASE_G{index:04d}.png');image.save(p)
    return binding(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');parser.add_argument('--out',default='diagnosis_v1');a=parser.parse_args()
    assert Path(a.out).name==a.out
    out=safe(W/a.out);assert not out.exists();out.mkdir()
    if not a.replay:old=prepare_nonpixel_inputs()
    else:
        sealed=json.loads(check(W/'SEALED_INPUTS.json').read_text())
        assert binding(W/'DIAGNOSIS_POLICY_BEFORE_NEW_CROP_RENDER.json')['sha256']==sealed['policy']['sha256']
        assert binding(Path(__file__).absolute())['sha256']==sealed['code']['sha256']
        for rel,b in sealed['inputs'].items():
            p=check(W/rel,b['sha256']);assert p.stat().st_size==b['size_bytes']
    result,details,visible,g,ledger,comparison=analyze()
    selected=[s['selected_index'] for s in result['core_selection'] if s['selected_index'] is not None]
    selected += [s['selected_index'] for s in result['additional_selection']]
    assert len(selected)==len(set(selected))==12
    js(out/'CASE_SELECTION_BEFORE_RENDER.json',{'completed':True,'core':result['core_selection'],'extra':result['additional_selection'],
        'prior_RGB_exposure':True,'note':'Prior versions/parallel audit inspected some crops. Fixed numeric selection, not fresh blinded inspection.'})
    if not a.replay:
        crop_binding=old['crop_cache']['binding']
        with np.load(bound(crop_binding),allow_pickle=False) as z:
            raw=z['RGB96_initial_geometry'][selected]
        if raw.shape[1:]==(3,96,96):raw=raw.transpose(0,2,3,1)
        assert raw.shape==(12,96,96,3) and raw.dtype==np.uint8
        np.savez_compressed(W/'analysis_inputs/CROPS_SELECTED.npz',global_index=np.array(selected),RGB96_HWC=raw)
        names=['LEDGER.csv','COMPARISON.csv','POINT_DETAILS.csv','FROZEN_GEOMETRY.npz','SOURCE_BINDINGS.json','CROPS_SELECTED.npz']
        js(W/'SEALED_INPUTS.json',{'completed':True,'policy':binding(W/'DIAGNOSIS_POLICY_BEFORE_NEW_CROP_RENDER.json'),
            'code':binding(Path(__file__).absolute()),'inputs':{f'analysis_inputs/{n}':binding(W/'analysis_inputs'/n) for n in names},
            'raw_crop_source':crop_binding,'selected_indices':selected,'new_training_model_forward_PnP':0})
    with np.load(W/'analysis_inputs/CROPS_SELECTED.npz',allow_pickle=False) as z:
        assert list(z['global_index'])==selected;crops=z['RGB96_HWC']
    cards=[]
    for j,i in enumerate(selected):
        cell=next((s for s in result['core_selection'] if s['selected_index']==i),None)
        role=cell['category'] if cell else next(s['role'] for s in result['additional_selection'] if s['selected_index']==i)
        census=cell['census_including_disputed'] if cell else None
        cards.append({'global_index':i,'role':role,'RGB96_HWC_sha256':hashlib.sha256(crops[j].tobytes()).hexdigest(),
                      'card':render_card(i,role,census,crops[j],g,ledger[i],comparison[i],out)})
    overview=Image.new('RGB',(1200,1500),'white')
    for y,i in [(0,370),(750,908)]:overview.paste(Image.open(out/f'CASE_G{i:04d}.png'),(0,y))
    overview.save(out/'NEAR_VS_DISTRACTOR_KO.png')
    js(out/'RESULTS.json',result);csv_write(out/'MISSED_EXTERNAL_67.csv',details);csv_write(out/'VISIBLE_MANUAL_854.csv',visible)
    csv_write(out/'VISIBLE_LOCATION_SUMMARY.csv',result['visible_location']);js(out/'CASE_DATA.json',cards)
    # A concise static readout; all underlying numbers and chosen crops are linked.
    stats=''.join('<tr><td>'+r['group']+'</td>'+''.join('<td>'+str(r['bins'][b])+'</td>' for b in BINS)+'</tr>' for r in result['census'])
    vis=''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in [r['group'],r['method'],f"{r['LE3']}/{r['N']}",f"{r['median_px']:.2f}",f"{r['P90_px']:.2f}"] )+'</tr>' for r in result['visible_location'])
    links=''.join(f'<li><a href="CASE_G{i:04d}.png">G{i} · '+next(x['role'] for x in cards if x['global_index']==i)+'</a></li>' for i in selected)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V265 · 가림 누락 원인과 보임 조건 위치 보정</title>
<style>body{font:17px/1.7 sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;color:#172431}img{max-width:100%}table{border-collapse:collapse;font-size:15px}td,th{border:1px solid #ccd4dd;padding:7px}aside{padding:18px;background:#fff0dc}</style>
<h1>외부 가림 누락을 나눠 보면 위치 문제와 판단 문제가 모두 있습니다</h1>
<aside>새 학습·모델 추론·PnP 없이 저장 결과와 실제 crop만 분석했습니다. 숨은 E 좌표는 직접 보이는 위치 정답이 아니므로 아래 거리는 ‘기존 참조 좌표와 차이’입니다.</aside>
<h2>가림 누락67점 분해</h2><table><tr><th>유형</th><th>참조 crop 밖</th><th>crop 안 ≤3px</th><th>crop 안 3–6px</th><th>crop 안 >6px</th></tr>CENSUS</table>
<p>누락67점 중49점은 전체 평균의 E가 위치 주변의 D로 바뀐 경우,18점은 양쪽 모두 D입니다. 초기와 heatmap 위치 모두 참조에서3px 이내인8점에서도 D를 출력해 위치 이탈만으로 모든 누락을 설명할 수 없습니다.</p>
<p>G370은 참조에서2.66px인데 콘의 빨간 몸체/흰 띠에 위치합니다. 기존 E 주석이 맞다면 가림 판단 실패입니다. G908·G256은 콘 받침/바닥 쪽으로 옮겨 다른 구조를 읽습니다.
G300·G1772는 목표 참조 위치가 검색 crop 밖입니다. G792·G1233·G1009처럼 팔레트 조각이 드러나는 경계는 정확한 native 대응 재검토가 필요하며, G996의 사용자 이견도 유지했습니다.</p>
<img src="NEAR_VS_DISTRACTOR_KO.png" alt="참조 근처에서도 가림 판단 실패와 다른 구조를 찾는 경우">
<p>공통 문제는 뚜렷한 콘 경계·띠·받침·바닥·근처 팔레트 조각이 위치 근거가 된다는 점입니다. E에 위치 heatmap 정답을 주지 않았고 softmax는 항상 봉우리를 만듭니다.
국소 Gaussian은96crop에서σ9px로 주변을 읽으며 DINO token 자체도 문맥을 포함합니다. 위치가 가까운 것과 지정 점의 직접 보임은 별개입니다.</p>
<h2>‘보이는 점을 보정’하는 역할이 더 맞습니다</h2><p>사람이 보임으로 주석하고 수동 좌표가 있는854점만 사후 분석했습니다. 원시 제안을 전부 적용한 결과는 실제 보임 승인 규칙을 검증한 결과가 아닙니다.</p>
<table><tr><th>유형</th><th>방법</th><th>≤3px</th><th>중앙값 px</th><th>P90 px</th></tr>VISIBLE</table>
<p>정사각형은 JOINT 원시 위치 제안으로223→352/531점, 중앙값3.68→2.23px,P90 9.11→5.91px입니다. 목재는153→136/323점으로 개선되지 않았습니다.
기존 승인 적용 JOINT의3px 경계 악화59점 중55점은 최종3–6px,4점은6–10px,10px 초과는0입니다. 작은 경계 초과를 모두 큰 실패로 부르지 않습니다.</p>
<h2>해결 방향</h2><ol><li>위치 모듈은 지정 키포인트의 보정 후보를 만드는 역할로 분리합니다. 보임이 별도로 확인된 점에서만 적용하며, 자기 가림 기하 필터가 외부 가림까지 제거하지는 않습니다.</li>
<li>가림 판단이 필요하면 초기 기대 위치/native ID를 유지한 채 그 위치와 보정 후보·가리는 물체를 함께 확인해야 합니다. 후보를 찾았다는 이유로 원래 점을 보임으로 바꾸지 않습니다. 목표가 crop 밖이면 검색 실패/보류로 구분합니다.</li>
<li>위치 실패는 다른 구조/검색 범위 문제로, 참조 근처 판단 실패는 가림 대상·직접 보임 정의/합성-실사 전이 문제로 나눕니다. 노출된 작은 모서리의 주석·대응을 먼저 재검토하고 임의 재라벨링하지 않습니다.</li>
<li>정사각형의 위치 제안을 우선 검증할 근거는 있으나 목재는 보류합니다. 이번 실사 결과로 새 threshold·crop·모델을 선택하지 않았고 배포 승인도 하지 않았습니다.</li></ol>
<h2>실제12개 crop</h2><ul>LINKS</ul><p><a href="RESULTS.json">전체 수치 JSON</a> · <a href="MISSED_EXTERNAL_67.csv">가림 누락67점</a> · <a href="VISIBLE_MANUAL_854.csv">수동 보임854점</a> · <a href="VISIBLE_LOCATION_SUMMARY.csv">위치 비교표</a> · <a href="CASE_SELECTION_BEFORE_RENDER.json">사례 선정·과거 RGB 노출 명시</a></p></html>'''
    (out/'INDEX_KO.html').write_text(page.replace('CENSUS',stats).replace('>VISIBLE</table>', '>'+vis+'</table>').replace('LINKS',links))
    files=[p for p in out.iterdir() if p.is_file()]
    js(out/'COMPLETE.json',{'completed':True,'new_training_model_forward_PnP':0,'outputs':{p.name:binding(p) for p in files},
        'SEALED_INPUTS':binding(W/'SEALED_INPUTS.json'),'code':binding(Path(__file__).absolute())})
    print(json.dumps({'completed':True,'output':str(out),'census':result['census'],'visible_summary':result['visible_location']},ensure_ascii=False))


if __name__=='__main__':main()
