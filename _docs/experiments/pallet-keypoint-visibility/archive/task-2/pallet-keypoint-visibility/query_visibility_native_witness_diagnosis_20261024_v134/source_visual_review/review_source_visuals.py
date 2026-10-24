"""Fixed saved-metadata selection then source-only RGB/crop diagnosis.

No model, fit, checkpoint, private, V112 or PnP access. Actual q is immutable.
"""
from pathlib import Path
import argparse, collections, hashlib, importlib.util, json
import numpy as np

H=Path(__file__).resolve().parent;P=H.parent.parent
V132=P/'query_visibility_actual_initializer_source_20261024_v132'
V133=P/'query_visibility_native_geometry_control_20261024_v133'
RUN=V133/'runs/C24_C25_MATCHED_G42_FIXED8000'
HELPER=V132/'report_preparation/render_saved_visibility.py'
SELECTION=H/'SELECTION_BEFORE_RGB.json';PACK=H/'SAVED_ROWS6144_BEFORE_RGB.json'
MODELS=('C16','C24','C25')
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def finite(v):return float(v)if np.isfinite(v)else None
def pred_name(v):return {-1:'보류',0:'보임',1:'외부 가림'}[int(v)]
def cohort(rows,mask):
    out=dict(original_rows=int(mask.sum()),teacher_states=dict(collections.Counter(r['raw_state']for take,r in zip(mask,rows)if take)))
    for m in MODELS:
        cm=np.zeros((2,3),np.int64)
        for take,r in zip(mask,rows):
            if take and r['raw_state']in('D','E'):
                y=0 if r['raw_state']=='D'else 1;p=r[m]['prediction'];cm[y,2 if p==-1 else p]+=1
        out[m]=dict(CM=cm.tolist(),known_DE=int(cm.sum()),known_U=int(cm[:,2].sum()))
    return out

def select():
    assert not SELECTION.exists()and not PACK.exists()
    control_path=RUN/'CONTROL_COMPLETE.json';control=read(control_path)
    assert control['completed']and control['PASS']and control['selected_arm_source_only']is None
    espath=V132/'saved_frozen_visibility_evaluation_v1/EVALUATION_SEAL.json';es=read(espath)
    cs_path=V132/'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json';cs=read(cs_path)
    assert es['completed']and es['PASS']and cs['completed']and cs['PASS']and cs['GT_read']is False
    source=[json.loads(x)for x in check(es['scores']).read_text().splitlines()if x]
    inp=read(check(cs['input_crop_facts']))['rows'];assert len(source)==len(inp)==6144
    bindings={};arrays={}
    for arm,model in(('C24_ZERO_GEOMETRY_CONTROL','C24'),('C25_NATIVE_GEOMETRY42','C25')):
        fit=read(check(control['arms'][arm]['fit']));assert fit['completed']and fit['optimizer_updates']==8000
        path=RUN/arm/'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz';bindings[model]=bind(path)
        with np.load(path,allow_pickle=False)as f:
            arrays[model]={k:np.array(f[k],copy=True)for k in('global_index','occurrence_id','query_xy','L','raw_head_logit_before_U','prediction','query_available','raw_teacher_state_CF')}
        assert np.array_equal(arrays[model]['global_index'],np.arange(6144))
    rows=[]
    for i,(r,x)in enumerate(zip(source,inp)):
        assert r['global_index']==x['occurrence_index']==i and r['query_id']==x['occurrence_id']
        assert r['query_xy']==x['query_xy']and r['roi_side']==x['L']
        row=dict(r);row['C16']=dict(prediction={'D':0,'E':1,'U':-1}[r['C16']['prediction']],raw_logit=r['C16']['raw_logit'],available=r['C16']['available'])
        for model in('C24','C25'):
            a=arrays[model];assert a['occurrence_id'][i]==r['query_id']and a['raw_teacher_state_CF'][i]==r['raw_state']
            q=np.array(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]);L=r['roi_side']if r['roi_side']is not None else np.nan
            assert np.array_equal(a['query_xy'][i],q,equal_nan=True)and(np.isnan(L)and np.isnan(a['L'][i])or a['L'][i]==L)
            raw=a['raw_head_logit_before_U'][i];av=bool(a['query_available'][i]);prediction=int(a['prediction'][i])
            assert prediction==int(raw>=0)if av else prediction==-1
            row[model]=dict(prediction=prediction,raw_logit=finite(raw),available=av)
        rows.append(row)
    val=np.array([r['split']=='VALIDATION'for r in rows]);nb=~np.array([r['convention_boundary']for r in rows],bool)
    le3=np.array([r['error_px']is not None and r['error_px']<=3 for r in rows])
    cohorts={name:cohort(rows,mask)for name,mask in(('VALIDATION_ALL',val),('VALIDATION_NONBOUNDARY',val&nb),('VALIDATION_NONBOUNDARY_LE3',val&nb&le3))}
    selected=[];pool_counts={};missing=[]
    for state,wrong,key in(('D',1,'D_FALSE_E'),('E',0,'E_FALSE_D')):
        eligible=lambda r:r['split']=='VALIDATION'and not r['convention_boundary']and r['raw_state']==state and any(r[m]['prediction']==wrong for m in('C24','C25'))
        near=[r for r in rows if eligible(r)and r['error_px']is not None and r['error_px']<=3]
        all_nonboundary=[r for r in rows if eligible(r)]
        pool_counts[key]=dict(nonboundary_LE3_rows=len(near),nonboundary_all_error_rows=len(all_nonboundary),
                             nonboundary_LE3_bases=len({r['base_index']for r in near}),nonboundary_all_bases=len({r['base_index']for r in all_nonboundary}))
        if not near:missing.append(key+'_NONBOUNDARY_LE3')
        seen=set();chosen=[]
        for tier,pool in(('NONBOUNDARY_LE3',near),('NONBOUNDARY_ALL_ERROR_FALLBACK',all_nonboundary)):
            for r in pool:
                if r['base_index']in seen or len(chosen)>=4:continue
                seen.add(r['base_index']);indices=[(r['base_index']*4+j)*8+r['native_slot']for j in range(4)]
                successes={m:sum(rows[j]['raw_state']in('D','E')and rows[j][m]['prediction']=={'D':0,'E':1}[rows[j]['raw_state']]for j in indices)for m in MODELS}
                chosen.append(dict(error_class=key,selection_tier=tier,global_index=r['global_index'],base_index=r['base_index'],native_slot=r['native_slot'],
                                   error_px=r['error_px'],foreign_family=r['factors']['foreign_family'],companion_indices=indices,samebase_known_correct_rows=successes))
        selected.extend(chosen)
    assert len(selected)<=8
    write(PACK,dict(completed=True,rows=rows,source_rows=es['scores'],C16_input_rows=cs['input_crop_facts'],GT_labels_unchanged=True,RGB_read=False))
    write(SELECTION,dict(completed=True,PASS=True,code=bind(__file__),helper=bind(HELPER),saved_rows=bind(PACK),control_complete=bind(control_path),
                         baseline_evaluation_seal=bind(espath),C16_prediction_seal=bind(cs_path),score_bindings=bindings,
                         cases=selected,maximum_case_groups=8,pool_counts=pool_counts,missing_categories=missing,full_cohort_counts=cohorts,
                         rule='For each union C24/C25 D→E andE→D: first4distinctbases inoriginalorder, NONBOUNDARY≤3 first, thenall-error NONBOUNDARY fallback; all4variants immutable. No RGB selection.',
                         source='SYNTHETIC_ONLY',actual_q_differs_across_variants=True,model_fit_private_V112_PnP=0,RGB_read=False))
    print(json.dumps(dict(selection=bind(SELECTION),cases=len(selected),pools=pool_counts,missing=missing)),flush=True)

def panel(case,ordinal,rows,inputs,pixels,h):
    from PIL import Image,ImageDraw
    image=Image.new('RGB',(1750,1420),'#f7f9fc');d=ImageDraw.Draw(image)
    heading={'D_FALSE_E':'보이는 점을 가렸다고 판단','E_FALSE_D':'가려진 점을 보인다고 판단'}[case['error_class']]
    h.text(d,(24,17),'합성 장면 · 실제 초기 추론 q · '+heading,30)
    h.text(d,(24,63),'기본 장면 %03d / 채널 %d / 선택행 %d / 오차 %.2fpx / %s'%(case['base_index'],case['native_slot'],case['global_index'],case['error_px'],case['foreign_family']),23)
    h.text(d,(24,104),'원래 q와 L을 유지했습니다. 네 장면의 q는 각각 달라지므로 같은 q를 고정한 대조가 아닙니다.',22,'#9c4500')
    h.text(d,(24,144),'노랑 + : 초기 q / 청록 + : 적용 convention reference. raw96에는 표시를 넣지 않았습니다.',22)
    h.text(d,(24,182),'선택 범위: '+case['selection_tier']+' | native 대응은 완전 인증되지 않았습니다.',21,'#53617a')
    for x,title in((24,'원본 장면 + 별도 표시'),(310,'저장 raw96 ×2'),(528,'별도 표시본'),(750,'원래 정답 정의·좌표·세 모델의 저장 출력')):h.text(d,(x,220),title,20)
    for vi,i in enumerate(case['companion_indices']):
        r=rows[i];inp=inputs[i];y=260+vi*270
        if i==case['global_index']:d.rounded_rectangle((14,y-6,1734,y+258),radius=8,outline='#c74c26',width=3)
        scene=Image.fromarray(pixels.scene(inp)).resize((230,230),Image.Resampling.BILINEAR);sd=ImageDraw.Draw(scene)
        q=inp['query_xy'];ref=r['reference_xy']
        if q is not None:h.cross(sd,[h.pixel_to_display(q[0],230/512),h.pixel_to_display(q[1],230/512)],'#ffe43b',7,3)
        if ref is not None:h.cross(sd,[h.pixel_to_display(ref[0],230/512),h.pixel_to_display(ref[1],230/512)],'#00e5e0',4,2)
        image.paste(scene,(24,y+10))
        original=pixels.crop(inp);raw=Image.fromarray(original).resize((192,192),Image.Resampling.NEAREST);image.paste(raw,(310,y+30))
        rawpath=H/'raw_crops'/('CASE%02d_%s_IDX%04d_RAW96.png'%(ordinal,r['variant'],i));Image.fromarray(original).save(rawpath)
        marked=raw.copy();md=ImageDraw.Draw(marked)
        if inp['crop_constructed']:
            h.cross(md,(95.5,95.5),'#ffe43b',11,4);local=h.reference_in_crop(q,ref,inp['L'])
            if local is not None:
                xy=[h.pixel_to_display(t,2)for t in local]
                if all(-10<=t<=201 for t in xy):h.cross(md,xy,'#00e5e0',7,2)
        image.paste(marked,(528,y+30))
        h.text(d,(750,y+3),'%s | teacher %s | error %spx | L %s | boundary %s'%(r['variant'],r['raw_state'],h.number(r['error_px'],2),h.number(inp['L'],2),r['convention_boundary']),22)
        h.text(d,(750,y+40),'형상 %s / 재질 %s / %s'%(r['factors']['geometry_family'],r['factors']['material'],r['factors']['aspect']),20)
        for mi,m in enumerate(MODELS):
            z=r[m];color='#53617a'
            if r['raw_state']in('D','E')and z['prediction']!=-1:color='#145d43'if z['prediction']=={'D':0,'E':1}[r['raw_state']]else'#9f3d31'
            h.text(d,(750,y+82+mi*34),'%s: %s | hidden %s | available %s'%(m,pred_name(z['prediction']),h.number(z['raw_logit']),z['available']),22,color)
        h.text(d,(750,y+192),'q %s / reference %s'%(h.coordinates(q),h.coordinates(ref)),20)
        h.text(d,(750,y+226),'적용 convention '+r['applied_convention']+' · source channel '+str(r['native_slot']),18,'#53617a')
    h.text(d,(24,1370),'D=보임, E=외부 가림, S=자기 가림, U=불확실. 점수는 hidden logit이며 0 이상이면 E. 현재 실사 결과가 아닙니다.',20,'#53617a')
    path=H/'figures'/('CASE%02d_%s_BASE%03d_SLOT%d.png'%(ordinal,case['error_class'],case['base_index'],case['native_slot']))
    image.save(path);return bind(path)

def render():
    selection=read(SELECTION);assert selection['completed']and selection['PASS']and selection['RGB_read']is False
    check(selection['code']);check(selection['helper']);check(selection['saved_rows'])
    packet=read(PACK);rows=packet['rows'];inputs=read(check(packet['C16_input_rows']))['rows'];assert len(rows)==len(inputs)==6144
    for folder in('figures','raw_crops'):(H/folder).mkdir()
    h=module(HELPER,'v134_reviewed_v132_display_helpers');pixels=h.SavedPixels()
    figures=[panel(c,i+1,rows,inputs,pixels,h)for i,c in enumerate(selection['cases'])]
    from PIL import Image,ImageDraw
    contacts=[]
    for errorclass in('D_FALSE_E','E_FALSE_D'):
        indices=[i for i,c in enumerate(selection['cases'])if c['error_class']==errorclass]
        if not indices:continue
        sheet=Image.new('RGB',(1750,1420),'#eef2f7')
        for j,i in enumerate(indices):
            im=Image.open(figures[i]['path']);im.thumbnail((875,710),Image.Resampling.LANCZOS);sheet.paste(im,((j%2)*875,(j//2)*710))
        path=H/'figures'/('CONTACT_'+errorclass+'.png');sheet.save(path);contacts.append(bind(path))
    result=dict(completed=True,PASS=True,selection=bind(SELECTION),saved_rows=bind(PACK),figures=figures,contacts=contacts,
                source='SYNTHETIC_ONLY',raw96_untouched=True,separate_display_marks=True,actual_model_fit_private_V112_PnP_calls=0,
                labels_edited=False,full_cohort_counts_unchanged=True,personal_visual_review_pending=True)
    write(H/'RENDER_COMPLETE.json',result)
    print(json.dumps(dict(completed=True,render=bind(H/'RENDER_COMPLETE.json'),figures=figures,contacts=contacts)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('select','render'));args=ap.parse_args()
    if args.stage=='select':select()
    else:render()
