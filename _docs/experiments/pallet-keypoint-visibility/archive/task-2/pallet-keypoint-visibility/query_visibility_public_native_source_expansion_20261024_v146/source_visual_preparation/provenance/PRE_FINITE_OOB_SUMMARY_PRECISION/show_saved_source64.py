"""Inert source-public contact sheets: original q/L96 + provided-pose lower roles.

Only --entry reads completed new source data. No learned model or label inference.
"""
from pathlib import Path
import argparse, ast, hashlib, html, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
CROP_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
NFRAMES=64; NSLOTS=512
ROLES=['FRONT_TOP_LEFT','FRONT_TOP_RIGHT','FRONT_BOTTOM_RIGHT','FRONT_BOTTOM_LEFT','REAR_TOP_LEFT','REAR_TOP_RIGHT','REAR_BOTTOM_RIGHT','REAR_BOTTOM_LEFT']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p): return json.loads(Path(p).read_text())
def check(b):
    assert sha(b['path'])==b['sha256'],b['path']
    return Path(b['path'])
def lines(p): return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float):return x if np.isfinite(x)else None
    if isinstance(x,(list,tuple)):return[clean(y)for y in x]
    return x
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def finite_xy(value):
    if value is None:return False
    try:return np.asarray(value,dtype=np.float64).shape==(2,)and bool(np.isfinite(np.asarray(value,dtype=np.float64)).all())
    except (TypeError,ValueError):return False
def write(p,obj):
    with Path(p).open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def complete(binding):
    p=check(binding);s=read(p);assert s['completed']and s.get('PASS',True);return s
def contains_binding(obj,binding):
    if isinstance(obj,dict):
        if obj.get('path')==binding['path']and obj.get('sha256')==binding['sha256']:return True
        return any(contains_binding(x,binding)for x in obj.values())
    if isinstance(obj,list):return any(contains_binding(x,binding)for x in obj)
    return False
def crop_operator():
    assert sha(CROP)==CROP_SHA
    nodes=[n for n in ast.parse(CROP.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='crop'];assert len(nodes)==1
    ns=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CROP)+'::pure_crop_only','exec'),ns);return ns['crop']

def prepare(policy):
    policy=Path(policy).resolve();assert policy.parent==V
    assert sha(policy)=='592ab0be3c8635cb75cbbd93c3457daf463892e7dfe34c5029c63213746d247e'
    # Metadata/code only. No new RGB/q/reference/pose/annotation body is opened.
    protocol=dict(completed=True,code=bind(__file__),policy=bind(policy),frames=64,slots=512,crop_operator=bind(CROP),
        input_contract='Completed canonical RGB transport manifest + actual-original initializer seal + provided-pose reference seal. All64 firstargmax and native8 q/L retained.',
        output_contract='64 fullscene raw/marked+8 untouched96 crops;512 input/crop facts and UNREVIEWED U templates;8 fixed input-order overviews;completed contact seal.',
        semantic_contract='Only lower2/3 whole-assembly virtual footprint roles are displayed as source reference. No automatic D/E/S label; input noquery/OOB separate from teacher. No nearest/permutation/q-L correction.',
        labels_are_assistant_weak_not_human_gold=True,full_visibility_trainready=False,current_model_fit_usage=0,
        RGB_model_query_pose_current_V112_body_reads_during_preparation=0)
    write(H/'PROTOCOL_BEFORE_CONTACT_RENDER.json',protocol)
    template=dict(root_authorized=True,policy=bind(policy),code=bind(__file__),protocol=bind(H/'PROTOCOL_BEFORE_CONTACT_RENDER.json'),
        RGB_transport_seal=None,RGB_manifest=None,initializer_prediction_seal=None,
        source_projection_seal=None,source_reference_manifest=None,output_subdir='saved_source64_contact_v1')
    write(H/'ROOT_ENTRY_TEMPLATE.json',template)
    print(json.dumps(dict(protocol=bind(H/'PROTOCOL_BEFORE_CONTACT_RENDER.json'),template=bind(H/'ROOT_ENTRY_TEMPLATE.json'))))

def run(entrypath):
    e=read(entrypath);assert e['root_authorized']is True
    assert check(e['code'])==Path(__file__).resolve();check(e['policy']);p=read(check(e['protocol']))
    assert p['code']==e['code']and p['policy']==e['policy']and p['frames']==64 and p['slots']==512
    transport=complete(e['RGB_transport_seal']);init=complete(e['initializer_prediction_seal']);projection=complete(e['source_projection_seal'])
    assert contains_binding(transport,e['RGB_manifest'])and contains_binding(projection,e['source_reference_manifest'])
    # The source projection must bind the same completed actual initializer.
    assert contains_binding(projection,e['initializer_prediction_seal'])
    rgbrows=read(check(e['RGB_manifest']))['rows'];refs=read(check(e['source_reference_manifest']))['rows']
    op={Path(b['path']).name:b for b in init['outputs']}
    props=lines(check(op['FROZEN_PROPOSALS_ALL64_INPUT_ONLY.jsonl']))
    with np.load(check(op['ORIGINAL_NATIVE_QUERY512_AND_VALID9X9_INPUT_ONLY.npz']),allow_pickle=False)as f:a={k:f[k].copy()for k in f.files}
    assert len(rgbrows)==len(props)==64 and len(refs)==512
    assert a['query_xy'].shape==(512,2)and a['L'].shape==(512,)and a['native_grid_valid'].shape==(512,9,9)
    assert len({r['id']for r in props})==len({r['source_id']for r in props})==64
    assert [r['frame_index']for r in props]==list(range(64))
    sequences=['0.55m_lab','1.23m_lab','0.55m_hall','1.23m_hall']
    assert {s:sum(r['source_id'].split(':')[0]==s for r in props)for s in sequences}=={s:16 for s in sequences}
    assert sum(r['split']=='TRAIN_LAB'for r in props)==sum(r['split']=='VAL_HALL'for r in props)==32
    assert list(a['global_index'])==list(range(512))
    assert list(a['frame_index'])==[j//8 for j in range(512)]and list(a['native_slot'])==[j%8 for j in range(512)]
    reference_by_key={}
    for r in refs:
        key=(r['frame_index'],r['native_slot']);assert key not in reference_by_key
        assert 0<=key[0]<64 and 0<=key[1]<8 and r['semantic_state']=='U';reference_by_key[key]=r
    assert len(reference_by_key)==512
    for i,(r,t)in enumerate(zip(props,rgbrows)):
        assert r['frame_index']==t['frame_index']==i and r['id']==t['id']and r['source_id']==t['source_id']
        assert r['split']==t['split']and r['split']in('TRAIN_LAB','VAL_HALL')
        assert r['source']=='PALLOX'
        selected=None if not r['candidates']else int(np.argmax([x['score']for x in r['candidates']]))
        assert selected==r['selected_index']
        for k in range(8):
            rr=reference_by_key[i,k];assert rr['id']==r['id']and rr['source_frame_id']==r['source_id']
            assert clean(a['query_xy'][i*8+k])==(None if selected is None else r['candidates'][selected]['keypoints_xy'][k])or (selected is None and np.isnan(a['query_xy'][i*8+k]).all())
    crop=crop_operator();out=V/e['output_subdir'];assert out.parent==V and not out.exists();out.mkdir()
    facts=[];sheets=[];marked_thumbs=[]
    for i,(r,t)in enumerate(zip(props,rgbrows)):
        rgb=np.load(check(t['canonical_RGB_NPY']),allow_pickle=False)
        assert rgb.dtype==np.uint8 and list(rgb.shape)==r['raw_hw']+[3]
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['decoded_RGB_sha256']==t['decoded_RGB_sha256']
        raw=Image.fromarray(rgb);marked=raw.copy();d=ImageDraw.Draw(marked);hh,ww=rgb.shape[:2]
        if r['selected_index']is not None:d.rectangle(r['candidates'][r['selected_index']]['box_xyxy'],outline=(255,145,0),width=2)
        crs=[];vs=[];statuses=[]
        for k in range(8):
            j=i*8+k;q=a['query_xy'][j];L=a['L'][j];rr=reference_by_key[i,k]
            constructed=bool(a['finite_non_sentinel'][j]and np.isfinite(L)and L>0)
            if constructed:cr,va=crop(rgb,q,L,96);cr=cr.transpose(1,2,0);va=va[0]
            else:cr=np.zeros((96,96,3),np.uint8);va=np.zeros((96,96),bool)
            crs.append(cr);vs.append(va)
            if np.isfinite(q).all()and 0<=q[0]<ww and 0<=q[1]<hh:
                x,y=q;d.ellipse((x-3,y-3,x+3,y+3),outline=(255,0,210),width=2)
                d.text((min(max(x+4,0),ww-24),min(max(y,0),hh-24)),str(k),font=font(17),fill=(255,0,210),stroke_width=1,stroke_fill='black')
            if k in (2,3):
                ref=rr['reference_xy'];statuses.append(f'n{k}:{rr["lower_role_continuity_status"]}')
                if finite_xy(ref)and 0<=ref[0]<ww and 0<=ref[1]<hh:
                    x,y=ref;d.line((x-5,y,x+5,y),fill=(0,255,255),width=2);d.line((x,y-5,x,y+5),fill=(0,255,255),width=2)
                    d.text((min(max(x+4,0),ww-36),min(max(y-18,0),hh-22)),f'R{k}',font=font(15),fill=(0,255,255),stroke_width=1,stroke_fill='black')
            facts.append(dict(global_index=j,frame_index=i,id=r['id'],source_id=r['source_id'],split=r['split'],native_slot=k,native_role=ROLES[k],
                query_xy=clean(q),L=clean(L),selected_index=r['selected_index'],no_detection=bool(a['no_detection'][j]),
                original_input_valid=bool(a['continuous_valid'][j]),finite_non_sentinel=bool(a['finite_non_sentinel'][j]),
                nearest_valid=bool(a['nearest_valid'][j]),RGBcentre_valid=bool(a['RGBcentre_valid'][j]),native_grid_any_valid=bool(a['native_grid_valid'][j].any()),
                crop_constructed=constructed,rgb96_SHA256=hashlib.sha256(cr.tobytes()).hexdigest(),valid96_SHA256=hashlib.sha256(va.tobytes()).hexdigest(),crop_valid_fraction=float(va.mean()),
                source_reference_xy=rr['reference_xy']if k in(2,3)else None,lower_role_continuity_status=rr['lower_role_continuity_status']if k in(2,3)else 'NOT_IN_LOWER_BRIDGE_SCOPE',
                semantic_state='U',semantic_reason_KO='아직 실제 개체/역할/가림을 검토하지 않은 초기 틀입니다. U는 negative가 아닙니다.',assistant_reviewed=False,
                input_availability_and_teacher_separate=True,original_q_L_firstargmax_unchanged=True,upper_or_rear_label_autofill=False,
                human_gold=False,precise_physical_vertex_gold=False,trainready=False,training_now=False))
        packet=out/f'FRAME_{i:02d}_FIXED96.npz';np.savez_compressed(packet,rgb96=np.stack(crs),valid96=np.stack(vs),global_index=np.arange(i*8,i*8+8,dtype=np.int32))
        width=960;height=round(hh*width/ww);canvas=Image.new('RGB',(1920,145+height+285),'white');cd=ImageDraw.Draw(canvas)
        cd.text((12,8),r['id']+f' | {r["split"]} | original firstargmax={r["selected_index"]} | candidates={len(r["candidates"])}',font=font(22),fill='black')
        cd.text((12,42),'PUBLIC SOURCE: left raw; right original q8 magenta / selected box orange / supplied-pose LOWER2/3 cyan.',font=font(21),fill='black')
        cd.text((12,73),'Cyan is a limited whole-assembly virtual-role reference, not physical pixel gold or automatic D/E/S.',font=font(20),fill='black')
        cd.text((12,104),'Below untouched factual96, NEAREST x2. Noquery zero placeholder is not E. '+', '.join(statuses),font=font(18),fill='black')
        canvas.paste(raw.resize((width,height),Image.Resampling.NEAREST),(0,145));canvas.paste(marked.resize((width,height),Image.Resampling.NEAREST),(960,145))
        for k in range(8):
            x=k*240;top=155+height;rr=facts[i*8+k]
            cd.text((x+4,top),f'native{k} | inputvalid={rr["original_input_valid"]}',font=font(17),fill='black')
            cd.text((x+4,top+24),'L='+str(None if rr['L']is None else round(rr['L'],2)),font=font(17),fill='black')
            canvas.paste(Image.fromarray(crs[k]).resize((192,192),Image.Resampling.NEAREST),(x+24,top+52))
        fig=out/f'FRAME_{i:02d}_WHOLESCENE_AND_FIXED96.png';canvas.save(fig)
        sheets.append(dict(frame_index=i,id=r['id'],sheet=bind(fig),crop_packet=bind(packet)))
        marked_thumbs.append(marked.copy())
    assert len(facts)==512
    overview=[]
    for start in range(0,64,8):
        ov=Image.new('RGB',(1280,2120),'white');d=ImageDraw.Draw(ov)
        for off in range(8):
            i=start+off;x=(off%2)*640;y=(off//2)*530
            d.text((x+5,y+3),f'{i:02d} {props[i]["source_id"]} / {props[i]["split"]}',font=font(18),fill='black')
            ov.paste(marked_thumbs[i].resize((640,480),Image.Resampling.NEAREST),(x,y+42))
        opath=out/f'OVERVIEW_{start//8:02d}_FRAMES_{start:02d}_TO_{start+7:02d}.png';ov.save(opath);overview.append(bind(opath))
    factpath=out/'ALL512_CROP_FACTS_AND_UNREVIEWED_TEMPLATE.jsonl'
    with factpath.open('x')as f:
        for r in facts:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    index=out/'INDEX.html'
    index.write_text('<!doctype html><meta charset="utf-8"><title>V146 public source64 original query crops</title><h1>공개 원본64장 · 원 q/L crop512개</h1><p>하부2/3 가상 역할 참조는 제한된 source 근거입니다. 자동 D/E 판정·사람 gold·학습 승인 상태가 아닙니다.</p>'+''.join(f'<p><a href="{html.escape(Path(s["sheet"]["path"]).name)}">{html.escape(s["id"])}</a></p>'for s in sheets))
    result=dict(completed=True,PASS=True,frames=64,slots=512,root_entry=bind(entrypath),code=bind(__file__),protocol=e['protocol'],
        policy=e['policy'],RGB_transport_seal=e['RGB_transport_seal'],initializer_prediction_seal=e['initializer_prediction_seal'],source_projection_seal=e['source_projection_seal'],
        crop_operator=bind(CROP),all512_crop_facts=bind(factpath),sheets=sheets,overviews=overview,index=bind(index),
        original_firstargmax_native8_q_L_unchanged=True,source_references_original_unchanged=True,all512_semantic_U_UNREVIEWED=True,
        input_valid=sum(r['original_input_valid']for r in facts),noquery=sum(r['no_detection']for r in facts),finite_OOB=sum(not r['original_input_valid']and not r['no_detection']for r in facts),
        source_RGB_only=True,RGB96_untouched=True,GT_center_or_nearest_permutation_or_q_L_correction=False,
        model_initializer_encoder_head_fit_current223_V112_PnP_calls_or_reads=0,human_gold=False,trainready=False,training_now=False)
    write(out/'CONTACT_RENDER_COMPLETE.json',result);print(json.dumps(bind(out/'CONTACT_RENDER_COMPLETE.json')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',metavar='ROOT_POLICY');g.add_argument('--entry');args=ap.parse_args()
    if args.prepare:prepare(args.prepare)
    else:run(args.entry)
