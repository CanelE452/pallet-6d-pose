"""Fixed V230 SOURCE illustrations: saved q/r evidence, selected RGB only, no model."""
import argparse
import ast
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import traceback

import numpy as np

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation'
POLICY=R/'V230_ROOT_YCB_QUERY_DIAGNOSIS/VISUAL_CASE_POLICY_BEFORE_QUERY_OBSERVATIONS.json'
POLICY_SHA='26eb0b6057a86d30dd819c502135f0d4d5723da0d81aa44829c4aeeff6dc2c7b'
SIDECAR_CODE=R/'V228_YCB_TEACHER_MATH_A_V2/build_query_observations_v2.py'
SIDECAR_CODE_SHA='4912259ab5fcd782a0b13a9ceb63219984a607ed6e5f6e2e0d272eb26f1a2f14'
SIDECAR_PROTOCOL=R/'V228_ROOT_QUERY_OBSERVATION/PROTOCOL_BEFORE_TRAIN_GT_OR_PIXEL_ACCESS.json'
SIDECAR_PROTOCOL_SHA='9f3d265782b1b3fc295892e5137eb58d0f7526c08b585cfac941688ed8bfa19f'
SIDECAR_OUT=R/'V228_ROOT_QUERY_OBSERVATION/query_observations_v1'
INPUTS=P/'query_visibility_ycb_real_surface_source_20261023_v109/query_preparation/INPUTS1000_QUERY_EMULATION_ONLY.jsonl'
INPUTS_SHA='f06d1d2b2231835160c3c748c3269e518a460a694a37fcbddf7b00f5dfe166f2'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
CROP_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
RAW_ROOT=P/'query_visibility_public_real_20261022_v63/ycb/data/train_real'
OUT=R/'V230_ROOT_YCB_QUERY_DIAGNOSIS/fixed6_q_vs_r_gallery_v1'
BUCKETS=(('D','UNIQUE_OWN'),('D','UNIQUE_FOREIGN'),('D','UNASSIGNED'),('E','UNIQUE_FOREIGN'),('E','UNIQUE_OWN'),('E','UNASSIGNED'))


def safe(value):
    text=str(value)
    for word in ('v112','current223','publicheld','oldval','v177','v178','source_holdout','v182','common'):
        assert word not in text.casefold(),f'Forbidden path before filesystem: {word}'
    p=Path(os.path.abspath(text))
    for part in reversed((p,*p.parents)):
        assert not part.is_symlink(),f'Symlink: {part}'
    return p


def read(path):
    fd=os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:return f.read()


def sha(data):return hashlib.sha256(data).hexdigest()


def ident(b):return str(Path(os.path.abspath(b['path']))),b['sha256']


def checked(b,pin=None):
    if pin is not None:assert ident(b)==pin
    raw=read(b['path']);assert sha(raw)==b['sha256'],b['path']
    if 'size_bytes' in b:assert len(raw)==b['size_bytes']
    return raw


def binding(path):
    raw=read(path);return {'path':str(path),'sha256':sha(raw),'size_bytes':len(raw)}


def write(path,value):
    with safe(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def choose(rows):
    """Prospective policy: <=3 pool first, unused frame then minimum slot; no pixels."""
    used=set();cases=[]
    for bucket,(state,owner) in enumerate(BUCKETS):
        pool=[r for r in rows if r['original_teacher_immutable']['raw_surface_teacher_state']==state and r['q_observation_not_semantic_target']['state']==owner]
        assert all(0<=r['constructed_jitter_distance_px']<=4 for r in pool)
        preferred=[r for r in pool if r['constructed_jitter_distance_px']<=3]
        eligible=preferred if preferred else pool
        if not eligible:
            cases.append({'bucket':bucket,'state':state,'q_owner':owner,'missing':True,'slot_index':None});continue
        unused=[r for r in eligible if r['frame_index'] not in used]
        row=min(unused if unused else eligible,key=lambda r:r['slot_index'])
        cases.append({'bucket':bucket,'state':state,'q_owner':owner,'missing':False,'slot_index':row['slot_index'],'frame_index':row['frame_index'],'fallback_distance_gt3':not bool(preferred),'reused_frame':row['frame_index'] in used})
        used.add(row['frame_index'])
    return cases


def local_address(reference,q,L):
    r=np.asarray(reference,np.float64);q=np.asarray(q,np.float64)
    assert r.shape==q.shape==(2,) and np.isfinite(r).all() and np.isfinite(q).all() and np.isfinite(L) and L>0
    return (r-q)/L*96+47.5


def crop_operator():
    raw=checked({'path':str(CROP),'sha256':CROP_SHA})
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='crop')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(CROP),'exec'),ns)
    return ns['crop']


def mark_local(ax,r_xy,q_xy=(47.5,47.5),extent=(-.5,95.5)):
    lo,hi=extent;ax.plot(*q_xy,'+',color='cyan',ms=15,mew=2)
    r=np.asarray(r_xy,float)
    if np.all((r>=lo)&(r<=hi)):ax.plot(*r,'x',color='magenta',ms=12,mew=2)
    else:
        edge=np.clip(r,lo+2,hi-2)
        ax.annotate('r outside',xy=edge,xytext=(.04,.04),textcoords='axes fraction',color='magenta',arrowprops={'arrowstyle':'->','color':'magenta'})
    ax.set_xlim(lo,hi);ax.set_ylim(hi,lo)


def diagram(ax,record,original,own):
    prefix='neighbour_' if original else ''
    keys=('inbounds','depth_raw_uint16','own_full','own_visible','visible_owner_count','owner_bits_uint64') if original else ('inbounds9','rawdepth9','ownfull9','ownvis9','ownercount9','ownerbits9')
    if original:
        keylist=['neighbour_inbounds','neighbour_depth_raw_uint16','neighbour_own_full','neighbour_own_visible','neighbour_visible_owner_count','neighbour_owner_bits_uint64']
    else:keylist=list(keys)
    inside,raw,full,visible,count,bits=[np.asarray(record[k]).reshape(9) for k in keylist]
    colors=[]
    for i in range(9):
        colors.append((.05,.05,.05) if not inside[i] else (1,.65,.15) if count[i]>1 else (.55,.55,.55) if count[i]==0 else (.25,.75,.35) if visible[i] else (.65,.4,.85))
        ids=[str(j) for j in range(64) if int(bits[i])&(1<<j)]
        ax.text(i%3,i//3,f'owner={"/".join(ids) or "none"}\nzraw={int(raw[i])}\nfull={int(full[i])}',ha='center',va='center',fontsize=9,color='white' if not inside[i] else 'black')
    ax.imshow(np.asarray(colors).reshape(3,3,3),extent=(-.5,2.5,2.5,-.5),interpolation='nearest')
    ax.set_xticks(range(3),[-1,0,1]);ax.set_yticks(range(3),[-1,0,1]);ax.set_title(('saved r' if original else 'saved q')+f' nearest 3x3; own instance={own}\nmask owner / raw depth / own-full support',fontsize=11)


def render(path,rgb,crop,valid,row):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    q=np.asarray(row['q'],float);r=np.asarray(row['r_GT_teacher_only'],float);L=float(row['L_input']);local=local_address(r,q,L)
    fig=plt.figure(figsize=(18,11),dpi=130);gs=fig.add_gridspec(2,3)
    ax=fig.add_subplot(gs[:,0]);ax.imshow(rgb);ax.plot(*q,'+',color='cyan',ms=14,mew=2);ax.plot(*r,'x',color='magenta',ms=12,mew=2);ax.add_patch(Rectangle(q-L/2,L,L,fill=False,edgecolor='cyan',linewidth=1.3));ax.set_title('Original SOURCE RGB\ncyan q + unchanged L; magenta teacher r',fontsize=12)
    ax=fig.add_subplot(gs[0,1]);ax.imshow(crop.transpose(1,2,0));mark_local(ax,local);ax.set_title(f'Original q/L RAW96; valid={int(valid.sum())}/9216\nr address=({local[0]:.3f},{local[1]:.3f})',fontsize=11)
    ax=fig.add_subplot(gs[0,2]);ax.imshow(crop[:,32:64,32:64].transpose(1,2,0));mark_local(ax,local-32,(15.5,15.5),(-.5,31.5));ax.set_title('Centre32 of the same original96\nNo r recrop or query correction',fontsize=11)
    diagram(fig.add_subplot(gs[1,1]),row['original_teacher_immutable'],True,row['source_instance_id'])
    diagram(fig.add_subplot(gs[1,2]),row['q_observation_not_semantic_target'],False,row['source_instance_id'])
    old=row['original_teacher_immutable'];state=old['raw_surface_teacher_state'];qowner=row['q_observation_not_semantic_target']['state']
    fig.suptitle(f"slot {row['slot_index']} | {row['input_id']} aux-slot {row['logical_aux_slot']} | associated-r {state} | literal-q {qowner} | jitter {row['constructed_jitter_distance_px']:.4f}px",fontsize=14)
    reason=old['reason'];fig.text(.02,.015,f'{reason}\nNative mapping UNKNOWN; constructed jitter is not actual initializer error. Saved r/q owner/depth are teacher-side diagnostics, not deployment inputs.\nMask/depth absence or foreign q pixels do not create semantic E. This display does not certify physical part identity or label errors.',fontsize=10)
    fig.subplots_adjust(top=.90,bottom=.12,wspace=.25,hspace=.30)
    fd=os.open(safe(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as f:fig.savefig(f,format='png')
    plt.close(fig)


def run(entry_path):
    eraw=read(entry_path);e=json.loads(eraw)
    assert e['approved_saved_observation_visuals'] is True and e['selected_source_RGB_only'] is True
    assert e['no_model_training_PnP_or_depth_mask_read'] is True
    checked(e['code'],(str(safe(__file__)),binding(__file__)['sha256']))
    policy=json.loads(checked(e['policy'],(str(POLICY),POLICY_SHA)));assert tuple(map(tuple,policy['buckets']))==BUCKETS
    complete=json.loads(checked(e['sidecar_complete'],(str(SIDECAR_OUT/'COMPLETE.json'),e['sidecar_complete']['sha256'])))
    assert complete['completed'] is True and complete['PASS'] is True and complete['all8000_preserved'] is True and complete['old_labels_and_input_q_L_unchanged'] is True
    assert ident(complete['code'])==(str(SIDECAR_CODE),SIDECAR_CODE_SHA) and ident(complete['protocol'])==(str(SIDECAR_PROTOCOL),SIDECAR_PROTOCOL_SHA)
    assert safe(e['output_directory'])==OUT and not OUT.exists();OUT.mkdir()
    write(OUT/'STARTED.json',{'entry':{'path':str(entry_path),'sha256':sha(eraw)},'code':e['code'],'policy':e['policy'],'automatic_retry':False})
    try:
        rows=[json.loads(line) for line in checked(complete['outputs']['QUERY_OBSERVATIONS8000.jsonl']).splitlines() if line]
        assert len(rows)==8000 and [r['slot_index'] for r in rows]==list(range(8000))
        assert all(r['native_mapping']=='UNKNOWN' and r['training_admission'] is False for r in rows)
        inputs=[json.loads(line) for line in checked(complete['inputs']['inputs1000'],(str(INPUTS),INPUTS_SHA)).splitlines() if line]
        assert len(inputs)==1000 and all(x['split']=='TRAIN' and x['native_mapping']=='UNKNOWN' for x in inputs)
        cases=choose(rows)
        # The ID receipt is written before the first RGB path stat/hash/decode.
        write(OUT/'FIXED6_CASE_IDS_BEFORE_RGB.json',{'completed':True,'policy':e['policy'],'sidecar_complete':e['sidecar_complete'],'cases':cases,'present_cases':sum(not c['missing'] for c in cases),'RGB_hash_or_decode_before_selection':0})
        crop_fn=crop_operator();facts=[];images=[]
        from PIL import Image,__version__ as pillow_version
        for case in cases:
            if case['missing']:facts.append(case);continue
            row=rows[case['slot_index']];inp=inputs[row['instance_row_index']]
            assert row['input_id']==inp['input_id'] and row['instance_row_index']==inp['instance_row_index']
            assert np.array_equal(row['q'],inp['initial_points'][row['logical_aux_slot']]) and row['L_input']==inp['roi_side']
            image_binding=inp['image'];p=safe(image_binding['path']);assert p.is_relative_to(RAW_ROOT) and p.parent.name=='rgb'
            encoded=checked(image_binding);rgb=np.array(Image.open(io.BytesIO(encoded)).convert('RGB'),dtype=np.uint8)
            assert rgb.shape==tuple(inp['raw_hw'])+(3,)
            local,valid=crop_fn(rgb,np.asarray(row['q'],np.float64),float(row['L_input']),96)
            assert local.shape==(3,96,96) and local.dtype==np.uint8 and valid.shape==(1,96,96) and valid.dtype==bool
            path=OUT/f"CASE_{case['bucket']:02d}_SLOT{row['slot_index']:05d}.png";render(path,rgb,local,valid,row)
            record=dict(case,source_row=row,source_RGB_binding=image_binding,source_raw_hw=inp['raw_hw'],new_decoded_RGB_sha256=sha(np.ascontiguousarray(rgb).tobytes()),historical_decoded_RGB_SHA_claimed=False,decoder={'Pillow':pillow_version,'numpy':np.__version__},local96_decoded_CHW_sha256=sha(local.tobytes()),valid96_bool_sha256=sha(valid.tobytes()),local96_reference_address=local_address(row['r_GT_teacher_only'],row['q'],row['L_input']).tolist(),panel=binding(path))
            facts.append(record);images.append(record['panel'])
        write(OUT/'CASE_DATA.json',{'completed':True,'cases':facts,'unique_source_frames':len({x['frame_index'] for x in cases if not x['missing']}),'new_native_labels':0,'training_admission':False})
        fields=('bucket','state','q_owner','missing','slot_index','frame_index','fallback_distance_gt3','reused_frame')
        with safe(OUT/'CASES.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:x.get(k) for k in fields} for x in cases)
        title='SOURCE q / r 고정 사례: 원 라벨과 literal q 관측 분리'
        sections=[]
        for x in facts:
            label=f"bucket {x['bucket']}: {x['state']} × {x['q_owner']}"
            sections.append('<section><h2>'+html.escape(label)+'</h2>'+('<p>고정 조건에 해당하는 사례 없음. 대체 없음.</p>' if x['missing'] else '<img style="width:100%;height:auto" src="'+html.escape(Path(x['panel']['path']).name)+'"><p>slot '+str(x['slot_index'])+'; 원 q/L crop 유지. Magenta r는 teacher 참조 표시입니다.</p>')+'</section>')
        page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+title+'</title><body style="max-width:1600px;margin:auto;font-family:sans-serif"><h1>'+title+'</h1><p>최대 6개 사전 고정 bucket입니다. Native mapping은 UNKNOWN이며, q는 GT 참조 r의 기존 구성 jitter입니다. q owner / depth는 배포 입력이나 새 D/E 정답이 아닙니다. 원 RGB를 r로 recrop하지 않았습니다. 관측 차이는 오라벨·모델 원인 증명이 아닙니다.</p>'+''.join(sections)+'<p><a href="CASES.csv">선정표</a> · <a href="CASE_DATA.json">원 저장 근거와 새 decoded RGB SHA</a> · <a href="FIXED6_CASE_IDS_BEFORE_RGB.json">RGB 읽기 전 고정 ID</a></p></body></html>'
        with safe(OUT/'PORTABLE_GALLERY_KO.html').open('x') as f:f.write(page)
        outputs={name:binding(OUT/name) for name in ('FIXED6_CASE_IDS_BEFORE_RGB.json','CASES.csv','CASE_DATA.json','PORTABLE_GALLERY_KO.html')}
        write(OUT/'COMPLETE.json',{'completed':True,'PASS':True,'entry':{'path':str(entry_path),'sha256':sha(eraw)},'code':e['code'],'policy':e['policy'],'sidecar_complete':e['sidecar_complete'],'outputs':outputs,'panels':images,'present_cases':len(images),'missing_buckets':[x['bucket'] for x in cases if x['missing']],'selected_original_RGB_decode_calls':len(images),'raw_depth_mask_scene_GT_camera_NPZ_model_forward_fit_PnP_reads_or_calls':0,'native_mapping':'UNKNOWN','new_labels_or_training_admission':False,'original_q_L_and_associated_r_labels_unchanged':True})
        print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(OUT/'COMPLETE.json')}),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise


def self_test():
    def row(slot,state,owner,dist,frame):return {'slot_index':slot,'frame_index':frame,'constructed_jitter_distance_px':dist,'original_teacher_immutable':{'raw_surface_teacher_state':state},'q_observation_not_semantic_target':{'state':owner}}
    cases=choose([row(0,'D','UNIQUE_OWN',3.1,0),row(7,'D','UNIQUE_OWN',2.9,1),row(1,'D','UNIQUE_FOREIGN',1.,1),row(8,'D','UNIQUE_FOREIGN',2.,2),row(4,'E','UNIQUE_FOREIGN',3.5,4)])
    assert cases[0]['slot_index']==7 and cases[1]['slot_index']==8 and cases[2]['missing'] and cases[3]['fallback_distance_gt3']
    assert np.array_equal(local_address([10,20],[10,20],96),[47.5,47.5])
    assert np.array_equal(local_address([11,18],[10,20],96),[48.5,45.5])
    crop_fn=crop_operator();rgb=np.arange(15*17*3,dtype=np.uint8).reshape(15,17,3);a,v=crop_fn(rgb,np.array([8.,7.]),10.,96)
    assert a.shape==(3,96,96) and a.dtype==np.uint8 and v.shape==(1,96,96) and v.dtype==bool
    print(json.dumps({'completed':True,'PASS':True,'synthetic_arrays_only':True,'actual_RGB_sidecar_payload_model_reads':0,'checks':['fixed bucket <=3 priority before <=4 fallback','unused frame then minimum slot','missing bucket retained','samecrop q47.5 and exact r algebra','pinned pure V90 crop CHW96/valid1HW types']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--entry');g.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:self_test()
    else:run(args.entry)
