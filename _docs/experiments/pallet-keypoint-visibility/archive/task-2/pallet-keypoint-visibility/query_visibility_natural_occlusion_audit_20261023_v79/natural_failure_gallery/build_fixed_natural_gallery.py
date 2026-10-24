"""Fixed source-only natural VAL four-cell gallery from saved outputs.

select stage freezes IDs/code before RGB or native mask/cached crop bodies.
render stage uses unchanged V76 input crops and verifies native RGB replay.
No model, backbone, GT repair, new teacher or current-evaluation access.
"""
from pathlib import Path
import argparse, ast, csv, hashlib, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
V76=P/'query_visibility_reference_local_support_20261023_v76'
V77=P/'query_visibility_controlled_pairs_20261023_v77'
V78=P/'query_visibility_dino_query_support_20261023_v78'
ENTRY=ROOT/'AUDIT_ENTRY.json'
ENTRY_SHA='bdefc2af54e4e9e38eb1dc312116ad230421a9e47eadfa4810b85b5b2b3987b6'
SALT='V79_NATURAL_STRICT_FOUR_CELLS_V1'
COHORT_NAME='STRONG_AND_ERROR_LE3PX_AND_CENTER_CELL'
COHORT=V77/'source_failure_audit/FIXED_COHORT_IDS.csv'
GT=V76/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz'
META=V76/'gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl'
MASKAUDIT=V76/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl'
QUERIES=V76/'public_query_emulation/INPUTS_RGB_ONLY.jsonl'
MANIFEST=V76/'public_input_pack/PACK_MANIFEST.json'
SCORES=dict(V76=V76/'mixed_fit/runs/MIXED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz',
    V77=V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz',
    V78=V78/'runs/SOURCE_DINO_V1/SELECTED_SOURCE_SCORES.npz')
CROP_CODE=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(binding):assert sha(binding['path'])==binding['sha256'],binding['path']
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def write(p,d):
    with p.open('x') as f:json.dump(clean(d),f,indent=2,allow_nan=False);f.write('\n')
def sigmoid(z):return float(1/(1+np.exp(-float(z))))
def validate():
    assert sha(ENTRY)==ENTRY_SHA
    entry=load(ENTRY);check(entry['decisions']);check(entry['prior_report'])
    decision=load(entry['decisions']['path'])
    assert decision['natural_failure_gallery']['scope'].startswith('YCBnaturalVAL')
    for binding in decision['bindings'].values():check(binding)
    completed=load(decision['bindings']['source_result']['path'])
    assert completed['completed'] is True;check(completed['scores'])
    assert completed['scores']==b(SCORES['V78'])
    oldcohort=load(V77/'source_failure_audit/PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.json')
    assert any(x==b(COHORT) for x in oldcohort['bindings'])
    corpusseal=load(V76/'gt_corpus_build/GT_ONLY_CORPUS_SEAL.json')
    assert any(x['path']==str(GT) and x['sha256']==sha(GT) for x in corpusseal['outputs'])
    return entry

def select():
    validate()
    assert not (OUT/'SELECTION_LOCK_BEFORE_RGB.json').exists()
    with GT.open('rb') as _:
        pass
    with np.load(GT,allow_pickle=False) as z:
        g={k:z[k] for k in ('id','source','split','label8','native_GTref8','rule8',
            'convex_self_proxy8_separate','center_M08','center_M48','rho_px','near_boundary8')}
    queries=lines(QUERIES);index=lines(META)
    qmap={x['id']:x for x in queries}
    imap={str(x):i for i,x in enumerate(g['id'])}
    with COHORT.open() as f:rows=[r for r in csv.DictReader(f) if r['cohort']==COHORT_NAME]
    assert len(rows)==535 and len({(r['id'],r['native_k']) for r in rows})==535
    saved={}
    for name,path in SCORES.items():
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z['id'][:5203],g['id'])
            assert np.array_equal(z['source'][:5203],g['source']) and np.array_equal(z['split'][:5203],g['split'])
            saved[name]=dict(logit=z['support_logit'][:5203],primary=z['primary_loss_mask'][:5203])
    assert np.array_equal(saved['V76']['primary'],saved['V77']['primary']) and np.array_equal(saved['V76']['primary'],saved['V78']['primary'])
    pool=[]
    for r in rows:
        i=imap[r['id']];k=int(r['native_k']);qrow=qmap[r['id']]
        assert g['source'][i]=='YCB_PUBLIC_REAL' and g['split'][i]=='val'
        assert saved['V78']['primary'][i,k]
        q=np.asarray(qrow['initial_points'],float)[k];ref=g['native_GTref8'][i,k];L=float(qrow['roi_side'])
        error=float(np.linalg.norm(ref-q));linf=float(np.max(abs((ref-q)/L)))
        assert str(g['rule8'][i,k]).startswith('STRONG_3X3') and error<=3 and linf<=1/18
        y=int(g['label8'][i,k]);z=float(saved['V78']['logit'][i,k]);pred=int(z>=0)
        assert y in (0,1)
        digest=hashlib.sha256((SALT+'\n'+r['id']+'\n'+str(k)).encode()).hexdigest()
        scores={name:dict(logit=float(s['logit'][i,k]),p_visible=sigmoid(s['logit'][i,k]),prediction='VISIBLE' if s['logit'][i,k]>=0 else 'COVERED') for name,s in saved.items()}
        pool.append(dict(id=r['id'],native_k=k,corpus_row=i,public_input_index=i-1003,
            GT_label=y,V78_prediction=pred,selection_SHA256=digest,
            q=q.tolist(),r=ref.tolist(),L=L,box=qrow['box'],error_px=error,
            error_over_L_Linf=linf,rule=str(g['rule8'][i,k]),self_visible_proxy_separate=int(g['convex_self_proxy8_separate'][i,k]),
            center_M0_at_reference=int(g['center_M08'][i,k]),center_M4_at_reference=int(g['center_M48'][i,k]),
            rho=float(g['rho_px'][i]),near_boundary=bool(g['near_boundary8'][i,k]),scores=scores,
            source_RGB=dict(path=qrow['image'],sha256=qrow['image_sha256']),
            source_instance=dict(scene_id=index[i]['scene_id'],frame_id=index[i]['frame_id'],instance_id=index[i]['instance_id'],object_id=index[i]['object_id'])))
    assert sum(r['GT_label']==0 for r in pool)==76 and sum(r['GT_label']==1 for r in pool)==459
    cells=[]
    for y,pred,name in [(0,1,'GT_COVERED_PRED_VISIBLE'),(0,0,'GT_COVERED_PRED_COVERED'),(1,1,'GT_VISIBLE_PRED_VISIBLE'),(1,0,'GT_VISIBLE_PRED_COVERED')]:
        candidates=[r for r in pool if r['GT_label']==y and r['V78_prediction']==pred]
        assert candidates
        chosen=min(candidates,key=lambda r:r['selection_SHA256'])
        cells.append(dict(cell=name,cell_population=len(candidates),**chosen))
    assert len({(x['id'],x['native_k']) for x in cells})==4
    lock=dict(completed=True,selection='One minSHA(id,native_k) per GTclass x V78 fixed binary outcome cell of already frozen strict535; outcome-conditioned diagnostic selection, not prevalence or a performance estimate.',
        selection_salt=SALT,hash_key="salt+'\\n'+id+'\\n'+str(k)",fixed_threshold=.5,
        cohort_name=COHORT_NAME,cohort_N=535,GT_COVERED_N=76,GT_VISIBLE_N=459,cases=cells,
        RGB_native_masks_cached_crops_opened=0,source_saved_logits_only=True,
        current_model_forward_fit_PnP=0,
        bindings=[b(p) for p in (ENTRY,COHORT,GT,META,MASKAUDIT,QUERIES,MANIFEST,CROP_CODE,Path(__file__).resolve(),*SCORES.values())])
    write(OUT/'SELECTION_LOCK_BEFORE_RGB.json',lock)
    print(json.dumps(dict(SELECTION=b(OUT/'SELECTION_LOCK_BEFORE_RGB.json'),cases=[dict(cell=c['cell'],id=c['id'],k=c['native_k'],population=c['cell_population']) for c in cells])),flush=True)

def pure_crop():
    node=next(n for n in ast.parse(CROP_CODE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='crop')
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(CROP_CODE),'exec'),env)
    return env['crop']

def maskcrop(mask,q,L,n=96):
    h,w=mask.shape;u=((np.arange(n)+.5)/n-.5)*L
    xx,yy=np.meshgrid(q[0]+u,q[1]+u);px=np.floor(xx+.5).astype(int);py=np.floor(yy+.5).astype(int)
    valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&(px>=0)&(px<w)&(py>=0)&(py<h)
    out=np.zeros((n,n),bool);out[valid]=mask[py[valid],px[valid]]
    return out,valid

def patch3(a,ref):
    x,y=np.floor(np.asarray(ref)+.5).astype(int)
    assert x>=1 and y>=1 and x+1<a.shape[1] and y+1<a.shape[0]
    return a[y-1:y+2,x-1:x+2].copy()

def mark(draw,point,color,radius=7):
    x,y=map(float,point);draw.line((x-radius,y,x+radius,y),fill=color,width=3);draw.line((x,y-radius,x,y+radius),fill=color,width=3)

def image_cell(mask,size):
    return Image.fromarray(mask.astype(np.uint8)*255).convert('RGB').resize(size,Image.Resampling.NEAREST)

def render():
    validate();lock=load(OUT/'SELECTION_LOCK_BEFORE_RGB.json')
    for binding in lock['bindings']:check(binding)
    assert not (OUT/'GALLERY_METADATA.json').exists()
    audits=lines(MASKAUDIT);index=lines(META);man=load(MANIFEST);crop=pure_crop()
    f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',23)
    cases=[];panels=[]
    for n,c in enumerate(lock['cases']):
        i=c['corpus_row'];pi=c['public_input_index'];k=c['native_k'];q=np.asarray(c['q']);ref=np.asarray(c['r']);L=c['L']
        assert audits[i]['id']==index[i]['id']==c['id']
        mm=audits[i]['mask_pair'];check(mm['M0']);check(mm['M4']);check(c['source_RGB'])
        rgb=np.asarray(Image.open(c['source_RGB']['path']).convert('RGB'))
        raw0=np.asarray(Image.open(mm['M0']['path']));raw4=np.asarray(Image.open(mm['M4']['path']))
        a,v=raw0>0,raw4>0
        assert list(rgb.shape[:2])==index[i]['raw_hw'] and a.shape==v.shape==rgb.shape[:2]
        assert not (v&~a).any() and int(a.sum())==mm['M0_target_area'] and int(v.sum())==mm['M4_target_area']
        ch=next(x for x in man['chunks'] if x['start']<=pi<x['end'])
        check(ch['file']);j=pi-ch['start']
        with np.load(ch['file']['path'],allow_pickle=False) as z:
            assert np.array_equal(z['query8_xy'][j,k],q) and float(z['L'][j])==L
            local=z['rgb96'][j,k];valid=z['crop_valid96'][j,k]
            target=z['target_rgb128'][j];targetvalid=z['target_valid128'][j]
            targetcenter=z['target_crop_center_xy'][j];targetside=float(z['target_crop_side'][j])
        rebuilt,rv=crop(rgb,q,L,96);rebt,rtv=crop(rgb,targetcenter,targetside,128)
        np.testing.assert_array_equal(local,rebuilt);np.testing.assert_array_equal(valid,rv)
        np.testing.assert_array_equal(target,rebt);np.testing.assert_array_equal(targetvalid,rtv)
        assert valid.any() and targetvalid.any()
        a96,nv=maskcrop(a,q,L);v96,nv2=maskcrop(v,q,L)
        assert np.array_equal(nv,nv2)
        p0,p4=patch3(raw0,ref),patch3(raw4,ref)
        assert (p0>0).all()
        if c['GT_label']==0:assert not (p4>0).any()
        else:assert (p4>0).all()
        base=f'{n+1}_{c["cell"]}'
        localpng=OUT/(base+'_RAW_RGB96.png');targetpng=OUT/(base+'_RAW_TARGET128.png')
        Image.fromarray(local.transpose(1,2,0)).save(localpng);Image.fromarray(target.transpose(1,2,0)).save(targetpng)
        m0png=OUT/(base+'_GT_M0_NEAREST96.png');m4png=OUT/(base+'_GT_M4_NEAREST96.png')
        image_cell(a96,(96,96)).save(m0png);image_cell(v96,(96,96)).save(m4png)
        # Display-only mask overlay, never a forward input.
        raw=local.transpose(1,2,0);over=raw.astype(float).copy()
        over[a96&~v96]=.55*over[a96&~v96]+.45*np.array([245,75,150])
        over[v96]=.60*over[v96]+.40*np.array([40,240,120])
        over=Image.fromarray(np.floor(np.clip(over,0,255)+.5).astype(np.uint8)).resize((288,288),Image.Resampling.NEAREST)
        draw=ImageDraw.Draw(over);mark(draw,[143.5,143.5],'yellow',8)
        loc=((ref-q)/L+.5)*96-.5;mark(draw,loc*3+1,'cyan',8)
        # True native 3x3 bounds around rounded teacher r, not a new crop.
        rp=np.floor(ref+.5).astype(int)
        lo=((rp-1.5-q)/L+.5)*288;hi=((rp+1.5-q)/L+.5)*288
        draw.rectangle([*lo,*hi],outline='cyan',width=2)
        scene=Image.fromarray(rgb).resize((512,384),Image.Resampling.LANCZOS)
        sd=ImageDraw.Draw(scene);scale=np.array([512/rgb.shape[1],384/rgb.shape[0]])
        sd.rectangle([*((q-L/2)*scale),*((q+L/2)*scale)],outline='yellow',width=2)
        if c['box'] is not None:sd.rectangle([*(np.array(c['box'][:2])*scale),*(np.array(c['box'][2:])*scale)],outline='#91d1ff',width=2)
        mark(sd,q*scale,'yellow',5);mark(sd,ref*scale,'cyan',5)
        scene_path=OUT/(base+'_SCENE_OVERVIEW_DISPLAY.png');scene.save(scene_path)
        panel=Image.new('RGB',(1800,820),'#111922');d=ImageDraw.Draw(panel)
        d.text((18,12),f'{n+1}. {c["cell"]} | {c["id"]} native{k}',font=bold,fill='white')
        d.text((18,47),'Fixed source diagnostic. Weak target-owned support teacher; not human D/E/S gold.',font=f,fill='#bfcbd5')
        d.text((18,76),f'q-r {c["error_px"]:.3f}px | Linf/L {c["error_over_L_Linf"]:.5f} | L {L:.3f} | {c["rule"]} | self-visible proxy {c["self_visible_proxy_separate"]}',font=f,fill='#bfcbd5')
        for x,title in [(18,'Source scene; target box blue, q yellow, r cyan'),(546,'Exact local RGB96'),(852,'GT overlay / teacher 3x3 cyan'),(1158,'Exact target RGB128'),(1438,'M0 / isolated support'),(1620,'M4 / visible support')]:d.text((x,112),title,font=small,fill='white')
        panel.paste(scene,(18,145));panel.paste(Image.fromarray(raw).resize((288,288),Image.Resampling.NEAREST),(546,145));panel.paste(over,(852,145))
        panel.paste(Image.fromarray(target.transpose(1,2,0)).resize((256,256),Image.Resampling.NEAREST),(1158,145))
        panel.paste(image_cell(a96,(160,160)),(1438,145));panel.paste(image_cell(v96,(160,160)),(1620,145))
        # Draw q/r markers on display mask maps; raw mask PNGs remain unmarked.
        d=ImageDraw.Draw(panel)
        for x in (1438,1620):mark(d,[x+79.5,224.5],'yellow',5);mark(d,[x+(loc[0]+.5)*160/96-.5,145+(loc[1]+.5)*160/96-.5],'cyan',5)
        d.text((546,451),'GT overlay: green=visible target; pink=M0 minus M4.',font=f,fill='#bfcbd5')
        d.text((546,477),'Same q/L, no mask repair or GT recenter. q yellow / fixed r cyan.',font=f,fill='#bfcbd5')
        y0=554
        for t,name in enumerate(('V76','V77','V78')):
            score=c['scores'][name];correct=(score['prediction']=='VISIBLE')==bool(c['GT_label'])
            d.text((20+t*350,y0),f'{name}: pV={score["p_visible"]:.6f}  {score["prediction"]}',font=bold,fill='#66e0a0' if correct else '#ff9d88')
        d.text((20,601),'Native teacher 3x3 at round(r); raw PNG values:',font=bold,fill='white')
        d.text((20,638),'M0: '+' | '.join(' '.join(f'{int(z):3d}' for z in rr) for rr in p0),font=f,fill='#c6d9e7')
        d.text((20,669),'M4: '+' | '.join(' '.join(f'{int(z):3d}' for z in rr) for rr in p4),font=f,fill='#c6d9e7')
        d.text((20,712),f'Original target instance: scene{c["source_instance"]["scene_id"]} / frame{c["source_instance"]["frame_id"]} / instance{c["source_instance"]["instance_id"]} / object{c["source_instance"]["object_id"]}.',font=f,fill='#c6d9e7')
        d.text((20,747),'Outcome-selected four examples are not a prevalence estimate. Visible mask uses source render/depth policy; exact generation mode unverified.',font=small,fill='#bfcbd5')
        path=OUT/(base+'_PANEL.png');panel.save(path);panels.append(panel)
        near=patch3(rgb,ref)
        record=dict(**c,raw_local_RGB96=b(localpng),raw_target_RGB128=b(targetpng),
            M0_display=b(m0png),M4_display=b(m4png),scene_overview=b(scene_path),panel=b(path),mask_pair=mm,
            cached_input_chunk=ch['file'],exact_native_RGB_to_cache_maxdiff=0,
            teacher_native_nearest_pixel=rp.tolist(),teacher_M0_raw3x3=p0.tolist(),teacher_M4_raw3x3=p4.tolist(),teacher_RGB_raw3x3=near.tolist(),
            teacher_strong3x3_replayed_exact=True,q_pixel_floor_plus_half=np.floor(q+.5).astype(int).tolist(),
            local96_counts=dict(native_label_valid=int(nv.sum()),visible=int(v96.sum()),nonvisible_expected=int((a96&~v96).sum()),outside_expected=int((nv&~a96).sum())),
            visual_interpretation_status='PENDING_SEPARATE_VISUAL_REVIEW_NOT_GOLD')
        cases.append(record)
    contact=Image.new('RGB',(1800,4*820),'#111922')
    for i,panel in enumerate(panels):contact.paste(panel,(0,i*820))
    contact.save(OUT/'CONTACT_SHEET.png')
    result=dict(completed=True,cases=cases,selection_lock=b(OUT/'SELECTION_LOCK_BEFORE_RGB.json'),
        exact_local96_target128_original_input_replay=True,teacher_native3x3_replay=True,
        q_L_reference_nativeorder_unchanged=True,source_query_producer='GT-derived public source emulation; not actual deployed initializer performance.',
        source_MASK_or_RGB_as_forward_input=False,interpretation='Weak support proxy only. BOP current19 code admits missing depth as visible; local historical generator not established. No automatic attribution to holes, semantic labels, or causes.',
        contact_sheet=b(OUT/'CONTACT_SHEET.png'),current_reads=0,model_forward=0,new_fit=0,PnP=0)
    write(OUT/'GALLERY_METADATA.json',result)
    write(OUT/'GALLERY_SEAL.json',dict(completed=True,entry=b(ENTRY),selection=b(OUT/'SELECTION_LOCK_BEFORE_RGB.json'),metadata=b(OUT/'GALLERY_METADATA.json'),
        code=b(Path(__file__).resolve()),outputs=[b(p) for p in sorted(OUT.glob('*.png'))],old_artifacts_modified=False,current_forward_fit_PnP=0))
    print(json.dumps(dict(completed=True,metadata=b(OUT/'GALLERY_METADATA.json'),contact=b(OUT/'CONTACT_SHEET.png'))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('select','render'));args=ap.parse_args()
    {'select':select,'render':render}[args.stage]()
