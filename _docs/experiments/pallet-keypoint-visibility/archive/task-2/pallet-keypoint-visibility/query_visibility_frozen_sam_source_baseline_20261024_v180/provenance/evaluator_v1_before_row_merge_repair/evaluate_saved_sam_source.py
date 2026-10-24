"""V180 independent saved-SAM evaluation. --prepare: metadata/invented only.
No torch/SAM import, inference, feature, checkpoint, threshold fitting or q rewrite.
"""
from pathlib import Path
import argparse, ast, csv, hashlib, json, sys, traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_SAM_CODE_WEIGHTS_OR_SOURCE_RGB.json'
MAN=P/'query_visibility_public_source_target_definition_audit_20261024_v179/source_train_goal_alignment_a/SAM_SMOKE_INPUTS128_METADATA_ONLY.jsonl'
V173=P/'query_visibility_full_source_query_causal_census_20261024_v173'
CENSUS=V173/'literal_raster_census_v1/COMPLETE.json'
S173=V173/'census_preparation/build_full_source_census.py'
S172=P/'query_visibility_source_mask_semantics_audit_20261024_v172/measurement_preparation/measure_saved_source_semantics.py'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
RASTER=P/'query_visibility_registered_native_diversity_20261023_v129/render_preparation/render_corpus_once.py'
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_SAM_OUTPUT_AND_GT_BODIES.json';TESTS=H/'GENERATED_INVENTED_CHECKS.json'
META=H/'SELECTED16_SCENE_GT_BINDINGS_METADATA_ONLY.json';OUT=V/'saved_sam_source_evaluation_c_v1'
OWNER=('INCIDENT_TARGET_DECLARED_CF','OTHER_TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT')
TRUTH=('TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT');PRED=('TARGET','NON_TARGET','U')
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def checked(b):
    p=Path(b['path']).resolve();require(p.is_relative_to(P),'outside project root')
    require(not any(x in str(p).lower() for x in ('_v112','current223','_v169','/private_evaluation/')),'forbidden binding')
    require(sha(p)==b['sha256'],'hash mismatch '+str(p));return p
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jlines(p,rs):
    with Path(p).open('x') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pure(path,names,ns):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names),'pure function names')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::PURE_ONLY','exec'),ns);return ns

def grid96(q,L,lo=0,hi=96):
    q=np.asarray(q,np.float64);require(q.shape==(2,) and np.isfinite(q).all() and not np.array_equal(q,[-1.,-1.]),'finite original q')
    require(np.isfinite(L) and L>0,'positive original L')
    u=(np.arange(lo,hi,dtype=np.float64)+.5)/96-.5
    x,y=np.meshgrid(q[0]+u*L,q[1]+u*L);return np.column_stack((x.ravel(),y.ravel()))
def footprint(points,hw):
    points=np.asarray(points,np.float64);require(points.ndim==2 and points.shape[1]==2 and np.isfinite(points).all(),'finite points')
    h,w=hw;x,y=points.T;ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy
    taps=[];support=np.zeros(len(points),np.float64)
    for ax,ay,wt in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx=ix+ax;yy=iy+ay;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h);support+=wt*inside;taps.append((xx,yy,wt,inside))
    domain=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    return taps,domain,support

def bilinear_values(a,points,crop_domain=False):
    a=np.asarray(a);require(a.ndim in (2,3),'HWC or HW source')
    taps,domain,support=footprint(points,a.shape[:2]);extra=a.shape[2:];out=np.zeros((len(points),)+extra,np.float64)
    for x,y,wt,inside in taps:
        weight=wt[inside].reshape((-1,)+(1,)*len(extra));out[inside]+=a[y[inside],x[inside]].astype(np.float64)*weight
    if crop_domain:out[~domain]=0
    return out,domain,support

def rgb96(rgb,q,L):
    vals,valid,_=bilinear_values(rgb,grid96(q,L),True)
    out=np.floor(np.clip(vals,0,255)+.5).astype(np.uint8).reshape(96,96,3)
    return out,valid.reshape(96,96)

def owner_classes(part,ambiguity):
    require(part.dtype==np.int32 and ambiguity.dtype==bool and part.shape==ambiguity.shape,'raster shape/dtype')
    # Literal V172 categories merged incident+other; ambiguity has precedence.
    out=np.where(part<0,2,np.where(part==999,1,0)).astype(np.int8);out[ambiguity]=3
    return out

def soft_at(classes,points,crop_domain=False):
    taps,domain,_=footprint(points,classes.shape);out=np.zeros((len(points),5),np.float64)
    for x,y,wt,inside in taps:
        cat=np.full(len(points),4,np.int8);cat[inside]=classes[y[inside],x[inside]]
        out[np.arange(len(points)),cat]+=wt
    if crop_domain:out[~domain]=0;out[~domain,4]=1
    require(np.all(out>=0) and np.allclose(out.sum(1),1,atol=2e-15,rtol=0),'categorical unit mass')
    return out,domain

def merge_owner6(a):
    a=np.asarray(a);require(a.dtype==np.float64 and a.shape[-1]==6,'owner6 float64')
    return np.stack((a[...,0]+a[...,1],a[...,2],a[...,3],a[...,4],a[...,5]),-1)

def predictions(logits,points,available,mode,crop_domain=False):
    points=np.asarray(points,np.float64);h,w=logits.shape
    if mode=='nearest':
        xy=np.floor(points+.5).astype(np.int64);good=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        values=np.zeros(len(points),np.float64);values[good]=logits[xy[good,1],xy[good,0]];support=good.astype(np.float64)
    else:
        require(mode=='bilinear','fixed mode');values,good,support=bilinear_values(logits,points,crop_domain)
    out=np.full(len(points),2,np.int8)
    if available:
        require(np.isfinite(values[good]).all(),'available finite SAM samples');out[good]=(values[good]<=0).astype(np.int8)
    return out,values,good,support

def soft_cm(truth,pred):
    require(truth.shape==(len(pred),5) and np.isfinite(truth).all(),'soft truth shape')
    require(np.isin(pred,[0,1,2]).all(),'prediction class');out=np.zeros((5,3),np.float64)
    for p in range(3):out[:,p]=truth[pred==p].sum(0)
    require(np.isclose(out.sum(),len(pred),atol=2e-10,rtol=0),'all mass conserved')
    return out

def describe_cm(cm):
    # No renormalization of ambiguity/invalid mass into T or NON-T.
    t=cm[0];n=cm[1]+cm[2];tp=float(t[0]);fp=float(n[0]);fn=float(t[1]);tn=float(n[1])
    ratio=lambda a,b:None if b==0 else float(a/b)
    return dict(truth_order=list(TRUTH),prediction_order=list(PRED),soft_confusion=cm.tolist(),sample_equivalent_mass=float(cm.sum()),TARGET_mass=float(t.sum()),NON_TARGET_known_mass=float(n.sum()),AMBIGUOUS_mass=float(cm[3].sum()),INVALID_SUPPORT_mass=float(cm[4].sum()),prediction_U_mass=float(cm[:,2].sum()),known_truth_prediction_U_mass=float(t[2]+n[2]),TARGET_precision_known_truth=ratio(tp,tp+fp),TARGET_recall_including_prediction_U=ratio(tp,t.sum()),NON_TARGET_recall_including_prediction_U=ratio(tn,n.sum()),binary_known_confusion_T_nonT_U=np.stack((t,n)).tolist(),denominator_note='Fractional raster point/sample mass; missing/ambiguous/invalid are retained separately, not physical native D/E labels.')

def region_metrics(logits,part,amb,available):
    classes=owner_classes(part,amb);valid=~amb;target=classes==0
    counts=dict(pixels=int(part.size),ambiguity_excluded=int(amb.sum()),valid_pixels=int(valid.sum()),target_pixels=int((target&valid).sum()),foreign_pixels=int(((classes==1)&valid).sum()),BG_pixels=int(((classes==2)&valid).sum()))
    if not available:return dict(**counts,prediction_U_pixels=int(valid.sum()),TP=None,FP=None,FN=None,TN=None,IoU=None,Dice=None)
    pred=logits>0;tp=int((valid&target&pred).sum());fp=int((valid&~target&pred).sum());fn=int((valid&target&~pred).sum());tn=int((valid&~target&~pred).sum())
    ratio=lambda a,b:None if b==0 else a/b
    return dict(**counts,prediction_U_pixels=0,TP=tp,FP=fp,FN=fn,TN=tn,FP_on_FOREIGN=int(((classes==1)&pred).sum()),FP_on_BG=int(((classes==2)&pred).sum()),IoU=ratio(tp,tp+fp+fn),Dice=ratio(2*tp,2*tp+fp+fn),TARGET_precision=ratio(tp,tp+fp),TARGET_recall=ratio(tp,tp+fn))

def groups(rows):
    yield 'ALL',np.ones(len(rows),bool)
    for key in ('native_slot','variant','material','aspect','geometry_family'):
        vals=[r[key] if key in r else r['factors'][key] for r in rows]
        for value in (range(8) if key=='native_slot' else sorted(set(vals))):yield key+'::'+str(value),np.array([x==value for x in vals],bool)

def invented():
    checks={};part=np.array([[0,999],[7,-1]],np.int32);amb=np.array([[0,1],[0,0]],bool)
    cat=owner_classes(part,amb);checks['ambiguity_first_alltarget_part7_not_foreign999']=cat.tolist()==[[0,3],[0,2]]
    wt,_=soft_at(cat,[[.5,.5]]);checks['soft_boundary_target_half_BGquarter_ambiguityquarter']=np.array_equal(wt,[[.5,0,.25,.25,0]])
    inv,_=soft_at(cat,[[-.25,0.]]);checks['padding_quarter_INVALID_not_BG']=np.array_equal(inv,[[.75,0,0,0,.25]])
    zero=np.zeros((2,2),np.float32);pred,_,_,_=predictions(zero,[[0,0]],True,'nearest');checks['official_strict_greater0_tie_nonTarget']=int(pred[0])==1
    logits=np.array([[1,-1],[-1,-1]],np.float32)
    near,_,_,_=predictions(logits,[[.49,.49]],True,'nearest');bil,_,_,_=predictions(logits,[[.49,.49]],True,'bilinear')
    checks['nearest_and_interpolated_logit_threshold_differ_no_mask_interpolation']=int(near[0])==0 and int(bil[0])==1
    u,_,_,_=predictions(logits,[[0,0],[1,1]],False,'nearest');checks['input_prompt_runtime_unavailability_U_first']=u.tolist()==[2,2]
    _,_,good,_=predictions(logits,[[1.5,0]],True,'bilinear',True);checks['fixed_crop_halfpixel_domain_rightedge_excluded']=not bool(good[0])
    truth=np.eye(5,dtype=np.float64);cm=soft_cm(truth,np.array([0,1,2,0,1],np.int8));d=describe_cm(cm)
    checks['softconfusion_conserves_all5truthclasses_and_U']=cm.sum()==5 and d['TARGET_mass']==1 and d['AMBIGUOUS_mass']==1 and d['INVALID_SUPPORT_mass']==1 and d['prediction_U_mass']==1
    six=np.array([[.1,.2,.3,.15,.05,.2]],np.float64);checks['incident_plus_other_exact_target_not_renormalized']=np.allclose(merge_owner6(six),[[.3,.3,.15,.05,.2]],rtol=0,atol=1e-15)
    checks['original96_central4_positions']=grid96([64.,64.],96.,47,49).tolist()==[[63.5,63.5],[64.5,63.5],[63.5,64.5],[64.5,64.5]]
    pinned=pure(S173,{'positions','sample_weights'},{'np':np})
    checks['all9216_grid_coordinates_exact_pinned_V173']=exact(grid96([37.25,44.75],103.5),pinned['positions']([37.25,44.75],103.5,0,96))
    pts=grid96([.1,.2],11.3);ours,_=soft_at(cat,pts,True);old,_=pinned['sample_weights'](cat,pts,5,True)
    checks['categorical_sampling_byteexact_V173_independentimplementation']=exact(ours,old)
    rng=np.random.default_rng(1800);rgb=rng.integers(0,256,(31,29,3),np.uint8);orig=rgb.copy();q=[1.2,30.1];L=43.7
    a,v=rgb96(rgb,q,L);ref=pure(CROP,{'crop'},{'np':np})['crop'](rgb,np.array(q,np.float64),L,96)
    checks['rgb96_valid_and_rounding_byteexact_pinned_crop_bordercase']=exact(a,ref[0].transpose(1,2,0)) and exact(v,ref[1][0]) and exact(rgb,orig)
    m=region_metrics(np.ones((2,2),np.float32),part,amb,True)
    checks['wholemask_ambiguityexcluded_targetbothparts_BG_FP']=m['TP']==2 and m['FP']==1 and m['ambiguity_excluded']==1 and m['IoU']==2/3
    checks['nearest_raster_half_up_tie_separate_from_softtruth']=nearest_class(cat,[.5,.5])==2
    toy=dict(original_global_index=np.arange(4096,dtype=np.int32),original_occurrence_id=np.array([f'INVENTED:{i}' for i in range(4096)]),scene_index=np.arange(4096,dtype=np.int32)//8,base_index=np.arange(4096,dtype=np.int32)//32,native_slot=(np.arange(4096)%8).astype(np.int8),query_xy=np.ones((4096,2),np.float64),L=np.full(4096,96.,np.float64),split=np.full(4096,'TRAIN'),query_available=np.ones(4096,bool))
    toy['query_xy'][0]=np.nan;toy['L'][0]=np.nan;toy['query_available'][0]=False
    tr=[]
    for i in (0,4095):tr.append(dict(original_global_index=i,original_occurrence_id=f'INVENTED:{i}',scene_index=i//8,base_index=i//32,native_slot=i%8,split='TRAIN',original_query_xy=None if i==0 else [1.,1.],original_L=None if i==0 else 96.,original_input_flags=dict(query_available=i!=0)))
    checks['exact_ID_join_preserves_missing_and_inputU_rows']=assert_train_alignment(tr,toy)==[0,4095]
    bad=[dict(r) for r in tr];bad[1]['native_slot']=6;rejected=False
    try:assert_train_alignment(bad,toy)
    except AssertionError:rejected=True
    checks['wrong_native_rowjoin_failclosed']=rejected
    bad=[dict(r) for r in tr];bad[1]['original_global_index']=0;rejected=False
    try:assert_train_alignment(bad,toy)
    except AssertionError:rejected=True
    checks['duplicated_original_ID_failclosed']=rejected
    require(all(checks.values()),'invented checks failed '+str(checks))
    return dict(completed=True,PASS=True,invented_only=True,number_checks=len(checks),checks=checks,source_payload_or_model_output_or_weight_body_header_hash_reads=0)

def make_previews(scene_rows,assets,row_results):
    """Fixed ALL16/ALL128. All overlays on copies; clean96 verified before save."""
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    nearest=Image.Resampling.NEAREST;overview=Image.new('RGB',(2560,4*330+70),'white');od=ImageDraw.Draw(overview)
    od.text((14,8),'ALL16 fixed TRAIN scenes: original RGB + input q8 box | SAM logit > 0 | all-target raster (ambiguity yellow)',font=font,fill='black')
    records=[];byid={r['original_global_index']:r for r in row_results};links=[]
    for j,s in enumerate(scene_rows):
        si=s['scene_index'];a=assets[si];rr=a['rows'];directory=OUT/('SCENE_%06d'%si);directory.mkdir()
        rgb=a['RGB'];classes=a['classes'];logits=a['logits'];available=a['scene_available']
        overlay=Image.fromarray(rgb.copy());d=ImageDraw.Draw(overlay);box=rr[0]['original_q8_context_box_xyxy'];d.rectangle(tuple(box),outline=(255,240,0),width=2)
        for r in rr:
            x,y=r['original_query_xy'];d.line((x-3,y,x+3,y),fill=(255,255,255),width=1);d.line((x,y-3,x,y+3),fill=(255,255,255),width=1)
        sm=np.zeros((512,512,3),np.uint8)
        if available:sm[logits>0]=(245,80,200)
        else:sm[:]=(110,90,150)
        gt=np.zeros((512,512,3),np.uint8);gt[classes==0]=(60,225,230);gt[classes==3]=(245,185,0)
        tx=(j%4)*640;ty=70+(j//4)*330
        overview.paste(overlay.resize((256,256),nearest),(tx,ty+30));overview.paste(Image.fromarray(sm).resize((128,128),nearest),(tx+266,ty+30));overview.paste(Image.fromarray(gt).resize((128,128),nearest),(tx+404,ty+30))
        od.text((tx+6,ty+3),f'scene {si} / base {s["base_index"]} / {s["variant"]}',font=small,fill='black')
        od.text((tx+266,ty+165),'SAM > 0',font=small,fill='black');od.text((tx+404,ty+165),'target / ambiguity',font=small,fill='black')
        od.text((tx+266,ty+190),'RGB: 0.5x nearest',font=small,fill='black');od.text((tx+266,ty+210),'masks: 0.25x nearest',font=small,fill='black')
        od.text((tx+6,ty+292),f'{rr[0]["factors"]["material"]} / {rr[0]["factors"]["aspect"]}',font=small,fill='black')
        card=Image.new('RGB',(1320,90+8*215),'white');cd=ImageDraw.Draw(card)
        cd.text((12,6),f'Scene {si}: ALL native 0..7 / unchanged per-variant q and L / TRAIN / class-agnostic proposal',font=font,fill='black')
        for x,title in [(12,'Original clean96 / 2x NN'),(222,'SAM bilinear logit>0 / 2x NN'),(432,'GT target weight96 / 2x NN')]:cd.text((x,36),title,font=small,fill='black')
        cd.text((12,60),'GT gray=target weight, amber=ambiguity, yellow=invalid; mask outside target is NON-T, not external/self.',font=small,fill='black')
        for k,r in enumerate(rr):
            i=r['original_global_index'];z=byid[i];cr=a['crops'][k];valid=a['valid'][k];wt=a['croptruth'][k];pr=a['croppred'][k]
            require(ash(cr)==r['saved_rgb96_sha256_inherited'] and ash(valid)==r['saved_valid96_sha256_inherited'],'preview raw input hash')
            clean=directory/('NATIVE_%d_CLEAN96.png'%k);Image.fromarray(cr).save(clean)
            decoded=np.array(Image.open(clean));require(exact(decoded,cr),'clean PNG decoded bytes')
            sam=np.zeros((96*96,3),np.uint8);sam[pr==0]=(255,255,255);sam[pr==2]=(110,90,150)
            gtgray=np.repeat(np.floor(np.clip(wt[:,0],0,1)*255+.5).astype(np.uint8)[:,None],3,1)
            # Pure display colors mark any unknown contribution; raw weights retained in NPZ.
            gtgray[wt[:,3]>0]=(220,145,0);gtgray[wt[:,4]>0]=(245,225,0)
            y=90+k*215
            card.paste(Image.fromarray(cr).resize((192,192),nearest),(12,y));card.paste(Image.fromarray(sam.reshape(96,96,3)).resize((192,192),nearest),(222,y));card.paste(Image.fromarray(gtgray.reshape(96,96,3)).resize((192,192),nearest),(432,y))
            texts=[f'native {k} / original global {i}',f'q=({r["original_query_xy"][0]:.3f}, {r["original_query_xy"][1]:.3f})  L={r["original_L"]:.3f}',f'input available={z["original_input_available"]}; prediction available={z["prediction_available"]}',f'point TARGET/F/BG weights: '+', '.join(f'{x:.3f}' for x in z['point_owner_soft5'][:3]),f'q prediction nearest={z["prediction_nearest"]}; bilinear={z["prediction_bilinear"]}',f'nearest raster={z["nearest_raster_class"]}; native labels unchanged', 'Source raster target includes every visible selected-pallet part.']
            for n,t in enumerate(texts):cd.text((650,y+4+n*25),t,font=small,fill='black')
            records.append(dict(original_global_index=i,native_slot=k,clean96=bind(clean),decoded_rgb96_sha256=ash(decoded),original_rgb96_sha256=r['saved_rgb96_sha256_inherited'],clean_pixels_unchanged=True,grid96_sha256=z['grid96_xy_sha256']))
        cp=directory/'CARD.png';card.save(cp);links.append(dict(scene_index=si,card=bind(cp),rows=8))
    op=OUT/'ALL16_SOURCE_SAM_RASTER_OVERVIEW.png';overview.save(op)
    jlines(OUT/'ALL128_CLEAN_PIXEL_RENDER_BINDINGS.jsonl',records)
    html=['<!doctype html><meta charset="utf-8"><title>V180 fixed source SAM diagnostics</title><style>body{font-family:system-ui;max-width:1500px;margin:2rem auto}img{max-width:100%}p{line-height:1.5}</style><h1>Fixed source TRAIN SAM smoke diagnostics</h1><p>All 16 scenes and all 128 original queries. No subset selection. Class-agnostic masks; not native-part/external/self or physical coordinate gold. All new training admission false. PNG visual QA is pending a separate actual view; HTML has not been browser rendered.</p><img src="ALL16_SOURCE_SAM_RASTER_OVERVIEW.png">']
    for x in links:
        rel=Path(x['card']['path']).relative_to(OUT);html.append(f'<h2>Scene {x["scene_index"]}: all native8</h2><a href="{rel}"><img src="{rel}"></a>')
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')
    return dict(overview=bind(op),cards=links,clean128=bind(OUT/'ALL128_CLEAN_PIXEL_RENDER_BINDINGS.jsonl'),gallery=bind(OUT/'index.html'),PNG_count=145,overview_count=1,scene_cards=16,clean_crops=128,actual_view_QA_pending=True,browser_render_performed=False)

def prepare():
    scope=read(SCOPE);require(scope['input_manifest']==bind(MAN),'scope fixed128 manifest')
    rows=lines(MAN);require(len(rows)==128 and len({r['original_global_index'] for r in rows})==128,'all128 unique')
    require(all(r['split']=='TRAIN' and r['training_admission'] is False for r in rows),'TRAIN-only')
    scenes=[]
    for si in dict.fromkeys(r['scene_index'] for r in rows):
        rr=[r for r in rows if r['scene_index']==si];require([r['native_slot'] for r in rr]==list(range(8)),'native8 original order')
        require(all(r['original_initial_points8']==rr[0]['original_initial_points8'] and r['raw_hw']==[512,512] for r in rr),'shared q8/raw shape')
        for r in rr:require(r['original_query_xy']==r['original_initial_points8'][r['native_slot']],'q is own original q8 slot')
        scenes.append(si)
    require(len(scenes)==16 and set(r['base_index'] for r in rows)=={0,1,16,17},'fixed16')
    census=read(CENSUS);require(census['completed'] and census['PASS'] and census['owner_class_order']==list(OWNER),'source census metadata')
    allmeta=lines(checked(census['scene_inventory']));selected=[next(s for s in allmeta if s['scene_index']==si) for si in scenes]
    require(all(s['split']=='TRAIN' for s in selected),'no held scenes')
    for s in selected:
        rr=[r for r in rows if r['scene_index']==s['scene_index']]
        require(s['scene_buffer']==rr[0]['full_RGB_packet'] and s['decoded_RGB_SHA']==rr[0]['decoded_RGB_sha256_inherited'],'GT scene same original RGB binding')
    write(META,dict(completed=True,metadata_only=True,scenes=selected,scene_order=scenes,rows128=bind(MAN)))
    test=invented();test['code']=bind(__file__);write(TESTS,test)
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),manifest=bind(MAN),selected_scene_metadata=bind(META),tests=bind(TESTS),source_census_complete=bind(CENSUS),TRAIN_sidecar=census['sidecars']['TRAIN'],source_authority_code=dict(V172=bind(S172),V173=bind(S173),V129_renderer=bind(RASTER),V90_crop=bind(CROP)),runtime=dict(executable=sys.executable,numpy=np.__version__),output=str(OUT),
      data_counts=dict(images=16,queries=128,bases=4,split='TRAIN',all_native8_each_scene=True),
      generic_execution='Same code --manifest --expected-rows --expected-scenes --predictions --outputdir, exact-bound by entry. Initial128/16 or prospective root-approved fullTRAIN4096/512 only. FullTRAIN extension requires separately bound expanded_TRAIN_scope and full_TRAIN_expansion_authorized_before_scores=true. Renderer/sample-array exports remain initial fixed16/128 only. Missing q/L stays explicit null/missing denominators, never NON-T.',
      prediction_binding='Dynamic completed prediction seal in separate ROOT entry; exact B frozen schema below. All prediction outputs must have been sealed before GT reads; no forward/import permitted.',
      prediction_schema=dict(seal='COMPLETE.json.scenes list16 in manifest order; original_input_metadata8 matches exact manifest row dictionaries; outputs binding or null; prompt/runtime flags/reasons',NPZ=dict(image_logits='float32[512,512]',lowres_logits='float32[256,256]',predicted_quality='float32[1]',mask_logits_gt0='bool[512,512]',box_xyxy='float64[4]',original_query_xy='float64[8,2]',original_L='float64[8]',native_slot='int8[8]',original_global_index='int64[8]',original_occurrence_id='string[8]',original_query_available='bool[8]',prediction_available='bool[8]')),
      truth='V172/V129 authoritative raster: ambiguity first; part999 foreign; part<0 BG; all other part>=0 target. Whole-image metrics exclude ambiguity with exact denominator. No incident-only or nearest/permutation rewrite. V173 TRAIN4096 scene_owner_point target=incident+other, keep foreign/BG/ambiguity/invalid separately. No teacher/depth/held sidecar read.',
      query_prediction='Two fixed diagnostics separately: nearest floor(q+.5) on full image logits then >0; float64 bilinear logits with zero border then >0. Ties0 NON_TARGET. Out-of-sample-domain or original input/prompt/runtime unavailable -> U. Bilinear at original q is not thresholding/interpolating a binary mask. Not external/self decisions.',
      exact96='Float64 original q+((index+.5)/96-.5)*L; RGB byte-parity to original saved rgb96/valid96 AND pure frozen V90 crop. SAM bilinear logits>0 on identical 9216 points; source categorical class interpolation preserves ambiguity/invalid mass. Outside frozen sample domain [-.5,w-.5)x[-.5,h-.5) -> U and INVALID. Source partial footprint outside is retained separately, not BG.',
      metrics='ALL128 retained: original/new U/reasons; full5truth-by3prediction soft point CM for both samplers and nearest raster hard CM separately; known T vs foreign+BG only without ambiguity/invalid renormalization. All/native0..7/material/aspect/geometry/variant strata. Whole-image TP/FP/FN/TN/IoU/Dice on nonambiguous source pixels, FP foreign and BG separate. Full96 sample-equivalent soft CM separately, not 9216 independent queries or native-region labels.',
      render_plan='Outcome-blind all16 scenes4x4 overview RGB+unchanged q8box/SAMmask/source targetmask, plus16 cards each all8 original clean96/SAM bilinearlogit>0/source targetweight96 aligned exactgrid, NN2x; all128 clean96 PNG decoded byte/hash verified. Display overlays on separate copies. Root actual view overview+16cards later; no browser rendering claim.',
      unchanged='Original q/L/native/variant/split/inputflags preserved. Original D/E/S/U only historical metadata in GT scene; not loaded as labels, not scored/relabelled. No model/prompt/threshold/data selection, PnP, training/admission or performance success criterion.',
      source_payload_or_SAM_output_or_weights_body_header_hash_reads_during_preparation=0,actual_evaluations=0,automatic_retry=False)
    write(PROTOCOL,p)
    template=dict(approved_saved_sam_source_evaluation=False,source_only=True,predictions_sealed_before_GT=True,new_model_fit_threshold_private_held_or_PnP=False,automatic_retry=False,code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],manifest=p['manifest'],tests=p['tests'],prediction_complete={'path':'ROOT_TO_BIND_COMPLETED_B_SAM_PREDICTION_COMPLETE','sha256':'ROOT_TO_BIND'},source_census_complete=p['source_census_complete'],TRAIN_sidecar=p['TRAIN_sidecar'],evaluation_manifest=p['manifest'],expected_rows=128,expected_scenes=16,output_directory=str(OUT),full_TRAIN_expansion_authorized_before_scores=False,expanded_TRAIN_scope=None)
    write(H/'ROOT_SAVED_EVALUATION_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),tests=p['tests'],selected_scene_metadata=p['selected_scene_metadata'],runtime=p['runtime'],actual_evaluations=0,source_payload_or_SAM_output_or_weights_body_header_hash_reads=0))
    print(json.dumps(dict(prepared=True,code=p['code'],protocol=bind(PROTOCOL),tests=bind(TESTS),preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)

def finiteq(q):return q is not None and np.asarray(q).shape==(2,) and bool(np.isfinite(q).all()) and not np.array_equal(q,[-1.,-1.])
def nearest_class(classes,q):
    if not finiteq(q):return None
    xy=np.floor(np.asarray(q,np.float64)+.5).astype(np.int64);x,y=xy;h,w=classes.shape
    return int(classes[y,x]) if 0<=x<w and 0<=y<h else 4

def assert_train_alignment(rows,train):
    ids=train['original_global_index'];require(ids.dtype==np.int32 and len(ids)==4096 and len(set(ids.tolist()))==4096,'TRAIN4096 unique IDs')
    lookup={int(i):j for j,i in enumerate(ids)};indices=[]
    require(len({r['original_global_index'] for r in rows})==len(rows),'manifest duplicate IDs')
    for r in rows:
        require(r['split']=='TRAIN' and r['original_global_index'] in lookup,'manifest TRAIN identity')
        k=lookup[r['original_global_index']];indices.append(k)
        for key,mkey in [('original_occurrence_id','original_occurrence_id'),('scene_index','scene_index'),('base_index','base_index'),('native_slot','native_slot')]:require(train[key][k].item()==r[mkey],'TRAIN identity '+key)
        require(str(train['split'][k])=='TRAIN','sidecar split')
        q=np.asarray([np.nan,np.nan] if r['original_query_xy'] is None else r['original_query_xy'],np.float64)
        L=np.nan if r['original_L'] is None else r['original_L']
        require(np.array_equal(train['query_xy'][k],q,equal_nan=True) and ((np.isnan(L) and np.isnan(train['L'][k])) or train['L'][k]==L),'TRAIN original q/L')
        for flag,value in r['original_input_flags'].items():require(bool(train[flag][k])==value,'TRAIN original flag '+flag)
    return indices

def runtime_inputs(entry,manifest,expected_rows,expected_scenes,predictions,outputdir):
    e=read(entry);require(e['approved_saved_sam_source_evaluation'] is True and e['source_only'] is True and e['predictions_sealed_before_GT'] is True,'root evaluation entry')
    require(not e['new_model_fit_threshold_private_held_or_PnP'] and not e['automatic_retry'],'no prohibited operation')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL),'frozen code/protocol');p=read(PROTOCOL)
    for key in ('scope','manifest','tests','source_census_complete','TRAIN_sidecar'):require(e[key]==p[key],'entry original binding '+key);checked(p[key])
    require(sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy'],'pinned runtime')
    require(read(checked(p['tests']))['PASS'],'invented checks')
    for b in p['source_authority_code'].values():checked(b)
    checked(p['selected_scene_metadata']);require(not outputdir.exists(),'preserve prior outputs; no overwrite/retry')
    require(e['evaluation_manifest']==bind(manifest) and e['prediction_complete']==bind(predictions),'dynamic manifest/prediction binding')
    require(e['expected_rows']==expected_rows and e['expected_scenes']==expected_scenes and Path(e['output_directory']).resolve()==outputdir.resolve(),'dynamic dimensions/output binding')
    require(outputdir.resolve().is_relative_to(V),'output inside V180')
    require((expected_rows,expected_scenes) in ((128,16),(4096,512)),'only initial smoke or explicit fullTRAIN')
    if expected_rows!=128:
        require(e['full_TRAIN_expansion_authorized_before_scores'] is True,'prospective fullTRAIN expansion');checked(e['expanded_TRAIN_scope'])
    else:require(bind(manifest)==p['manifest'],'smoke unchanged')
    rows=lines(checked(e['evaluation_manifest']));require(len(rows)==expected_rows and all(r['split']=='TRAIN' for r in rows),'all expected TRAIN rows')
    order=list(dict.fromkeys(r['scene_index'] for r in rows));require(len(order)==expected_scenes,'all expected scenes')
    for si in order:
        rr=[r for r in rows if r['scene_index']==si];require([r['native_slot'] for r in rr]==list(range(8)),'native8 in each scene')
        require(all(r['original_initial_points8']==rr[0]['original_initial_points8'] for r in rr),'q8 metadata consistency')
        for r in rr:
            if finiteq(r['original_query_xy']):require(r['original_initial_points8'][r['native_slot']]==r['original_query_xy'],'own q8 slot')
    # Completed seal is read before any truth or source pixel payload.
    seal=read(checked(e['prediction_complete']));require(seal['completed'] is True,'prediction stage terminal completed')
    saved=seal['scenes'];require([s['scene_index'] for s in saved]==order,'prediction/manifest scene order')
    for s,si in zip(saved,order):require(s['original_input_metadata8']==[r for r in rows if r['scene_index']==si],'prediction input metadata byte-semantic equality')
    return e,p,rows,saved

def run(entry,manifest,expected_rows,expected_scenes,predictions,outputdir):
    global OUT
    OUT=outputdir.resolve();e,p,rows,saved=runtime_inputs(entry,manifest,expected_rows,expected_scenes,predictions,OUT)
    OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),prediction_complete=bind(predictions),evaluation_only=True))
    try:
        # TRAIN only: VALIDATION sidecar is never opened or hashed.
        census=read(checked(p['source_census_complete']));require(census['sidecars']['TRAIN']==p['TRAIN_sidecar'],'frozen TRAIN binding')
        source_scene_meta={s['scene_index']:s for s in lines(checked(census['scene_inventory']))}
        flags=list(rows[0]['original_input_flags'])
        keys=['original_global_index','original_occurrence_id','scene_index','base_index','native_slot','query_xy','L','split','owner_class_order','scene_owner_point','scene_owner_central4','scene_owner_central32']+flags
        with np.load(checked(p['TRAIN_sidecar']),allow_pickle=False) as z:train={k:z[k].copy() for k in keys}
        require(train['owner_class_order'].tolist()==list(OWNER),'owner6 class order');indices=assert_train_alignment(rows,train);trainindex={r['original_global_index']:k for r,k in zip(rows,indices)}
        selected=read(checked(p['selected_scene_metadata']))['scenes'];render_scenes={s['scene_index'] for s in selected}
        output_rows=[];images=[];assets={};used={};cache_path=None;cache=None;sample_files=[]
        for s in saved:
            si=s['scene_index'];rr=[r for r in rows if r['scene_index']==si];meta=source_scene_meta[si];require(meta['split']=='TRAIN' and meta['scene_buffer']==rr[0]['full_RGB_packet'],'same source scene')
            # Only GT raster IDs/ambiguity and original RGB for fixed previews/cache parity.
            buffer_path=checked(meta['scene_buffer']);used[str(buffer_path)]=meta['scene_buffer']
            with np.load(buffer_path,allow_pickle=False) as z:part=z['part_id'].copy();amb=z['ambiguity'].copy();rgb=z['RGB_uint8'].copy()
            require(part.shape==amb.shape==(512,512) and part.dtype==np.int32 and amb.dtype==bool and rgb.shape==(512,512,3) and rgb.dtype==np.uint8,'source raster/RGB schema')
            require(ash(rgb)==meta['decoded_RGB_SHA']==rr[0]['decoded_RGB_sha256_inherited'],'source decoded RGB SHA')
            classes=owner_classes(part,amb);scene_available=bool(s['prompt_available'] and s['runtime_available'])
            logits=np.zeros((512,512),np.float32);prediction_available=np.zeros(8,bool)
            if s['outputs'] is not None:
                packet=checked(s['outputs']);used[str(packet)]=s['outputs']
                expected=set(p['prediction_schema']['NPZ'])
                with np.load(packet,allow_pickle=False) as z:
                    require(set(z.files)==expected,'SAM strict saved packet keys');a={k:z[k].copy() for k in expected}
                logits=a['image_logits'];require(logits.dtype==np.float32 and logits.shape==(512,512) and np.isfinite(logits).all(),'full SAM logits')
                require(a['lowres_logits'].dtype==np.float32 and a['lowres_logits'].shape==(256,256) and np.isfinite(a['lowres_logits']).all(),'lowres saved metadata')
                require(a['predicted_quality'].dtype==np.float32 and a['predicted_quality'].shape==(1,) and np.isfinite(a['predicted_quality']).all(),'uncalibrated quality metadata')
                require(a['mask_logits_gt0'].dtype==bool and np.array_equal(a['mask_logits_gt0'],logits>0),'official strictzero mask parity')
                for key,vals,dtype in [('native_slot',range(8),np.int8),('original_global_index',[r['original_global_index'] for r in rr],np.int64),('original_occurrence_id',[r['original_occurrence_id'] for r in rr],None),('original_query_xy',[[np.nan,np.nan] if r['original_query_xy'] is None else r['original_query_xy'] for r in rr],np.float64),('original_L',[np.nan if r['original_L'] is None else r['original_L'] for r in rr],np.float64),('original_query_available',[r['original_input_flags']['query_available'] for r in rr],bool)]:
                    val=np.asarray(list(vals) if isinstance(vals,range) else vals,dtype=dtype);require(a[key].shape==val.shape and (dtype is None or a[key].dtype==val.dtype),'saved '+key+' schema')
                    require(np.array_equal(a[key],val,equal_nan=True) if val.dtype.kind=='f' else np.array_equal(a[key],val),'saved '+key+' exact metadata')
                require(a['box_xyxy'].dtype==np.float64 and np.array_equal(a['box_xyxy'],np.asarray(rr[0]['original_q8_context_box_xyxy'],np.float64)),'unchanged q8box')
                prediction_available=a['prediction_available'];require(prediction_available.dtype==bool and prediction_available.shape==(8,),'saved prediction flags')
                require(np.array_equal(prediction_available,np.array([r['original_input_flags']['query_available'] and scene_available for r in rr])),'U-first availability')
            else:require(not scene_available,'available scene missing outputs')
            im=dict(scene_index=si,base_index=meta['base_index'],variant=meta['variant'],factors=meta['factors'],prompt_available=bool(s['prompt_available']),runtime_available=bool(s['runtime_available']),prompt_reason=s['prompt_reason'],runtime_reason=s['runtime_reason'],metrics=region_metrics(logits,part,amb,scene_available),source_buffer=meta['scene_buffer'],prediction_output=s['outputs'])
            images.append(im);crops=[];valids=[];croptruth=[];croppred=[]
            for k,r in enumerate(rr):
                i=r['original_global_index'];q=r['original_query_xy'];L=r['original_L'];ti=trainindex[i];available=bool(prediction_available[k]);original_available=bool(r['original_input_flags']['query_available'])
                row=dict(**r,original_input_available=original_available,prediction_available=available,prompt_available=bool(s['prompt_available']),runtime_available=bool(s['runtime_available']),prompt_reason=s['prompt_reason'],runtime_reason=s['runtime_reason'],native_semantic_labels_not_replaced=True,training_admission=False)
                finite=finiteq(q);point_truth=merge_owner6(train['scene_owner_point'][ti]);missing=not np.isfinite(point_truth).all();require(missing==(not finite),'missing original q/soft mass');require(np.isnan(point_truth).all() if missing else np.isfinite(point_truth).all(),'all-null or all-finite point mass')
                if finite:
                    points=np.asarray(q,np.float64)[None];direct,_=soft_at(classes,points);require(np.allclose(direct[0],point_truth,rtol=0,atol=2e-13),'independent point soft parity V173')
                    near,nv,ng,_=predictions(logits,points,available,'nearest');bil,bv,bg,support=predictions(logits,points,available,'bilinear');nearest=nearest_class(classes,q)
                    row.update(point_owner_soft5=point_truth.tolist(),point_truth_missing=False,prediction_nearest=PRED[int(near[0])],prediction_bilinear=PRED[int(bil[0])],prediction_nearest_index=int(near[0]),prediction_bilinear_index=int(bil[0]),nearest_logit=float(nv[0]) if available and ng[0] else None,bilinear_logit=float(bv[0]) if available and bg[0] else None,bilinear_inbounds_weight=float(support[0]),nearest_raster_class=TRUTH[nearest],nearest_raster_class_index=nearest)
                else:row.update(point_owner_soft5=None,point_truth_missing=True,prediction_nearest='U',prediction_bilinear='U',prediction_nearest_index=2,prediction_bilinear_index=2,nearest_logit=None,bilinear_logit=None,bilinear_inbounds_weight=None,nearest_raster_class='MISSING_ORIGINAL_Q',nearest_raster_class_index=None)
                if finite and L is not None and np.isfinite(L) and L>0:
                    grid=grid96(q,L);wt,valid=soft_at(classes,grid,True);pred,lv,_,supp=predictions(logits,grid,available,'bilinear',True);cr,vr=rgb96(rgb,q,L)
                    require(exact(vr,valid.reshape(96,96)),'same frozen96 valid grid')
                    for key,lo,hi in [('central4',47,49),('central32',32,64)]:
                        got=wt.reshape(96,96,5)[lo:hi,lo:hi].mean((0,1));ref=merge_owner6(train['scene_owner_'+key][ti]);require(np.allclose(got,ref,atol=2e-13,rtol=0),'independent '+key+' source soft parity')
                    if cache_path!=r['saved_crop_chunk']['path']:
                        cache_path=r['saved_crop_chunk']['path'];checked(r['saved_crop_chunk']);used[cache_path]=r['saved_crop_chunk']
                        with np.load(cache_path,allow_pickle=False) as z:cache={key:z[key].copy() for key in ('global_index','rgb96','valid96','crop_constructed')}
                    ck=r['saved_crop_row'];require(int(cache['global_index'][ck])==i and bool(cache['crop_constructed'][ck]),'saved crop original identity')
                    require(exact(cr,cache['rgb96'][ck]) and exact(vr,cache['valid96'][ck]),'original clean96/valid byteparity')
                    require(ash(cr)==r['saved_rgb96_sha256_inherited'] and ash(vr)==r['saved_valid96_sha256_inherited'],'original crop hashes')
                    row.update(grid96_xy_sha256=ash(grid),rgb96_original_byteparity=True,valid96_original_byteparity=True,crop96_sample_equivalent_confusion=soft_cm(wt,pred).tolist(),crop96_outside_sample_domain=int((~valid).sum()),crop96_partial_source_footprint_samples=int(((supp<1)&valid).sum()))
                    if si in render_scenes:crops.append(cr);valids.append(vr);croptruth.append(wt);croppred.append(pred)
                    if si in render_scenes:
                        sf=OUT/('SCENE_%06d_NATIVE%d_SAMPLES.npz'%(si,k));np.savez_compressed(sf,grid96_xy=grid.reshape(96,96,2),sam_logit96=np.where(pred!=2,lv,np.nan).reshape(96,96),sam_prediction96=pred.reshape(96,96),raster_owner_soft5_96=wt.reshape(96,96,5),valid96=valid.reshape(96,96));sample_files.append(bind(sf))
                else:row.update(grid96_xy_sha256=None,rgb96_original_byteparity=None,valid96_original_byteparity=None,crop96_sample_equivalent_confusion=None,crop96_outside_sample_domain=None,crop96_partial_source_footprint_samples=None)
                output_rows.append(row)
            if si in render_scenes:
                require(len(crops)==8,'fixed smoke scene all8 constructed');assets[si]=dict(RGB=rgb,classes=classes,logits=logits,scene_available=scene_available,rows=rr,crops=crops,valid=valids,croptruth=croptruth,croppred=croppred)
        require(len(output_rows)==expected_rows and len(images)==expected_scenes,'all retained')
        # Exact original manifest ordering, not ID sorting or outcome selection.
        require([r['original_global_index'] for r in output_rows]==[r['original_global_index'] for r in rows],'all row sequence')
        aggregates=[]
        for name,mask in groups(rows):
            subset=[r for r,m in zip(output_rows,mask) if m];present=[r for r in subset if not r['point_truth_missing']]
            block=dict(group=name,N=len(subset),original_inputU=sum(not r['original_input_available'] for r in subset),predictionU=sum(not r['prediction_available'] for r in subset),point_truth_missing_N=len(subset)-len(present),point_truth_measured_N=len(present),point={},nearest_raster={},crop96={})
            for mode in ('nearest','bilinear'):
                truth=np.asarray([r['point_owner_soft5'] for r in present],np.float64).reshape(-1,5);pred=np.asarray([r['prediction_'+mode+'_index'] for r in present],np.int8)
                block['point'][mode]=describe_cm(soft_cm(truth,pred))
                hard=np.zeros((5,3),np.int64)
                for r in present:hard[r['nearest_raster_class_index'],r['prediction_'+mode+'_index']]+=1
                block['nearest_raster'][mode]=dict(truth_order=list(TRUTH),prediction_order=list(PRED),confusion=hard.tolist(),N=int(hard.sum()),note='Nearest categorical raster diagnostic; separate from fractional point truth.')
            cms=[np.asarray(r['crop96_sample_equivalent_confusion'],np.float64) for r in subset if r['crop96_sample_equivalent_confusion'] is not None]
            block['crop96']=describe_cm(sum(cms,np.zeros((5,3),np.float64)));block['crop96']['original_query_N']=len(cms);block['crop96']['missing_grid_N']=len(subset)-len(cms)
            aggregates.append(block)
        rowfile=OUT/'ALL_QUERIES_SAM_AND_LITERAL_TARGET_DIAGNOSTICS.jsonl';jlines(rowfile,output_rows)
        imagefile=OUT/'ALL_IMAGES_TARGET_RASTER_OVERLAP.jsonl';jlines(imagefile,images)
        summary=OUT/'FIXED_CONFUSIONS_AND_STRATA.json';write(summary,dict(completed=True,query_N=expected_rows,image_N=expected_scenes,groups=aggregates,interpretation=p['unchanged']))
        previews=make_previews(selected,assets,output_rows)
        for b in used.values():checked(b)
        checked(p['TRAIN_sidecar']);checked(e['prediction_complete']);checked(e['evaluation_manifest'])
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),prediction_complete=bind(predictions),manifest=bind(manifest),source_TRAIN_sidecar=p['TRAIN_sidecar'],query_N=expected_rows,image_N=expected_scenes,all_rows_and_U_retained=True,rows=bind(rowfile),images=bind(imagefile),summary=bind(summary),sample_arrays=sample_files,previews=previews,original_q_L_native_flags_and_RGB96_valid_byteparity=True,no_model_fit_forward_or_selection=True,new_native_labels_or_training_admission=False,selected_model=None,fully_solved=False,automatic_retry=False))
        print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'),query_N=expected_rows,image_N=expected_scenes)),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path)
    ap.add_argument('--manifest',type=Path,default=MAN);ap.add_argument('--expected-rows',type=int,default=128);ap.add_argument('--expected-scenes',type=int,default=16)
    ap.add_argument('--predictions',type=Path);ap.add_argument('--outputdir',type=Path,default=OUT);args=ap.parse_args()
    if args.prepare and args.entry is None and args.predictions is None:prepare()
    elif args.entry is not None and args.predictions is not None and not args.prepare:run(args.entry,args.manifest,args.expected_rows,args.expected_scenes,args.predictions,args.outputdir)
    else:raise SystemExit('Use --prepare (invented only), or separate root-approved --entry plus --predictions; no automatic execution')
