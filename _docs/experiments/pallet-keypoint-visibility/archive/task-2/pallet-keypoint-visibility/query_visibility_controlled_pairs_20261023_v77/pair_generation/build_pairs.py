"""V77 locked opaque source visibility pairs. No models/features/current IO."""
import argparse,ast,hashlib,importlib.util,json,time
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from opaque_pair_geometry import donor_geometry,rotate,compose

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
V76=P/'query_visibility_reference_local_support_20261023_v76'
GT=V76/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz'
INDEX=V76/'gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl'
PERROW=V76/'gt_corpus_build/PER_ROW_TEACHER_GT_ONLY.jsonl'
SYNTH=V76/'source_inventory/UNCHANGED_MODEL_INPUT_LINKS.jsonl'
PUBLIC=V76/'public_query_emulation/INPUTS_RGB_ONLY.jsonl'
YCB=P/'query_visibility_public_real_20261022_v63/ycb/instances.jsonl'
TEACHER=V76/'teacher_preparation/reference_local_support_teacher.py'
GRID=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
BOX=P/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'
SALT='V77_SAME_REFERENCE_OPAQUE_PAIR_V1'
INPUT_KEYS=['frame_index','id','source','split','image','image_sha256','raw_hw','initial_points','roi_side','box']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()
def b(p):return {'path':str(p),'sha256':sha(p)}
def check(x):assert sha(x['path'])==x['sha256'],x['path']
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(x)for x in Path(p).read_text().splitlines()]
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return{str(k):clean(x)for k,x in v.items()}
    if isinstance(v,(tuple,list)):return[clean(x)for x in v]
    if isinstance(v,(np.bool_,np.integer)):return v.item()
    if isinstance(v,(float,np.floating)):return float(v)if np.isfinite(v)else None
    return v
def save(p,d):
    with p.open('x')as f:json.dump(clean(d),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def jsonline(f,d):f.write(json.dumps(clean(d),separators=(',',':'),allow_nan=False)+'\n')
def pure(p,name):
    n=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
    d={'np':np};exec(compile(ast.Module(body=[n],type_ignores=[]),str(p),'exec'),d);return d[name]
def digest(kind,identity):return hashlib.sha256((SALT+'\n'+kind+'\n'+identity).encode()).hexdigest()

def freeze():
    test=read(OUT/'GENERATED_DISCRETE_COVERAGE_TESTS.json');assert test['PASS']and test['cases']==480
    bindings={k:b(p)for k,p in {'root_decisions':ROOT/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json',
        'original_teacher_corpus':GT,'original_teacher_seal':V76/'gt_corpus_build/GT_ONLY_CORPUS_SEAL.json',
        'original_metadata_index':INDEX,'original_mask_teacher_rows':PERROW,'synth_inputs':SYNTH,'public_query_inputs':PUBLIC,
        'public_query_seal':V76/'public_query_emulation/SEAL.json','YCB_donor_instances':YCB,'teacher_helper':TEACHER,
        'teacher_contract':V76/'teacher_preparation/TARGET_CONTRACT_PROPOSAL_V2.json','native_grid':GRID,'box_rule':BOX,
        'source_proposal':V76/'source_failure_diagnosis/SOURCE_INTERPRETATION_AND_NEXT_STEP_PROPOSAL.json',
        'geometry':OUT/'opaque_pair_geometry.py','generated_tests':OUT/'GENERATED_DISCRETE_COVERAGE_TESTS.json','builder':Path(__file__)}.items()}
    save(OUT/'PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json',{'schema':'V77_SOURCE_OPAQUE_VISIBILITY_PAIR_BUILD_V1',
        'status':'FROZEN_BEFORE_CORPUS_SELECTION_AND_COMPOSITION','bindings':bindings,'original_rows':5203,'original_splits_unchanged':True,
        'eligibility':'At leastone original V76primary&label8VISIBLE(1); rederiveexactfactual geometryadmission from locked q/L/box/HW, no model scores or quality selection.',
        'original_primary':'teacherknown0/1 &finite/nonmissingq&continuous+nearestcenterbounds&native9ROIany&bilinearlocal96any&target128any&-.5<=(r-q)/L<.5 allaxes',
        'anchor':'One pereligible originalrow: minimum SHA256(salt+newline+anchor+newline+id+newline+nativek), fixed nativek tieorder.',
        'salt':SALT,'donors':'All actual nonempty officialYCBmask_visib instances in SAMEtrain/valpartition, includingthin donors; zeroarea ledgeronly. No donorquality/model filtering.',
        'donor_compatibility':'Not sameRGBsha; iftargetYCB, donorobjectID differs. SortedcanonicaldonorID pool, SHA256 salt/donor/originalid modulo compatiblecount; no retries.',
        'quarterturn':'SHA256 salt/turn/originalid modulo4; rotatesRGB andbinarymask andinteriorcenter together.',
        'geometry':'Donortightbbox paddedonezero pixel; EDT max firstrow-major interiorcenter c, d>=1. s=(rho+2)/(d-sqrt(2)/2-1e-6), WITHOUT max1. Targetnativep maps c+(p-r)/s. masknearestfloor+.5; RGBbilinear, blackOOB; fullyopaque. Destination clippedtoframe only.',
        'coverage_proof':'Every targetnativepixel withinR=rho+2 mapsdist<=d-sqrt2/2-eps; nearestsourcecenter distance<d, so donorinside.480generateddiscretecases cover1px/thin/holes/fractionalr/quarterturns/frameedges.',
        'teacher':'M0 unchanged; M4_after=M4_original&~pastedmask. Exact frozenV76 strong3x3-before-footprint helper rerunALL8, references/rho/selfunchanged. Syntheticraw>127 including1/2background; BOPraw>0.',
        'failures':'Everyeligible originalgetsafterrow; anchor actualteacher mustCOVERED0. Ifnot, retainRGB/actualteacher andexplicitfailedflag; pairedanchortrainingtargetU/loss0. No forceC, no resample/scale search/filter.',
        'nonanchors':'Keep all8teacherlabels; primaryknownnonanchors includeunchangedVISIBLE controls andcorrectC/U transitions. No global-anyoccluder relabel.',
        'unchanged_inputs':'q8/L/box/nativeorder/geometrycontext unchanged; afterRGBrequiresnewP3/local/targetRGB sampling in separateauthorizedproducer. PublicqremainsGT-derivedsourceemulation, not actualdeployedinitializer.',
        'input_manifest_whitelist':INPUT_KEYS,'input_lineage_whitelist':['original_row_index','original_id','after_id','after_frame_index','source','split','query_producer'],
        'GT_only':'anchor/donor/transforms/masks/ref/labels/self/admission/pairtargets remain separateGT_ONLY files; nativek enteredmodel onlythroughinitializerchannel convention.',
        'PNG':'ChangedfullnativeRGBPNG compress_level1 andpastedbinarymaskPNG; M4afterGTPNG. No overwrite.',
        'diagnostics_not_filters':['extent','nativeframeclipping','paintedtargetfraction','scale','othernativeVISIBLEretention','anchorcoverage'],
        'prohibited':{'current_data_results_scores':True,'model_scores_for_selection':True,'model_P3_weights_forward_fit':True,'PnP':True}})
    print('PROTOCOL_FROZEN',sha(OUT/'PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json'),flush=True)

def validate_protocol():
    p=read(OUT/'PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json')
    for x in p['bindings'].values():check(x)
    return p

def anycrop(q,L,hw,n):
    if not np.isfinite(q).all()or not np.isfinite(L)or L<=0:return False
    h,w=hw;u=((np.arange(n)+.5)/n-.5)*L
    return bool(np.any((q[0]+u>=-.5)&(q[0]+u<w-.5))and np.any((q[1]+u>=-.5)&(q[1]+u<h-.5)))

def source_metadata():
    idx=lines(INDEX);synth=lines(SYNTH);public=lines(PUBLIC)
    assert len(idx)==5203 and len(synth)==1003 and len(public)==4200
    fields=['id','source','split','label8','rule8','native_GTref8','rho_px','convex_self_proxy8_separate','weighted_expected_mass_A8']
    with np.load(GT,allow_pickle=False)as z:gt={k:z[k]for k in fields}
    grid=pure(GRID,'native_grid');boxfun=pure(BOX,'initial_query_box');rows=[]
    for i,meta in enumerate(idx):
        if i<1003:
            x=synth[i];q=np.asarray(x['unchanged_initial_query9'],float)[:8];L=float(x['input_L']);hw=x['raw_RGB']['raw_hw'];im=x['raw_RGB']['path'];ims=x['raw_RGB']['sha256'];producer='ORIGINAL_SEALED_N3_SYNTH_INITIALIZER'
        else:
            x=public[i-1003];q=np.asarray(x['initial_points'],float);L=float(x['roi_side']);hw=x['raw_hw'];im=x['image'];ims=x['image_sha256'];producer='GT_DERIVED_PUBLIC_SOURCE_CORRELATED_QUERY_EMULATION_NOT_DEPLOYED_INITIALIZER'
        assert meta['id']==x['id']==str(gt['id'][i])and meta['split']==str(gt['split'][i])
        box=boxfun(q,hw);_,gv,L2=grid(q,hw);assert L==L2
        h,w=hw;good=np.isfinite(q).all(1)&~(q==-1).all(1);px=np.floor(np.where(good[:,None],q,-1e8)+.5)
        qvalid=good&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)&(px[:,0]>=0)&(px[:,0]<w)&(px[:,1]>=0)&(px[:,1]<h)
        roi=gv.reshape(8,-1).any(1);local=np.array([anycrop(p,L,hw,96)for p in q]);target=False if box is None else anycrop((box[:2]+box[2:])/2,1.25*float(max(box[2:]-box[:2])),hw,128)
        off=(gt['native_GTref8'][i]-q)/L;square=np.isfinite(off).all(1)&((off>=-.5)&(off<.5)).all(1)
        geometry=qvalid&roi&local&target&square;primary=np.isin(gt['label8'][i],[0,1])&geometry
        anchors=np.flatnonzero(primary&(gt['label8'][i]==1))
        anchor=None if not len(anchors)else int(min(anchors,key=lambda k:(digest('anchor',meta['id']+'\n'+str(k)),int(k))))
        rows.append({'original_row_index':i,'original_id':meta['id'],'source':meta['source'],'split':meta['split'],
            'image':im,'image_sha256':ims,'raw_hw':hw,'initial_points':q,'roi_side':L,'box':box,'query_producer':producer,
            'geometry_admission8':geometry,'original_primary8':primary,'query_center_valid8':qvalid,'reference_in_local_square8':square,
            'original_visible_admitted_slots':anchors,'anchor_k':anchor,'eligible':anchor is not None})
    return idx,gt,rows

def index():
    validate_protocol();idx,gt,rows=source_metadata();donors=lines(YCB);assert len(donors)==4200
    pool=[]
    for i,y in enumerate(donors):
        raw=np.asarray(Image.open(y['mask_vis']));assert raw.shape==(y['height'],y['width'])
        mask=raw>0;area=int(mask.sum());identity=f'YCB:{y["scene_id"]:06d}:{y["frame_id"]:06d}:{y["instance_id"]:06d}'
        rec={'donor_row_index':i,'id':identity,'split':y['split'],'object_id':y['object_id'],'image':y['image'],'image_sha256':y['image_sha256'],
            'mask_vis':b(Path(y['mask_vis'])),'raw_hw':[y['height'],y['width']],'actual_visible_pixels':area,'usable_nonempty':area>0}
        if area:
            geom=donor_geometry(mask);rec.update(tight_box_xyxy=geom['box'],center_in_padded_tight_crop_xy=geom['center_xy'],EDT_max=geom['interior_distance'])
        pool.append(clean(rec))
        if(i+1)%1000==0:print('DONOR_POOL_FACTUAL_MASKS',i+1,'/4200',flush=True)
    save(OUT/'DONOR_POOL_LOCK.json',{'completed':True,'all4200rows':pool,'selection_filter':'Only actualempty mask unusable; no otherqualityfilter','nonempty':sum(d['usable_nonempty']for d in pool),'zeroarea_unusable':sum(not d['usable_nonempty']for d in pool)})
    selected=[];counts=Counter()
    with(OUT/'ORIGINAL_5203_ELIGIBILITY_GT_ONLY.jsonl').open('x')as f:
        for row in rows:
            jsonline(f,row);counts[row['source']+'/'+row['split']+'/all']+=1
            if not row['eligible']:counts[row['source']+'/'+row['split']+'/noneligible']+=1;continue
            i=row['original_row_index'];ownobj=idx[i].get('object_id')
            compatible=sorted([d for d in pool if d['usable_nonempty']and d['split']==row['split']and d['image_sha256']!=row['image_sha256']and(ownobj is None or d['object_id']!=ownobj)],key=lambda d:d['id'])
            assert compatible,'No compatible factualdonor; stop instead ofsubstitute'
            choice=int(digest('donor',row['original_id']),16)%len(compatible);donor=compatible[choice]
            selected.append({**clean(row),'after_frame_index':len(selected),'after_id':row['original_id']+'::V77_OPAQUE_AFTER',
                'donor':donor,'compatible_donor_count':len(compatible),'selected_donor_index':choice,'quarterturn':int(digest('turn',row['original_id']),16)%4})
            counts[row['source']+'/'+row['split']+'/eligible']+=1
    with(OUT/'PAIR_SELECTION_GT_ONLY.jsonl').open('x')as f:
        for row in selected:jsonline(f,row)
    save(OUT/'METADATA_SELECTION_LOCK_BEFORE_COMPOSITION.json',{'completed':True,'original_rows':5203,'after_rows':len(selected),'counts':dict(counts),
        'RGB_bodies_or_compositions_read':0,'all_originalrows_ledger':b(OUT/'ORIGINAL_5203_ELIGIBILITY_GT_ONLY.jsonl'),'selection':b(OUT/'PAIR_SELECTION_GT_ONLY.jsonl'),
        'donor_pool':b(OUT/'DONOR_POOL_LOCK.json'),'protocol':b(OUT/'PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json'),'model_scores_current_reads':0})
    print('ALL_METADATA_LOCKED',len(selected),dict(counts),flush=True)

@lru_cache(maxsize=96)
def rgb(path,expected):
    assert sha(path)==expected,path
    return np.asarray(Image.open(path).convert('RGB'))

def build():
    validate_protocol();lock=read(OUT/'METADATA_SELECTION_LOCK_BEFORE_COMPOSITION.json')
    for key in ['all_originalrows_ledger','selection','donor_pool','protocol']:check(lock[key])
    idx,gt,rows=source_metadata();selected=lines(OUT/'PAIR_SELECTION_GT_ONLY.jsonl');assert len(selected)==lock['after_rows']
    originals=lines(PERROW);assert len(originals)==5203
    spec=importlib.util.spec_from_file_location('frozen_local_support_teacher',TEACHER);T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
    dirs={k:OUT/k for k in ['images','pasted_masks_GT_ONLY','after_visible_masks_GT_ONLY']}
    for d in dirs.values():d.mkdir(exist_ok=False)
    fields=['label','rule','GTref_valid_xy','center_M0','center_M4','strong_3x3_window_valid','strong_3x3_full_M0','strong_3x3_visible','strong_3x3_covered',
        'expected_support_pixels_A','visible_support_pixels_V','weighted_expected_mass_A','weighted_visible_mass_V','visible_over_expected_ratio','footprint_support_adequate',
        'valid_footprint_pixels','full_disk_lattice_pixels','footprint_valid_fraction','near_boundary']
    collected={k:[]for k in fields};success=[];stats=Counter();start=time.monotonic()
    paths={n:(OUT/n).open('x')for n in ['INPUTS_RGB_ONLY.jsonl','INPUT_LINEAGE.jsonl','PAIR_LINEAGE_GT_ONLY.jsonl','AFTER_TEACHERS_GT_ONLY.jsonl']}
    try:
        for ai,s in enumerate(selected):
            i=s['original_row_index'];row=rows[i];meta=idx[i];k=s['anchor_k'];assert ai==s['after_frame_index']and s['original_id']==row['original_id']and row['anchor_k']==k
            q=np.asarray(s['initial_points'],float);r=gt['native_GTref8'][i];self_flag=gt['convex_self_proxy8_separate'][i];rho=float(gt['rho_px'][i])
            assert row['original_primary8'][k]and gt['label8'][i,k]==1
            for name in ['M0','M4']:check(originals[i]['mask_pair'][name])
            threshold=127 if row['source']=='SYNTHETIC'else 0
            M0=np.asarray(Image.open(meta['M0_path']))>threshold;M4=np.asarray(Image.open(meta['M4_path']))>threshold
            assert not np.any(M4&~M0)
            before=T.reference_local_support(M0,M4,r,self_proxy8=self_flag)
            np.testing.assert_array_equal(before['label'],gt['label8'][i]);assert float(before['rho_px'])==rho
            donor=s['donor'];check(donor['mask_vis']);dm=np.asarray(Image.open(donor['mask_vis']['path']))>0;g=donor_geometry(dm)
            np.testing.assert_array_equal(g['box'],donor['tight_box_xyxy']);np.testing.assert_array_equal(g['center_xy'],donor['center_in_padded_tight_crop_xy']);assert g['interior_distance']==donor['EDT_max']
            drgb=rgb(donor['image'],donor['image_sha256']);x0,y0,x1,y1=g['box'];dcut=np.pad(drgb[y0:y1,x0:x1],((1,1),(1,1),(0,0)),constant_values=0)
            mask,color,c=rotate(g['mask'],dcut,g['center_xy'],s['quarterturn'])
            original=rgb(row['image'],row['image_sha256']);assert list(original.shape[:2])==row['raw_hw']
            after,paste,transform=compose(original,color,mask,c,g['interior_distance'],r[k],rho)
            M4after=M4&~paste;assert not np.any(M4after&~M0)
            teacher=T.reference_local_support(M0,M4after,r,self_proxy8=self_flag)
            np.testing.assert_array_equal(teacher['weighted_expected_mass_A'],before['weighted_expected_mass_A'])
            assert not np.any((gt['label8'][i]==0)&(teacher['label']!=0))and not np.any((gt['label8'][i]!=1)&(teacher['label']==1))
            ok=bool(transform['disk_all_covered']and teacher['label'][k]==0);success.append(ok)
            name=f'{ai:06d}';im=dirs['images']/(name+'.png');pm=dirs['pasted_masks_GT_ONLY']/(name+'.png');vm=dirs['after_visible_masks_GT_ONLY']/(name+'.png')
            Image.fromarray(after).save(im,compress_level=1);Image.fromarray(paste.astype(np.uint8)*255).save(pm,compress_level=1);Image.fromarray(M4after.astype(np.uint8)*255).save(vm,compress_level=1)
            inp=dict(frame_index=ai,id=s['after_id'],source=row['source'],split=row['split'],image=str(im),image_sha256=sha(im),raw_hw=row['raw_hw'],initial_points=q,roi_side=row['roi_side'],box=row['box'])
            assert list(inp)==INPUT_KEYS;jsonline(paths['INPUTS_RGB_ONLY.jsonl'],inp)
            lineage=dict(original_row_index=i,original_id=row['original_id'],after_id=s['after_id'],after_frame_index=ai,source=row['source'],split=row['split'],query_producer=row['query_producer'])
            jsonline(paths['INPUT_LINEAGE.jsonl'],lineage)
            after_primary=np.isin(teacher['label'],[0,1])&row['geometry_admission8'];nonanchor=np.arange(8)!=k
            targetfraction=float(np.sum(paste&M0)/max(int(M0.sum()),1));paintfraction=float(paste.mean())
            pair=dict(**lineage,anchor_k=k,anchor_success_COVERED=ok,anchor_training_target=0 if ok else-1,original_teacher_label8=gt['label8'][i],
                original_primary8=row['original_primary8'],after_primary8=after_primary,query_center_valid8=row['query_center_valid8'],reference_in_local_square8=row['reference_in_local_square8'],
                native_GTref8=r,convex_self_visible_proxy8_separate=self_flag,donor=donor,quarterturn=s['quarterturn'],transform=transform,
                M0_unchanged=originals[i]['mask_pair']['M0'],M4_original=originals[i]['mask_pair']['M4'],M4_after=b(vm),pasted_mask=b(pm),after_RGB=b(im),before_RGB={'path':row['image'],'sha256':row['image_sha256']},
                painted_expected_target_fraction=targetfraction,painted_frame_fraction=paintfraction,geometry_context_q_L_box_unchanged=True,no_score_selection=True)
            jsonline(paths['PAIR_LINEAGE_GT_ONLY.jsonl'],pair)
            jsonline(paths['AFTER_TEACHERS_GT_ONLY.jsonl'],dict(after_frame_index=ai,original_row_index=i,id=s['after_id'],source=row['source'],split=row['split'],anchor_k=k,anchor_success=ok,teacher=teacher))
            for key in fields:collected[key].append(teacher[key])
            prefix=row['source']+'/'+row['split']
            stats[prefix+'/rows']+=1;stats[prefix+'/anchor_C_success']+=int(ok);stats[prefix+'/anchor_failure_U']+=int(not ok)
            stats[prefix+'/disk_coverage_fail']+=int(not transform['disk_all_covered']);stats[prefix+'/extent_clipped']+=int(transform['destination_out_of_bounds'])
            for label in [-1,0,1]:
                stats[prefix+'/teacher_label_'+str(label)]+=int(np.sum(teacher['label']==label));stats[prefix+'/primary_label_'+str(label)]+=int(np.sum(after_primary&(teacher['label']==label)))
                stats[prefix+'/nonanchor_primary_label_'+str(label)]+=int(np.sum(after_primary&nonanchor&(teacher['label']==label)))
            stats[prefix+'/nonanchor_originalV_stillV']+=int(np.sum(nonanchor&(gt['label8'][i]==1)&(teacher['label']==1)))
            for old,new in zip(gt['label8'][i],teacher['label']):stats[prefix+f'/transition_{old}_to_{new}']+=1
            if(ai+1)%100==0 or ai+1==len(selected):print('OPAQUE_PAIRS',ai+1,'/',len(selected),'seconds',round(time.monotonic()-start,1),'anchor_fail',sum(not x for x in success),flush=True)
    finally:
        for f in paths.values():f.close()
    ii=np.asarray([s['original_row_index']for s in selected]);anchors=np.asarray([s['anchor_k']for s in selected],np.int8)
    output={k+'8':np.asarray(v)for k,v in collected.items()}
    output.update(after_row_index=np.arange(len(selected),dtype=np.int32),original_row_index=ii,id=np.asarray([s['after_id']for s in selected]),
        original_id=gt['id'][ii],anchor_k=anchors,pair_anchor_success=np.asarray(success,bool),anchor_training_target=np.where(success,0,-1).astype(np.int8),
        native_GTref8=gt['native_GTref8'][ii],convex_self_proxy8_separate=gt['convex_self_proxy8_separate'][ii],split=gt['split'][ii],source=gt['source'][ii],
        rho_px=gt['rho_px'][ii],original_label8=gt['label8'][ii],original_primary8=np.asarray([rows[i]['original_primary8']for i in ii]),
        query_valid8=np.asarray([rows[i]['query_center_valid8']for i in ii]),reference_in_local_square8=np.asarray([rows[i]['reference_in_local_square8']for i in ii]),
        human_semantic_visibility8=np.full((len(selected),8),-1,np.int8))
    np.savez_compressed(OUT/'AFTER_GT_ONLY.npz',**output)
    summary={'completed':True,'original_baseline_rows':5203,'after_rows':len(selected),'all_eligible_retained':True,'all_noneligible_retained_in_ledger':True,
        'anchor_success':sum(success),'anchor_failure_retained_U':sum(not x for x in success),'by_source_split':dict(stats),'seconds':time.monotonic()-start,
        'field_whitelist':INPUT_KEYS,'new_model_P3_fit_current_PnP':0,'source_query_emulation_caveat':'Public q remainsGT-derivedsourceemulation; no realdeployedinitializer claim',
        'self_flag':'Separate convexvisible1/mayvisible,0selfhidden,-1U; nevermerged withsupport',
        'nan_ratios':'NPZNaN andJSONnull; noinvalidsubstitution','failed_anchor_no_retry_or_force':True}
    save(OUT/'BUILD_SUMMARY.json',summary)
    verify()

def verify():
    validate_protocol();selection=lines(OUT/'PAIR_SELECTION_GT_ONLY.jsonl');manifest=lines(OUT/'INPUTS_RGB_ONLY.jsonl');lineage=lines(OUT/'INPUT_LINEAGE.jsonl')
    assert len(selection)==len(manifest)==len(lineage)
    with np.load(OUT/'AFTER_GT_ONLY.npz',allow_pickle=False)as z:
        assert z['label8'].shape==(len(selection),8)and np.array_equal(z['after_row_index'],np.arange(len(selection)))
        assert z['id'].tolist()==[r['id']for r in manifest]and z['original_row_index'].tolist()==[r['original_row_index']for r in lineage]
    for i,(s,m,l)in enumerate(zip(selection,manifest,lineage)):
        assert list(m)==INPUT_KEYS and m['frame_index']==i and m['id']==l['after_id']==s['after_id']
        assert l['original_row_index']==s['original_row_index']and m['source']==s['source']and m['split']==s['split']
        np.testing.assert_array_equal(m['initial_points'],s['initial_points']);assert m['roi_side']==s['roi_side']and m['box']==s['box']
        assert set(l)=={'original_row_index','original_id','after_id','after_frame_index','source','split','query_producer'}
        assert s['donor']['split']==s['split']and s['donor']['image_sha256']!=s['image_sha256']
    save(OUT/'ALL_METADATA_IDENTITY_AND_WHITELIST_QA.json',{'PASS':True,'after_rows':len(manifest),'all_q_L_box_source_split_nativeorder_exact':True,
        'input_GT_anchor_donor_reference_label_fields':0,'all_original_index_joins_exact':True,'model_forward_fit_current_PnP':0})
    outputs=[b(p)for p in sorted(OUT.iterdir())if p.is_file()and p.name!='SEAL.json']
    save(OUT/'SEAL.json',{'completed':True,'schema':'V77_ACTUAL_SOURCE_OPAQUE_PAIR_DATA_SEAL','protocol':b(OUT/'PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json'),
        'input_manifest':b(OUT/'INPUTS_RGB_ONLY.jsonl'),'input_lineage':b(OUT/'INPUT_LINEAGE.jsonl'),'GT':b(OUT/'AFTER_GT_ONLY.npz'),
        'pair_lineage_GT':b(OUT/'PAIR_LINEAGE_GT_ONLY.jsonl'),'outputs':outputs,'all_native_RGB_and_masks_bound_perrow':True,
        'source_only':True,'fit_current_authorization':False})
    print('PAIRS_SEALED',len(manifest),sha(OUT/'SEAL.json'),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','index','build']);args=a.parse_args();{'freeze':freeze,'index':index,'build':build}[args.stage]()
