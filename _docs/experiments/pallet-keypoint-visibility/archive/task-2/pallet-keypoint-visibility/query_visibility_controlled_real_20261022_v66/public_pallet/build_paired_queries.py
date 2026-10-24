#!/usr/bin/env python3
"""Frozen paired-public-source queries and partial labels, CPU only.

Feature inputs are saved independently from teacher/depth/pose/evidence paths.
No feature/model call or fitting is performed. Labels follow the actual jittered
native query, never the preferred anchor category.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, importlib.util, json, sys, time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
V66 = ROOT.parent
OUT = ROOT / 'PAIRED_QUERIES'
RECIPE = V66 / 'PAIRED_QUERY_RECIPE.json'
RECIPE_SHA = '513b6cf28855f18a35c683dfcc1b1133f4b4e571a86ff0a0bd08ae942133fb8b'
EVIDENCE = ROOT / 'SYMMETRIC_ALL50_V2/PAIR_EVIDENCE_MANIFEST.jsonl'
LOCK = OUT / 'QUERY_INPUT_LOCK.json'
INTERPRETATION = OUT / 'QUERY_RECIPE_INTERPRETATION.json'
GRID_PATH = V66.parent / 'query_visibility_amodal_global_20261022_v65/global_recipe.py'
CERT_KEYS = ('visible_positive', 'visible_negative', 'full_positive', 'full_negative')
cv2.setNumThreads(1)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ref(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p))

def bind(r):
    assert sha(r['path']) == r['sha256'], r['path']

def write(p, j):
    Path(p).write_text(json.dumps(j, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def seed_for(pair_id, stage):
    data = ('20261022:' + pair_id + ':' + stage).encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'little')

def freeze():
    assert sha(RECIPE) == RECIPE_SHA
    recipe = json.loads(RECIPE.read_text())
    for b in recipe['bindings']:
        bind(b)
    assert not OUT.exists(), 'Preserve immutable query outputs'
    OUT.mkdir()
    interpretation = dict(schema='PAIRED_QUERY_RECIPE_ROOT_APPROVED_INTERPRETATION_V1',
        original_recipe=ref(RECIPE), authority='Root explicitly approved both interpretations before query generation.',
        geometric_fallback='First/even slot within category uses rounded roughbox TL; second/odd uses BR. Geometric fallback and fixed slots6/7 may reuse anchors. All actual pixel pools exclude globally used anchors.',
        exhausted_preferred='An otherwise nonempty pool with all anchors already used proceeds to the next frozen fallback. Never reuse pixel-pool anchors.',
        anchor_RNG='One anchor-stage default_rng used sequentially for slots0..5. Row-major eligible flattened native-pixel array passed to rng.choice. Geometric fallback and fixed slots6/7 consume no anchor RNG draws.',
        box_RNG='Draw uniform[-.05,.05] shape2 for center shift, then uniform[.9,1.1] shape2 for width/height scale; union teacher bbox uses maxcoords+1 before jitter/clamp.',
        jitter_RNG='Distinct jitter-stage default_rng uniform(-3,3,size=(8,2)); identical float64 q8 for both views; no post-jitter repair.',
        split='Feature manifest TRAIN/VALIDATION, with source_split TRAIN_LAB/VAL_HALL metadata. Label NPZ files use TRAIN/VAL name aliases for existing source-label joins.',
        query_alias='query and query8 are identical input-only lists for source feature-worker compatibility.',
        source='PALLOX', global_id='id',
        native9='float64 q+(arange9-4)*side/9; independent x/y meshgrid; nearest floor(xy+.5).',
        global64='Unchanged bound v65 global_recipe.grid(box) and native_valid; 1.25 expansion, 64x64 half-cell grid.',
        invalid='valid=false, allowed63, exact/radius2/joint=-1, joint_known=false; invalid cells never supervise. Invalid radius2 count0/strict flagsfalse are non-label placeholders.',
        partial_labels='No implicit O/full-negative/self. Shared full-positive Va|Vb only under frozen conditional-teacher assumptions.',
        no_models_private_GPU_PnP_or_fit=True)
    write(INTERPRETATION, interpretation)
    source = rows(EVIDENCE)
    assert len(source) == 50
    bindings = {}
    for r in source:
        bind(r['evidence']); bindings[r['evidence']['path']] = r['evidence']['sha256']
        for view in ['A', 'B']:
            for b in r['files'][view].values():
                bind(b); bindings[b['path']] = b['sha256']
    deny = V66.parent / 'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json'
    code = [Path(__file__), V66/'contract/paired_truth.py', V66/'contract/partial_truth_grid.py',
            V66/'contract/PARTIAL_TRUTH_GRID_MOCK_PROOF.json', GRID_PATH]
    lock = dict(schema='PAIRED_SHARED_QUERY_INPUT_LOCK_BEFORE_CPU_GENERATION_V1',
        recipe=ref(RECIPE), interpretation=ref(INTERPRETATION),
        source_evidence_manifest=ref(EVIDENCE), label_protocol=ref(V66/'PAIRED_REAL_LABEL_PROTOCOL.json'),
        source_split_lock=ref(ROOT/'SPLIT_LOCK_BEFORE_FIT.json'),
        conditional_admission=ref(ROOT/'INTAKE_ADMISSION_STATUS.json'),
        denylist=ref(deny), code=[ref(p) for p in code],
        source_files=[dict(path=p, sha256=s) for p,s in sorted(bindings.items())],
        expected=dict(pairs=50,targets=100,queries=800,TRAIN_targets=44,VALIDATION_targets=56),
        runtime=dict(python=sys.executable,numpy=np.__version__,cv2=cv2.__version__),
        feature_extraction_started=False, GPU_models_fit_private_PnP_calls=0)
    write(LOCK, lock)
    print('QUERY_INPUT_LOCK', sha(LOCK), 'source_files',len(bindings),flush=True)

def sample_grid(grid, xy, global_recipe):
    pixel = np.floor(xy + .5).astype(np.int64)
    h,w = grid['exact_state'].shape
    valid = global_recipe.native_valid(xy,pixel,[h,w])
    x,y = np.clip(pixel[...,0],0,w-1),np.clip(pixel[...,1],0,h-1)
    result = dict(valid=valid)
    for key in ['exact_state','radius2_state','joint_state']:
        result[key] = np.where(valid,grid[key][y,x],-1).astype(np.int8)
    result['allowed'] = np.where(valid,grid['joint_compatible_mask'][y,x],63).astype(np.uint8)
    result['joint_known'] = valid & grid['joint_known'][y,x]
    for key in ['exact_possible_mask','radius2_possible_mask']:
        result[key] = np.where(valid,grid[key][y,x],7).astype(np.uint8)
    for key in ['radius2_strict_E_core','radius2_strict_O_core','radius2_clipped_by_frame']:
        result[key] = valid & grid[key][y,x]
    result['radius2_window_pixels'] = np.where(valid,grid['radius2_window_pixels'][y,x],0).astype(np.uint8)
    return result

def anchor_queries(r, evidence):
    pair_id = r['pair_id']
    seeds = {stage:seed_for(pair_id,stage) for stage in ['box','anchor','jitter']}
    rgbox, rganchor, rgjitter = [np.random.default_rng(seeds[stage]) for stage in ['box','anchor','jitter']]
    originals=[]
    for v in ['A','B']:
        path=r['files'][v]['mask_teacher']['path']
        originals.append(cv2.imread(path,cv2.IMREAD_UNCHANGED)>0)
    mask=originals[0]|originals[1]
    y,x=np.where(mask);assert len(x),'Empty original teacher union: fail, do not silently drop pair'
    original_box=np.array([x.min(),y.min(),x.max()+1,y.max()+1],np.float64)
    size=original_box[2:]-original_box[:2];center=(original_box[2:]+original_box[:2])/2
    shift=rgbox.uniform(-.05,.05,size=2);scale=rgbox.uniform(.9,1.1,size=2)
    newcenter=center+shift*size;newsize=size*scale
    box=np.r_[newcenter-newsize/2,newcenter+newsize/2]
    box=np.clip(box,[0,0,0,0],[639,479,639,479])
    assert (box[2:]>box[:2]).all(),'Degenerate box: fail explicitly'
    corners=np.floor(box.reshape(2,2)+.5).astype(np.int64)
    Va,Vb,Ea,Eb=[evidence[k] for k in ['visible_A','visible_B','external_A','external_B']]
    union=Va|Vb
    erode=lambda m:cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    pools={'E_core_union':erode(Ea)|erode(Eb),'E_union':Ea|Eb,
           'V_union':union,'commonV_core':erode(Va&Vb),'V_boundary':union&~erode(union)}
    chains={0:['E_core_union','E_union','V_union'],1:['E_core_union','E_union','V_union'],
            2:['commonV_core','V_union'],3:['commonV_core','V_union'],
            4:['V_boundary','V_union'],5:['V_boundary','V_union']}
    category=['E_core','E_core','commonV','commonV','V_boundary','V_boundary','roughbox_TL','roughbox_BR']
    used=set();anchors=[];selection=[]
    for k in range(8):
        chosen=None;attempts=[]
        if k<6:
            for level,name in enumerate(chains[k]):
                flat=np.flatnonzero(pools[name]);eligible=np.array([n for n in flat if int(n)not in used],np.int64)
                attempts.append(dict(pool=name,pool_pixels=int(len(flat)),eligible_unused_pixels=int(len(eligible))))
                if eligible.size:
                    n=int(rganchor.choice(eligible));chosen=np.array([n%640,n//640],np.int64)
                    selected=name;fallback=level;geometric=False;break
        if chosen is None:
            chosen=corners[k%2].copy();selected='roughbox_TL'if k%2==0 else'roughbox_BR'
            fallback=len(chains[k])if k<6 else 0;geometric=True
        anchors.append(chosen);used.add(int(chosen[1])*640+int(chosen[0]))
        selection.append(dict(slot=k,category=category[k],selected_pool=selected,fallback_level=fallback,
                              geometric_reuse_permitted=geometric,pool_attempts=attempts,anchor_xy=chosen.tolist()))
    anchors=np.asarray(anchors,np.int64);jitter=rgjitter.uniform(-3,3,size=(8,2));query=anchors.astype(np.float64)+jitter
    side=float(np.clip(.25*max(np.ptp(query[:,0]),np.ptp(query[:,1])),16,640))
    return dict(pair_id=pair_id,rng_seeds=seeds,original_teacher_union_box=original_box.tolist(),
                box_shift_fraction=shift.tolist(),box_scale=scale.tolist(),box=box.tolist(),
                anchors=anchors.tolist(),selection=selection,jitter=jitter.tolist(),query=query.tolist(),side=side)

def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)

def make_sheet(inputs, selections, qrows, manifest):
    fixed=set(json.loads((ROOT/'FIXED_SIX_PAIR_SELECTION.json').read_text())['pair_ids'])
    byid={r['id']:r for r in inputs};qm={r['sample_id']:r for r in qrows};sm={r['pair_id']:r for r in selections}
    canvas=Image.new('RGB',(1160,110+6*285),'white');d=ImageDraw.Draw(canvas)
    d.text((12,8),'Fixed six public pairs: slot0 actual shared query, native96 crop',font=font(23),fill='black')
    d.text((12,40),'A full / A crop / B full / B crop. Magenta=query; green=V teacher; orange=E teacher.',font=font(18),fill='black')
    d.text((12,70),'Labels are read at actual jittered query; anchor category is not truth. Native alignment conditional, no model scores.',font=font(16),fill='black')
    for i,p in enumerate([r for r in manifest if r['pair_id']in fixed]):
        y=110+i*285;group='PALLOXPAIR:'+p['pair_id'];q=np.array(sm[p['pair_id']]['query'][0]);center=np.floor(q+.5).astype(int)
        d.text((12,y),p['pair_id']+'  q='+np.array2string(q,precision=2),font=font(18),fill='black')
        with np.load(p['evidence']['path'])as z:
            for vi,view in enumerate(['A','B']):
                r=byid[group+':'+view];qr=qm[r['id']+':0'];rgb=np.array(Image.open(r['image']).convert('RGB'))
                full=Image.fromarray(rgb);fd=ImageDraw.Draw(full);fd.line((q[0]-9,q[1],q[0]+9,q[1]),fill=(255,0,220),width=3);fd.line((q[0],q[1]-9,q[0],q[1]+9),fill=(255,0,220),width=3)
                canvas.paste(full.resize((280,210),Image.Resampling.BILINEAR),(12+vi*580,y+28))
                xx0,yy0=center[0]-48,center[1]-48;xx1,yy1=xx0+96,yy0+96
                crop=np.full((96,96,3),30,np.uint8);vp=np.zeros((96,96),bool);ep=np.zeros_like(vp)
                sx0,sy0=max(0,xx0),max(0,yy0);sx1,sy1=min(640,xx1),min(480,yy1)
                if sx1>sx0 and sy1>sy0:
                    sl=(slice(sy0-yy0,sy1-yy0),slice(sx0-xx0,sx1-xx0));ss=(slice(sy0,sy1),slice(sx0,sx1))
                    crop[sl]=rgb[ss];vp[sl]=z['visible_'+view][ss];ep[sl]=z['external_'+view][ss]
                over=crop.astype(float);over[vp]=.7*over[vp]+.3*np.array([0,255,50]);over[ep]=.5*over[ep]+.5*np.array([255,125,0])
                im=Image.fromarray(over.astype(np.uint8));cd=ImageDraw.Draw(im);cx,cy=q[0]-xx0,q[1]-yy0;cd.line((cx-5,cy,cx+5,cy),fill=(255,0,220),width=1);cd.line((cx,cy-5,cx,cy+5),fill=(255,0,220),width=1)
                canvas.paste(im.resize((210,210),Image.Resampling.NEAREST),(310+vi*580,y+28))
                text=f"{view}: exact {qr['exact_state']} near {qr['radius2_state']} allowed {qr['joint_compatible_mask']} {'valid'if qr['valid']else'invalid'}"
                d.text((12+vi*580,y+247),text,font=font(16),fill='black')
    canvas.save(OUT/'FIXED_SIX_SLOT0_NATIVE96.png')

def build():
    lock=json.loads(LOCK.read_text())
    for key in ['recipe','interpretation','source_evidence_manifest','label_protocol','source_split_lock','conditional_admission','denylist']:
        bind(lock[key])
    for b in lock['code']+lock['source_files']:
        bind(b)
    assert not(OUT/'INPUTS_RGB_ONLY.jsonl').exists(),'Preserve previous generation'
    grid_recipe=load_module(GRID_PATH,'paired_query_global_recipe')
    partial=load_module(V66/'contract/partial_truth_grid.py','paired_query_partial_truth_grid')
    paired=load_module(V66/'contract/paired_truth.py','paired_query_paired_truth')
    manifest=rows(EVIDENCE);denied=set(json.loads(Path(lock['denylist']['path']).read_text())['image_sha256'])
    all_inputs=[];all_labels=[];selections=[];dst={};counts={};started=time.monotonic()
    for split in ['TRAIN','VALIDATION']:
        keys=['sample_id','global_id','target_index','native_corner','dense_allowed','dense_valid','dense_exact_state','dense_radius2_state',
              'center_allowed','center_valid','center_joint_known','center_joint_state','center_exact_state','center_radius2_state',
              'global_allowed','global_valid','global_exact_state','global_radius2_state','image_sha256','global_image_sha256','query','side','global_box']
        dst[split]={k:[]for k in keys};counts[split]=Counter()
    for pi,p in enumerate(manifest):
        with np.load(p['evidence']['path'])as z:e={k:z[k]for k in z.files}
        selection=anchor_queries(p,e);selections.append(selection);q=np.asarray(selection['query'],np.float64);side=selection['side'];box=selection['box']
        a=(np.arange(9,dtype=np.float64)-4)*side/9;xx,yy=np.meshgrid(a,a);native_xy=q[:,None,None,:]+np.stack((xx,yy),-1)[None]
        global_xy,_,_,_=grid_recipe.grid(box)
        split='TRAIN'if p['split']=='TRAIN_LAB'else'VALIDATION';dest=dst[split];count=counts[split]
        for view in ['A','B']:
            group='PALLOXPAIR:'+p['pair_id'];ident=group+':'+view;image=p['files'][view]['image']
            assert image['sha256']not in denied;bind(image)
            inp=dict(id=ident,global_id=ident,group_id=group,split=split,source_split=p['split'],source='PALLOX',
                     image=image['path'],image_sha256=image['sha256'],raw_hw=[480,640],query=q.tolist(),query8=q.tolist(),side=side,box=box)
            assert not any(k in inp for k in ['depth','pose','mask','evidence','labels']);all_inputs.append(inp)
            cert={k:e[k+'_'+view]for k in CERT_KEYS}
            assert not cert['full_negative'].any()
            grid=partial.partial_truth_grid(cert);local=sample_grid(grid,native_xy,grid_recipe);glob=sample_grid(grid,global_xy,grid_recipe)
            ti=len(dest['global_id']);dest['global_id'].append(ident);dest['global_image_sha256'].append(image['sha256']);dest['global_box'].append(box)
            for key in ['allowed','valid','exact_state','radius2_state']:dest['global_'+key].append(glob[key])
            dest['sample_id'].extend([ident+':'+str(k)for k in range(8)]);dest['target_index'].extend([ti]*8);dest['native_corner'].extend(range(8));dest['image_sha256'].extend([image['sha256']]*8);dest['query'].extend(q);dest['side'].extend([side]*8)
            for key in ['allowed','valid','exact_state','radius2_state']:dest['dense_'+key].append(local[key])
            for key in ['allowed','valid','joint_known','joint_state','exact_state','radius2_state']:dest['center_'+key].extend(local[key][:,4,4])
            for k in range(8):
                s=selection['selection'][k];n=q[k];c={key:(bool(value[k,4,4])if value.dtype==bool else int(value[k,4,4]))for key,value in local.items()}
                qr=dict(sample_id=ident+':'+str(k),id=ident,group_id=group,pair_id=p['pair_id'],view=view,slot=k,split=split,
                        anchor_category=s['category'],anchor_selection=s,anchor_xy=selection['anchors'][k],jitter=selection['jitter'][k],query=n.tolist(),
                        valid=c['valid'],exact_state=c['exact_state'],radius2_state=c['radius2_state'],joint_known=c['joint_known'],joint_state=c['joint_state'],
                        joint_compatible_mask=c['allowed'],joint_compatible_codes=[i for i in range(6)if c['allowed']&(1<<i)],
                        exact_possible_mask=c['exact_possible_mask'],radius2_possible_mask=c['radius2_possible_mask'],
                        radius2_strict_E_core=c['radius2_strict_E_core'],radius2_clipped_by_frame=c['radius2_clipped_by_frame'],
                        radius2_window_pixels=c['radius2_window_pixels'],semantic_self=None,corner_semantic_truth=None,evidence_tier='CONDITIONAL_TEACHER_NOT_GOLD')
                # Independent scalar contract verifies each actual query center.
                scalar=paired.query_truth({**cert,'exact_state':np.where(cert['visible_positive'],0,np.where(cert['visible_negative'],1,-1)).astype(np.int8)},n)
                assert scalar['valid']==qr['valid']
                for key in ['exact_state','radius2_state','joint_known','joint_state','joint_compatible_mask']:
                    expected=scalar[key]if scalar['valid']else(63 if key=='joint_compatible_mask'else False if key=='joint_known'else -1)
                    assert qr[key]==expected,(ident,k,key,qr[key],expected)
                all_labels.append(qr)
                count[('center_exact',qr['exact_state'])]+=1;count[('center_radius2',qr['radius2_state'])]+=1;count[('center_allowed',qr['joint_compatible_mask'])]+=1
                count[('center_valid',int(qr['valid']))]+=1;count[('center_joint_known',int(qr['joint_known']))]+=1
                count[('selected_pool',s['selected_pool'])]+=1;count[('anchor_fallback_level',s['fallback_level'])]+=1
            for name,data in [('dense_allowed',local['allowed'][local['valid']]),('global_allowed',glob['allowed'][glob['valid']])]:
                for value,n in zip(*np.unique(data,return_counts=True)):count[(name,int(value))]+=int(n)
        if(pi+1)%10==0:print('PAIR_QUERIES',pi+1,50,'seconds',round(time.monotonic()-started,2),flush=True)
    assert len(all_inputs)==100 and len(all_labels)==800 and len(selections)==50
    assert len({r['sample_id']for r in all_labels})==800 and len({r['id']for r in all_inputs})==100
    for i in range(0,100,2):
        for key in ['group_id','query','query8','box','side','split']:assert all_inputs[i][key]==all_inputs[i+1][key]
    hashes={s:{r['image_sha256']for r in all_inputs if r['split']==s}for s in ['TRAIN','VALIDATION']};assert not hashes['TRAIN']&hashes['VALIDATION']
    for filename,rr in [('INPUTS_RGB_ONLY.jsonl',all_inputs),('QUERIES_LABELS.jsonl',all_labels),('ANCHOR_SELECTION.jsonl',selections)]:
        with(OUT/filename).open('w')as f:
            for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    outputs=[]
    for split in ['TRAIN','VALIDATION']:
        arr={key:np.concatenate(value,axis=0)if key.startswith('dense_')else np.asarray(value)for key,value in dst[split].items()}
        arr['target_index']=arr['target_index'].astype(np.int64);arr['native_corner']=arr['native_corner'].astype(np.uint8)
        for key in ['center_joint_state','center_exact_state','center_radius2_state']:arr[key]=arr[key].astype(np.int8)
        for key in ['center_joint_known','center_valid']:arr[key]=arr[key].astype(bool)
        arr['center_allowed']=arr['center_allowed'].astype(np.uint8)
        n,t=len(arr['sample_id']),len(arr['global_id']);assert n==8*t
        assert arr['dense_allowed'].shape==(n,9,9)and arr['global_allowed'].shape==(t,64,64)
        assert not((arr['dense_exact_state']==2)&arr['dense_valid']).any()
        assert not((arr['global_exact_state']==2)&arr['global_valid']).any()
        assert(arr['dense_allowed'][~arr['dense_valid']]==63).all()and(arr['global_allowed'][~arr['global_valid']]==63).all()
        path=OUT/('PALLOX_'+('TRAIN'if split=='TRAIN'else'VAL')+'.npz');np.savez_compressed(path,**arr)
        outputs.append(dict(**ref(path),bytes=path.stat().st_size,split=split,targets=t,queries=n,schema={k:dict(shape=list(a.shape),dtype=str(a.dtype))for k,a in arr.items()}))
    make_sheet(all_inputs,selections,all_labels,manifest)
    for key in ['recipe','interpretation','source_evidence_manifest','label_protocol','source_split_lock','conditional_admission','denylist']:bind(lock[key])
    for b in lock['code']+lock['source_files']:bind(b)
    report=dict(schema='PAIRED_PUBLIC_SHARED_QUERY_AND_PARTIAL_LABEL_SEAL_V1',completed=True,
        input_lock=ref(LOCK),recipe=ref(RECIPE),interpretation=ref(INTERPRETATION),pairs=50,targets=100,queries=800,
        targets_by_split=dict(Counter(r['split']for r in all_inputs)),queries_by_split=dict(Counter(r['split']for r in all_labels)),
        independent_actual_query_scalar_contract_checks=800,center_scalar_mismatches=0,pair_shared_queries_boxes_sides=True,
        source_current_RGB_SHA_overlap=0,source_split_RGB_SHA_overlap=0,
        outputs=[ref(OUT/n)for n in ['INPUTS_RGB_ONLY.jsonl','QUERIES_LABELS.jsonl','ANCHOR_SELECTION.jsonl','FIXED_SIX_SLOT0_NATIVE96.png']]+outputs,
        summaries={s:[dict(metric=k[0],value=k[1],N=n)for k,n in sorted(c.items(),key=lambda x:(x[0][0],str(x[0][1])))]for s,c in counts.items()},
        source_guided_query_conditioning='Sampling uses source teacher/E/V support; deployment N0 queries do not have this source-guided selection. Report this gap; do not claim matching deployment query distribution.',
        conditional_teacher_not_gold=True,full_negative_O_labels_generated=0,semantic_self_truth_generated=False,
        all_pairs_kept_including_empty_E=True,threshold_or_model_output_selection=False,
        models_features_GPU_training_private_PnP_calls=0,pose_alignment_or_optimization=False,
        fit_admission='No model fit authorized by this source query seal',seconds=time.monotonic()-started)
    write(OUT/'SEAL.json',report)
    print('PAIRED_QUERY_SEAL',sha(OUT/'SEAL.json'),'targets100 queries800 seconds',round(report['seconds'],2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','build']);a=parser.parse_args()
    freeze()if a.mode=='freeze'else build()
