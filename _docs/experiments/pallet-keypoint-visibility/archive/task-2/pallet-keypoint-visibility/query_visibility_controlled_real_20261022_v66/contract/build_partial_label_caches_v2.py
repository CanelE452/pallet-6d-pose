"""Label-only CPU source caches; freeze code/recipe/input manifests before building.

No RGB reader, backbone, model, feature-cache read/write, private labels or PnP.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,importlib.util,json,time
import cv2
import numpy as np
import controlled_truth
from partial_truth_grid import partial_truth_grid

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'label_cache_v2'
LOCK=ROOT/'provenance/PARTIAL_LABEL_CACHE_INPUT_LOCK_V2.json'
RECIPE=ROOT/'PARTIAL_LABEL_CACHE_RECIPE_V2.json'
KEYS=('visible_positive','visible_negative','full_positive','full_negative')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def bind(b):assert sha(b['path'])==b['sha256'],b['path']
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def jsonl(p):return [json.loads(l)for l in Path(p).open()]


def freeze():
    assert not LOCK.exists()and not RECIPE.exists(),'Recipe/input lock already frozen'
    recipe=dict(schema='V66_LABEL_ONLY_PARTIAL_SOURCE_CACHE_RECIPE_BEFORE_BUILD_V2',
     split_alias={'TRAIN':'TRAIN','VALIDATION':'VAL','VAL':'VAL'},
     amendment='V1 aborted before any NPZ writes on first VALIDATION target. Only explicit output split alias corrected; frozen V1 artifacts preserved.',
     scope='CPU labels only, ORIGINAL_YCB2000 and CONTROLLED1999; no features, RGB, models or current evaluation reads',
     inputs='Immutable TARGETS/COMPOSITES/QUERIES, source mask files and controlled canonical known-bit NPZs',
     output_names=['ORIGINAL_YCB_TRAIN.npz','ORIGINAL_YCB_VAL.npz','CONTROLLED_TRAIN.npz','CONTROLLED_VAL.npz'],
     source_order='Preserve each input manifest order per split. Each target contributes its unchanged8 queries in native_corner0..7 order. Feature join must use sample_id/global_id.',
     IDs='Original global_id=target id, sample_id=original_sample_ids. Controlled global_id=composite id, sample_id=sealed QUERIES.csv sample_id. target_index indexes that cache own global_id list.',
     original_truth='construct_certainty(original_visible, original_full, paste=zeros, visible_negative_trusted=True). Visible negatives explicit BOP completeness assumption. Full-only remains full UNKNOWN, never hardE.',
     controlled_truth='Read visible_positive/visible_negative/full_positive/full_negative verbatim from generation-sealed NPZ; enforce closed certainty.',
     dense_native_grid='query + (i-4)*side/9, x/y meshgrid i0..8, float64; nearest floor(xy+.5); continuous AND rounded native bounds',
     global_native_grid='Unchanged v65global_recipe.grid(box):1.25 box expansion,64x64 half-cell grid; nearest floor(xy+.5); continuous AND rounded native bounds',
     label_grid='Proven partial_truth_grid native5x5, clipped frame window, exact/shared-neighbor correlated6joint completions',
     joint_order=[['V','V'],['E','V'],['E','E'],['O','V'],['O','E'],['O','O']],
     invalid='allowed=63 (all6states), valid=false, state=-1, joint_known=false. Invalid points never supervise.',
     data_fields={'sample_id':'Nunicode','global_id':'Tunicode','target_index':'Nint64','native_corner':'Nuint8',
       'dense_allowed':'N9x9uint8','dense_valid':'N9x9bool','dense_exact_state':'N9x9int8','dense_radius2_state':'N9x9int8',
       'center_joint_known':'Nbool','center_joint_state':'Nint8','center_exact_state':'Nint8','center_radius2_state':'Nint8',
       'global_allowed':'T64x64uint8','global_valid':'T64x64bool','global_exact_state':'T64x64int8','global_radius2_state':'T64x64int8',
       'image_sha256':'Nunicode of saved inputRGBSHA','global_image_sha256':'Tunicode','query':'N2float64','side':'Nfloat64','global_box':'T4float64'},
     checks=['Source saved RGB SHAs directly compared to current223 exclusion manifest; no current images opened',
       'Actual read source-mask/controlled-NPZ SHA matches immutable input metadata',
       'All15992 controlled centers exact/r2/joint masks/known/state/valid match sealed QUERIES.csv',
       'All15992 controlled original-baseline centers match sealed zero-occlusion truth',
       'Every original full-only pixel exact UNKNOWN with no fullpositive/fullnegative certificate',
       'Unchanged q8/side/box and IDs; split-disjoint saved RGB SHA; compressed label outputs total<100MB'],
     original_hidden_proxy='Original full-minus-vis hardE retained only in source input masks/legacy audit files. Never written as primary training hardE in these caches.',
     no_extra_jitter=True,no_features_models_RGB_private_PnP=True)
    write(RECIPE,recipe)
    generation=json.loads((ROOT/'provenance/GENERATION_INPUT_LOCK.json').read_text())
    seal=json.loads((ROOT/'controlled/SEAL.json').read_text())
    for b in generation['source_bindings']+generation['code_bindings']+[generation['targets'],generation['occluders'],generation['recipe']]:bind(b)
    for key in ['lock','recipe','label_protocol','label_helper']:bind(seal[key])
    inputs=[ROOT/'provenance/GENERATION_INPUT_LOCK.json',ROOT/'controlled/SEAL.json',ROOT/'provenance/TARGETS.jsonl',
            ROOT/'controlled/COMPOSITES.jsonl',ROOT/'controlled/QUERIES.csv',ROOT/'DATA_LABEL_PROTOCOL.json',
            ROOT.parent/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json']
    code=[Path(__file__),ROOT/'contract/partial_truth_grid.py',ROOT/'contract/controlled_truth.py',
          ROOT/'contract/PARTIAL_TRUTH_GRID_MOCK_PROOF.json',ROOT/'contract/smoke_label_cache_sampling.py',
          ROOT/'contract/LABEL_CACHE_SAMPLING_MOCK_PROOF.json',ROOT.parent/'query_visibility_amodal_global_20261022_v65/global_recipe.py']
    lock=dict(schema='V66_PARTIAL_LABEL_CACHE_IMMUTABLE_INPUT_LOCK',recipe=ref(RECIPE),
              inputs=[ref(p)for p in inputs],code=[ref(p)for p in code],
              source_generation_seal_completed=seal['completed'],source_generation_accepted=seal['accepted'],
              source_mask_binding='Per-target visible/full metadata paths and SHAs from frozen TARGETS; controlled canonical NPZ SHAs from frozen COMPOSITES and original generation seal outputs',
              runtime=dict(python=__import__('sys').executable,numpy=np.__version__,cv2=cv2.__version__),
              private_RGB_labels_scores_read=False,new_models_training_GPU_PnP_calls=0)
    write(LOCK,lock);print('FROZEN',sha(LOCK),flush=True)


def load_recipe():
    lock=json.loads(LOCK.read_text());bind(lock['recipe'])
    for b in lock['inputs']+lock['code']:bind(b)
    path=ROOT.parent/'query_visibility_amodal_global_20261022_v65/global_recipe.py'
    spec=importlib.util.spec_from_file_location('frozen_input_only_global_recipe',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return lock,module


def sample(grid,xy,global_recipe):
    pixel=np.floor(xy+.5).astype(np.int64);h,w=grid['exact_state'].shape
    valid=global_recipe.native_valid(xy,pixel,(h,w))
    xx=np.clip(pixel[...,0],0,w-1);yy=np.clip(pixel[...,1],0,h-1)
    result=dict(allowed=np.where(valid,grid['joint_compatible_mask'][yy,xx],63).astype(np.uint8),valid=valid,
                exact_state=np.where(valid,grid['exact_state'][yy,xx],-1).astype(np.int8),
                radius2_state=np.where(valid,grid['radius2_state'][yy,xx],-1).astype(np.int8),
                joint_known=valid&grid['joint_known'][yy,xx],joint_state=np.where(valid,grid['joint_state'][yy,xx],-1).astype(np.int8))
    assert ((result['allowed']>=1)&(result['allowed']<=63)).all()
    return result


def check_center(got,rows,prefix=''):
    for k,row in enumerate(rows):
        for outkey,csvkey in [('allowed','joint_compatible_mask'),('exact_state','exact_state'),('radius2_state','radius2_state')]:
            key=prefix+csvkey;assert int(got[outkey][k,4,4])==int(row[key]),(row['sample_id'],key)
        if not prefix:
            assert bool(got['valid'][k,4,4])==(row['valid']=='True')
            assert bool(got['joint_known'][k,4,4])==(row['joint_known']=='True')
            assert int(got['joint_state'][k,4,4])==int(row['joint_state'])


def build():
    assert not OUT.exists(),'Append-only label outputs already exist'
    lock,global_recipe=load_recipe();cv2.setNumThreads(1);OUT.mkdir()
    write(OUT/'BUILD_STARTED.json',dict(lock=ref(LOCK),builder=ref(__file__)))
    targets=jsonl(ROOT/'provenance/TARGETS.jsonl');composites=jsonl(ROOT/'controlled/COMPOSITES.jsonl')
    assert len(targets)==2000 and len(composites)==1999
    original_by_id={r['id']:r for r in targets};assert len(original_by_id)==2000
    controlled_by_original={r['target_original']['id']:r for r in composites};assert len(controlled_by_original)==1999
    csvrows=list(csv.DictReader((ROOT/'controlled/QUERIES.csv').open()))
    csvindex={(r['composite_id'],int(r['native_corner'])):r for r in csvrows};assert len(csvindex)==15992
    deny=json.loads((ROOT.parent/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json').read_text())
    denied=set(deny['image_sha256']);assert len(denied)==223
    seal=json.loads((ROOT/'controlled/SEAL.json').read_text());sealed_outputs={b['path']:b['sha256']for b in seal['outputs']}
    destinations={};statistics={};split_shas={'TRAIN':set(),'VAL':set()}
    for kind in ['ORIGINAL_YCB','CONTROLLED']:
        for split in ['TRAIN','VAL']:
            destinations[(kind,split)]={k:[]for k in ['sample_id','global_id','target_index','native_corner','dense_allowed','dense_valid','dense_exact_state','dense_radius2_state','center_joint_known','center_joint_state','center_exact_state','center_radius2_state','global_allowed','global_valid','global_exact_state','global_radius2_state','image_sha256','global_image_sha256','query','side','global_box']}
            statistics[(kind,split)]=Counter()
    start=time.monotonic();original_unknown_pixels=0;controlled_centers=baseline_centers=0;mask_files=0
    for i,t in enumerate(targets):
        split={'TRAIN':'TRAIN','VALIDATION':'VAL','VAL':'VAL'}[t['split']];assert split in split_shas and len(t['query'])==8
        assert t['image']['sha256']not in denied;split_shas[split].add(t['image']['sha256'])
        for key in ['visible','full_proxy']:bind(t[key]);mask_files+=1
        vis,full=controlled_truth.decode_public_masks(cv2.imread(t['visible']['path'],cv2.IMREAD_UNCHANGED),cv2.imread(t['full_proxy']['path'],cv2.IMREAD_UNCHANGED))
        cert=controlled_truth.construct_certainty(vis,full,np.zeros_like(vis),visible_negative_trusted=True)
        hidden=full&~vis;assert (cert['exact_state'][hidden]==-1).all()
        assert not cert['full_positive'][hidden].any()and not cert['full_negative'][hidden].any()
        original_unknown_pixels+=int(hidden.sum())
        query=np.asarray(t['query'],np.float64);side=float(t['side']);a=(np.arange(9,dtype=np.float64)-4)*side/9
        xx,yy=np.meshgrid(a,a);local_xy=query[:,None,None,:]+np.stack((xx,yy),-1)[None]
        global_xy,_,_,_=global_recipe.grid(t['box'])
        original_grid=partial_truth_grid(cert);original_local=sample(original_grid,local_xy,global_recipe)
        original_global=sample(original_grid,global_xy,global_recipe)
        c=controlled_by_original.get(t['id'])
        variants=[('ORIGINAL_YCB',t['id'],list(t['original_sample_ids']),t['image']['sha256'],original_local,original_global)]
        if c:
            assert c['target_original']==t
            for key in ['query','side','box']:assert c[key]==t[key]
            assert c['image']['sha256']not in denied and c['occluder_original']['image']['sha256']not in denied
            assert c['labels']['sha256']==sealed_outputs[c['labels']['path']];bind(c['labels'])
            with np.load(c['labels']['path'],allow_pickle=False)as z:cc={key:z[key]for key in KEYS}
            controlled_grid=partial_truth_grid(cc);local=sample(controlled_grid,local_xy,global_recipe);glob=sample(controlled_grid,global_xy,global_recipe)
            rows=[csvindex[(c['id'],k)]for k in range(8)]
            check_center(local,rows);check_center(original_local,rows,'original_zero_occlusion_')
            controlled_centers+=8;baseline_centers+=8
            variants.append(('CONTROLLED',c['id'],[r['sample_id']for r in rows],c['image']['sha256'],local,glob))
        for kind,gid,sids,imagesha,local,glob in variants:
            dst=destinations[(kind,split)];idx=len(dst['global_id']);count=statistics[(kind,split)]
            assert len(sids)==8 and all(sid==gid+':'+str(k)for k,sid in enumerate(sids))
            dst['global_id'].append(gid);dst['global_image_sha256'].append(imagesha);dst['global_box'].append(t['box'])
            for key in ['allowed','valid','exact_state','radius2_state']:dst['global_'+key].append(glob[key])
            dst['sample_id'].extend(sids);dst['target_index'].extend([idx]*8);dst['native_corner'].extend(range(8))
            dst['image_sha256'].extend([imagesha]*8);dst['query'].extend(query);dst['side'].extend([side]*8)
            for key in ['allowed','valid','exact_state','radius2_state']:dst['dense_'+key].append(local[key])
            for key in ['joint_known','joint_state','exact_state','radius2_state']:dst['center_'+key].extend(local[key][:,4,4])
            for label,data in [('center_allowed',local['allowed'][:,4,4]),('dense_allowed',local['allowed'][local['valid']]),('global_allowed',glob['allowed'][glob['valid']])]:
                for v,n in zip(*np.unique(data,return_counts=True)):count[(label,int(v))]+=int(n)
            count[('dense_valid',1)]+=int(local['valid'].sum());count[('dense_valid',0)]+=int((~local['valid']).sum())
            count[('global_valid',1)]+=int(glob['valid'].sum());count[('global_valid',0)]+=int((~glob['valid']).sum())
        if (i+1)%200==0:print('PARTIAL_LABEL_CACHE',i+1,'of2000','seconds',round(time.monotonic()-start,2),flush=True)
    assert not split_shas['TRAIN']&split_shas['VAL']
    assert controlled_centers==baseline_centers==15992
    outputs=[];summaries=[]
    for(kind,split),dst in destinations.items():
        arrays={}
        for key,value in dst.items():
            if key.startswith('dense_'):arrays[key]=np.concatenate(value,axis=0)
            else:arrays[key]=np.asarray(value)
        arrays['target_index']=arrays['target_index'].astype(np.int64);arrays['native_corner']=arrays['native_corner'].astype(np.uint8)
        arrays['center_joint_state']=arrays['center_joint_state'].astype(np.int8);arrays['center_joint_known']=arrays['center_joint_known'].astype(bool)
        N=len(arrays['sample_id']);T=len(arrays['global_id']);assert N==8*T
        assert len(set(arrays['sample_id']))==N and len(set(arrays['global_id']))==T
        assert np.array_equal(arrays['target_index'],np.repeat(np.arange(T),8))
        assert arrays['dense_allowed'].shape==(N,9,9)and arrays['global_allowed'].shape==(T,64,64)
        assert arrays['dense_allowed'].dtype==arrays['global_allowed'].dtype==np.uint8
        assert (arrays['dense_allowed'][~arrays['dense_valid']]==63).all()and (arrays['global_allowed'][~arrays['global_valid']]==63).all()
        path=OUT/(kind+'_'+split+'.npz');np.savez_compressed(path,**arrays)
        outputs.append(dict(**ref(path),bytes=path.stat().st_size,N=N,T=T))
        summaries.append(dict(kind=kind,split=split,N=N,T=T,counts=[dict(metric=k[0],value=k[1],N=n)for k,n in sorted(statistics[(kind,split)].items())],
                              schema={key:dict(shape=list(a.shape),dtype=str(a.dtype))for key,a in arrays.items()}))
    total=sum(b['bytes']for b in outputs);assert total<100_000_000
    for b in lock['inputs']+lock['code']+[lock['recipe']]:bind(b)
    report=dict(schema='V66_PARTIAL_SOURCE_LABEL_CACHE_BUILD_SEAL',completed=True,lock=ref(LOCK),recipe=ref(RECIPE),outputs=outputs,
      total_compressed_bytes=total,original_targets=2000,controlled_targets=1999,original_rows=16000,controlled_rows=15992,
      original_full_only_native_pixels_preserved_full_UNKNOWN=original_unknown_pixels,source_mask_files_SHA_verified=mask_files,
      controlled_centers_verified_against_sealed_CSV=controlled_centers,original_baseline_centers_verified_against_sealed_CSV=baseline_centers,
      source_saved_RGB_SHA_denylist_intersection=0,source_split_RGB_SHA_overlap=0,source_RGB_files_opened=0,
      private_images_labels_scores_read=False,features_models_training_GPU_PnP_calls=0,old_feature_cache_rewrites=0,
      original_hidden_proxy_promoted_hardE=0,invalid_allowed_mask=63,fit_admission='NO_MODEL_FIT_AUTHORIZED_BY_THIS_LABEL_SEAL',
      summaries=summaries,seconds=time.monotonic()-start)
    write(OUT/'SEAL.json',report);print('COMPLETE',json.dumps({k:report[k]for k in ['total_compressed_bytes','original_rows','controlled_rows','seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','build']);args=parser.parse_args()
    freeze()if args.mode=='freeze'else build()
