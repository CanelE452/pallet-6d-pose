"""Independent stable-ID permutation/offset and rejection tests, mock arrays only."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib,importlib.util,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'model/merge_partial_caches.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    spec=importlib.util.spec_from_file_location('merge_partial_under_test',CODE);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    pairs=np.array([(0,0),(1,0),(1,1),(2,0),(2,1),(2,2)],np.int8);rejected=[]
    with TemporaryDirectory(prefix='merge_mock_',dir=ROOT/'source_audit')as tmp:
        root=Path(tmp)
        def fixture(tag):
            ids=[tag+':B:1',tag+':A:0',tag+':B:0',tag+':A:1'];gids=[tag+':A',tag+':B'];n,t=4,2
            sha_by_global={gids[0]:'a'*64,gids[1]:'b'*64};target=np.array([1,0,1,0],np.int64)
            q_by_sample={sid:np.array([i+1.25,i+2.5])for i,sid in enumerate(sorted(ids))}
            side_by_sample={sid:16.+i for i,sid in enumerate(sorted(ids))}
            f=dict(x=np.zeros((n,65,9,9),np.float16),rgb_native=np.zeros((n,75,9,9),np.uint8),sam=np.zeros((n,2,9,9),np.float16),
              context=np.zeros((n,64),np.float32),box_grid=np.zeros((n,4,9,9),np.float16),sample_id=np.array(ids),group_id=np.array([gids[j]for j in target]),
              image_sha256=np.array([sha_by_global[gids[j]]for j in target]),target_index=target,
              global_index64=np.zeros((n,9,9),np.int64),global_index8=np.zeros((n,9,9),np.int64),global_query_valid=np.ones((n,9,9),bool),
              global_rgb=np.zeros((t,3,64,64),np.uint8),global_sam=np.zeros((t,1,64,64),np.float16),global_prompt=np.zeros((t,1,64,64),bool),
              global_id=np.array(gids),global_image_sha256=np.array([sha_by_global[s]for s in gids]),global_box=np.array([[0,0,20,20],[1,2,30,32]],np.float64),
              query=np.array([q_by_sample[s]for s in ids]),roi_side=np.array([side_by_sample[s]for s in ids]))
            f['x'][:,64]=1;f['x'][:,0,0,0]=np.arange(n)
            order=np.array([3,2,1,0]);go=np.array([1,0]);queryids=f['sample_id'][order];globalids=f['global_id'][go]
            gi={str(s):i for i,s in enumerate(globalids)}
            masks={s:1<<i for i,s in enumerate(sorted(ids))}
            g=dict(sample_id=queryids,global_id=globalids,target_index=np.array([gi[str(f['global_id'][f['target_index'][j]])]for j in order]),
              image_sha256=f['image_sha256'][order],global_image_sha256=f['global_image_sha256'][go],global_box=f['global_box'][go],
              query=f['query'][order],side=f['roi_side'][order],dense_allowed=np.array([np.full((9,9),masks[str(s)],np.uint8)for s in queryids]),
              dense_valid=np.ones((n,9,9),bool),global_allowed=np.array([np.full((64,64),16 if s.endswith(':A')else 32,np.uint8)for s in globalids]),global_valid=np.ones((t,64,64),bool))
            g['dense_exact_state']=np.array([np.full((9,9),pairs[int(np.log2(masks[str(s)])),0],np.int8)for s in queryids])
            g['dense_radius2_state']=np.array([np.full((9,9),pairs[int(np.log2(masks[str(s)])),1],np.int8)for s in queryids])
            g['global_exact_state']=np.full((t,64,64),2,np.int8);g['global_radius2_state']=np.array([np.full((64,64),1 if s.endswith(':A')else 2,np.int8)for s in globalids])
            g['dense_valid'][0,0,0]=False;g['dense_allowed'][0,0,0]=63;g['dense_exact_state'][0,0,0]=g['dense_radius2_state'][0,0,0]=-1
            # A valid but wholly unknown cell stays valid63; no invented label.
            g['global_allowed'][0,1,1]=63;g['global_exact_state'][0,1,1]=g['global_radius2_state'][0,1,1]=-1
            fp=root/(tag+'_features.npz');lp=root/(tag+'_labels.npz');np.savez_compressed(fp,**f);np.savez_compressed(lp,**g)
            return f,g,fp,lp,masks
        parts={}
        for source in ['ORIGINAL_YCB','CONTROLLED_YCB','PALLOX']:
            f,g,fp,lp,masks=fixture(source);d=mod.align(source,fp,lp)
            for key in mod.QUERY_INPUTS+mod.GLOBAL_INPUTS:assert np.array_equal(d[key],f[key]),(source,key)
            assert np.array_equal(d['target_index'],f['target_index'])
            for i,sid in enumerate(f['sample_id']):assert d['dense_allowed'][i,4,4]==masks[str(sid)]
            for i,gid in enumerate(f['global_id']):assert d['global_allowed'][i,4,4]==(16 if str(gid).endswith(':A')else 32)
            assert d['global_valid'][1,1,1]and d['global_allowed'][1,1,1]==63
            parts[source]=d
        f,g,fp,lp,_=fixture('SYNTH');f['joint']=np.tile(np.arange(81).reshape(9,9)%6,(4,1,1)).astype(np.int8)
        f['dense_valid']=np.ones((4,9,9),bool);f['global_joint']=np.tile(np.arange(4096).reshape(64,64)%6,(2,1,1)).astype(np.int8);f['global_valid']=np.ones((2,64,64),bool)
        f['dense_valid'][0,0,0]=False;f['joint'][0,0,0]=-1;np.savez_compressed(fp,**f)
        d=mod.align('SYNTH',fp)
        assert d['dense_allowed'][0,0,0]==63 and not d['dense_valid'][0,0,0]
        for prefix,j in [('dense','joint'),('global','global_joint')]:
            valid=f[prefix+'_valid'];assert np.array_equal(d[prefix+'_allowed'][valid],(1<<f[j][valid]).astype(np.uint8))
        parts['SYNTH']=d
        def reject(name,changed_f=None,changed_g=None,source='ORIGINAL_YCB'):
            basef,baseg,_,_,_=fixture('BAD_'+name);basef.update(changed_f(basef)if callable(changed_f)else changed_f or {});baseg.update(changed_g(baseg)if callable(changed_g)else changed_g or {})
            pf=root/'BAD_FEATURE.npz';pg=root/'BAD_LABEL.npz';np.savez_compressed(pf,**basef);np.savez_compressed(pg,**baseg)
            try:mod.align(source,pf,pg)
            except (ValueError,IndexError,AssertionError):rejected.append(name)
            else:raise AssertionError('Expected rejection '+name)
        reject('query_SHA',changed_g=lambda x:dict(image_sha256=np.full_like(x['image_sha256'],'c'*64)))
        reject('global_SHA',changed_g=lambda x:dict(global_image_sha256=np.full_like(x['global_image_sha256'],'c'*64)))
        reject('global_box',changed_g=lambda x:dict(global_box=x['global_box']+1))
        reject('query_target_mapping',changed_g=lambda x:dict(target_index=1-x['target_index']))
        reject('query_coordinates',changed_g=lambda x:dict(query=x['query']+.1))
        reject('ROI_side',changed_g=lambda x:dict(side=x['side']+1))
        reject('duplicate_ID',changed_g=lambda x:dict(sample_id=np.repeat(x['sample_id'][:1],4)))
        reject('label_outside_native_canvas',changed_f=lambda x:dict(x=np.zeros_like(x['x'])))
        reject('invalid_cell_not63',changed_g=lambda x:dict(dense_allowed=np.ones_like(x['dense_allowed'])))
        f['joint'][0,4,4]=-1;np.savez_compressed(fp,**f)
        try:mod.align('SYNTH',fp)
        except (ValueError,IndexError,AssertionError):rejected.append('SYNTH_valid_unknown_joint')
        else:raise AssertionError('Valid synthetic UNKNOWN silently relabeled')
        allsources=['SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX']
        joined=mod.concatenate([parts[s]for s in allsources]);offset=0
        for i,s in enumerate(allsources):
            n=len(parts[s]['sample_id']);got=joined['target_index'][i*4:(i+1)*4]
            assert np.array_equal(got,parts[s]['target_index']+offset);offset+=len(parts[s]['global_id'])
            assert np.array_equal(joined['global_id'][got],parts[s]['global_id'][parts[s]['target_index']])
        val0=mod.concatenate([parts[s]for s in allsources]);val1=mod.concatenate([parts[s]for s in allsources])
        assert all(np.array_equal(val0[k],val1[k])for k in val0)
        train0=mod.concatenate([parts[s]for s in allsources[:3]]);train1=mod.concatenate([parts[s]for s in allsources])
        assert len(train0['sample_id'])==12 and len(train1['sample_id'])==16
    report=dict(schema='V66_INDEPENDENT_PARTIAL_CACHE_MERGE_MOCK_AUDIT',PASS=True,merge_code_sha256=sha(CODE),
      feature_row_and_target_table_order_preserved=True,independent_GT_row_and_target_permutation_joined_by_ID=True,
      source_target_index_offsets_correct=True,common4source_validation_arrays_identical=True,TRAIN3vs4sources=True,
      synthetic_valid_hard_onehot_invalid63=True,valid_unknown63_retained=True,negative_fixtures_rejected=rejected,
      actual_source_or_private_RGB_feature_GT_cache_reads=0,new_models_training_GPU_PnP_calls=0)
    out=ROOT/'source_audit/INDEPENDENT_PARTIAL_CACHE_MERGE_MOCK_AUDIT.json';assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
