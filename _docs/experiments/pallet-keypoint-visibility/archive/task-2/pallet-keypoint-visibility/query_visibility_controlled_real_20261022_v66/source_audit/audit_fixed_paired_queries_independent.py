"""Fixed-six RNG/native-label replay, no query-builder/grid-helper imports."""
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];P=ROOT/'public_pallet';Q=P/'PAIRED_QUERIES'
sys.path.insert(0,str(ROOT/'contract'))
from paired_truth import construct_certainty,query_truth

def resolve(p):return Path(p)if Path(p).is_absolute()else P/p
def sha(p):return hashlib.sha256(resolve(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(l)for l in Path(p).open()]
def erode(a):return np.lib.stride_tricks.sliding_window_view(np.pad(a,2,constant_values=False),(5,5)).all((-2,-1))
def rng(pair,stage):return np.random.default_rng(int.from_bytes(hashlib.sha256(('20261022:'+pair+':'+stage).encode()).digest()[:8],'little'))


def main():
    cv2.setNumThreads(1);start=time.monotonic()
    fixed=set(json.loads((P/'FIXED_SIX_PAIR_SELECTION.json').read_text())['pair_ids']);assert len(fixed)==6
    evidence={r['pair_id']:r for r in rows(P/'SYMMETRIC_ALL50_V2/PAIR_EVIDENCE_MANIFEST.jsonl')}
    selection={r['pair_id']:r for r in rows(Q/'ANCHOR_SELECTION.jsonl')}
    inputs={r['id']:r for r in rows(Q/'INPUTS_RGB_ONLY.jsonl')}
    sourceNPZ={s:np.load(Q/('PALLOX_'+s+'.npz'),allow_pickle=False)for s in ['TRAIN','VAL']}
    gp=ROOT.parent/'query_visibility_amodal_global_20261022_v65/global_recipe.py'
    spec=importlib.util.spec_from_file_location('fixed_query_global_recipe',gp);global_recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(global_recipe)
    cells=0;views=0;cases=[]
    for pairid in sorted(fixed):
        r=evidence[pairid];saved=selection[pairid]
        assert sha(r['evidence']['path'])==r['evidence']['sha256']
        with np.load(resolve(r['evidence']['path']),allow_pickle=False)as z:
            va,vb,ea,eb=[z[k]for k in ['visible_A','visible_B','external_A','external_B']]
        sam=[]
        for v in ['A','B']:
            b=r['files'][v]['mask_teacher'];assert sha(b['path'])==b['sha256'];m=cv2.imread(str(resolve(b['path'])),cv2.IMREAD_UNCHANGED)
            assert set(np.unique(m))<={0,255};sam.append(m>0)
        y,x=np.where(sam[0]|sam[1]);base=np.array([x.min(),y.min(),x.max()+1,y.max()+1],np.float64)
        size=base[2:]-base[:2];center=(base[2:]+base[:2])/2;br=rng(pairid,'box')
        shift=br.uniform(-.05,.05,2);scale=br.uniform(.9,1.1,2)
        newcenter=center+shift*size;newsize=size*scale
        box=np.clip(np.r_[newcenter-newsize/2,newcenter+newsize/2],[0,0,0,0],[639,479,639,479])
        corners=np.floor(box.reshape(2,2)+.5).astype(int);un=va|vb
        pools=[erode(ea)|erode(eb),ea|eb,un,erode(va&vb),un&~erode(un)]
        chains=[[0,1,2],[0,1,2],[3,2],[3,2],[4,2],[4,2]]
        used=set();anchors=[];ar=rng(pairid,'anchor')
        for slot in range(8):
            selected=None
            if slot<6:
                for pool in chains[slot]:
                    candidates=[int(k)for k in np.flatnonzero(pools[pool])if int(k)not in used]
                    if candidates:
                        selected=candidates[int(ar.integers(len(candidates)))];break
            if selected is None:
                point=corners[slot%2].copy();selected=int(point[1])*640+int(point[0])
            else:point=np.array([selected%640,selected//640])
            anchors.append(point);used.add(selected)
        anchors=np.array(anchors);jitter=rng(pairid,'jitter').uniform(-3,3,(8,2));query=anchors.astype(float)+jitter
        side=float(np.clip(.25*max(np.ptp(query[:,0]),np.ptp(query[:,1])),16,640))
        for key,value in [('original_teacher_union_box',base),('box',box),('anchors',anchors),('jitter',jitter),('query',query)]:
            assert np.array_equal(np.asarray(saved[key]),value),(pairid,key)
        assert saved['side']==side
        for view,vp,vn in [('A',va,ea),('B',vb,eb)]:
            ident='PALLOXPAIR:'+pairid+':'+view;inp=inputs[ident]
            assert inp['query']==query.tolist()and inp['box']==box.tolist()and inp['side']==side
            z=sourceNPZ['TRAIN'if inp['split']=='TRAIN'else'VAL']
            sid={str(s):i for i,s in enumerate(z['sample_id'])};gid={str(s):i for i,s in enumerate(z['global_id'])}
            indices=np.array([sid[ident+':'+str(k)]for k in range(8)]);target=gid[ident]
            assert (z['target_index'][indices]==target).all()
            cert=construct_certainty(vp,vn,va|vb,np.zeros_like(vp))
            a=(np.arange(9)-4)*side/9;xx,yy=np.meshgrid(a,a)
            xy=query[:,None,None,:]+np.stack([xx,yy],-1)[None]
            for label,coords,allow,valid,exact,near in [
              ('dense',xy,z['dense_allowed'][indices],z['dense_valid'][indices],z['dense_exact_state'][indices],z['dense_radius2_state'][indices]),
              ('global',global_recipe.grid(box)[0],z['global_allowed'][target],z['global_valid'][target],z['global_exact_state'][target],z['global_radius2_state'][target])]:
                for idx in np.ndindex(coords.shape[:-1]):
                    truth=query_truth(cert,coords[idx]);assert bool(valid[idx])==truth['valid'],(ident,label,idx)
                    expected=truth['joint_compatible_mask']if truth['valid']else 63
                    assert int(allow[idx])==expected,(ident,label,idx,'allowed')
                    assert int(exact[idx])==truth['exact_state']and int(near[idx])==truth['radius2_state'],(ident,label,idx,'hardstate')
                    cells+=1
            views+=1
        cases.append(dict(pair_id=pairid,anchors_query_box_side_bit_exact=True,views=2))
    for z in sourceNPZ.values():z.close()
    report=dict(schema='V66_INDEPENDENT_FIXED_PAIRED_QUERY_NATIVE_AUDIT',PASS=True,
      recipe_sha256=sha(ROOT/'PAIRED_QUERY_RECIPE.json'),query_seal_sha256=sha(Q/'SEAL.json'),
      auditor_sha256=sha(__file__),query_builder_imported=False,vector_grid_helper_imported=False,
      fixed_pairs=6,view_targets=views,actual_native9_global64_cells_scalar_checked=cells,
      RNG_box_anchor_jitter_query_side_bit_exact=True,invalid63_validfalse=True,
      source_RGB_feature_model_private_data_reads=0,GPU_training_PnP_calls=0,cases=cases,seconds=time.monotonic()-start)
    out=ROOT/'source_audit/INDEPENDENT_FIXED_PAIRED_QUERY_AUDIT.json';assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
