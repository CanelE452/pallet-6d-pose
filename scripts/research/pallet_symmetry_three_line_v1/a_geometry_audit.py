"""Derive object-specific target permutations from immutable source 3D geometry."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import env as E
from audit_math import derive_permutations
from preflight import rotations
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
def main():
    records=E.read(E.C.LINE/'SOURCE_MANIFEST.json')['records'];geometry=dict(np.load(E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'))
    index={str(k):i for i,k in enumerate(geometry['stems'])}
    contract=E.read(E.ROOT/'_docs/experiments/pallet_translation_loss_v1/LOSS_SYMMETRY_CONTRACT.json')
    counts=Counter();rows=[];missing=[];maximum=0
    def one(r):
        if r['source'] in ['P0','TEX']:return r,'scene.usd'
        p=E.ROOT/r['renderer_annotation_locator_provenance_only']
        return r,E.read(p)['objects'][0]['source_asset']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number,(r,asset) in enumerate(pool.map(one,records)):
            i=index[r['id']];entry=contract['assets'].get(asset,{})
            n=entry.get('max_valid_order',1) if entry.get('status')=='CONFIRMED' else 1
            corners=geometry['Xcf'][i];x=np.concatenate([corners,corners.mean(0,keepdims=True)])
            p=derive_permutations(x,rotations(n),np.array(EDGES))
            assert geometry['match_err'][i]<.05;maximum=max(maximum,float(geometry['match_err'][i]))
            counts[(r['source_split'],f'C{n}',asset)]+=1
            rows.append(dict(id=r['id'],source_split=r['source_split'],source_partition=r['partition'],asset=asset,
              group_order=n,permutations=p.tolist(),geometry_row=i,dimensions_xyz_m=geometry['dims'][i].tolist(),
              label_sha256=r['label_sha256'],image_sha256=r['image_sha256'],GT_row_count=len(r['targets'])))
            if number%10000==0:print('A 3D geometry',number,flush=True)
    E.freeze(E.RAW/'A/rect_target_sidecar.json',dict(records=rows,geometry_source=E.bound(E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'),original_labels_unchanged=True))
    E.write(E.DOC/'A/DATA_GEOMETRY_AUDIT.json',dict(status='COMPLETE',counts={str(k):v for k,v in counts.items()},
      max_3D_to_annotation_reprojection_discrepancy_px=maximum,rows=len(rows),
      train=sum(r['source_split']=='train' for r in rows),val=sum(r['source_split']=='val' for r in rows),
      permutation_derived_from_stored_3D=True,source_axis='source world Rz(pi), converted pose frame Y height',
      source_split_used_for_A_not_refiner_matched_subset=True,actual_updates=0,FINAL_access=False))
if __name__=='__main__':main()
