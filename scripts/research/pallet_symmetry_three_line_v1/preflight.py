"""Hash/geometry/data audit, never opens FINAL or mutates source data."""
import json, math, subprocess, unittest, io
from collections import Counter
import numpy as np
import cv2
import env as E
from audit_math import derive_permutations, edge_channel_permutations
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES

def geometry(x,y,z):
    return np.array([[-x/2,-y/2,-z/2],[x/2,-y/2,-z/2],[x/2,y/2,-z/2],[-x/2,y/2,-z/2],
                     [-x/2,-y/2,z/2],[x/2,-y/2,z/2],[x/2,y/2,z/2],[-x/2,y/2,z/2],[0,0,0]])
def rotations(n):
    return np.array([[[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]] for a in np.arange(n)*2*math.pi/n])
def main():
    assert E.sha(E.R0)==E.R0_SHA and E.sha(E.DHT)==E.DHT_SHA
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    subprocess.check_call(['git','merge-base','--is-ancestor','44fdd7b415c5ba8cdaaf41ff487217d59a639de1','origin/main'])
    contract=[]
    reg=E.read(E.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')
    for obj in reg['objects']:
        d=obj['physical_dimensions_m'];x=geometry(d['x'],d['y'],d['z'])
        n=4 if obj['object_type']=='plastic_standard_110x110x15' else 2
        p=derive_permutations(x,rotations(n),np.array(EDGES))
        contract.append(dict(object_type=obj['object_type'],dimensions_m=d,corners_centroid=x.tolist(),
          frame='canonical x right, y down (height), z depth; centroid origin for this sidecar',
          group_order=n,rotations=rotations(n).tolist(),permutations=p.tolist(),
          edge_channel_permutations=edge_channel_permutations(p,np.array(EDGES)).tolist(),
          basis='approved square task equivalence, not visual identity' if n==4 else 'paper declared C2 benchmark convention, not physical inspection',
          source_registry_status=obj['symmetry_status']))
    E.freeze(E.DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json',dict(objects=contract,edges=EDGES,
      extension='New experiment extends approved task equivalence to 2D targets and evaluation; original contracts unchanged.',
      source_frame='Renderer asset C2 is about source world Z; converted renderer pose frame uses Y height. Per-frame 3D corner correspondence must be audited, not inferred from 2D x sorting.',
      prediction_slots='camera-facing training channels; retained, no front classifier',
      centroid='index 8; not a front-face pose origin',visibility='Physical visibility UNKNOWN unless separately verified; annotation-valid is not visibility.',
      runtime_type='B only frozen paper population. Source asset is supplied scenario metadata, never a GT pose/selected evaluation symmetry.',
      controller_compatibility=False))
    square={}; hashes={}; parent={}
    for split in ['train','val']:
        rows=[];masks=[]
        for ip in sorted((E.SQUARE/'images'/split).glob('*')):
            if not ip.is_file():continue
            lp=E.SQUARE/'labels'/split/(ip.stem+'.txt')
            if not lp.exists():raise FileNotFoundError(lp)
            im=cv2.imread(str(ip));assert im is not None
            a=np.array(lp.read_text().split(),float).reshape(-1,32);assert len(a)==1
            k=a[0,5:].reshape(9,3);masks.append(k[:,2]>0)
            # Prepared originals, not augmentation: recipe crop_dir/aug_dir null.
            rows.append(dict(id=ip.stem,image=str(ip.relative_to(E.ROOT)),label=str(lp.relative_to(E.ROOT)),
              image_sha256=E.sha(ip),raw_pixel_sha256=__import__('hashlib').sha256(im[100:-100,100:-100].tobytes()).hexdigest(),
              label_sha256=E.sha(lp),raw_hw=[im.shape[0]-200,im.shape[1]-200],parent_id=ip.stem))
        hashes[split]={r['raw_pixel_sha256'] for r in rows};parent[split]={r['parent_id'] for r in rows}
        perms=np.array(contract[-1]['permutations']);v=np.array(masks)
        square[split]=dict(count=len(rows),mask_noninvariant_frames=int(np.any(v[:,perms]!=v[:,None],axis=(1,2)).sum()) if len(rows) else 0,
                          records=rows)
    overlap=hashes['train']&hashes['val'];parent_overlap=parent['train']&parent['val']
    original_n=square['train']['count']
    square['train']['records']=[r for r in square['train']['records'] if r['raw_pixel_sha256'] not in overlap and r['parent_id'] not in parent_overlap]
    square['train']['count']=len(square['train']['records'])
    E.freeze(E.RAW/'A/square_membership.json',square)
    source=E.read(E.C.LINE/'SOURCE_MANIFEST.json');counts=Counter((r['source_split'],r['source']) for r in source['records'])
    source_checks=dict(image_missing=[],label_missing=[],label_sha_mismatch=[])
    for i,r in enumerate(source['records']):
        if not E.Path(r['image']).is_file():source_checks['image_missing'].append(r['id'])
        if not E.Path(r['label']).is_file():source_checks['label_missing'].append(r['id'])
        elif E.sha(r['label'])!=r['label_sha256']:source_checks['label_sha_mismatch'].append(r['id'])
        if i%10000==0:print('source check',i,flush=True)
    splitids={};splitsha={}
    for split in ['calibration','synth_val']:
        records=E.read(E.EXPORT/(split+'.json'))['records']
        splitids[split]={r['frame_id'] for r in records};splitsha[split]={r['source_image_sha256'] for r in records}
    assert not splitids['calibration']&splitids['synth_val']
    assert not splitsha['calibration']&splitsha['synth_val']
    data=dict(square={k:{a:b for a,b in v.items() if a!='records'} for k,v in square.items()},
      square_recipe=E.read(E.SQUARE/'_prepare_live_gt.json'),square_parent_overlap=sorted(parent_overlap),
      square_raw_hash_overlap=sorted(overlap),square_train_removed=original_n-square['train']['count'],
      square_limitation='Session-interleaved reused DEV; not independent confirmation; manual 4-fold-normalised labels.',
      synthetic_source_counts={str(k):v for k,v in counts.items()},synthetic_checks=source_checks,
      missing_historical_probe_manifest=not (E.ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K.jsonl').exists(),
      replacement_manifest=str((E.C.LINE/'SOURCE_MANIFEST.json').relative_to(E.ROOT)),
      B_split_counts={k:len(v) for k,v in splitids.items()},B_cross_split_id_or_image_overlap=0,FINAL_reads=0)
    E.freeze(E.DOC/'DATA_PARTITIONS_AND_QA.json',data)
    files=[E.R0,E.DHT,E.C.B/'P_SELECTION.json',E.C.LINE/'SOURCE_MANIFEST.json',E.C.LINE/'cache/CACHE_MANIFEST.json',
      E.EXPORT/'calibration.json',E.EXPORT/'synth_val.json',E.SQUARE/'_prepare_live_gt.json',E.SQUARE/'data.yaml',
      E.ROOT/'pallet_yolo_loss/c4.py',E.ROOT/'_docs/experiments/pallet_translation_loss_v1/LOSS_SYMMETRY_CONTRACT.json',
      E.ROOT/'challenge/config/SQUARE_PALLET_SYMMETRY_CONTRACT.json',E.ROOT/'challenge/real_gt_v2/SYMMETRY_CONTRACT.json',
      E.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz']
    files += [E.C.BRAW/f'runs/seed{s}/last.pt' for s in [1,2,3]]
    import ultralytics.utils.loss as loss
    files.append(E.Path(loss.__file__))
    bindings=[]
    for p in files:
        if p.is_relative_to(E.ROOT):bindings.append(E.bound(p))
        else:bindings.append(dict(path=str(p),sha256=E.sha(p),bytes=p.stat().st_size))
    E.freeze(E.DOC/'SOURCE_BINDING.json',dict(files=bindings))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(E.HERE),pattern='test_audit_math.py'))
    E.write(E.DOC/'REGRESSION_TESTS.json',dict(core_tests=result.testsRun,core_pass=result.wasSuccessful(),log=stream.getvalue(),repository_adapter='PENDING',training_completed=False))
    print(json.dumps({k:v for k,v in data.items() if k!='synthetic_checks'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
