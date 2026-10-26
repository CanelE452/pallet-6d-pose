"""Extract ONLY camera metadata from DEV annotation containers after T0.

No protected FINAL file is opened. References/poses/keypoints are never accessed.
Registered dimensions and hypotheses come unchanged from frozen experiment code.
"""
from common import *
import numpy as np
from t3_adapter import ast_functions

def main():
    meta=read(OUT/'cache/METADATA_RECORDINGS.json')
    frames=core_manifest()
    protected_ids=set(meta['protected_final_frame_ids'])
    protected_sha=set(meta['protected_final_image_sha256'])
    assert len(frames)==292
    assert not protected_ids.intersection(r['id'] for r in frames)
    assert not protected_sha.intersection(r['image_sha256'] for r in frames)
    paths=[FLIP/'MANIFEST.json',OUT/'cache/METADATA_RECORDINGS.json',FLIP/'run_type.py']
    pools={kind:read(FLIP/kind/'POOL.json') for kind in KINDS}
    paths.extend(FLIP/k/'POOL.json' for k in KINDS)
    start('DEV_geometry_metadata',paths)
    functions=ast_functions(FLIP/'run_type.py',('cuboid','hypotheses'))
    dimensions={}
    for kind,pool in pools.items():
        rows=pool['records']
        unique={json.dumps(r['dimensions_xyz_m'],sort_keys=True) for r in rows}
        assert len(rows)==1000 and len(unique)==1,(kind,unique)
        dimensions[kind]=json.loads(next(iter(unique)))
    result=[];inputs={str(p):sha(p) for p in paths}
    for frame in frames:
        kind=kind_of(frame)
        assert frame['id'] not in protected_ids and frame['image_sha256'] not in protected_sha
        path=Path(frame['annotation_path'])
        actual=sha(path);assert actual==frame['annotation_sha256'],(frame['id'],'annotation binding changed')
        # JSON syntax is decoded, but only this metadata field is accessed.
        # No object annotation, keypoint, pose or reference field is inspected.
        camera=read(path)['camera_data']
        assert camera['width']==frame['width'] and camera['height']==frame['height']
        q=camera['intrinsics']
        K=[[float(q['fx']),0.,float(q['cx'])],[0.,float(q['fy']),float(q['cy'])],[0.,0.,1.]]
        assert np.isfinite(K).all() and K[0][0]>0 and K[1][1]>0
        item={k:frame[k] for k in ('id','width','height','image_sha256','object_type')}
        item.update(kind=kind,K=K,dimensions_xyz_m=dimensions[kind],
                    K_source={'path':str(path),'sha256':actual,'json_pointer':'/camera_data/intrinsics',
                              'native_size_pointer':'/camera_data/width,/camera_data/height'},
                    dimensions_source={'path':str(FLIP/kind/'POOL.json'),'json_pointer':'/records/*/dimensions_xyz_m'})
        item['hypotheses']=[{'name':name,'xyz':xyz.tolist()} for name,xyz in functions['hypotheses'](item)]
        inputs[str(path)]=actual;result.append(item)
    write(OUT/'cache/DEV_GEOMETRY_INPUTS.json',dict(complete=True,records=result,
          provenance={'inputs_sha256':inputs,'DEV_annotation_fields_accessed':['camera_data'],
                      'protected_FINAL_body_reads':0,'reference_or_pose_coordinates_accessed':False,
                      'K_not_estimated_or_refitted':True,'metadata_only_T0_complete_before_execution':True}))
    log('step_complete',task='DEV_geometry_metadata',records=len(result),protected_FINAL_body_reads=0,
        path=str(OUT/'cache/DEV_GEOMETRY_INPUTS.json'),sha256=sha(OUT/'cache/DEV_GEOMETRY_INPUTS.json'))
    print('DEV_GEOMETRY_METADATA_PASS',len(result),dimensions,flush=True)

if __name__=='__main__':main()
