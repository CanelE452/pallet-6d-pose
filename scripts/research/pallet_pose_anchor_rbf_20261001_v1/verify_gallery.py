"""Independent fixed-RBF RGB, frozen pose, projection and displayed metric audit.

This helper reads cached outputs and source RGB only. It never opens real GT,
fits a model, runs inference, or solves PnP. Call verify(report) from the public
verifier after the report has been generated. No files are written here.
"""
from collections import Counter
import csv
import hashlib
import itertools
from pathlib import Path
import numpy as np
from PIL import Image
from . import common as C

MODELS = ('R0', 'UNION_s1', 'UNION_s2', 'UNION_s3')
SIGNS = tuple(itertools.product((-1., 1.), repeat=3))
EDGES = tuple((i,j) for i in range(8) for j in range(i+1,8)
              if sum(x != y for x,y in zip(SIGNS[i],SIGNS[j])) == 1)


def bound_read(binding):
    C.verify(binding)
    return C.read(C.ROOT / binding['path'])


def scalar_projection(pose, K):
    """Scalar pinhole projection independent of report's batch matrix code."""
    assert pose['available']
    rotation=np.asarray(pose['R_cf'],np.float64)
    translation=np.asarray(pose['centroid'],np.float64)
    dimensions=np.asarray(pose['cf_extents'],np.float64)
    matrix=np.asarray(K,np.float64)
    assert rotation.shape == matrix.shape == (3,3)
    assert translation.shape == dimensions.shape == (3,) and (dimensions>0).all()
    corners=[];camera=[];uv=[]
    for sign in SIGNS:
        local=[sign[j]*float(dimensions[j])/2 for j in range(3)]
        point=[sum(float(rotation[j,k])*local[k] for k in range(3))+float(translation[j]) for j in range(3)]
        homogeneous=[sum(float(matrix[j,k])*point[k] for k in range(3)) for j in range(3)]
        assert np.isfinite(homogeneous).all() and homogeneous[2] != 0
        corners.append(local);camera.append(point)
        uv.append([homogeneous[0]/homogeneous[2],homogeneous[1]/homogeneous[2]])
    return np.asarray(corners),np.asarray(camera),np.asarray(uv)


def selfcheck():
    pose={'available':True,'R_cf':np.eye(3).tolist(),'centroid':[0.,0.,2.], 'cf_extents':[2.,2.,2.]}
    local,camera,uv=scalar_projection(pose,[[100,0,50],[0,100,40],[0,0,1]])
    np.testing.assert_array_equal(local[0],[-1,-1,-1])
    np.testing.assert_array_equal(camera[0],[-1,-1,1])
    np.testing.assert_array_equal(uv[0],[-50,-60])
    assert len(EDGES)==12
    print('GALLERY_PROJECTION_SELFCHECK_PASS artifact_reads=0')


def verify(report=None):
    """Return JSON-serializable evidence for the final PUBLIC_REVIEW receipt."""
    if report is None:
        report=C.read(C.DOC/'REPORT_DATA.json')
    C.verify(C.read(C.DOC/'REAL_PROTOCOL_SHA.json'))
    protocol=C.read(C.DOC/'REAL_PROTOCOL.json')
    lock=C.read(C.DOC/'REAL_ROUTING_LOCK.json')
    assert protocol['schema']=='pallet_pose_anchor_rbf_learned_real_v1'
    assert report['real_protocol']==C.bind(C.DOC/'REAL_PROTOCOL.json')
    assert report['real_routing']==C.bind(C.DOC/'REAL_ROUTING_LOCK.json')
    assert lock['runtime_uses_fixed_rbf'] and protocol['runtime_uses_fixed_rbf']
    assert lock['basis_SHA_bind']==protocol['basis_SHA_bind']
    C.verify(protocol['basis_SHA_bind'])
    assert lock['protocol']==C.bind(C.DOC/'REAL_PROTOCOL.json')
    assert lock['whole_pose_selection'] and not lock['real_reference_values_read']
    assert not lock['runtime_uses_margin'] and not lock['runtime_safe_mask']
    assert protocol['inputs']==lock['inputs']
    metadata=bound_read(protocol['inputs']['metadata'])
    groups=bound_read(protocol['inputs']['groups'])
    poses=bound_read(protocol['inputs']['poses'])
    choices=bound_read(lock['choices'])
    rows={r['id']:r for r in metadata}
    assert len(rows)==173 and choices['ids']==[r['id'] for r in metadata]
    result=C.read(C.DOC/'REAL_RESULTS.json')
    assert report['real_results']==C.bind(C.DOC/'REAL_RESULTS.json')
    assert result['routing_lock']==C.bind(C.DOC/'REAL_ROUTING_LOCK.json')
    csv_path=C.DOC/'REAL_FRAME_RESULTS.csv'
    csv_binding=next(b for b in result['artifacts'] if b['path']==str(csv_path.relative_to(C.ROOT)))
    C.verify(csv_binding)
    with csv_path.open(newline='') as handle:
        table=list(csv.DictReader(handle))
    lookup={(r['model'],r['id']):r for r in table}
    assert len(table)==len(lookup)==2249
    # Prior published report fixes the six diagnostic display frames. This is
    # not outcome selection among the current three learned UNION models.
    old_path=C.ANCHOR_DOC/'REPORT_DATA.json'
    manifest=C.read(C.ANCHOR_DOC/'PUBLICATION_MANIFEST.json')
    old_binding=next(b for b in manifest['files'] if b['path']==str(old_path.relative_to(C.ROOT)))
    old=bound_read(old_binding)
    published=Path('/tmp/pallet-pose-github-review-20260930')/old_path.relative_to(C.ROOT)
    assert published.is_file() and old_path.read_bytes()==published.read_bytes()
    historical=old['illustrations']; assert len(historical)==6
    natural=groups['NATURAL99']; assert len(natural)==99
    recordings=sorted({rows[fid]['recording'] for fid in natural}); assert len(recordings)==6
    chosen=[min((fid for fid in natural if rows[fid]['recording']==rec),
        key=lambda fid:(-float(lookup['R0',fid]['translation_cm']),fid)) for rec in recordings]
    assert [r['id'] for r in historical]==chosen
    details=report['illustrations'];assert len(details)==6
    assert [r['id'] for r in details]==chosen and len(set(chosen))==6
    assert report['actual_RGB_images']==6
    selection=bound_read(report['gallery_selection'])
    assert selection['complete'] and selection['illustrations']==details
    assert selection['source_selection_report']==old_binding
    assert selection['routing']==C.bind(C.DOC/'REAL_ROUTING_LOCK.json')
    assert selection['metadata']==protocol['inputs']['metadata']
    assert selection['poses']==protocol['inputs']['poses']
    assert selection['metrics'] in result['artifacts']
    C.verify(selection['metrics'])
    assert selection['actual_RGB_images']==6 and selection['panels']==24
    assert selection['new_metric_calls']==selection['new_image_forwards']==selection['new_PnP_solves']==0
    maximum_uv_difference=0.; maximum_camera_difference=0.; results=[];panel_count=0
    for detail,prior in zip(details,historical):
        fid=detail['id']; row=rows[fid]
        assert detail['recording']==row['recording']==prior['recording']
        assert detail['dimensions_m']==row['xyz']==prior['dimensions_m']
        assert detail['image']==row['image']==prior['image']
        assert detail['K']==row['K']
        assert detail['corner_signs']==[list(s) for s in SIGNS]
        assert detail['edges']==[list(e) for e in EDGES]
        C.verify(detail['image'])
        with Image.open(C.ROOT/detail['image']['path']) as im:
            width,height=im.size
            rgb=np.asarray(im.convert('RGB'))
        assert [height,width]==row['hw']==detail['image_hw']
        assert rgb.shape==(height,width,3) and rgb.dtype==np.uint8
        assert [p['model'] for p in detail['panels']]==list(MODELS)
        checked=[]
        for panel in detail['panels']:
            model=panel['model']
            choice=choices['records'][model][fid] if model!='R0' else None
            pose=choice['pose'] if choice else poses['R0'][fid]['GEO_pose']
            expected_hyp=choice['hypothesis'] if choice else poses['R0'][fid]['GEO_name']
            parent=choice['parent'] if choice else 'R0'
            assert panel['parent']==parent and panel['hypothesis']==expected_hyp
            assert panel['pose']==pose and pose['available']
            if choice:
                expected=next(h['pose'] for h in poses[parent][fid]['hypotheses'] if h['name']==expected_hyp)
                assert expected==pose and not choice['fallback']
            dimensions=np.array(row['xyz']);cf=np.array(pose['cf_extents'])
            assert np.allclose(cf,dimensions,rtol=0,atol=1e-9) or np.allclose(cf,dimensions[[2,1,0]],rtol=0,atol=1e-9)
            local,camera,uv=scalar_projection(pose,row['K'])
            np.testing.assert_allclose(panel['corners_cf'],local,rtol=0,atol=1e-12)
            np.testing.assert_allclose(panel['corners_camera'],camera,rtol=0,atol=1e-12)
            np.testing.assert_allclose(panel['projected_uv'],uv,rtol=0,atol=1e-8)
            uv_difference=float(np.max(np.abs(uv-np.asarray(panel['projected_uv']))))
            camera_difference=float(np.max(np.abs(camera-np.asarray(panel['corners_camera']))))
            maximum_uv_difference=max(maximum_uv_difference,uv_difference)
            maximum_camera_difference=max(maximum_camera_difference,camera_difference)
            edges=[(i,j) for i,j in EDGES if min(camera[i,2],camera[j,2])>0]
            assert panel['drawn_edges']==[list(e) for e in edges]
            values=lookup[model,fid]
            assert panel['T_cm']==float(values['translation_cm'])
            assert panel['R_deg']==float(values['rotation_deg'])
            assert values['pose_available']=='True' and values['parent']==parent and values['hypothesis']==expected_hyp
            checked.append(dict(model=model,parent=parent,hypothesis=expected_hyp,
                full_pose_exact=True,drawn_edges=len(edges),UV_max_difference_px=uv_difference,
                T_cm=panel['T_cm'],R_deg=panel['R_deg']))
            panel_count+=1
        results.append(dict(id=fid,recording=row['recording'],image=detail['image'],image_hw=[height,width],
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),dimensions_m=row['xyz'],
            calibration_exact=True,panels=checked))
    assert panel_count==24
    jpgs=[b for b in report['artifacts'] if b['path'].endswith('.jpg')]
    assert len(jpgs)==3
    montages=[]
    for b in jpgs:
        C.verify(b)
        with Image.open(C.ROOT/b['path']) as image:
            dimensions=image.size;image.verify()
        montages.append(dict(binding=b,width=dimensions[0],height=dimensions[1]))
    return dict(complete=True,PASS=True,code=C.bind(Path(__file__)),
        actual_RGB_images=6,learned_seed_panels='All3 frozen UNION seeds shown; no best-seed selection.',
        panels=24,corner_projections=192,projected_coordinate_values=384,
        input_pixels_and_dimensions_verified=True,current_frozen_pose_parity=True,
        maximum_UV_difference_px=maximum_uv_difference,UV_tolerance_px=1e-8,
        maximum_camera_difference_m=maximum_camera_difference,
        panel_T_R_values_equal_to_public_CSV=48,original_case_selection_equal=True,
        selection_rule='One maximum operational R0 T error per natural recording; lexical ID tie. Same prior published6frames.',
        prior_report=old_binding,gallery_selection=report['gallery_selection'],CSV=csv_binding,metadata=protocol['inputs']['metadata'],
        poses=protocol['inputs']['poses'],choices=lock['choices'],frames=results,montages=montages,
        frozen_RBF_basis=protocol['basis_SHA_bind'],current_real_protocol=report['real_protocol'],
        real_reference_reads=0,new_PnP_solves=0,image_forwards=0,
        limitation='Projection metadata and source pixels verified; final figure legibility and drawn overlay appearance are separately visually reviewed by root.')


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selfcheck','verify'))
    args=parser.parse_args()
    if args.action=='selfcheck':selfcheck()
    else:print(json.dumps(verify(),ensure_ascii=False))
