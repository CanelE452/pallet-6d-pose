"""Independent historical R0-only RGB, dimensions, pose and projection audit.

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

MODELS = ('R0',)
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


def published_binding(doc, path):
    manifest=C.read(doc/'PUBLICATION_MANIFEST.json')
    assert manifest['complete']
    matches=[b for b in manifest['files'] if b['path']==str(path.relative_to(C.ROOT))]
    assert len(matches)==1
    C.verify(matches[0])
    published=Path('/tmp/pallet-pose-github-review-20260930')/path.relative_to(C.ROOT)
    assert published.is_file() and path.read_bytes()==published.read_bytes()
    return matches[0]


def verify(report=None):
    """Historical baseline only; return evidence for root's PUBLIC_REVIEW."""
    if report is None:report=C.read(C.DOC/'REPORT_DATA.json')
    not_run=bound_read(C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json'))
    assert not_run['complete'] and not_run['status']=='NOT_RUN_SOURCE_GATE_FAILED'
    assert not_run['failed_stage']=='SOURCE_VAL'
    assert not_run['learned_real_routes']==not_run['real_metric_calls']==not_run['raw_real_reference_reads']==0
    assert not_run['method_training_fits_completed']==4
    assert all(not_run['absence_checks'].values())
    for path in not_run['absence_checks']:assert not (C.ROOT/path).exists()
    gate=bound_read(not_run['source_gate'])
    assert gate['complete'] and not gate['PASS'] and not gate['real_routing_authorized']
    assert gate['checks_total']==not_run['checks_total']==45
    assert gate['checks_passed']==not_run['checks_passed']
    assert gate['failed_checks']==not_run['failed_checks']
    complete=bound_read(not_run['method_training_complete'])
    assert complete['complete'] and complete['all_certified'] and complete['fit_count']==4
    assert not not_run['method_success'] and not not_run['goal_complete']

    for path in (C.DOC/'REAL_PROTOCOL.json',C.DOC/'REAL_ROUTING_LOCK.json',
                 C.RAW/'REAL_CHOICES.json',C.DOC/'REAL_RESULTS.json',
                 C.DOC/'REAL_FRAME_RESULTS.csv',C.RAW/'POSE_METRICS.json'):
        assert not path.exists()
    prior_path=C.RBF_DOC/'REPORT_DATA.json'
    prior_binding=published_binding(C.RBF_DOC,prior_path)
    prior=bound_read(prior_binding)
    selected_path=C.ANCHOR_DOC/'REPORT_DATA.json'
    selection_binding=published_binding(C.ANCHOR_DOC,selected_path)
    original_selection=bound_read(selection_binding)
    historical=prior['illustrations']; original=original_selection['illustrations']
    assert len(historical)==len(original)==6
    fixed_ids=[r['id'] for r in original]
    assert [r['id'] for r in historical]==fixed_ids
    prior_protocol_binding=published_binding(C.RBF_DOC,C.RBF_DOC/'REAL_PROTOCOL.json')
    assert prior['real_protocol']==prior_protocol_binding
    protocol=bound_read(prior_protocol_binding)
    metadata=bound_read(protocol['inputs']['metadata'])
    groups=bound_read(protocol['inputs']['groups'])
    poses=bound_read(protocol['inputs']['poses'])
    rows={r['id']:r for r in metadata};assert len(rows)==173
    csv_binding=published_binding(C.RBF_DOC,C.RBF_DOC/'REAL_FRAME_RESULTS.csv')
    with (C.ROOT/csv_binding['path']).open(newline='') as handle:
        table=list(csv.DictReader(handle))
    assert len(table)==2249 and len({(r['model'],r['id']) for r in table})==2249
    lookup={r['id']:r for r in table if r['model']=='R0'}
    assert set(lookup)==set(rows)
    natural=groups['NATURAL99']; assert len(natural)==99
    recordings=sorted({rows[fid]['recording'] for fid in natural});assert len(recordings)==6
    chosen=[min((fid for fid in natural if rows[fid]['recording']==rec),
        key=lambda fid:(-float(lookup[fid]['translation_cm']),fid)) for rec in recordings]
    assert chosen==fixed_ids
    details=report['illustrations'];assert len(details)==6
    assert [r['id'] for r in details]==fixed_ids and len(set(fixed_ids))==6
    assert report['actual_RGB_images']==6
    assert report['learned_real_evaluated'] is False
    assert report['method_success'] is False and report['goal_complete'] is False
    assert report['baseline_panels']==6 and report['current_learned_real_panels']==0
    selection=bound_read(report['gallery'])
    assert selection['complete'] and selection['status']=='HISTORICAL_R0_INPUT_ILLUSTRATIONS_ONLY'
    assert not selection['current_learned_real_evaluated']
    assert selection['illustrations']==details
    assert selection['source_selection_report']==selection_binding
    assert selection['historical_baseline_report']==prior_binding
    assert selection['not_run']==C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json')
    assert selection['actual_RGB_images']==6 and selection['panels']==6
    assert selection['new_metric_calls']==selection['new_image_forwards']==selection['new_PnP_solves']==0
    maximum_uv_difference=0.;maximum_camera_difference=0.;results=[];panel_count=0
    for detail,prior_detail,original_detail in zip(details,historical,original):
        fid=detail['id'];row=rows[fid]
        assert detail['recording']==row['recording']==prior_detail['recording']==original_detail['recording']
        assert detail['dimensions_m']==row['xyz']==prior_detail['dimensions_m']==original_detail['dimensions_m']
        assert detail['image']==row['image']==prior_detail['image']==original_detail['image']
        assert detail['K']==row['K']==prior_detail['K']
        assert detail['current_learned_real_evaluated'] is False
        assert detail['corner_signs']==[list(s) for s in SIGNS] and detail['edges']==[list(e) for e in EDGES]
        C.verify(detail['image'])
        with Image.open(C.ROOT/detail['image']['path']) as image:
            width,height=image.size;rgb=np.asarray(image.convert('RGB'))
        assert [height,width]==row['hw']==detail['image_hw']==prior_detail['image_hw']
        assert rgb.shape==(height,width,3) and rgb.dtype==np.uint8
        assert [p['model'] for p in detail['panels']]==['R0']
        panel=detail['panels'][0]
        historical_panel=next(p for p in prior_detail['panels'] if p['model']=='R0')
        pose=poses['R0'][fid]['GEO_pose'];hypothesis=poses['R0'][fid]['GEO_name']
        assert panel['model']==panel['parent']=='R0' and panel['hypothesis']==hypothesis
        assert panel['pose']==historical_panel['pose']==pose and pose['available']
        assert panel['pose_source']=='Historical operational R0 input illustration only; not the current signed-axis Newton model or a new performance calculation.'
        dimensions=np.asarray(row['xyz']);cf=np.asarray(pose['cf_extents'])
        assert np.allclose(cf,dimensions,rtol=0,atol=1e-9) or np.allclose(cf,dimensions[[2,1,0]],rtol=0,atol=1e-9)
        local,camera,uv=scalar_projection(pose,row['K'])
        for key,expected,tolerance in (('corners_cf',local,1e-12),('corners_camera',camera,1e-12),('projected_uv',uv,1e-8)):
            assert panel[key]==historical_panel[key]
            np.testing.assert_allclose(panel[key],expected,rtol=0,atol=tolerance)
        uv_difference=float(np.max(np.abs(uv-np.asarray(panel['projected_uv']))))
        camera_difference=float(np.max(np.abs(camera-np.asarray(panel['corners_camera']))))
        maximum_uv_difference=max(maximum_uv_difference,uv_difference)
        maximum_camera_difference=max(maximum_camera_difference,camera_difference)
        edges=[(i,j) for i,j in EDGES if min(camera[i,2],camera[j,2])>0]
        assert panel['drawn_edges']==historical_panel['drawn_edges']==[list(e) for e in edges]
        values=lookup[fid]
        assert panel['T_cm']==historical_panel['T_cm']==float(values['translation_cm'])
        assert panel['R_deg']==historical_panel['R_deg']==float(values['rotation_deg'])
        assert values['pose_available']=='True' and values['parent']=='R0' and values['hypothesis']==hypothesis
        results.append(dict(id=fid,recording=row['recording'],image=detail['image'],image_hw=[height,width],
            decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),dimensions_m=row['xyz'],
            calibration_exact=True,historical_R0_full_pose_exact=True,hypothesis=hypothesis,
            drawn_edges=len(edges),UV_max_difference_px=uv_difference,
            historical_T_cm=panel['T_cm'],historical_R_deg=panel['R_deg'],current_learned_prediction=False))
        panel_count+=1
    assert panel_count==6
    jpgs=[b for b in report['artifacts'] if b['path'].endswith('.jpg')]
    assert len(jpgs)==3
    assert {Path(b['path']).name for b in jpgs}=={f'baseline_input_rgb_dimensions_{i}.jpg' for i in (1,2,3)}
    montages=[]
    for binding in jpgs:
        C.verify(binding)
        with Image.open(C.ROOT/binding['path']) as image:
            dimensions=image.size;image.verify()
        montages.append(dict(binding=binding,width=dimensions[0],height=dimensions[1]))
    return dict(complete=True,PASS=True,code=C.bind(Path(__file__)),actual_RGB_images=6,
        panels=6,historical_R0_panels=6,current_learned_prediction_panels=0,
        corner_projections=48,projected_coordinate_values=96,
        input_pixels_and_dimensions_verified=True,historical_frozen_pose_parity=True,
        maximum_UV_difference_px=maximum_uv_difference,UV_tolerance_px=1e-8,
        maximum_camera_difference_m=maximum_camera_difference,
        historical_panel_T_R_values_equal_to_previous_public_CSV=12,
        original_case_selection_equal=True,
        selection_rule='Same prior six IDs: maximum historical operational R0 T per natural recording; lexical ID tie. No current model output or performance exists.',
        prior_report=prior_binding,original_selection_report=selection_binding,gallery_selection=report['gallery'],
        historical_CSV=csv_binding,metadata=protocol['inputs']['metadata'],poses=protocol['inputs']['poses'],
        evaluation_not_run=C.bind(C.DOC/'REAL_EVALUATION_NOT_RUN.json'),frames=results,montages=montages,
        real_reference_reads=0,new_reference_metric_calculations=0,new_PnP_solves=0,image_forwards=0,
        new_learned_real_routes=0,method_success=False,goal_complete=False,
        limitation='Historical baseline/input illustrations only. Source RGB and projection metadata verified; final rendered legibility separately reviewed by root.')


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selfcheck','verify'))
    args=parser.parse_args()
    if args.action=='selfcheck':selfcheck()
    else:print(json.dumps(verify(),ensure_ascii=False))
