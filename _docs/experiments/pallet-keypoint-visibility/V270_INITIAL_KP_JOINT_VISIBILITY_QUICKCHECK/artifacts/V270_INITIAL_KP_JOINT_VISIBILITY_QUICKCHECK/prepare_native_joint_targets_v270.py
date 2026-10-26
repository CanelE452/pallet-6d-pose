"""Existing SYN1003 native-coordinate plus external-ray-proxy joint targets.

No model, RGB decode, M4 supervision, current223 annotation, optimizer, or PnP.
Stored cuboid/xy/fractions already share camera-dynamic slot order.  perm_v4
is retained as provenance and is deliberately not applied a second time.
"""
from pathlib import Path
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import time
import types

import numpy as np

HERE = Path(__file__).absolute().parent
TASK = Path('/home/minjae/Documents/Codex/2026-10-20/task-2')
SOURCE_COMPLETE = TASK / 'pallet-keypoint-visibility/publication_root_preparation/V267_FULL_IMAGE_CLIP_VISIBLE_MASK/data_audit/COMPLETE.json'
SOURCE_COMPLETE_SHA = 'ef2a2e0ac722df7a2f9cfc84d7333726107df20b1b39feba0db7ce673da2d483'
GT = Path('/home/minjae/Documents/pallet-research/pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz')
GT_SHA = '578a24d67db0959a0b014ece9802279c803ee05c9845c2fbe38e8e8c389bdeb6'
EXCERPT = TASK / 'virtual_cuboid_ray_oracle_20261022_v57/proxy_source_excerpt.py'
EXCERPT_SHA = '8aa96899db235c99e02b5ca32747e9bf3a47f6b15c064578040d4f4822a585ee'
WARNING = ('EXTERNAL_RAY_BLOCKAGE_PROXY_NOT_PHYSICAL_NATIVE_VISIBILITY; '
           'virtual cuboid anchors may lack physical material support; '
           'source self geometry is convex-box eligibility, not mesh visibility; '
           '9-ray aggregate is not a central pixel or human D/E label')


def safe(value):
    p = Path(value)
    if not p.is_absolute() or any(t in str(p).lower() for t in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def read(path, expected=None):
    p = safe(path)
    raw = p.read_bytes()
    bound = {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    if expected is not None:
        assert bound['sha256'] == expected, bound
    return raw, bound


def binding(path):
    return read(path)[1]


def checked(bound):
    raw, actual = read(bound['path'], bound['sha256'])
    assert actual == bound, (actual, bound)
    return raw


def write_json(path, obj):
    p = safe(path)
    assert not p.exists(), p
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    p = safe(path)
    assert not p.exists(), p
    p.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for r in rows))


def run(out):
    started = time.time()
    out = safe(out)
    assert not out.exists(), out
    complete_raw, complete_bound = read(SOURCE_COMPLETE, SOURCE_COMPLETE_SHA)
    source_complete = json.loads(complete_raw)
    assert source_complete['PASS'] and source_complete['completed'] and source_complete['frames'] == 1003
    frame_bound = source_complete['outputs']['FRAME_MANIFEST.jsonl']
    frames = [json.loads(line) for line in checked(frame_bound).decode().splitlines()]
    assert len(frames) == 1003 and [r['input']['frame_index'] for r in frames] == list(range(1003))
    gt_raw, gt_bound = read(GT, GT_SHA)
    gt = {r['id']: r for r in map(json.loads, gzip.decompress(gt_raw).decode().splitlines())}
    source_raw, source_bound = read(EXCERPT, EXCERPT_SHA)
    source = types.ModuleType('v270_pinned_source_self_functions')
    exec(compile(source_raw, str(EXCERPT), 'exec'), source.__dict__)
    out.mkdir()
    protocol = {
        'completed': True, 'schema': 'V270_NATIVE_DYNAMIC8_JOINT_PROXY_TARGET_PROTOCOL_V1',
        'code': binding(Path(__file__).absolute()), 'source_complete': complete_bound,
        'source_frames': frame_bound, 'source_gt': gt_bound, 'source_self_excerpt': source_bound,
        'source_excerpt_limit': 'Current Windows self/aggregate excerpt; full historical renderer byte identity is unverified. Original 9-ray hits/owners/offset rays are not present in annotation.',
        'order': 'Annotation projected_cuboid, cuboid and fraction[0:8] already camera_dynamic_0123_v4; same saved projected_GT_native_xy. No permutation applied. perm_v4 provenance retained. Centroid index8 excluded.',
        'visibility_teacher': 'Only positive-camera-depth raster-inframe source-convex-self-visible points: D_proxy iff external_fraction==0, E_proxy iff external_fraction>=0.5; fractions in (0,0.5) UNKNOWN. Source 0.5 fixed before any real evaluation.',
        'xy_teacher': 'Stored renderer projected virtual cuboid coordinate, including hidden points. loc_valid positive z and raster [0,W)x[0,H); amodal geometry target, not physical visible-corner gold.',
        'normalization': 'xy01=(xy_px+0.5)/[raw_W,raw_H], no clipping; invalid points excluded by loc_valid.',
        'visibility_unknown_reasons': ['OUTSIDE_OR_NONPOSITIVE_DEPTH', 'CONVEX_BOX_SELF', 'PARTIAL_RAY_BLOCKAGE_LT_HALF'],
        'split': 'Unchanged original TRAIN768 / DEV235, source RGB/scenario disjoint; repeatedly exposed development split, incomplete asset-family lineage.',
        'label_warning': WARNING,
        'M4_supervision_reads': 0, 'RGB_decodes': 0, 'current223_reads': 0, 'model_forwards': 0,
        'selection_or_fit': 0,
    }
    write_json(out/'TARGET_PROTOCOL_BEFORE_LABEL_JOIN.json', protocol)
    shape = (1003, 8)
    xy = np.zeros((*shape, 2), np.float32)
    xy01 = np.zeros_like(xy)
    loc_valid = np.zeros(shape, bool)
    vis = np.full(shape, -1, np.int8)
    self_visible = np.zeros(shape, bool)
    fractions = np.zeros(shape, np.float32)
    depth = np.zeros(shape, np.float32)
    fit = np.asarray([r['input']['split'] == 'train' for r in frames], bool)
    raw_hw = np.asarray([r['input']['raw_hw'] for r in frames], np.int32)
    perm = np.zeros(shape, np.int8)
    assert int(fit.sum()) == 768 and int((~fit).sum()) == 235
    counts = {fold: collections.Counter() for fold in ('train', 'validation')}
    labels, point_rows, maximum = [], [], 0.0
    source_aggregate_mismatches = []
    annotation_bounds = []
    permutations = collections.Counter()
    for i, r in enumerate(frames):
        inp = r['input']; name = inp['id']; fold = inp['split']; g = gt[name]
        assert g['split'] == fold and g['raw_hw'] == inp['raw_hw']
        ann_bound = r['target']['annotation']
        objj = json.loads(checked(ann_bound))
        assert len(objj['objects']) == 1
        obj = objj['objects'][0]
        assert g['renderer_annotation_sha256'] == ann_bound['sha256']
        assert obj['keypoint_convention'] == 'camera_dynamic_0123_v4'
        points = np.asarray(obj['projected_cuboid'], np.float64)
        assert points.shape == (8,2) and np.isfinite(points).all()
        assert np.array_equal(points, np.asarray(g['projected_GT_native_xy'], np.float64))
        xyz = np.asarray(g['native_GT_xyz'], np.float64)
        pc = xyz @ np.asarray(g['R_fixed'], np.float64).T + np.asarray(g['t'], np.float64)
        proj = pc @ np.asarray(g['K'], np.float64).T
        reproj = proj[:,:2] / proj[:,2:3]
        error = float(np.max(np.abs(reproj-points)))
        maximum = max(maximum,error)
        assert error < 0.001, (name,error)
        p = np.asarray(obj['perm_v4'], np.int8)
        assert p.shape == (8,) and sorted(p.tolist()) == list(range(8))
        perm[i] = p
        permutations[str(p.tolist())] += 1
        sv = np.asarray(source.self_visible_corner_mask(obj['cuboid'],objj['camera_data']['location_worldframe']), bool)
        f9 = np.asarray(obj['v2_labels']['occlusion_fraction'], np.float64)
        assert f9.shape == (9,) and np.isfinite(f9).all() and (f9>=0).all() and (f9<=1).all()
        f = f9[:8]
        h,w = inp['raw_hw']
        inside = (pc[:,2]>0)&(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
        source_inside = (pc[:,2]>0)&(points[:,0]>=0)&(points[:,0]<=w)&(points[:,1]>=0)&(points[:,1]<=h)
        agg = source.external_corner_gate_metrics(source_inside, f, self_visible=sv)
        for k,v in [('V_inframe','V_actual'),('V_vis','V_vis_actual'),('ext_occ_corners','ext_occ_corners_actual')]:
            if int(agg[k]) != int(obj['v2_labels'][v]):
                source_aggregate_mismatches.append({'id':name,'metric':k,'replayed':int(agg[k]),'stored':int(obj['v2_labels'][v])})
        eligible = inside&sv
        y = np.full(8,-1,np.int8)
        y[eligible&(f==0)] = 0
        y[eligible&(f>=0.5)] = 1
        xy[i] = points.astype(np.float32)
        xy01[i] = ((points+0.5)/np.asarray([w,h],np.float64)).astype(np.float32)
        loc_valid[i] = inside
        vis[i] = y
        self_visible[i] = sv
        fractions[i] = f.astype(np.float32)
        depth[i] = pc[:,2].astype(np.float32)
        c = counts[fold]
        c['frames'] += 1; c['points'] += 8; c['loc_valid'] += int(inside.sum())
        c['outside_or_nonpositive_depth'] += int((~inside).sum())
        c['self_inside'] += int((inside&~sv).sum())
        c['eligible_before_ambiguous_filter'] += int(eligible.sum())
        c['D_clear0'] += int((y==0).sum());c['E_ge_half'] += int((y==1).sum())
        c['unknown_partial_between0_half'] += int((eligible&(f>0)&(f<0.5)).sum())
        c['E_full1_subset'] += int((eligible&(f==1)).sum())
        c['source_inclusive_vs_raster_mismatch'] += int((inside!=source_inside).sum())
        assert int((y>=0).sum()) == int((y==0).sum()+(y==1).sum())
        reasons=[]
        for k in range(8):
            reason=('OUTSIDE_OR_NONPOSITIVE_DEPTH' if not inside[k] else
                    'CONVEX_BOX_SELF' if not sv[k] else
                    'PARTIAL_RAY_BLOCKAGE_LT_HALF' if y[k]<0 else
                    'CLEAR_EXTERNAL_RAYS' if y[k]==0 else 'BLOCKED_EXTERNAL_RAYS_GE_HALF')
            reasons.append(reason)
            point_rows.append({'frame_index':i,'id':name,'split':fold,'native_slot':k,
                'gt_x_px':float(points[k,0]),'gt_y_px':float(points[k,1]),'gt_depth':float(pc[k,2]),
                'loc_valid':bool(inside[k]),'source_convex_self_visible':bool(sv[k]),
                'external_ray_fraction':float(f[k]),'visibility_y':int(y[k]),
                'visibility_valid':bool(y[k]>=0),'visibility_reason':reason,
                'source_dynamic_slot':k,'canonical_asset_corner_via_perm_provenance':int(p[k]),
                'physical_native_visibility_gold':False})
        labels.append({'frame_index':i,'id':name,'split':fold,'raw_hw':inp['raw_hw'],'RGB':inp['RGB'],
            'annotation':ann_bound,'projected_native_dynamic8_xy':points.tolist(),'loc_valid8':inside.tolist(),
            'visibility_y8':y.tolist(),'visibility_reason8':reasons,
            'source_convex_self_visible8':sv.tolist(),'external_ray_fraction8':f.tolist(),
            'perm_v4_dynamic_to_canonical_provenance_only':p.tolist(),'permutation_applications':0,
            'source_asset':obj.get('source_asset'),'pallet_type':obj['v2_labels'].get('pallet_type'),
            'material_variant':obj['v2_labels'].get('material_variant_actual'),'label_warning':WARNING})
        annotation_bounds.append(ann_bound)
    assert not source_aggregate_mismatches, source_aggregate_mismatches[:10]
    write_jsonl(out/'FRAME_TARGETS1003.jsonl',labels)
    with safe(out/'POINT_TARGETS8024.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(point_rows[0]));writer.writeheader();writer.writerows(point_rows)
    np.savez_compressed(safe(out/'TARGETS.npz'),xy_px=xy,xy01=xy01,loc_valid=loc_valid,
        visibility_y=vis,visibility_valid=(vis>=0),source_convex_self_visible=self_visible,
        external_ray_fraction=fractions,gt_depth=depth,raw_hw=raw_hw,fit=fit,
        frame_index=np.arange(1003,dtype=np.int32),perm_v4_provenance=perm)
    write_json(out/'ANNOTATION_SOURCE_BINDINGS.json',{'annotations':annotation_bounds,
        'original_RGB_binding_count':1003,'original_RGB_decodes':0,'M4_supervision_reads':0})
    # Fixed diagnostic selection by source order and labels, before any RGB/mask arrays.
    selections=[]
    for fold in ('train','validation'):
        for material in sorted({r['material_variant'] for r in labels}):
            candidates=[r for r in labels if r['split']==fold and r['material_variant']==material and 0 in r['visibility_y8'] and 1 in r['visibility_y8']]
            if candidates:
                r=candidates[0]
                selections.append({'kind':'SAME_FRAME_CLEAR_AND_BLOCKED_PROXY_PAIR','frame_index':r['frame_index'],
                    'id':r['id'],'split':fold,'material_variant':material,
                    'clear_slot':r['visibility_y8'].index(0),'blocked_slot':r['visibility_y8'].index(1)})
    write_json(out/'EXAMPLES_SELECTED_BEFORE_RGB.json',{'rule':'First source-order frame per split and material with at least one strict clear and blocked label; not controlled counterfactual pairs.',
        'examples':selections,'RGB_arrays_read':0,'current223_reads':0})
    summary={'completed':True,'PASS':True,'schema':'V270_SYN1003_NATIVE_JOINT_PROXY_TARGETS_V1',
        'code':binding(Path(__file__).absolute()),'source_complete':complete_bound,'source_frames':frame_bound,
        'source_gt':gt_bound,'source_self_excerpt':source_bound,'counts':{k:dict(v) for k,v in counts.items()},
        'label_warning':WARNING,'permutation_applications':0,'stored_xy_saved_gt_exact_match_frames':1003,
        'xyz_pose_K_reprojection_max_absolute_px':maximum,'source_aggregate_match_frames':1003,
        'source_aggregate_mismatches':source_aggregate_mismatches,'perm_v4_provenance_counts':dict(permutations),
        'RGB_decodes':0,'M4_supervision_reads':0,'current223_reads':0,'model_forwards':0,'optimizer_updates':0,
        'xy_target_kind':'AMODAL_VIRTUAL_CUBOID_PROJECTION_NOT_PHYSICAL_ENDPOINT',
        'elapsed_seconds':time.time()-started}
    write_json(out/'TARGET_AUDIT_SUMMARY.json',summary)
    names=['TARGET_PROTOCOL_BEFORE_LABEL_JOIN.json','FRAME_TARGETS1003.jsonl','POINT_TARGETS8024.csv','TARGETS.npz',
        'ANNOTATION_SOURCE_BINDINGS.json','EXAMPLES_SELECTED_BEFORE_RGB.json','TARGET_AUDIT_SUMMARY.json']
    result={**{k:v for k,v in summary.items() if k in ('completed','PASS','schema','counts','label_warning','code')},
        'outputs':{n:binding(out/n) for n in names},'current223_reads':0,'RGB_decodes':0,
        'model_forwards':0,'optimizer_updates':0}
    write_json(out/'COMPLETE.json',result)
    print(json.dumps({'COMPLETE':binding(out/'COMPLETE.json'),'counts':summary['counts']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='syn_joint_targets_v1')
    args=parser.parse_args();run(HERE/args.out)
