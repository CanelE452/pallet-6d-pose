"""Read-only existing DINO token-cache compatibility audit; no GT members or model calls.

Only the new private cache_contract.json may be written. NPZ containers include
legacy labels, but member access is restricted to the seven input-only keys
below. Legacy source loaders, target arrays, weights and inference are not used.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
from PIL import Image
from scripts.research.pallet_sensors_submission_v1.posefix_contract_math import axis_aligned_crop_matrix

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RAW=ROOT/'data/pallet/results'/HERE.name
OUTPUT=RAW/'cache_contract.json'
DOCS=ROOT/'_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1'
SOURCE_DOC=ROOT/'_docs/experiments/pallet_pose_union_selection_20261001_v1'
SOURCE_RAW=ROOT/'data/pallet/results/pallet_pose_union_selection_20261001_v1'
LEGACY=ROOT/'data/pallet/results/pallet_line_pose_v1/cache'
PHASES=('dino_source_diversity','dino_tail_sampling','dino_localization')
SAFE_NPZ={'feature','matrix','points','valid','original_points','bbox_diagonal','protocol_sha256'}
SAFE_LEGACY=('points','boxes','point_valid','input_shape','record_index','gain')
READ_MEMBERS=Counter()
BINDINGS={}


def read(p):return json.loads(Path(p).read_text())


def bind(p):
    p=Path(p).resolve();key=str(p.relative_to(ROOT))
    if key not in BINDINGS:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        BINDINGS[key]=dict(path=key,sha256=h.hexdigest(),bytes=p.stat().st_size)
    return BINDINGS[key]


def verify(b):
    actual=bind(ROOT/b['path'])
    assert actual['sha256']==b['sha256'],('SHA',b['path'])
    if 'bytes'in b:assert actual['bytes']==b['bytes'],('BYTES',b['path'])
    return actual


def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes());return h.hexdigest()


def member(z,key):
    assert key in SAFE_NPZ,('FORBIDDEN_CACHE_MEMBER',key)
    READ_MEMBERS[key]+=1;return z[key]


def guard():
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1]
        if not path.is_relative_to(ROOT):return
        if isinstance(mode,str) and any(k in mode for k in 'wax+'):
            assert path==OUTPUT,('WRITE_SCOPE',path);return
        name=str(path)
        assert not any(k in name for k in ('/data/evaluation/','/annotations/','GEOMETRY_SIDETABLE','GEOMETRY_RESOLVED_POSE_GT','SOURCE_TRAIN_LABELS','SYNTH_LABELS','SOURCE_VAL_METRICS','POSE_METRICS','/RESULTS.json','/FIT_','/fits/')),('FORBIDDEN',name)
        assert path.name not in ('gt_points.npy','gt_valid.npy','gt_support.npy','matched.npy','matched_gt_index.npy','matched_iou.npy')
    sys.addaudithook(hook)


def groups(row,eligible):
    return ['all5120','original_'+row['split']]+(['eligibleTRAIN2598'] if row['id'] in eligible else [])


def legacy_matrix(row,cache_row,arrays,wide):
    h,w=row['hw'];ih,iw=map(int,arrays['input_shape'][cache_row])
    gain=min(640/h,640/w);rw,rh=round(w*gain),round(h*gain)
    offset=np.array([round((iw-rw)/2-.1),round((ih-rh)/2-.1)],np.float64)
    assert (offset>=0).all() and abs(float(arrays['gain'][cache_row])-gain)<=1e-9
    box=((arrays['boxes'][cache_row].reshape(2,2)-offset)/gain).reshape(4)
    matrix=axis_aligned_crop_matrix(box)
    if wide:matrix[:2,2]+=[144,192]
    points=(arrays['points'][cache_row]-offset)/gain
    return matrix,points,box


def run():
    assert not OUTPUT.exists(),'Immutable audit output already exists.'
    guard();started=time.monotonic()
    lock=read(SOURCE_DOC/'SOURCE_PREDICTIONS_LOCK.json')
    assert lock['complete'] and lock['frames']==5120
    metadata=verify(lock['metadata']);rows=read(ROOT/metadata['path'])
    assert len(rows)==len({r['id'] for r in rows})==5120
    assert Counter(r['split'] for r in rows)=={'TRAIN':4096,'VAL':1024}
    r0receipt=read(ROOT/lock['receipts']['R0']['path']);verify(lock['receipts']['R0'])
    assert r0receipt['complete'] and r0receipt['frames']==5120 and r0receipt['protocol']==lock['protocol']
    assert r0receipt['checkpoint']['sha256']=='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
    train_audit_path=ROOT/'_docs/experiments/pallet_pose_residual_direction_audit_20261001_v1/FEATURE_AUDIT.json'
    train_audit=read(train_audit_path);assert train_audit['complete'] and train_audit['PASS']
    verify(train_audit['directions'])
    # No direction values or source label values needed to establish fixed membership.
    with np.load(ROOT/train_audit['directions']['path'],allow_pickle=False) as z:
        eligible=z['ids'].tolist();indices=z['source_index'].copy()
    assert len(eligible)==len(set(eligible))==2598
    assert [rows[int(i)]['id'] for i in indices]==eligible and all(rows[int(i)]['split']=='TRAIN' for i in indices)
    eligible=set(eligible)
    manifest=read(LEGACY/'CACHE_MANIFEST.json');manifest_binding=bind(LEGACY/'CACHE_MANIFEST.json')
    assert manifest['n_records']==60000 and manifest['backbone_sha256']==r0receipt['checkpoint']['sha256']
    arrays={k:np.load(LEGACY/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in SAFE_LEGACY}
    by_phase={};identity={r['id']:[] for r in rows};checked_caches={};current_predictions={};images_checked={}
    expected_protocols={}
    for ph in ('dino_wide',*PHASES):
        pp=DOCS/ph/'PROTOCOL.json';expected_protocols[bind(pp)['sha256']]=bind(pp)
    backbone=read(DOCS/'dino_source_diversity/BACKBONE.json')
    # Checkpoint bytes are hashed only: no torch load/import or model creation.
    for b in [backbone['checkpoint'],backbone['loader'],backbone['license'],*backbone['code']]:verify(b)
    assert backbone['model']=='dinov2_vits14' and backbone['commit']=='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
    for ph in PHASES:
        protocol=read(DOCS/ph/'PROTOCOL.json');receipt=read(DOCS/ph/'CACHE_COMPLETE.json')
        verify(receipt['protocol']);assert receipt['protocol']==bind(DOCS/ph/'PROTOCOL.json')
        source={r['id']:r for r in protocol['source_records']}
        cached={r['id']:r for r in receipt['records'] if r['domain']=='source'}
        assert len(source)==len(cached) and set(source)==set(cached)
        ordered=sorted(r['row'] for r in protocol['source_records'] if r['partition']=='train')+protocol['source_held_rows']
        if ph=='dino_localization':ordered=sorted(ordered)  # Original N.source_cache_signature uses np.unique(all rows).
        signatures={k:array_sha(arrays[k][ordered]) for k in SAFE_LEGACY}
        for key in SAFE_LEGACY:assert signatures[key]==protocol['source_cache_signature'][key],(ph,key)
        assert any(b==manifest_binding for b in protocol['sources'])
        relevant_code=[b for b in protocol['sources'] if b['path'].startswith('scripts/')]
        for b in relevant_code:verify(b)
        wide=ph!='dino_localization';expected_shape=(384,56,42) if wide else (384,28,21)
        counts={g:Counter() for g in ('all5120','original_TRAIN','original_VAL','eligibleTRAIN2598')};matches=[]
        for i,row in enumerate(rows):
            gg=groups(row,eligible)
            for g in gg:counts[g]['total']+=1
            if row['id'] not in source:
                for g in gg:counts[g]['not_in_cache_source_population']+=1
                continue
            prior=source[row['id']];entry=cached[row['id']]
            assert entry['row']==prior['row'] and entry['key']=='S'+str(prior['row'])
            assert manifest['record_ids'][prior['row']]==row['id']
            for g in gg:counts[g]['ID_match']+=1
            if prior['image']['sha256']!=row['image']['sha256']:
                for g in gg:counts[g]['image_SHA_mismatch']+=1
                continue
            for b in (row['image'],prior['image']):verify(b)
            for g in gg:counts[g]['actual_image_SHA_equal']+=1
            if row['id'] not in images_checked:
                with Image.open(ROOT/row['image']['path']) as image:hw=[image.height,image.width]
                assert hw==row['hw'];images_checked[row['id']]=hw
            cb=verify(entry['cache']);path=ROOT/cb['path']
            if cb['path'] not in checked_caches:
                with np.load(path,allow_pickle=False) as z:
                    matrix=member(z,'matrix').copy();q=member(z,'original_points').copy()
                    features=member(z,'feature');tag=str(member(z,'protocol_sha256'))
                    assert features.dtype==np.float16 and np.isfinite(features).all()
                    shape=tuple(features.shape);feature_sha=array_sha(features)
                    assert tag in expected_protocols
                checked_caches[cb['path']]=dict(matrix=matrix,points=q,shape=shape,protocol_sha=tag,feature_sha=feature_sha)
            token=checked_caches[cb['path']];assert token['shape']==expected_shape
            historical,oldq,oldbox=legacy_matrix(row,prior['row'],arrays,wide)
            np.testing.assert_array_equal(token['matrix'],historical)
            np.testing.assert_array_equal(token['points'],oldq)
            if i not in current_predictions:
                binding=r0receipt['files'][i];assert binding['path'].endswith(f'/R0/{i:05d}.json');verify(binding)
                saved=read(ROOT/binding['path']);assert saved['id']==row['id'] and saved['model']=='R0'
                assert saved['checkpoint_sha']==r0receipt['checkpoint']['sha256']
                assert saved['protocol_sha']==lock['protocol']['sha256']
                current_predictions[i]=(saved['prediction'],binding)
            prediction,pbinding=current_predictions[i];selected=prediction['selected_index']
            if selected is None:
                for g in gg:counts[g]['current_R0_no_selected_box']+=1
                continue
            current=prediction['candidates'][selected];newbox=np.asarray(current['box_xyxy'],np.float64)
            newmatrix=axis_aligned_crop_matrix(newbox)
            if wide:newmatrix[:2,2]+=[144,192]
            matrix_exact=np.array_equal(newmatrix,historical)
            for g in gg:
                counts[g]['crop_matrix_exact' if matrix_exact else 'crop_matrix_mismatch']+=1
                counts[g]['strict_reusable']+=int(matrix_exact)
            info=dict(id=row['id'],source_index=i,split=row['split'],eligible_TRAIN=row['id'] in eligible,
                old_partition=prior['partition'],old_cache_row=prior['row'],old_image=prior['image'],current_image=row['image'],
                cache=cb,current_R0_prediction=pbinding,grid=list(expected_shape),dtype='float16',
                feature_array_sha=token['feature_sha'],generation_protocol=expected_protocols[token['protocol_sha']],
                old_matrix=historical.tolist(),current_matrix=newmatrix.tolist(),matrix_exact=matrix_exact,
                max_matrix_absolute_difference=float(np.max(np.abs(historical-newmatrix))),
                max_box_absolute_difference=float(np.max(np.abs(oldbox-newbox))),
                max_original_keypoint_absolute_difference=float(np.max(np.abs(oldq-np.asarray(current['keypoints_xy'])))),
                historical_matrix_reconstructed_exact=True,strict_reusable=matrix_exact,
                reason='EXACT_TRANSFORM_MATCH' if matrix_exact else 'OLD_FLOAT32_CANVAS_BOX_INVERSE_DIFFERS_FROM_CURRENT_R0_BOX')
            matches.append(info);identity[row['id']].append(dict(phase=ph,cache=cb['path'],strict_reusable=matrix_exact))
        gaps=[r['max_matrix_absolute_difference'] for r in matches]
        by_phase[ph]=dict(protocol=bind(DOCS/ph/'PROTOCOL.json'),receipt=bind(DOCS/ph/'CACHE_COMPLETE.json'),
            cache_rows=len(receipt['records']),source_rows=len(source),real_rows=sum(r['domain']=='real' for r in receipt['records']),
            counts={k:dict(v) for k,v in counts.items()},legacy_input_signatures=signatures,
            feature_shape=list(expected_shape),max_matrix_absolute_difference=max(gaps) if gaps else None,
            median_matrix_absolute_difference=float(np.median(gaps)) if gaps else None,matches=matches)
        print('CACHE_PHASE_AUDITED',ph,by_phase[ph]['counts'],flush=True)
    mapping=[]
    for i,row in enumerate(rows):
        mm=identity[row['id']];wide=[r for r in mm if r['phase']!='dino_localization']
        mapping.append(dict(id=row['id'],source_index=i,split=row['split'],eligible_TRAIN=row['id'] in eligible,
            matched_caches=mm,wide_input_cache_present=bool(wide),strict_wide_reusable=any(r['strict_reusable'] for r in wide),
            reason='CACHE_POPULATION_MISSING' if not wide else ('EXACT_TRANSFORM_MATCH' if any(r['strict_reusable'] for r in wide) else 'CROP_AFFINE_MISMATCH')))
    summary={}
    for group,selected in [('all5120',mapping),('eligibleTRAIN2598',[r for r in mapping if r['eligible_TRAIN']]),('VAL1024',[r for r in mapping if r['split']=='VAL'])]:
        summary[group]=dict(rows=len(selected),wide_image_identity_matches=sum(r['wide_input_cache_present'] for r in selected),
            strict_reusable=sum(r['strict_wide_reusable'] for r in selected),reasons=dict(Counter(r['reason'] for r in selected)))
    result=dict(complete=True,PASS=True,status='AUDITED_NOT_FULLY_REUSABLE',scope='Cache identity and input compatibility only; no downstream descriptor, policy, target, error or performance evaluation.',
        code=bind(__file__),current_metadata=metadata,current_prediction_lock=bind(SOURCE_DOC/'SOURCE_PREDICTIONS_LOCK.json'),
        current_R0_receipt=lock['receipts']['R0'],eligible_TRAIN_membership=bind(train_audit_path),eligible_TRAIN_ID_container=train_audit['directions'],
        backbone=bind(DOCS/'dino_source_diversity/BACKBONE.json'),backbone_checkpoint=backbone['checkpoint'],
        backbone_note='BACKBONE.patch_shape [28,21,384] documents original loader smoke. Actual wide feature arrays and generation code establish [384,56,42].',
        preprocessing=dict(crop_hw=[768,576],affine='Original expansion1.25 aspect4:3 crop384x288 matrix then translation+[144,192]. Same scale, twice field in both axes.',
            image='Already reflect101-padded prepared RGB; do not pad again; current K/keypoint coordinates stay in that prepared canvas.',
            warp='cv2.warpAffine INTER_LINEAR, BORDER_CONSTANT zero, BGR to RGB; float32 subtract then restore [123.68,116.78,103.94].',
            resize='Torch float32 RGB/255, bilinear to784x588,align_corners=False.',
            normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]),
            tokens='Frozen official DINOv2 ViT-S/14 final normalized patch tokens only; [56,42,384] row-major to [384,56,42], then float16.',
            token_grid_in_crop='For zero-based pixel-center coordinates, token patch centers are ((j+.5)*14-.5,(i+.5)*14-.5) in resized image. The align_corners=False inverse is ((j+.5)*576/42-.5,(i+.5)*768/56-.5). This documents geometry only; no token sampling/interpolation is performed.',
            no_crop_pixel_or_token_equivalence_inferred_from_small_affine_error=True,RGB_crop_comparisons_run=False),
        reuse_rule='Exact current image bytes AND exact current R0-defined affine AND same feature layer/grid/preprocessing/backbone are required. Any affine mismatch is rejected without tolerance relaxation.',
        summary=summary,phases=by_phase,identity_mapping=mapping,
        inspected_source_images=len(images_checked),verified_unique_token_caches=len(checked_caches),
        allowed_NPZ_member_reads=dict(READ_MEMBERS),GT_NPZ_member_reads=0,legacy_GT_array_reads=0,
        containers_with_label_members_disclosed=True,legacy_source_adapter_instantiated=False,
        code_and_artifact_bindings=list(BINDINGS.values()),new_forwards=0,new_fits=0,new_argmin=0,new_pose_errors=0,
        current_VAL_input_metadata_and_cached_predictions_examined=True,current_VAL_quality_read=False,real_data_read=False,
        caveats=['Historical DINO source membership was selected under its own source split and existing-label eligibility; it is not the current fixed split.',
                 'Tail sampling contributes17 additional shared IDs but was historically selected by source difficulty; no difficulty values or target members were read here.',
                 'Small affine differences do not prove large feature differences, but exact transformed pixels/tokens are not established and are rejected by the fixed strict reuse rule.',
                 'Original localization grid28x21 is a different crop/token contract and cannot substitute for wide56x42.'],
        execution_history=[dict(session=17965,exit_code=1,output_created=False,new_forwards=0,reason='Audit assumed concatenate(train,held) signature row order for original localization; localization producer uses sorted np.unique. Diversity and tail provenance had passed. Corrected according to frozen producer code; no cached data changed.',failed_audit_code_sha256='a7e34e21817f8d44ed7a9363450bb74eade385fe7eebed6fb8b493c43bcb79ac')],
        wall_seconds=time.monotonic()-started)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    # No own-output hash read under a producer guard; root can bind after exit.
    print('CACHE_CONTRACT_PASS',json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();sys.dont_write_bytecode=True;run()
