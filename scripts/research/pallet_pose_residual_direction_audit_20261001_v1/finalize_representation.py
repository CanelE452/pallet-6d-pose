"""Finalize the immutable partial audit after a verified own-output-read failure.

The sealed producer remains unchanged. This separate operational recovery guard
allows reading its existing partial NPZ, never writing it. All original pure
mathematics and input validation are reused. Every reconstructed array must be
byte-identical before final JSON/Markdown can be emitted. No fit or selection.
"""
from . import representation_audit as A
from .representation_audit import (C, config, bound_read, array_sha, load_inputs,
    prepared, collision_audit, input_span_audit, report_text, NOVELTY_FROBENIUS_MAX)
from pathlib import Path
import os
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
READS=None

def install_guard():
    global READS
    assert READS is None and A.READS is None
    READS=set();A.READS=READS;sys.dont_write_bytecode=True
    json_allow={C.DOC/'REPRESENTATION_PROTOCOL.json',C.DOC/'REPRESENTATION_PROTOCOL_SHA.json',
        C.DOC/'FEATURE_AUDIT.json',C.DOC/'AUDIT_PROTOCOL.json',C.DOC/'AUDIT_PROTOCOL_SHA.json',
        C.PARENT_DOC/'TRAIN_PROTOCOL.json',C.SIGN_DOC/'PREFIT_REVIEW.json',
        C.PARENT_DOC/'SOURCE_CONTRACT.json',C.PARENT_DOC/'SOURCE_FEATURE_LOCK.json',
        C.RBF_DOC/'RBF_BASIS.json',C.DOC/'OUTPUT_BINDING_INCIDENT.json',C.DOC/'RECOVERY_PLAN.json'}
    arrays={C.PARENT_RAW/'SOURCE_FEATURES.npz',C.PARENT_RAW/'SOURCE_TRAIN_LABELS.npz',C.RAW/'TRAIN_DIRECTIONS.npz',C.RAW/'REPRESENTATION_ARRAYS.npz'}
    outputs={C.DOC/'REPRESENTATION_AUDIT.json',C.DOC/'REPRESENTATION_AUDIT_KO.md'}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();name=str(p);mode,flags=args[1:3]
        writing=((isinstance(mode,str) and any(c in mode for c in 'wax+')) or
            (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))))
        if writing:
            assert p in outputs,('FINALIZATION_WRITE_DENIED',name);return
        assert not any(token in name for token in ('/data/evaluation/','/real_gt_v2/','/annotations/',
            'SOURCE_VAL','REAL_CHOICES','REAL_FEATURE','POSE_METRICS','GEOMETRY_RESOLVED_POSE_GT',
            'GEOMETRY_SIDETABLE','SOURCE_MANIFEST','SYNTH_LABELS','SYNTH_RECORDS',
            '/model_parameters/','/fits/','REJECTED_')),('FINALIZATION_QUALITY_OR_WEIGHT_DENIED',name)
        assert p.suffix.lower() not in ('.png','.jpg','.jpeg','.pt','.pth','.onnx'),('FINALIZATION_IMAGE_WEIGHT_DENIED',name)
        if p.suffix=='.npz':assert p in arrays,('FINALIZATION_ARRAY_DENIED',name)
        if p.is_relative_to(C.ROOT):
            if p.suffix=='.json':assert p in json_allow,('FINALIZATION_JSON_DENIED',name)
            READS.add(str(p.relative_to(C.ROOT)))
    sys.addaudithook(hook)


def run():
    started=time.monotonic();install_guard()
    plan=bound_read(C.bind(C.DOC/'RECOVERY_PLAN.json'))
    C.verify(plan['recovery_code']);C.verify(plan['original_producer']);C.verify(plan['partial_array'])
    assert plan['recovery_code']==C.bind(__file__)
    incident=bound_read(plan['incident'])
    assert incident['partial_array']==plan['partial_array'] and incident['exit_code']==1
    data=load_inputs()
    if (C.DOC/'REPRESENTATION_AUDIT.json').exists():raise AssertionError('ALREADY_FROZEN_NO_REPEAT')
    models={};arrays=dict(ids=data['ids'],source_index=data['source_index'],anchor_index=data['index'],mean18=data['mean18'],std18=data['std18'])
    with threadpool_limits(limits=1):
        for model in C.MODEL_NAMES:
            p=prepared(model,data);valid=p['valid'];nonanchor=valid.copy()
            rows=np.flatnonzero(valid.any(1));nonanchor[rows,data['index'][rows]]=False
            collisions,oldgroup,newgroup=collision_audit(p['x'],p['extra'],p['target'],valid,data['index'],data['ids'],p['names'])
            span=input_span_audit(p['x'][nonanchor],p['extra'][nonanchor])
            models[model]=dict(names=p['names'],prior_six_hashes=p['hashes'],
                extra18_sha=array_sha(p['extra']),extended271_sha=array_sha(np.concatenate([p['x'],p['extra']],axis=2)),
                collisions=collisions,input_span=span,forced_anchor_zero=True,all_invalid_rows_retained=1)
            for key,value in [('extra18',p['extra']),('axis_target',p['target']),('valid',valid),
                              ('old_group_ids',oldgroup),('extended_group_ids',newgroup)]:arrays[model+'_'+key]=value
            print('REPRESENTATION_INPUT_AUDITED',model,flush=True)
    insufficient=all(r['input_span']['frobenius_residual_ratio']<=NOVELTY_FROBENIUS_MAX for r in models.values())
    artifact=C.RAW/'REPRESENTATION_ARRAYS.npz'
    matched={}
    with np.load(artifact,allow_pickle=False) as prior:
        assert set(prior.files)==set(arrays)
        for key,value in arrays.items():
            saved=prior[key]
            assert saved.dtype==value.dtype and saved.shape==value.shape
            assert saved.tobytes()==value.tobytes(),key
            matched[key]=array_sha(value)
    assert C.bind(artifact)==plan['partial_array']
    print('RECOVERY_PARTIAL_ARRAYS_EXACT',len(matched),flush=True)
    out=dict(complete=True,PASS=True,diagnostic_only=True,created_at=C.now(),code=C.bind(A.__file__),execution_code=C.bind(__file__),
        finalization_recovery=dict(plan=C.bind(C.DOC/'RECOVERY_PLAN.json'),incident=plan['incident'],
            preserved_partial_array=plan['partial_array'],array_keys_exactly_reconstructed=matched,
            original_producer_run_completed=False,separate_guard_used=True,separate_guard_array_write_allowed=False,
            same_sealed_math_recomputed=True,diagnostic_attempt=3),
        protocol=C.bind(C.DOC/'REPRESENTATION_PROTOCOL.json'),inputs=data['bindings'],representation=config(),
        source_TRAIN_only=True,frames=2598,available_anchor_rows=2597,failed_rows_retained=1,
        source_ids_sha=array_sha(data['ids']),source_index_sha=array_sha(data['source_index']),
        anchor_index_sha=array_sha(data['index']),normalization94_sha=array_sha(np.stack([data['mean'],data['std']])),
        normalization18=dict(mean=data['mean18'],std=data['std18'],sha=array_sha(np.stack([data['mean18'],data['std18']]))),
        models=models,arrays=C.bind(artifact),novelty=dict(all_models_ratio_le1e_4=insufficient,
            status='INSUFFICIENT_ADDITIONAL_INPUT_EVIDENCE' if insufficient else 'INPUT_NONREDUNDANCY_ONLY_NOT_PERFORMANCE_EVIDENCE',
            interpretation='모든 모델의 새 입력 잔차 비율이1e−4 이하라 추가 특징의 근거가 부족하다.' if insufficient else '일부 모델에서 입력 비중복이 관측되지만 학습·성능·전이의 증거는 아니며 fit을 승인하지 않는다.'),
        float64_signed_zero_canonicalization=True,approximate_collision_rounding=False,
        source_feature_container_includes_VAL_inputs=True,VAL_quality_read=False,real_targets_read=False,
        raw_source_reference_reads=0,new_reference_metric_calculations=0,prior_weights_read=False,
        optimizer_steps=0,new_fits=0,actual_data_weight_probes=0,actual_data_objective_probes=0,new_policy_argmin=0,
        image_forwards=0,new_PnP_calls=0,method_success=False,goal_complete=False,read_paths=sorted(READS),
        wall_seconds=time.monotonic()-started)
    C.save(C.DOC/'REPRESENTATION_AUDIT_KO.md',report_text(out))
    C.save(C.DOC/'REPRESENTATION_AUDIT.json',out)
    print('REPRESENTATION_AUDIT_PASS',out['novelty']['status'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()
