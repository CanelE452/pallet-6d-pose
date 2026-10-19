"""Metadata-only core schedule compatibility; zero scoring/pose/fit calls."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from . import kernel_extra as K
from . import fullnormal_parallel as PARALLEL, fullnormal_dispatch as DISPATCH


def constructed():
    frame=dict(id='CONSTRUCTED_METADATA_ONLY',K=[[1,0,0],[0,1,0],[0,0,1]],native_xyz=[],raw_hw=[8,8],initial_pose=dict(available=False),image_sha256='0'*64)
    result=dict(id=frame['id'],packet=dict(id=frame['id'],metadata_only=True),families={})
    feature=dict(id=frame['id'],candidates=[]);config=dict(none_prior=.01)
    payload=dict(frame={key:frame[key]for key in ('K','native_xyz','raw_hw','initial_pose','image_sha256')},
        feature_configuration=config,source=K.CORE.source_binding())
    helper=K.U.sha(PARALLEL.__file__)
    old=dict(schema='BI_FULL_NORMAL_PARALLEL_EXECUTION_V1',driver_sha256=helper,numerical_kernel_source_binding=payload['source'])
    ordered=dict(old,schema='FULL_NORMAL_ORDERED_DISPATCH_V1',row_helper_sha256=helper,orchestration_sha256=K.U.sha(DISPATCH.__file__))
    with TemporaryDirectory(prefix='kernel-schedule-metadata-')as temporary:
        folder=Path(temporary);rp=folder/'RESULT.json.gz';fp=folder/'FEATURE.json.gz'
        K.Q.write(rp,result);K.Q.write(fp,feature)
        seal=dict(complete=True,id=frame['id'],source_binding=payload['source'],cache_payload_sha256=K.Q.P.digest(payload),
            artifacts={path.name:K.U.sha(path)for path in (rp,fp)})
        for lock in (old,ordered):
            assert K.parallel_lock(lock)
            bound=K.bound_core_feature(frame,result,feature,config,lock,seal,rp,fp,context_payload=payload)
            assert bound.binding==K._context(frame,result,feature,config)
        for field in ('driver_sha256','row_helper_sha256','orchestration_sha256'):
            broken=dict(ordered);broken[field]='0'*64
            try:K.bound_core_feature(frame,result,feature,config,broken,seal,rp,fp,context_payload=payload)
            except ValueError:pass
            else:raise AssertionError('Changed ordered schedule binding accepted: '+field)
        wrong=deepcopy(payload);wrong['feature_configuration']['none_prior']=.2
        try:K.bound_core_feature(frame,result,feature,config,ordered,seal,rp,fp,context_payload=wrong)
        except ValueError:pass
        else:raise AssertionError('Schedule metadata bypassed exact feature context')
        assert not K.parallel_lock(dict(schema='UNREGISTERED_PARALLEL_V1'))
    return dict(complete=True,schemas=list(K.PARALLEL_SCHEMAS),metadata_only=True,new_individual_cost_evaluations=0,
        new_pose_optimizer_calls=0,new_H1_fits=0,current_reference_reads=0,
        driver_rowhelper_orchestration_corruptions_rejected=True,exact_context_check_preserved=True,
        source_sha256=K.U.sha(K.__file__),row_helper_sha256=helper,orchestration_sha256=K.U.sha(DISPATCH.__file__))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    receipt=constructed();K.U.write(args.output,receipt);print(K.U.canonical(receipt),flush=True)
