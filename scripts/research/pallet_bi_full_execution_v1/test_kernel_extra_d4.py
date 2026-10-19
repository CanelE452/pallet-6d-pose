"""Constructed schedule/context metadata only; no rows,poses,costs or references."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from . import kernel_extra as K
from . import fullnormal_dispatch as D3,fullnormal_dispatch_four as D4,fullnormal_parallel as ROW

def tests():
    source=K.CORE.source_binding();helper=K.U.sha(ROW.__file__)
    lock=dict(schema='FULL_NORMAL_FOUR_CPU_DISPATCH_V1',driver_sha256=helper,row_helper_sha256=helper,
        orchestration_sha256=K.U.sha(D4.__file__),preserved_D3_validation_sha256=K.U.sha(D3.__file__),
        numerical_kernel_source_binding=source,workers=4,dispatch_window=4,
        resource_policy=dict(CPU_workers=4,BLAS_threads_per_worker=1,observation_or_backbone_training=False))
    assert K.parallel_lock(lock)
    assert D4.source_bindings()['preserved_D3_validation_sha256']==lock['preserved_D3_validation_sha256']
    frame=dict(id='CONSTRUCTED_METADATA_ONLY',K=[[1,0,0],[0,1,0],[0,0,1]],native_xyz=[],raw_hw=[2,2],initial_pose=dict(available=False),image_sha256='CONSTRUCTED_NO_IMAGE')
    result=dict(packet=dict(id=frame['id'],constructed_metadata_only=True),families={})
    feature=dict(id=frame['id'],candidates=[],constructed_metadata_only=True);config=dict(constructed_metadata_only=True)
    context=dict(feature_configuration=config,source=source,frame={key:frame[key]for key in ('K','native_xyz','raw_hw','initial_pose','image_sha256')})
    with TemporaryDirectory(prefix='kernel-d4-metadata-')as tmp:
        rp=Path(tmp)/'RESULT.json.gz';fp=Path(tmp)/'FEATURE.json.gz';K.Q.write(rp,result);K.Q.write(fp,feature)
        seal=dict(complete=True,id=frame['id'],source_binding=source,cache_payload_sha256=K.Q.P.digest(context),artifacts={p.name:K.U.sha(p)for p in (rp,fp)})
        bound=K.bound_core_feature(frame,result,feature,config,lock,seal,rp,fp,context_payload=context)
        assert isinstance(bound,K._BoundFeature)
        bad_locks=[]
        for key,value in [('orchestration_sha256',K.U.sha(D3.__file__)),('preserved_D3_validation_sha256','wrongD3'),
                          ('driver_sha256','wrongDriver'),('row_helper_sha256','wrongHelper'),('workers',3),('workers',4.),('dispatch_window',3)]:
            bad_locks.append(dict(lock,**{key:value}))
        for key,value in [('CPU_workers',3),('BLAS_threads_per_worker',2)]:
            bad_locks.append(dict(lock,resource_policy=dict(lock['resource_policy'],**{key:value})))
        for bad in bad_locks:
            try:K.bound_core_feature(frame,result,feature,config,bad,seal,rp,fp,context_payload=context)
            except ValueError:pass
            else:raise AssertionError('Invalid four-worker schedule binding accepted')
        changed=deepcopy(context);changed['frame']['image_sha256']='DIFFERENT_CONSTRUCTED_IMAGE_HASH'
        try:K.bound_core_feature(frame,result,feature,config,lock,seal,rp,fp,context_payload=changed)
        except ValueError:pass
        else:raise AssertionError('Changed exact core context accepted')
        for previous in [dict(schema='FULL_NORMAL_ORDERED_DISPATCH_V1',driver_sha256=helper,row_helper_sha256=helper,
                              orchestration_sha256=K.U.sha(D3.__file__),numerical_kernel_source_binding=source),
                         dict(schema='BI_FULL_NORMAL_PARALLEL_EXECUTION_V1',driver_sha256=helper,numerical_kernel_source_binding=source)]:
            assert K.parallel_lock(previous)
            K.bound_core_feature(frame,result,feature,config,previous,seal,rp,fp,context_payload=context)
    return dict(PASS=True,constructed_schedule_context_metadata_only=True,actual_dataset_rows_read=0,
        numerical_cost_pose_solve_forward_fit_calls=0,reference_decodes=0,images_opened=0,
        D4_lock_and_bound_context_accepted=True,wrong_D4_SHA_D3_SHA_driver_helper_workers_window_BLAS_rejected=True,
        changed_context_rejected=True,old_D3_and_parallel_contexts_accepted=True,
        kernel_source_sha256=K.U.sha(K.__file__),D4_source_sha256=K.U.sha(D4.__file__),D3_source_sha256=K.U.sha(D3.__file__),row_helper_sha256=helper)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=tests();K.U.write(a.output,r);print(K.U.canonical(r))
