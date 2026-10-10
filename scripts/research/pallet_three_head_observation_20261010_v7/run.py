"""Freeze all three heads and two point-only supplies; seal245 before scoring."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import gzip
import json
import os
import time
import traceback
from . import common as C

class RowWriter:
    """Exclusive streaming JSONL output; publish only closed, fsynced gzip data.

    The pending filename is an execution artifact.  The gzip header retains the
    final basename, and each row uses the exact original ``C.save_rows`` encoding.
    No diagnostic field is removed or summarized by this writer.
    """
    def __init__(self, final_path, *, interrupted_path=None):
        self.final_path=Path(final_path)
        self.pending_path=self.final_path.with_name('PENDING_'+self.final_path.name)
        self.interrupted_path=None if interrupted_path is None else Path(interrupted_path)
        for path in (self.final_path,self.pending_path,self.interrupted_path):
            if path is not None:
                C.require(not path.exists() and not path.is_symlink(),'preserve '+path.name)
        self.final_path.parent.mkdir(parents=True,exist_ok=True)
        self.count=0;self.closed=False;self.published=False
        self.close_errors=[];self.preservation_errors=[]
        self.raw=self.pending_path.open('xb')
        try:
            self.zipped=gzip.GzipFile(filename=self.final_path.name,fileobj=self.raw,mode='wb',mtime=0)
        except BaseException:
            self.raw.close()
            raise

    def write(self,row):
        C.require(not self.closed,'cannot write a closed row stream')
        payload=(json.dumps(C.finite(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
        self.zipped.write(payload)
        self.count+=1

    def close(self):
        if self.closed:return
        self.closed=True
        for phase,action in (('gzip_close',self.zipped.close),('raw_flush',self.raw.flush),
                             ('raw_fsync',lambda:os.fsync(self.raw.fileno())),('raw_close',self.raw.close)):
            try:action()
            except BaseException as error:
                self.close_errors.append(dict(phase=phase,type=type(error).__name__,message=str(error)))

    def promote(self):
        self.close()
        C.require(not self.close_errors,'row stream close/fsync failed: '+self.final_path.name)
        C.require(not self.published,'row stream already promoted')
        # link is exclusive: unlike rename it never replaces an existing target.
        os.link(self.pending_path,self.final_path)
        self.published=True
        self.pending_path.unlink()
        self._sync_directory()

    def _sync_directory(self):
        descriptor=os.open(self.final_path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)

    def preserve_interrupted(self):
        self.close()
        if self.interrupted_path is None:return
        source=self.final_path if self.published else self.pending_path
        try:
            os.link(source,self.interrupted_path)
            self._sync_directory()
        except BaseException as error:
            self.preservation_errors.append(dict(type=type(error).__name__,message=str(error)))

class FrameStreams:
    """Keep one frame's full packets in RAM, plus small population counters."""
    NAMES={'geometry':'GEOMETRY_SEALED.jsonl.gz','fixed':'FIXED_GEOMETRY_SEALED.jsonl.gz',
           'observations':'OBSERVATIONS.jsonl.gz'}
    INTERRUPTED={'geometry':'INTERRUPTED_GEOMETRY.jsonl.gz','fixed':'INTERRUPTED_FIXED_GEOMETRY.jsonl.gz',
                 'observations':'INTERRUPTED_OBSERVATIONS.jsonl.gz'}
    def __init__(self,args):
        self.writers={};self.keys={kind:set() for kind in self.NAMES}
        self.method_counts=Counter();self.fixed_counts=Counter();self.head_counts=Counter()
        try:
            for kind,name in self.NAMES.items():
                self.writers[kind]=RowWriter(C.output_path(args,name),
                    interrupted_path=C.output_path(args,self.INTERRUPTED[kind]))
        except BaseException:
            self.preserve_interrupted()
            raise

    def append(self,kind,rows):
        for row in rows:
            field='head_arm' if kind=='observations' else 'method'
            key=(row[field],row['id'])
            C.require(key not in self.keys[kind],'duplicate streamed '+kind+' row')
            self.writers[kind].write(row)
            self.keys[kind].add(key)
            counter=self.head_counts if kind=='observations' else self.fixed_counts if kind=='fixed' else self.method_counts
            counter[row[field]]+=1

    def counts(self):return {kind:writer.count for kind,writer in self.writers.items()}

    def publish(self):
        for writer in self.writers.values():writer.close()
        C.require(not any(writer.close_errors for writer in self.writers.values()),'stream close/fsync failed')
        for writer in self.writers.values():writer.promote()

    def preserve_interrupted(self):
        for writer in self.writers.values():writer.preserve_interrupted()

    def diagnostics(self):
        return dict(mode='per_frame_full_witness_stream',diagnostic_fields_removed=0,row_order_unchanged=True,
            close_and_fsync_before_final_publish=True,
            streams={kind:dict(rows=writer.count,published=writer.published,pending_basename=writer.pending_path.name,
                final_basename=writer.final_path.name,close_errors=writer.close_errors,
                interruption_preservation_errors=writer.preservation_errors)
                for kind,writer in self.writers.items()})

def quiet():
    from ..pallet_corner_mechanism_audit_20261010_v1.deployment_smoke import quiet as probe
    value=probe()
    C.require(value['quiet'] and value['temperature_under_80'],'pending competing workload/thermal guard')
    return value

def freeze(args):
    from ..pallet_boundary_corner_refiner_20261010_v2 import protection
    from .pipeline import POLICY
    from ..pallet_cornerwise_independent_20261010_v4.pose import POLICY as POSE_POLICY
    C.cohort_frames(args)
    checks=C.read(C.DOC/'PIPELINE_CHECKS.json')
    C.require(checks['complete'] and checks['passed'],'synthetic selection checks required')
    for b in checks['code'].values():C.bound(C.REPO/b['path'],b,'tested code')
    cal_checks=C.read(C.DOC/'SOURCE_CALIBRATION_CONTRACT_CHECKS.json')
    C.require(cal_checks['passed'] and cal_checks['check_groups']==9,'source calibration contract checks required')
    for b in cal_checks['code'].values():C.bound(C.REPO/b['path'],b,'tested source calibration code')
    C.require(C.read(C.DOC/'SOURCE_CALIBRATION_CHECKS.json')['passed'],'independent source calibration arithmetic checks required')
    stream_checks=C.read(C.DOC/'STREAMING_CHECKS.json')
    C.require(stream_checks['passed'] and stream_checks['complete'] and stream_checks['check_groups']==4,
        'complete/interrupted serialization checks required')
    for b in stream_checks['code'].values():C.bound(C.REPO/b['path'],b,'tested streaming code')
    extra=[]
    for name in ('pallet_boundary_bootstrap_audit_20261010_v1','pallet_source_neck_contract_20261010_v1'):
        for area in ('scripts/research','_docs/experiments'):
            for path in sorted((C.REPO/area/name).rglob('*')):
                if path.is_file():extra.append(C.binding(path))
    if not Path(args.prior_bindings).exists():
        C.write_new(args.prior_bindings,dict(tracked=protection.snapshot(args.source_root),additional_completed_audits=extra))
    C.protect(args)
    C.validate_calibrations(args,C.ARMS)
    C.write_new(args.protocol,dict(schema='one_shot_fixed_three_head_point_observation_protocol_v7',
        methods=list(C.METHODS),primary=C.PRIMARY,frames=245,method_rows=1960,fixed_rows=490,contrasts=[list(pair) for pair in C.CONTRASTS],
        new_training_updates=0,new_RGB=0,new_seeds=0,GT_tuning=False,
        checkpoint_selection='corrected last step3000 of all three original arms, no best-head selection',supply_policy=POLICY,pose_policy=POSE_POLICY,
        source_calibration='same fixed source CAL128 procedure per head, existing ROLE coefficients reused byte-exact; new256source exposures/16headcalls',
        self_hidden='masked arms exclude frozen initial H from final fit; no-mask control uses H=[]; no initial numeric prior in all8 arms; new valid R,t projects each applied H once; never refit projections',
        inputs=C.inputs(args),CPU_checks=C.binding(C.DOC/'PIPELINE_CHECKS.json'),
        CPU_streaming_checks=dict(code=C.binding(C.CODE/'test_streaming.py'),receipt=C.binding(C.DOC/'STREAMING_CHECKS.json')),
        evaluation='original C1/E7 missing comparison under latest decoder: all3 corrected heads x boundary-only/cornerwise-hybrid plus nativeH/no-mask; one fresh245, no real-score threshold/head/supply/seed/weight selection; existing repeatedly viewed geometric proxy DEV',
        timing='separate quiet1500 complete fresh API calls:10 methods x (20warmup+26x5); each learned path forwards only its requested head',
        performance_success='primary operational245 mean T and R both below fixed seed1 N3_SUBPIX',
        missing_insufficient_ambiguous='record distinct statuses and N3 fallback; no mask-error frame veto',
        hypothesis_reuse='same-coordinate four-subset bank reused across masks; refit calls counted separately',
        limitations=['Boundary-only uses no native N3 numeric fit/LOO gate; hybrid keeps unchanged native consensus and conditional H/Base proposal dependencies; no independent physical truth.',
                    'Severe74 excluded from new runs by direct cohort instruction; historical319 retained.',
                    'Source controlled variants and independent role-feature replay are not completed by this run.']))
    C.verify_protocol(args);C.protect(args)
    print('CORNERWISE_FROZEN',C.sha(args.protocol),flush=True)

def infer(args):
    import cv2
    import numpy as np
    from .pipeline import Pipeline,preserve_prediction
    C.verify_protocol(args);C.protect(args)
    frames=C.cohort_frames(args)
    names=('INFERENCE_STARTED.json','OBSERVATIONS.jsonl.gz','GEOMETRY_SEALED.jsonl.gz',
           'FIXED_GEOMETRY_SEALED.jsonl.gz','GEOMETRY_SEAL.json','BASE_N3_PARITY.json','INFERENCE_RECEIPT.json')
    for name in names:C.output_path(args,name)
    for kind,name in FrameStreams.NAMES.items():
        C.output_path(args,'PENDING_'+name)
        C.output_path(args,FrameStreams.INTERRUPTED[kind])
    probes=[]
    def guard(phase,completed=0):
        from ..pallet_corner_mechanism_audit_20261010_v1.deployment_smoke import quiet as probe
        value=dict(phase=phase,completed_frames=completed,**probe())
        C.write_new(C.output_path(args,'RESOURCE_%03d.json'%len(probes)),value)
        probes.append(value)
        C.require(value['quiet'] and value['temperature_under_80'],'pending competing workload/thermal guard; exact probe retained')
    guard('before_models')
    C.write_new(C.output_path(args,names[0]),dict(protocol=C.binding(args.protocol),
        configured_frames=245,configured_rows=1960,configured_fixed_rows=490,no_automatic_retry=True))
    parity=[];banks=[];calls={};model_calls={};cv_calls={};streams=None
    start=time.monotonic();pipeline=None;complete=False;primitive=None;head_hooks=[];head_actual=Counter();cleanup_error=None
    inference_error=None;current_frame_id=None
    try:
        streams=FrameStreams(args)
        with C.inference_canary():pipeline=Pipeline(args)
        head_hooks=[head.register_forward_pre_hook(lambda *_, arm=arm:head_actual.update({arm:1}))
                    for arm,head in pipeline.heads.items()]
        with C.primitive_counter() as primitive:
            for frame in frames:
                current_frame_id=frame['id']
                path=Path(args.source_root)/frame['image']
                C.require(C.sha(path)==frame['image_sha256'],'original RGB SHA changed')
                image=cv2.imread(str(path),cv2.IMREAD_COLOR)
                C.require(image is not None and list(image.shape[:2])==frame['raw_hw'],'RGB shape/decode differs')
                meta={key:frame[key] for key in ('id','session','object_type')}
                with C.inference_canary(),pipeline.torch.no_grad():
                    captured=pipeline.capture(image,frame['K'],frame['xyz'],meta,need_arms=C.ARMS,need_base_pose=True)
                    preserve_prediction(captured['raw'],captured['prediction'])
                    result,ledger=pipeline.outcomes(captured)
                    fixed=[pipeline.fixed_result(captured,arm) for arm in ('BASE','N3_SUBPIX')]
                # Stored prediction parity is diagnostic after fresh inference;
                # no stored coordinates or pose are handed to selection/PnP.
                idx=captured['raw']['selected_index']
                metadata=None if idx is None else {k:v for k,v in captured['raw']['candidates'][idx].items() if k!='keypoints_xy'}
                C.require(idx==frame['selected_index'] and C.finite(metadata)==C.finite(frame['candidate_metadata']),'detector metadata differs')
                base=pipeline.runtime._point_parity(captured['original_base_points'],frame['points']['BASE'],1e-7,frame['id'])
                n3=pipeline.runtime._point_parity(captured['native_N3_points'],frame['points']['N3_SUBPIX'],1e-7,frame['id'])
                C.require(np.array_equal(captured['native_N3_points'][8],captured['original_base_points'][8],equal_nan=True),'center changed')
                frame_observations=[]
                for arm in C.ARMS:
                    frame_observations.append(dict(id=frame['id'],session=frame['session'],head_arm=arm,GT_input=False,
                        original_base_points=captured['original_base_points'],native_N3_points=captured['native_N3_points'],
                        initial_N3_pose=captured['initial_pose'],predicted_N3_hidden=captured['hidden'],
                        **captured['observations'][arm]))
                streams.append('geometry',result)
                streams.append('fixed',fixed)
                streams.append('observations',frame_observations)
                parity.append(dict(id=frame['id'],BASE=base,N3_SUBPIX=n3,metadata_preserved=True,
                    stored_parity_not_prediction_input=True,center_preserved=True))
                banks.append(dict(id=frame['id'],**ledger))
                # Writers own serialized bytes only, not the full frame packets.
                del result,fixed,frame_observations,captured
                if len(parity)%32==0:
                    guard('after_frame',len(parity))
                    print('THREE_HEAD_GEOMETRY',len(parity),245,flush=True)
            cv_calls=dict(primitive)
        calls=dict(pipeline.counts)
        model_calls=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards,heads=dict(head_actual))
        C.require(streams.counts()==dict(geometry=1960,fixed=490,observations=735),'incomplete population')
        C.require(dict(streams.method_counts)=={method:245 for method in C.METHODS} and
            dict(streams.fixed_counts)=={method:245 for method in ('BASE','N3_SUBPIX')} and
            dict(streams.head_counts)=={arm:245 for arm in C.ARMS},'incomplete streamed method/head population')
        C.require(calls['detector_calls']==calls['N3_route_calls']==calls['initial_pose_calls']==245 and
            calls['final_pose_paths']==1960 and calls['base_control_pose_calls']==245,'actual path counts differ')
        C.require(model_calls['N3']==245 and model_calls['detector']==246 and
            model_calls['heads']=={arm:245 for arm in C.ARMS} and
            calls['feature_initial_pose_calls']==245 and calls['head_calls']==735,
            'actual frozen model/feature path counts differ')
        guard('after_geometry',len(parity))
        C.verify_protocol(args);preserved=C.protect(args)
        streams.publish()
        C.write_new(C.output_path(args,'BASE_N3_PARITY.json'),dict(passed=True,rows=parity,rtol=0,atol=1e-7))
        seal=dict(schema='fixed_three_head_point_observation_geometry_seal_v7',complete=True,GT_read_allowed=False,
            frames=245,rows=1960,fixed_rows=490,methods=list(C.METHODS),protocol=C.binding(args.protocol),
            observations=C.binding(Path(args.output)/'OBSERVATIONS.jsonl.gz'),
            geometry=C.binding(Path(args.output)/'GEOMETRY_SEALED.jsonl.gz'),
            fixed_geometry=C.binding(Path(args.output)/'FIXED_GEOMETRY_SEALED.jsonl.gz'),
            parity=C.binding(Path(args.output)/'BASE_N3_PARITY.json'),actual_calls=calls,
            actual_model_forwards=model_calls,actual_OpenCV_entry_calls=cv_calls,banks=banks,
            model_bindings=pipeline.bindings,resource_snapshots=probes,protection=preserved,
            wall_seconds=time.monotonic()-start,runtime_benchmark=False)
        C.write_new(C.output_path(args,'GEOMETRY_SEAL.json'),seal);complete=True
    except BaseException as error:
        inference_error=dict(type=type(error).__name__,message=str(error),frame_id=current_frame_id,
            traceback=traceback.format_exc())
        raise
    finally:
        if primitive is not None:cv_calls=dict(primitive)
        if pipeline is not None:
            calls=dict(pipeline.counts)
            model_calls=dict(detector=pipeline.models.detector_forwards,N3=pipeline.models.n3_forwards,heads=dict(head_actual))
            hook_errors=[]
            for hook in head_hooks:
                try:hook.remove()
                except BaseException as error:
                    hook_errors.append(dict(type=type(error).__name__,message=str(error)))
            if hook_errors:
                cleanup_error=dict(type='HookCleanupErrors',errors=hook_errors);complete=False
            try:pipeline.close()
            except BaseException as error:
                cleanup_error=dict(type=type(error).__name__,message=str(error),hook_errors=hook_errors);complete=False
        if not complete:
            if streams is not None:streams.preserve_interrupted()
        row_counts=dict(geometry=0,fixed=0,observations=0) if streams is None else streams.counts()
        C.write_new(C.output_path(args,'INFERENCE_RECEIPT.json'),dict(complete=complete,
            actual_complete_frames=len(parity),observation_rows=row_counts['observations'],
            method_rows=row_counts['geometry'],fixed_rows=row_counts['fixed'],
            actual_calls=calls,actual_model_forwards=model_calls,actual_OpenCV_entry_calls=cv_calls,
            cleanup_error=cleanup_error,inference_error=inference_error,
            serialization=None if streams is None else streams.diagnostics(),
            new_training_updates=0,new_RGB=0,GT_access_during_inference=False,no_automatic_retry=True,
            wall_seconds=time.monotonic()-start))
    C.require(complete,'inference did not complete; preserved prefix receipt')
    print('CORNERWISE_SEALED',row_counts['geometry'],row_counts['fixed'],'before_GT',flush=True)

def main():
    p=C.parser(__doc__,('freeze','preflight','infer'))
    args=p.parse_args()
    if args.stage=='freeze':freeze(args)
    elif args.stage=='preflight':C.verify_protocol(args);C.protect(args);quiet();print('CORNERWISE_PREFLIGHT_PASS',flush=True)
    else:infer(args)

if __name__=='__main__':main()
