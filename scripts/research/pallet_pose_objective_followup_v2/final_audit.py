"""Independent CPU snapshot audit. Only V2 AUDIT.json is written.

Existing artifact/code locks are verified, never repaired. Fits, GPU, Git,
historical namespaces, shared state, and resource ledgers are read-only here.
An incomplete experiment/report is PENDING, not an invented PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import io
from pathlib import Path
import re
import time
import unittest

import numpy as np
from . import common as C
from . import metric_baseline as M
from . import eval_student as E
from scripts.research.pallet_oracle_mechanism_followup_v1 import final_audit as V1

REQUIRED_DOCS=('GOAL_LOCK.md','METRIC_AND_SELECTION_LOCK.json','INPUT_BINDINGS.json',
    'POOL_AND_SUPERVISION_AUDIT.md','LOSS_SIGNAL_AUDIT.md','CANDIDATE_MATRIX.md',
    'EXPERIMENT_LOG.md','FINAL_TABLES.md','REPORT_KO.md','CLAIM_IMPACT.md','NEXT_DECISION.md',
    'REPRODUCE.md','STATE.json','RESOURCE_LEDGER.json')
ARRAY_KEYS=V1.FORBIDDEN_ARRAY_KEYS|{'image','img','rgb','pixels','weights','state_dict','bbox','keypoint_annotations'}


def require(condition,reason):
    if not condition:raise AssertionError(reason)


def close(a,b):
    E.O.D.close(a,b)


def verify_bindings(value):
    rows=list(V1.bindings(value))
    for binding in rows:
        C.verify(binding)
        if 'bytes' in binding:require((C.ROOT/binding['path']).stat().st_size==binding['bytes'],'Bound file size differs')
    return len(rows)


def privacy_paths(value,prefix='$'):
    out=[]
    if isinstance(value,dict):
        for key,child in value.items():
            path=prefix+'.'+key
            # Figure-manifest width/height metadata is not a raw pixel image.
            dimension_metadata=(key=='pixels' and isinstance(child,list) and len(child)==2
                                and all(isinstance(x,int) and x>0 for x in child))
            if key in ARRAY_KEYS and V1.numeric_array(child) and not dimension_metadata:out.append(path)
            if key in ('state_dict','model_weights') and isinstance(child,dict):out.append(path)
            out.extend(privacy_paths(child,path))
    elif isinstance(value,list):
        for index,child in enumerate(value):out.extend(privacy_paths(child,f'{prefix}[{index}]'))
    elif isinstance(value,str) and (value.startswith('data:image/') or value.startswith('data:application/octet-stream')):
        out.append(prefix)
    return out


def _field_count(value,key):
    if isinstance(value,dict):return int(key in value)+sum(_field_count(v,key) for v in value.values())
    if isinstance(value,list):return sum(_field_count(v,key) for v in value)
    return 0


def old_assets():
    lock=C.read(C.DOC/'INPUT_BINDINGS.json')
    require(len(lock['files'])==91,'Original V2 inventory is not the frozen 91 files')
    count=verify_bindings(lock['files'])
    papers=[b for b in lock['files'] if b['path'].startswith('_docs/paper/selftraining_submission_v1/')]
    require(papers,'Original paper preservation inventory missing')
    metric=C.read(C.DOC/'METRIC_AND_SELECTION_LOCK.json')
    count+=verify_bindings(metric)
    return dict(original_91_preserved=True,original_paper_files_preserved=len(papers),bindings_verified=count,
                scope='Hash preservation of bound assets. Complete tracked/untracked Git scope requires parent Git audit; this tool never invokes Git.')


def public_privacy():
    files=[p for p in sorted(C.DOC.rglob('*.json')) if p.name!='AUDIT.json']
    violations=[]
    for path in files:
        fields=privacy_paths(C.read(path))
        if fields:violations.append(dict(file=str(path.relative_to(C.ROOT)),fields=fields))
    require(not violations,'Public JSON contains named private coordinate/camera/image/weight arrays')
    binaries=[p for p in C.DOC.rglob('*') if p.is_file() and p.suffix.lower() in ('.pt','.pth','.ckpt','.npy','.npz','.pkl')]
    require(not binaries,'Checkpoint or raw-array binary exists in public docs')
    aggregate=[]
    for name in ('BASELINE_POSE_RESULTS.json','TR_ORACLE_RESULTS.json'):
        path=C.DOC/name;value=C.read(path)
        require(_field_count(value,'frame_id')==_field_count(value,'id')==0,'Public aggregate contains per-frame record IDs')
        aggregate.append(dict(file=name,bytes=path.stat().st_size,lines=len(path.read_text().splitlines()),
                              per_frame_record_ID_fields=0,private_array_fields=0))
    return dict(JSON_files_scanned=len(files),violations=violations,large_aggregate_artifacts=aggregate,
                scope='Named-field numeric-array/embedded-image scan and forbidden binary suffixes; arbitrary renamed or encoded payloads are not a semantic information-flow proof',
                hash_path_metadata_allowed=True,large_files_reason='Repeated arm/group/paired/common-valid aggregate quantile summaries, not raw coordinates or weights')


def _independent_pose_fields(metric,pose,truth):
    row=M.extend_metric(metric,pose,truth)
    if not row['available']:return row
    R=np.asarray(pose['R_physical']);G=np.asarray(truth['R']);yaw=[]
    for Q in E.O.D.Pose.rotations(truth['order']):
        relative=(G@Q).T@R
        yaw.append(abs(float((np.degrees(np.arctan2(relative[0,2],relative[2,2]))+180)%360-180)))
    require(np.isclose(min(yaw),row['yaw_deg'],atol=1e-7,rtol=1e-7),'Existing C2 yaw convention differs')
    return row


def baseline_and_oracles():
    baseline=C.read(C.DOC/'BASELINE_POSE_RESULTS.json');oracle=C.read(C.DOC/'TR_ORACLE_RESULTS.json')
    verified=verify_bindings(baseline)+verify_bindings(oracle)
    members=C.read(M.RAW/'POPULATION_LOCK_PRIVATE.json');private=C.read(M.RAW/'FRAME_METRICS_PRIVATE.json')
    _,truth=E.O.D.Pose.metadata('REAL_DEV');checked=0;groups_checked=0
    for material in M.MAIN_ALIASES:
        oldroot=C.P.RAW if material=='PLASTIC' else C.M.RAW
        pp=C.read(oldroot/'POSE_PREDICTIONS.json');pose_map={a:pp[b] for a,b in M.MAIN_ALIASES[material].items()}
        for cycle,aliases in [(M.OLD_CYCLES[0],{'C2_RAW':'NEW_RAW','C2_REF':'NEW_REF'}),(M.OLD_CYCLES[1],{'C3_RAW9':'RAW9','C3_MANUAL9':'MANUAL9'})]:
            pp=C.read(M.OLD_RAW/'cycles'/cycle/f'{material}_POSES.json')
            pose_map.update({a:pp[b] for a,b in aliases.items()})
        for arm,rows in private[material].items():
            for fid,metric in rows.items():
                close(_independent_pose_fields(metric,pose_map[arm][fid],truth[fid]),metric);checked+=1
        for group,ids in members[material]['groups'].items():
            for arm,rows in private[material].items():
                actual=M.summarize([rows[fid] for fid in ids]);saved=baseline['materials'][material]['groups'][group][arm]
                close(actual,{k:saved[k] for k in actual});groups_checked+=1
            for name,saved in baseline['materials'][material]['contrasts'][group].items():
                after,before=name.split('-minus-')
                close(M.paired(private[material][before],private[material][after],ids),saved)
        cache=C.read(M.OLD_RAW/'pose_oracle'/f'{material}_METRICS.json')['arms']
        for group,ids in members[material]['groups'].items():
            for arm in E.O.ARMS[material]:
                current={fid:cache[arm][fid]['current'] for fid in ids}
                fixed={fid:cache[arm][fid]['hypotheses'] for fid in ids}
                actual,_=M.oracle_for(ids,fixed,current)
                close(actual,oracle['materials'][material][group]['fixed_D9_candidates'][arm])
                pool={fid:[dict(name=a,metric=cache[a][fid]['current']) for a in E.O.ARMS[material]] for fid in ids}
                actual,_=M.oracle_for(ids,pool,current)
                close(actual,oracle['materials'][material][group]['whole_output_expert_pool'][arm])
    require(checked==1384,'Baseline frame/arm denominator changed')
    require(len(members['PLASTIC']['groups'][M.PRIMARY])==99,'Primary population no longer99')
    require(oracle['GT_DEPENDENT'] and oracle['DIAGNOSTIC_ONLY'] and not oracle['production_input'],'Oracle isolation flags changed')
    return dict(independently_checked_centroid_T_fullR_yaw_camera_components=checked,
                baseline_groups_reaggregated=groups_checked,all_baseline_paired_groups_recomputed=True,
                separate_T_and_R_oracle_objectives_recomputed=True,bindings_verified=verified,
                scope='Centroid T/fullR/yaw recomputed from stored poses; ADD/IoU caches reaggregated, no new solver or IoU execution')


def checkpoint_protected(fit,initials):
    import torch
    from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import pose_parameter
    binding=fit['initialization'];verify_bindings(binding);key=binding['sha256']
    if key not in initials:
        model=torch.load(C.ROOT/binding['path'],map_location='cpu',weights_only=False)['model'].float()
        state=model.state_dict();buffers={name for name,_ in model.named_buffers()}
        protected={name for name in state if name in buffers or not pose_parameter(name)}
        require(len(protected)==747,'Current architecture has unexpected protected-state inventory')
        initials[key]=(state,protected)
    base,protected=initials[key]
    final=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
    require(set(base)==set(final),'Checkpoint state inventory changed')
    require(all(torch.equal(base[name],final[name]) for name in protected),'Protected state changed')
    changed={name for name in base if not torch.equal(base[name],final[name])}
    require(changed and all(pose_parameter(name) for name in changed),'Unexpected nonpose update or no update')
    require(changed==set(fit['changed_tensors']),'Declared changed tensors differ from checkpoint')
    return len(protected),len(changed)


def check_pair_traces(raw,ref,declared):
    require(len(raw)==len(ref)==declared['batches'],'Pair trace lengths differ')
    differences=Counter();totals={arm:{role:Counter() for role in ('SOURCE','REAL')} for arm in ('RAW','REF')}
    occ=Counter()
    for a,b in zip(raw,ref):
        for key in ('batch','epoch','names','images','boxes','batch_idx'):
            require(a[key]==b[key],'RAW/REF input, box or sample order differs')
        for key in ('support','coordinates'):differences[key]+=a[key]!=b[key]
        require(a['roles']['SOURCE']==b['roles']['SOURCE'],'Source effective supervision differs between pair')
        for label,row in [('RAW',a),('REF',b)]:
            for role in totals[label]:totals[label][role].update(row['roles'][role])
        require(len(a['occlusion'])==len(b['occlusion']),'Occlusion trace lengths differ')
        for ai,bi in zip(a['occlusion'],b['occlusion']):
            for key in ('role','seed','rectangle','fill_seed','applied'):
                require(ai.get(key)==bi.get(key),'Paired occlusion placement/fill differs')
            occ[ai['role']+'_images']+=1
            if ai['role']=='REAL':
                occ['applied']+=ai['applied'];occ['REF_masked_supervised']+=bi.get('actual_covered',0)
                occ['RAW_masked_supervised']+=ai.get('actual_covered',0)
    require(dict(differences)==declared['differences'],'Support/coordinate differences were not accurately declared')
    flat={role+'_'+key:value for role,values in totals['REF'].items() for key,value in values.items()}
    require(flat==declared['exposures'],'Declared REF exposure differs from actual trace')
    require(dict(occ)==declared['occlusion'],'Declared occlusion exposure differs from actual trace')
    if differences['support']:
        require(bool(declared.get('support_difference_interpretation')),'Support differences are undisclosed')
    return dict(batches=len(raw),differences=dict(differences),
                exposures={arm:{role:dict(value) for role,value in rows.items()} for arm,rows in totals.items()},
                paired_RGB_names_boxes_exact=True,source_effective_support_equal=True,
                raw_ref_supervision_masks_globally_exact=differences['support']==0,
                mask_difference_contract='Existing coordinate-dependent clipping retained and explicitly counted; not hidden as global mask equality')


def new_fits_and_pairs():
    initials={};fits=[];pairs={};pending=[];parent_protocols=set();binding_count=0
    metric_time=datetime.fromisoformat(C.read(C.DOC/'METRIC_AND_SELECTION_LOCK.json')['created_at'])
    for path in sorted((C.RAW/'cycles').glob('*/FIT_*.json')):
        fit=C.read(path)
        if not fit.get('complete'):pending.append(str(path.relative_to(C.ROOT)));continue
        required={'cycle','material','target','seed','optimizer_steps','initialization','changed_tensors'}
        if not required<=set(fit):pending.append('UNSUPPORTED_FIT_SCHEMA:'+str(path.relative_to(C.ROOT)));continue
        binding_count+=verify_bindings(fit)
        require(fit['exact_R0_initialization'] and fit['protected_state_exact'],'Fit initialization/protection not asserted')
        protocol=C.read(C.ROOT/fit['protocol']['path'])
        parent=C.read(C.ROOT/fit['parent_protocol']['path'])
        require(fit['initialization']==parent['initialization']==C.checkpoint(fit['material'],'R0'),
                'Initialization is not the original material R0 checkpoint')
        require(protocol['locked_before_fit'] and fit['optimizer_steps']==protocol['updates_per_fit']<=640,'Fit update/protocol budget differs')
        require(fit['manual_added']==0,'New manual supervision requires separate audit')
        state_path=path.parent/f'START_{fit["arm"]}.json';start=C.read(state_path);verify_bindings(start)
        require(datetime.fromisoformat(start['utc'])>=metric_time,'Fit began before metric lock')
        protected,changed=checkpoint_protected(fit,initials)
        trace=C.read(C.ROOT/fit['trace']['path']);require(len(trace)==fit['optimizer_steps'],'Trace/update count differs')
        key=(fit['cycle'],fit['material'],fit['seed']);pairs.setdefault(key,{})[fit['target']]=trace
        fits.append(dict(arm=fit['arm'],cycle=fit['cycle'],steps=fit['optimizer_steps'],protected_tensors=protected,changed_tensors=changed))
        if fit['parent_protocol']['path'] not in parent_protocols:
            binding_count+=verify_bindings(parent['inputs']+parent['sources']+[parent['initialization']])
            parent_protocols.add(fit['parent_protocol']['path'])
    pair_results={}
    for (cycle,material,seed),pair in pairs.items():
        name=f'{cycle}/{material}_S{seed}';path=C.DOC/'cycles'/cycle/f'PARITY_{material}_S{seed}.json'
        if set(pair)!={'RAW','REF'} or not path.exists():pending.append('UNFINISHED_PAIR:'+name);continue
        pair_results[name]=check_pair_traces(pair['RAW'],pair['REF'],C.read(path))
    return dict(status='PENDING' if pending else 'PASS',completed_fits=fits,paired_actual_traces=pair_results,
                verified_parent_protocols=len(parent_protocols),bindings_verified=binding_count,pending=pending,
                initial_state_assertion_scope='Exact R0 start assertion reused from bound completed training trace; final protected747 independently bit-compared')


def student_scores():
    from scripts.research.pallet_material_selftrain_closure_v1.score_eval import score_frame
    paths=sorted((C.DOC/'cycles').glob('*/RESULTS_*_S*.json'));done=[];checked=0;pending=[]
    _,truth_pose=E.O.D.Pose.metadata('REAL_DEV');truth=C.read(C.P.TRUTH)
    for path in paths:
        result=C.read(path);cycle,material,seed=[result[k] for k in ('cycle','material','seed')]
        p=E.paths(cycle,material,seed);lock=E.inference_binding_checks(p)
        verify_bindings(result['scoring_sources']+result['private_artifacts'])
        start=C.read(p['raw']/f'SCORING_START_{p["tag"]}.json');verify_bindings(start)
        require(datetime.fromisoformat(lock['created_at'])<=datetime.fromisoformat(start['utc']),'Scoring preceded paired prediction lock')
        rows=C.read(p['metadata']);poses=C.read(p['poses']);predictions=C.read(p['predictions'])
        frames,fixed,metrics=[C.read(p['raw']/f'{name}_{p["tag"]}.json') for name in ('FRAME_METRICS','FIXED_ID_METRICS','POSE_METRICS')]
        for arm,values in metrics.items():
            for fid,metric in values.items():
                close(_independent_pose_fields(metric,poses[arm][fid],truth_pose[fid]),metric);checked+=1
                if arm in ('NEW_RAW','NEW_REF'):
                    close(score_frame(fid,predictions[arm][fid],truth[fid]),frames[arm][fid])
                    close(score_frame(fid,predictions[arm][fid],truth[fid],fixed=True),fixed[arm][fid])
        groups,summaries=E.group_summaries(rows,material,frames,fixed,metrics);close(summaries,result['groups'])
        n,corners,matched=E.EXPECTED[material]
        for value in summaries['ALL'].values():
            require(value['frames']==n and value['twoD']['corners']==value['fixed_ID']['corners']==corners and value['twoD']['matched']==matched,'Scored population/denominators changed')
        for group,ids in groups.items():
            for name,saved in result['contrasts'][group].items():
                after,before=name.split('-minus-')
                actual=M.paired(metrics[before],metrics[after],ids)
                close(actual,{key:saved[key] for key in actual})
        population=M.PRIMARY if material=='PLASTIC' else 'ALL'
        for reference in ('OLD_REF','R0','NEW_RAW'):
            close(M.classify_candidate(summaries[population]['NEW_REF'],summaries[population][reference]),result['classification']['versus'][reference])
        if material=='PLASTIC':
            summary,point_rows=E.verified66(predictions,rows,truth)
            close(summary,result['verified66'])
            close(point_rows,C.read(p['raw']/f'VERIFIED66_POINT_METRICS_{p["tag"]}.json'))
        done.append(dict(cycle=cycle,material=material,seed=seed,frames=n,primary_frames=len(groups[population]),pose_frame_arm_checks=len(metrics)*n,
                         raw_ref_2D_recomputed=True,all_group_T_R_reaggregated=True,all_paired_T_R_recomputed=True,classification_recomputed=True))
    # Completed pair without a result is valid progress, but not a completed audit.
    for parity in (C.DOC/'cycles').glob('*/PARITY_*_S*.json'):
        if not parity.with_name(parity.name.replace('PARITY_','RESULTS_',1)).exists():pending.append(str(parity.relative_to(C.ROOT)))
    return dict(status='PENDING' if pending else 'PASS',completed_results=done,independent_pose_frame_arm_checks=checked,pending_scoring=pending,
                scope='New T/R/yaw and raw2D metrics rechecked; cached ADD/IoU aggregated with original contracts, no selector re-inference')


def ledger_snapshot():
    ledger=C.read(C.DOC/'RESOURCE_LEDGER.json');caps=ledger['caps'];events=ledger['events']
    authorized={'fits':12,'optimizer_updates':7680,'GPU_seconds':21600,'wall_seconds':36000,'main_hypothesis_cycles':3}
    require(all(caps[k]<=v for k,v in authorized.items()),'Declared cap exceeds current request')
    names=[e['event'] for e in events];require(len(names)==len(set(names)),'Resource event duplicate')
    sums={k:sum(e[k] for e in events) for k in ('fits','optimizer_updates','gpu_seconds')}
    for key,total in sums.items():
        require(all(e[key]>=0 for e in events),'Negative resource consumption')
        require(np.isclose(total,ledger['totals'][key],atol=1e-6,rtol=1e-12),'Event sum/total mismatch')
        cap=caps['GPU_seconds' if key=='gpu_seconds' else key];require(total<=cap,'Resource cap exceeded')
    require(ledger['totals']['elapsed_wall_seconds']<=caps['wall_seconds'],'Recorded active-wall cap exceeded')
    completed=[];missing=[]
    for path in (C.RAW/'cycles').glob('*/FIT_*.json'):
        fit=C.read(path)
        if not fit.get('complete'):continue
        require(fit['optimizer_steps']<=640,'Per-fit640 update cap exceeded')
        matching=[event for event in events if event['fits']>0 and
                  any(b['sha256']==fit['checkpoint']['sha256'] for b in V1.bindings(event.get('details')))]
        if not matching:missing.append('FIT_EVENT:'+str(path.relative_to(C.ROOT)))
        else:
            require(len(matching)==1 and matching[0]['fits']==1 and matching[0]['optimizer_updates']==fit['optimizer_steps'],'Completed fit resource matching differs')
        completed.append(fit)
    require(len(completed)<=caps['fits'] and sum(f['optimizer_steps'] for f in completed)<=caps['optimizer_updates'],'Observed artifacts exceed resource cap')
    failures=[]
    for path in (C.RAW/'cycles').glob('*/FAILURE_*.json'):
        failure=C.read(path);steps=failure.get('executed_optimizer_steps')
        if steps is None:missing.append('UNSUPPORTED_FAILURE_SCHEMA:'+str(path.relative_to(C.ROOT)));continue
        failures.append(dict(path=str(path.relative_to(C.ROOT)),steps=steps,fit_consumption=int(steps>0)))
    expected_fits=len(completed)+sum(f['fit_consumption'] for f in failures)
    expected_steps=sum(f['optimizer_steps'] for f in completed)+sum(f['steps'] for f in failures)
    if sums['fits']!=expected_fits or sums['optimizer_updates']!=expected_steps:missing.append('FIT_OR_FAILURE_RESOURCE_RECONCILIATION_PENDING')
    main=[];controls=[]
    for path in (C.DOC/'cycles').glob('*/PROTOCOL.json'):
        protocol=C.read(path)
        if path.parent.name=='BASELINE_REPEAT' or protocol.get('is_replication') or protocol.get('is_material_applicability'):
            controls.append(path.parent.name)
        else:main.append(path.parent.name)
    require(len(main)<=caps['main_hypothesis_cycles'],'Main hypothesis cycle cap exceeded')
    return dict(status='PENDING' if missing else 'PASS',event_sums=sums,completed_fits=len(completed),completed_updates=sum(f['optimizer_steps'] for f in completed),
                failed_attempts=failures,pending=missing,main_hypothesis_protocols=main,declared_replication_controls=controls,
                recorded_wall_seconds=ledger['totals']['elapsed_wall_seconds'],ledger_binding=C.bind(C.DOC/'RESOURCE_LEDGER.json'),
                inference_CPU_diagnostic_event_completeness='Not inferred from completed fits; parent must record all timed intervals including technical attempts',
                wall_scope='Recorded last resource event, not elapsed calendar time at later audit rerun')


def json_pointer(value,pointer):
    require(pointer=='' or pointer.startswith('/'),'Invalid JSON pointer')
    if not pointer:return value
    for piece in pointer[1:].split('/'):
        piece=piece.replace('~1','/').replace('~0','~')
        value=value[int(piece)] if isinstance(value,list) else value[piece]
    return value


def check_numeric_trace(entry,sources=None,documents=None):
    sources={} if sources is None else sources;documents={} if documents is None else documents
    binding=entry['source'];key=(binding['path'],binding['sha256'])
    if key not in sources:
        C.verify(binding);sources[key]=C.read(C.ROOT/binding['path'])
    source=sources[key]
    value=json_pointer(source,entry['json_pointer'])
    rendered=format(value,entry['format']);require(rendered==entry['rendered'],'Rendered number differs from bound source')
    path=(C.DOC/entry['document']).resolve();require(path.is_relative_to(C.DOC),'Report trace escaped public namespace')
    if path not in documents:documents[path]=path.read_text().splitlines()
    lines=[line for line in documents[path] if entry['line_contains'] in line]
    require(len(lines)==1 and rendered in lines[0],'Numeric trace missing/ambiguous in report context')


def report_tables():
    baseline=C.read(C.DOC/'BASELINE_POSE_RESULTS.json');oracle=C.read(C.DOC/'TR_ORACLE_RESULTS.json')
    text=(C.DOC/'METRIC_CONTRACT.md').read_text();checked=0
    for line in text.splitlines():
        cells=[c.strip() for c in line.split('|')[1:-1]]
        if len(cells)!=6:continue
        if cells[0] in baseline['materials']['PLASTIC']['groups'][M.PRIMARY]:
            row=baseline['materials']['PLASTIC']['groups'][M.PRIMARY][cells[0]]
            require(cells[1]==f"{row['valid_pose']}/{row['frames']}",'Baseline report coverage differs')
            for cell,key in zip(cells[2:5],M.KEYS):
                expected=' / '.join(f"{row['conditional'][key][q]:.6f}" for q in ('median','P90'))
                require(cell==expected,'Baseline report T/R/yaw differs')
            require(int(cells[5])==row['axis_mismatch_count'],'Baseline report axis count differs');checked+=1
        elif cells[0].startswith(('PLASTIC/','WOOD/')):
            material,group=cells[0].split('/',1);arm='REF_LR5' if material=='PLASTIC' else 'WOOD_REF_LR5'
            row=oracle['materials'][material][group]['fixed_D9_candidates'][arm]
            for cell,name in zip(cells[1:4],('current','translation_optimal','rotation_optimal')):
                require(cell==' / '.join(f"{row[name]['conditional'][key]['median']:.6f}" for key in M.KEYS[:2]),'Oracle report conflates T/R optimum')
            require(int(cells[4])==row['frames_with_same_candidate_joint_gain'] and int(cells[5])==row['T_R_optimal_choices_different'],'Oracle report counts differ');checked+=1
    require(checked==11,'Expected8 baseline and3 oracle table rows')
    trace=C.DOC/'REPORT_NUMERIC_TRACE.json';extra=0
    if trace.exists():
        source_cache={};document_cache={}
        for entry in C.read(trace)['entries']:check_numeric_trace(entry,source_cache,document_cache);extra+=1
    missing=[name for name in REQUIRED_DOCS if not (C.DOC/name).exists()]
    figure_links=0
    for path in C.DOC.rglob('*.md'):
        for target in V1.markdown_image_targets(path.read_text()):
            require((path.parent/target).resolve().exists(),'Broken local report figure link');figure_links+=1
    return dict(status='PENDING' if missing else 'PASS' if extra else 'LIMITED',
                metric_contract_table_rows_source_verified=checked,optional_numeric_trace_fields=extra,
                missing_required_docs=missing,local_figure_links_checked=figure_links,
                numeric_coverage='Only the stated11 contract rows and supplied trace entries checked; no claim of arbitrary prose-number verification')


def public_figures():
    images=[p for p in C.DOC.rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.svg')]
    if not images:return dict(status='PENDING',reason='No public figures yet; no raw RGB published by this audit')
    manifest=C.DOC/'FIGURE_MANIFEST.json'
    if not manifest.exists():return dict(status='PENDING',reason='Public images require source/approval manifest',image_files=len(images))
    value=C.read(manifest);count=verify_bindings(value)
    files=value.get('files',[row['file'] for row in value.get('figures',[])])
    declared={(C.ROOT/b['path']).resolve() for b in files}
    require({p.resolve() for p in images}==declared,'Unregistered public image file')
    for example in value.get('examples',[]):
        if example.get('status')=='NA':continue
        require('previous_publication' in example,'Raw RGB example lacks existing publication approval source')
        prior=C.read(C.ROOT/example['previous_publication']['path'])
        allowed={e.get('id',e.get('frame_id')) for e in prior['examples']}
        require(example.get('id',example.get('frame_id')) in allowed,'New raw RGB frame has no prior publication ID')
    return dict(image_files=len(images),bindings_verified=count,approved_existing_examples=sum(e.get('status')!='NA' for e in value.get('examples',[])),
                note='Aggregate plots are not raw RGB; raw examples require original approved publication ID')


def cpu_tests():
    modules=['test_metric_baseline','test_eval_student','test_final_audit']
    suite=unittest.defaultTestLoader.loadTestsFromNames(['scripts.research.pallet_pose_objective_followup_v2.'+name for name in modules])
    from . import test_occlusion
    for name in sorted(n for n in dir(test_occlusion) if n.startswith('test_')):
        suite.addTest(unittest.FunctionTestCase(getattr(test_occlusion,name)))
    modules.append('test_occlusion')
    result=unittest.TextTestRunner(stream=io.StringIO(),verbosity=0).run(suite)
    return dict(status='FAIL' if not result.wasSuccessful() else 'LIMITED' if result.skipped else 'PASS',modules=modules,
                run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=[dict(test=str(test),reason=reason) for test,reason in result.skipped],
                unsuccessful=[str(test) for test,_ in result.failures+result.errors],
                scope='Listed CPU-only tests; future loss/gradient modules require their own explicit audit, not silently treated as tested')


def overall(checks):
    statuses={check['status'] for check in checks.values()}
    return 'FAIL' if 'FAIL' in statuses else 'PARTIAL_PENDING' if 'PENDING' in statuses else 'PASS_WITH_LIMITATIONS' if statuses-{'PASS'} else 'PASS'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-tests',action='store_true');args=parser.parse_args(argv)
    start=time.perf_counter();cpu=time.process_time();checks={}
    operations=[old_assets,public_privacy,baseline_and_oracles,new_fits_and_pairs,student_scores,ledger_snapshot,report_tables,public_figures]
    if not args.skip_tests:operations.append(cpu_tests)
    for operation in operations:
        tick=time.perf_counter()
        try:
            value=operation();value.setdefault('status','PASS')
        except FileNotFoundError:
            value=dict(status='PENDING',reason='Required artifact not yet present; no pass inferred')
        except Exception as error:
            value=dict(status='FAIL',error_type=type(error).__name__,
                reason='Named verification raised; rerun this function locally for details. Exception text omitted to avoid publishing private arrays.')
        value['CPU_wall_seconds']=time.perf_counter()-tick;checks[operation.__name__]=value
        print(operation.__name__,value['status'],flush=True)
    if args.skip_tests:checks['cpu_tests']=dict(status='NOT_RUN',reason='Explicit --skip-tests')
    result=dict(created_at=C.now(),status=overall(checks),checks=checks,
                CPU_wall_seconds=time.perf_counter()-start,CPU_process_seconds=time.process_time()-cpu,
                GPU_seconds=0,new_fits=0,optimizer_updates=0,
                code=C.bind(Path(__file__)),tests=C.bind(Path(__file__).with_name('test_final_audit.py')),
                reused_helper_code=C.bind(Path(V1.__file__)),
                integrity='Snapshot while research may still be active; rerun after final results/ledger/reports. No old files, locked code, ledger, state, or Git are modified.',
                rerun='python -m scripts.research.pallet_pose_objective_followup_v2.final_audit')
    require(not privacy_paths(result),'Audit would publish a private array')
    C.save(C.DOC/'AUDIT.json',result)
    print('AUDIT',result['status'],flush=True)
    return int(result['status']=='FAIL')


if __name__=='__main__':raise SystemExit(main())
