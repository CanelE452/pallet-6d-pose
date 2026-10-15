"""Freeze pre-mask full-WD choice, then use its exact saved d-family pose.

Cheap all238 selection control: original CAL46/TEST49 and fresh remaining143.
No fitting, image queries, reference-based routing, or old-pose replacement.
All238 predictions are sealed before this control opens coordinate references.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import shutil

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import select_frozen_wd_partial_refinement_v4_13e as ORIGINAL
from . import evaluate_two_wd_families_v4_9 as WD
from . import score_remainder_derived_reference_v4_14 as REF
from . import score_remainder_closed_sources_v4_14 as CLOSED
from .scoring import pose_metric,distribution

OUTPUT=ORIGINAL.OUTPUT.parent/'early_wd_partial_refinement_eval_v4_15'
FRESH=REF.REMAINDER/'pose_evaluation_v4_14'
ARM='EARLY_WD_FACE_AWARE_PARTIAL_BI'
CURRENT=ORIGINAL.ARMS[2]
RULE='FULL_WD_SEARCH (prior0 BI/legacy partial/OT/U before global mask) family fixed; exact d same-family saved numeric pose, including its own fallback'


def _bind(path):return str(Path(path).relative_to(C.ROOT)),C.sha(path)


def _verified_journal(folder,count):
    protocol=C.read(folder/'PROTOCOL.json');seal=C.read(folder/'PREDICTION_SEAL.json')
    journal=folder/'PREDICTIONS.jsonl.gz'
    if (seal['rows']!=count or C.sha(journal)!=seal['predictions_sha256']
        or C.sha(folder/'PROTOCOL.json')!=seal['protocol_sha256']):
        raise ValueError('Complete source journal seal required: '+str(folder))
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    return protocol,seal,journal


def select_early_family(old,partial,early):
    name=early['selected_family']
    if name not in ('ORIGINAL_WD','SWAPPED_WD'):raise ValueError('Explicit early family required')
    if early.get('original_states',old['states'])!=old['states']:
        raise ValueError('Early search and final states differ')
    for key in ('states','image_sha256','frozen_BI_fit','u_region_observation'):
        if key in early and early[key]!=old[key]:raise ValueError('Early/final observation binding differs: '+key)
    new=partial['arms'][ORIGINAL.D.ARMS[1]]
    for family in early['families']:
        before=next(f for f in old['families'] if f['name']==family['name'])
        after=next(f for f in new['families'] if f['name']==family['name'])
        for key in ('native_xyz','cf_extents','physical_registration_Q'):
            if not np.allclose(family[key],before[key],rtol=0,atol=1e-12) or not np.allclose(family[key],after[key],rtol=0,atol=1e-12):
                raise ValueError('Early/global/d family geometry binding differs: '+key)
    # Reuse the existing exact-family validator and conservative ambiguity
    # propagation; this proxy changes only the predicted family selector.
    proxy=deepcopy(old);proxy['selected_family']=name;proxy['family_ambiguity']=deepcopy(early['family_ambiguity'])
    selected=ORIGINAL.select_same_family(proxy,partial)
    chosen=new['arms'][name]
    for key in ('actual_pose','accepted','states','frozen_visible_measurements'):
        if key in chosen and selected[key]!=chosen[key]:raise ValueError('Exact d numeric/readout archive changed: '+key)
    if selected['solution']['R']!=chosen['solution']['R'] or selected['solution']['t']!=chosen['solution']['t']:
        raise ValueError('No old-best-pose or cross-family fallback permitted')
    selected['method']=ARM
    selected['hierarchical_correspondence_audit'].update(rule=RULE,
        initial_family_choice_stage='FULL_WD_SEARCH_BEFORE_GLOBAL_MASK',
        predicted_early_family=name,GT_inputs=False,new_pose_fits=0)
    return selected


def _wd_sources():
    protocol=C.read(WD.OUTPUT/'PROTOCOL.json');complete=C.read(WD.OUTPUT/'PREDICTION_COMPLETE.json')
    if complete['rows']!=95 or complete['severe']!=0 or not complete['before_any_new_pose_scoring']:
        raise ValueError('Original complete95 early-WD search required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    result={};bindings=dict([_bind(WD.OUTPUT/'PROTOCOL.json'),_bind(WD.OUTPUT/'PREDICTION_COMPLETE.json')])
    for part,count in (('CAL',46),('TEST',49)):
        path=WD.OUTPUT/f'{part}_PREDICTION_SEAL.json';seal=C.read(path);journal=WD.OUTPUT/f'{part}_PREDICTIONS.jsonl.gz'
        if (seal['rows']!=count or C.sha(journal)!=seal['predictions_sha256']
            or C.sha(path)!=complete['ordered_partition_prediction_seal_sha256'][part]
            or seal['protocol_sha256']!=C.sha(WD.OUTPUT/'PROTOCOL.json')):
            raise ValueError('Original partition-WD seal changed')
        values=list(C.rows(journal))
        if len(values)!=count:raise ValueError('Keep every early-WD row')
        for row in values:
            if row['contract']['prior_weight']!=0:raise ValueError('Early family must use prior0 only')
            result[row['id']]={key:deepcopy(row[key]) for key in ('id','partition','session','image_sha256','states',
                'frozen_BI_fit','partial_boundary_observation','u_region_observation','selected_family','family_ambiguity')}
            result[row['id']]['families']=[{key:deepcopy(f[key]) for key in ('name','native_xyz','cf_extents','physical_registration_Q')}for f in row['families']]
        bindings.update([_bind(path),_bind(journal)])
    return result,protocol,bindings


def predict(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every previous selector control')
    closure=CLOSED.verify_additional_source_closure();wd,wd_protocol,bindings=_wd_sources()
    protocols={};journals={};sources={};protected=None
    for name,folder,count in (('old',ORIGINAL.OLD,95),('d',ORIGINAL.PARTIAL,95),('e',ORIGINAL.OUTPUT,95),('fresh',FRESH,143)):
        protocol,seal,journal=_verified_journal(folder,count);protocols[name]=protocol;journals[name]=journal
        if protected is not None and protected!=protocol['protected_user_sha256']:raise ValueError('Protected contracts differ')
        protected=protocol['protected_user_sha256'];sources.update(protocol['source_sha256']);bindings.update(protocol['input_sha256'])
        for path in (folder/'PROTOCOL.json',folder/'PREDICTION_SEAL.json',journal):bindings.update([_bind(path)])
    sources.update(wd_protocol['source_sha256'])
    for path in (Path(__file__),Path(ORIGINAL.__file__),Path(CLOSED.__file__),Path(REF.__file__),Path(pose_metric.__code__.co_filename)):
        sources.update([_bind(path)])
    bindings.update([_bind(CLOSED.CLOSURE/'ADDITIONAL_EXECUTED_SOURCE_BINDINGS_V1.json'),
                     _bind(REF.OUTPUT/'PLAN.json'),_bind(REF.OUTPUT/'SCORE_SEAL.json')])
    # Targets are byte-bound here; no coordinate/pose JSON is decoded.
    for target in (E.COHORT/'OOF_pose_v1/ISOLATED_CAL_SCORING.json',E.COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json'):
        bindings.update([_bind(target)])
    fresh_ids=protocols['fresh']['ids'];old_ids=protocols['old']['ids']
    if (protocols['d']['ids']!=old_ids or protocols['e']['ids']!=old_ids
        or len(old_ids)!=95 or len(fresh_ids)!=143 or len(set(old_ids+fresh_ids))!=238):
        raise ValueError('Original ordered95 plus fresh143 required')
    for value in (sources,bindings,protected):E._verify_sources(value)
    output.mkdir(parents=True)
    C.write(output/'PROTOCOL.json',dict(schema='EARLY_FULL_WD_FAMILY_THEN_EXACT_D_ALL238_V4_15',
        rule=RULE,arms=[CURRENT,ARM],ids=old_ids+fresh_ids,rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,source_closure=closure,
        family_policy_fixed_before_own_metrics=True,all238_sealed_before_own_reference_decode=True,
        new_pose_or_line_fits=0,new_image_or_state_queries=0,reference_based_pose_or_family_choice=False,
        old_best_pose_fallback=False,all_fallbacks_and_native_or_reference_flags_retained=True,
        previously_seen_e_metrics_motivated_control=True,independent_holdout_claim=False,
        fresh_reference_policy=REF.POLICY))
    for relative,digest in sources.items():
        path=output/'code_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path)!=digest:raise ValueError('Source snapshot changed')
    results=[]
    for old,d,current in zip(C.rows(journals['old']),C.rows(journals['d']),C.rows(journals['e'])):
        if old['id']!=d['id'] or old['id']!=current['id']:raise ValueError('Exact ordered source row binding')
        early=wd[old['id']]
        for key in ('id','partition','session'):
            if early[key]!=old[key]:raise ValueError('Early/source frame mismatch: '+key)
        if early['partial_boundary_observation']!=old['partial_boundary_observation']:
            raise ValueError('Legacy partial observations differ before family choice')
        selected=select_early_family(old,d,early)
        results.append(dict(id=old['id'],partition=old['partition'],session=old['session'],states=old['states'],
            image_sha256=old['image_sha256'],current_family=old['selected_family'],early_family=early['selected_family'],
            family_changed=old['selected_family']!=early['selected_family'],
            arms={CURRENT:current['arms'][CURRENT],ARM:selected},source_exact_d_family=True))
    for row in C.rows(journals['fresh']):
        early=row['full_WD_search'];old=row['old_global_outline'];d=row['partial_global_outline']
        if early['effective_prior_weight']!=0 or early['solver_configuration']['prior_weight']!=0:
            raise ValueError('Fresh full-WD pre-mask family must have no prior')
        selected=select_early_family(old,d,early)
        results.append(dict(id=row['id'],partition='REMAINDER',session=row['session'],states=row['states'],
            image_sha256=row['image_sha256'],current_family=row['frozen_family'],early_family=early['selected_family'],
            family_changed=row['frozen_family']!=early['selected_family'],
            arms={CURRENT:row['arms'][CURRENT],ARM:selected},source_exact_d_family=True))
    if [r['id']for r in results]!=old_ids+fresh_ids:raise ValueError('Keep every ordered238 frame')
    for value in (sources,bindings,protected):E._verify_sources(value)
    CLOSED.verify_additional_source_closure()
    C.save_rows(output/'PREDICTIONS.jsonl.gz',results)
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        ordered_ids=old_ids+fresh_ids,predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),before_own_reference_coordinate_decode=True,
        all238_including_fallbacks_kept=True,new_fits_and_image_queries=0))


def score(output=OUTPUT):
    output=Path(output).resolve()
    if (output/'SCORE_SEAL.json').exists():raise FileExistsError('Preserve every prior score')
    protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREDICTION_SEAL.json');journal=output/'PREDICTIONS.jsonl.gz'
    if (seal['rows']!=238 or not seal['before_own_reference_coordinate_decode']
        or C.sha(journal)!=seal['predictions_sha256'] or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']):
        raise ValueError('Seal all238 before own metrics')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    CLOSED.verify_additional_source_closure();rows=list(C.rows(journal))
    if [r['id']for r in rows]!=protocol['ids']:raise ValueError('No population filtering')
    reference_plan=C.read(REF.OUTPUT/'PLAN.json');annotations={r['id']:r for r in reference_plan['annotations']}
    reports={}
    for partition,count in (('CAL',46),('TEST',49),('REMAINDER',143)):
        population=[r for r in rows if r['partition']==partition]
        if len(population)!=count:raise ValueError('Every partition row required')
        labels=None
        if partition!='REMAINDER':
            target=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if partition=='CAL'else'test_reused_v1/ISOLATED_TEST_SCORING.json')
            labels=C.read(target)['rows']
        records=[]
        for row in population:
            if labels is not None:
                truth=labels[row['id']]['target']['truth']
                metrics={arm:pose_metric(row['arms'][arm]['actual_pose'],truth)for arm in (CURRENT,ARM)}
                reference_metadata=labels[row['id']]['metadata']
            else:
                annotation=annotations[row['id']]
                if C.sha(C.ROOT/annotation['annotation'])!=annotation['annotation_sha256']:raise ValueError('Fixed143 reference changed')
                truth=REF.reference_from_document(C.read(C.ROOT/annotation['annotation']))
                metrics={arm:REF.metric_for_reference(row['arms'][arm]['actual_pose'],truth)for arm in (CURRENT,ARM)}
                reference_metadata=dict(audit=truth['audit'],metadata=truth['metadata'])
            records.append(dict(id=row['id'],session=row['session'],family_changed=row['family_changed'],
                current_family=row['current_family'],early_family=row['early_family'],metrics=metrics,
                accepted={arm:row['arms'][arm]['accepted']for arm in (CURRENT,ARM)},reference_metadata=reference_metadata))
        summary={}
        for arm in (CURRENT,ARM):
            summary[arm]=dict(translation_cm=distribution([r['metrics'][arm]['translation_cm']for r in records]),
                rotation_deg=distribution([r['metrics'][arm]['rotation_deg']for r in records if r['metrics'][arm]['rotation_deg']is not None]),
                translation_le5=sum(r['metrics'][arm]['translation_cm']<=5 for r in records),
                successes_5cm_5deg=sum(bool(r['metrics'][arm]['success_5cm_5deg'])for r in records),
                fallback_count=sum(not r['accepted'][arm]for r in records),
                max_translation_id=max(records,key=lambda r:r['metrics'][arm]['translation_cm'])['id'])
        report=dict(partition=partition,rows=records,count=count,summary=summary,
            family_changes=sum(r['family_changed']for r in records),all_rows_and_flags_kept=True,rule=RULE,
            reference_or_metrics_not_used_for_choice=True,independent_physical_GT_or_untouched_holdout_claim=False)
        C.write(output/f'{partition}_SCORE.json',report);reports[partition]=report
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    CLOSED.verify_additional_source_closure()
    C.write(output/'SCORE_SEAL.json',dict(prediction_seal_sha256=C.sha(output/'PREDICTION_SEAL.json'),rows=238,
        score_sha256={part:C.sha(output/f'{part}_SCORE.json')for part in reports},all238_prediction_seal_before_own_metrics=True))
    print({p:dict(family_changes=r['family_changes'],summary=r['summary'])for p,r in reports.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('predict','score','all'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    if args.stage in ('predict','all'):predict(args.output)
    if args.stage in ('score','all'):score(args.output)
