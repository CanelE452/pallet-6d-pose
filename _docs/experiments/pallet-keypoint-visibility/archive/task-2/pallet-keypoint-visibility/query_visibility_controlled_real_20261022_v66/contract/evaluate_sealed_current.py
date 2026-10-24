"""Replay selected V66 PRIVATE saved outputs only after complete inference seal.

Before the gate checks finish this module opens no current GT, score or input
file. Source-support classes E/O remain distinct diagnostic outputs; binary
negative is ABSENT_NONVISIBLE, never an assertion of external mechanism.
"""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

import saved_evaluate_current_scores as source_base
source_base.rows=source_base.readcsv
from source_support_contract import CLASS_NAMES,JOINT_STATES,confident_joint_map,hierarchical_states,joint_marginals

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PALLET=ROOT.parent
MODEL='V66_SOURCE_SELECTED'
SOURCE_SEAL_SHA=None
PAIRS=np.asarray(JOINT_STATES)
OUTPUT_NAMES=['VISIBLE_SUPPORT_PROXY','EXTERNAL_SUPPORT_PROXY','OUTSIDE_FULL_SUPPORT_PROXY','U']


def complete_gate(gate_path,inference_seal,scores_path):
    """Complete source+cached-inference metadata gate before GT/score reads."""
    code=ROOT/'inference/infer_cached_current.py'
    spec=importlib.util.spec_from_file_location('v66_cached_inference_gate_only',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    gate,candidate,freeze,prep=module.validate_gate(gate_path)
    if inference_seal.resolve()!=Path(candidate['output_directory']).resolve()/'INFERENCE_SEAL.json':
        raise PermissionError('Unapproved inference seal path')
    if scores_path.resolve()!=Path(candidate['output_directory']).resolve()/'REAL_QUERY_SCORES.csv':
        raise PermissionError('Unapproved score path')
    seal=json.loads(inference_seal.read_text())
    if seal.get('schema')!='V66_CACHED_SELECTED_PRIVATE_INFERENCE_SEAL_V1'or seal.get('completed')is not True:
        raise PermissionError('Complete V66 inference seal required')
    if seal.get('frames')!=223 or seal.get('slots')!=1784:raise PermissionError('Sealed cohort dimensions changed')
    if seal['candidate']!=candidate['name']or seal['gate']['sha256']!=source_base.sha(gate_path):
        raise PermissionError('Sealed source candidate/gate mismatch')
    for key,binding in [('checkpoint',candidate['checkpoint']),('freeze',candidate['freeze']),('source_selection_seal',gate['source_selection_seal']),
                        ('fit_input_lock',gate['fit_input_lock']),('fit_protocol',gate['fit_protocol']),('preparation_seal',gate['v65_preparation_seal'])]:
        if not module.same(seal[key],binding):raise PermissionError('Sealed binding mismatch '+key)
    if seal['inference_code_sha256']!=candidate['inference_code_sha256']or seal['cached_inputs']!=gate['private_inputs']:
        raise PermissionError('Inference code/cached inputs changed')
    if seal['thresholds']!=module.THRESHOLDS:raise PermissionError('Sealed decoder changed')
    for key in ['optimizer_updates','PnP_calls','new_backbone_forward','SAM_forwards','real_RGB_images_opened','semantic_annotation_reads','real_CAL_model_reads']:
        if seal[key]!=0:raise PermissionError('Inference scope violation '+key)
    for key in ['private_ref_or_human_GT_read','performance_tables_read','old_current_scores_read','threshold_or_model_selection','geometry_self_overrides_learned_probability']:
        if seal[key]is not False:raise PermissionError('Inference exclusion failed '+key)
    for key in ['model_state_unchanged','original_query_coordinates_preserved','input_validity_flags_preserved','geometry_flags_preserved']:
        if seal[key]is not True:raise PermissionError('Inference invariant failed '+key)
    if seal['device']!='cpu'or seal['head_torch']!=gate['runtime']['head_torch']or seal['head_executable']!=str(Path(gate['runtime']['head_executable']).resolve()):
        raise PermissionError('Inference runtime changed')
    registered={str(Path(b['path']).resolve()):b for b in seal['outputs']}
    if str(scores_path.resolve())not in registered:raise PermissionError('Scores not sealed')
    # Score hashes/metadata access begin only after inference completeness/scope.
    for b in registered.values():module.check(b,'sealed V66 output')
    for role,b in gate['private_inputs'].items():module.check(b,'unchanged cached input '+role)
    return candidate,seal


def validate_scores(records):
    """No GT needed: exact/r2 marginals, input validity and decision parity."""
    if len(records)!=1784:raise AssertionError('Every registered native slot must be saved before GT evaluation')
    seen=set();valid=0
    for row in records:
        key=(row['id'],row['native_corner'])
        if key in seen:raise AssertionError('Duplicate saved native slot')
        seen.add(key)
        good=row['valid'].lower()=='true'
        if not good:
            if row.get('p_visible',''):
                raise AssertionError('Invalid raw network values must not be main binary scores')
            continue
        p=np.array([float(row['joint_p'+str(j)])for j in range(6)])
        marginal=joint_marginals(p)
        for scale,name in enumerate(['exact','radius2']):
            expected={'p_visible_'+name:marginal[scale,0],
                      'p_external_support_'+name:marginal[scale,1],
                      'p_outside_full_'+name:marginal[scale,2],
                      'p_full_'+name:marginal[scale,:2].sum()}
            for field,value in expected.items():
                if not math.isclose(float(row[field]),float(value),abs_tol=1e-6):raise AssertionError('Saved marginal mismatch: '+field)
        if not math.isclose(float(row['p_visible']),float(marginal[0,0]),abs_tol=1e-6):raise AssertionError('Primary binary score must be exact pV')
        cm=confident_joint_map(p)
        expectedU=not bool(cm['accepted'])
        if (row['confidence_U'].lower()=='true') != expectedU:
            raise AssertionError('Saved confidence-U mismatch')
        valid+=1
    return dict(rows=len(records),valid=valid,invalid=len(records)-valid)


def main():
    global MODEL,SOURCE_SEAL_SHA
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gate',required=True,type=Path)
    ap.add_argument('--inference-seal',required=True,type=Path)
    ap.add_argument('--scores',required=True,type=Path)
    ap.add_argument('--output-prefix',default='V66_CURRENT')
    a=ap.parse_args()
    # No evaluation GT, score CSV or input data may be read before this gate.
    candidate,inference=complete_gate(a.gate,a.inference_seal,a.scores)
    MODEL='V66_'+candidate['name']
    SOURCE_SEAL_SHA=inference['source_selection_seal']['sha256']
    if list(HERE.glob(a.output_prefix+'_*'))or(HERE/'PRIVATE_PRIMARY_RESULTS.json').exists():raise FileExistsError('Preserve prior V66 evaluation outputs')
    binding=json.loads((HERE/'PRIVATE_SAVED_EVALUATOR_COPY_BINDING.json').read_text())
    copied=HERE/'saved_evaluate_current_scores.py'
    if source_base.sha(copied)!=binding['sha256']:raise AssertionError('Copied v63 evaluator changed')
    score_records=source_base.rows(a.scores)
    score_audit=validate_scores(score_records)
    prefix=HERE/a.output_prefix
    if not prefix.resolve().is_relative_to(HERE):raise ValueError('Outputs must stay in v66contract')
    command=[sys.executable,str(copied),'--scores',str(a.scores),'--freeze',candidate['freeze']['path'],
             '--score-column','p_visible','--model',MODEL,'--output-prefix',a.output_prefix,
             '--include-geometry','--allow-unresolved-scores']
    subprocess.run(command,check=True)
    # Predeclared radius2 diagnostic uses its own learned target, never replaces
    # exact-pixel primary or selects a radius from private results.
    radius_prefix=a.output_prefix+'_R2_SUPPORT_DIAGNOSTIC'
    subprocess.run([sys.executable,str(copied),'--scores',str(a.scores),'--freeze',candidate['freeze']['path'],
                    '--score-column','p_visible_radius2','--model',MODEL+'_R2_SUPPORT_DIAGNOSTIC',
                    '--output-prefix',radius_prefix,'--include-geometry','--allow-unresolved-scores'],check=True)
    oldroot=PALLET/'query_visibility_public_real_20261022_v63'
    oldname='MIXED_CONTEXT_RGB_SAM_V4'
    oldscores=oldroot/'private_evaluation'/oldname/'REAL_QUERY_SCORES.csv'
    oldfreeze=oldroot/'runs'/oldname/'MODEL_FREEZE.json'
    oldinference=oldscores.with_name('INFERENCE_SEAL.json')
    historical=json.loads(oldinference.read_text())
    historical_score=next(x for x in historical['outputs']if Path(x['path']).resolve()==oldscores.resolve())
    if source_base.sha(oldscores)!=historical_score['sha256']:raise AssertionError('Historical v63 score changed after inference seal')
    oldprefix=a.output_prefix+'_V63_PRIMARY_REPLAY'
    subprocess.run([sys.executable,str(copied),'--scores',str(oldscores),'--freeze',str(oldfreeze),
                    '--score-column','p_visible','--model',oldname,'--output-prefix',oldprefix,
                    '--include-geometry','--allow-unresolved-scores'],check=True)
    v65root=PALLET/'query_visibility_amodal_global_20261022_v65'
    v65name='V65_MIXED_GLOBAL'
    v65scores=v65root/'private_evaluation/MIXED_GLOBAL_UNWEIGHTED/REAL_QUERY_SCORES.csv'
    v65freeze=v65root/'runs/MIXED_GLOBAL_UNWEIGHTED/MODEL_FREEZE.json'
    v65inference=v65scores.with_name('INFERENCE_SEAL.json')
    v65history=json.loads(v65inference.read_text())
    if v65history.get('completed')is not True:raise PermissionError('Historical V65 inference incomplete')
    v65registered=next(b for b in v65history['outputs']if Path(b['path']).resolve()==v65scores.resolve())
    if source_base.sha(v65scores)!=v65registered['sha256']:raise PermissionError('Historical V65 scores changed')
    v65prefix=a.output_prefix+'_V65_PRIMARY_REPLAY'
    subprocess.run([sys.executable,str(copied),'--scores',str(v65scores),'--freeze',str(v65freeze),
      '--score-column','p_visible','--model',v65name,'--output-prefix',v65prefix,'--include-geometry','--allow-unresolved-scores'],check=True)
    # Current GT reads start here only after the inference seal and binary replay.
    spec=importlib.util.spec_from_file_location('v66_pure_saved_semantic_evaluator',copied)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    manifest=PALLET/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv'
    truth=base.readcsv(manifest)
    lookup={(r['id'],r['native_corner']):r for r in score_records}
    probabilities={key:float(r['p_visible'])if r['p_visible']else None for key,r in lookup.items()}
    geometry={key:True if r['geometry_self_hidden'].lower()=='true'else False if r['geometry_self_hidden'].lower()=='false'else None for key,r in lookup.items()}
    rules=dict(threshold_comparison='>=',geometry_flags=geometry,geometry_model=MODEL+'+GEOMETRY')
    oldrecords=base.readcsv(oldscores)
    oldprob={(r['id'],r['native_corner']):float(r['p_visible'])if r['p_visible']else None for r in oldrecords}
    oldgeometry={(r['id'],r['native_corner']):True if r['geometry_self_hidden'].lower()=='true'else False if r['geometry_self_hidden'].lower()=='false'else None for r in oldrecords}
    oldrules=dict(threshold_comparison='>=',geometry_flags=oldgeometry,geometry_model=oldname+'+GEOMETRY')
    v65records=base.readcsv(v65scores)
    v65prob={(r['id'],r['native_corner']):float(r['p_visible'])if r['p_visible']else None for r in v65records}
    v65geometry={(r['id'],r['native_corner']):True if r['geometry_self_hidden'].lower()=='true'else False if r['geometry_self_hidden'].lower()=='false'else None for r in v65records}
    v65rules=dict(threshold_comparison='>=',geometry_flags=v65geometry,geometry_model=v65name+'+GEOMETRY')
    # Six states + U are source-support predictions, not seven human GT classes.
    exports=[];distributions=[]
    for row in truth:
        key=(row['id'],row['native_corner']);score=lookup.get(key)
        if score is None:raise AssertionError('Missing native slot export')
        out={k:row[k]for k in ['id','native_corner','group','human_state','initial_reference_error_px']}
        good=score['valid'].lower()=='true'
        out['valid']=good
        if good:
            p=np.array([float(score['joint_p'+str(j)])for j in range(6)])
            c=confident_joint_map(p);j=int(c['joint'])
            out.update(joint_MAP=int(p.argmax()),confident_joint_or_U=j,
                       exact_source_support_or_U=OUTPUT_NAMES[int(c['classes'][0])]if j>=0 else'U',
                       radius2_source_support_or_U=OUTPUT_NAMES[int(c['classes'][1])]if j>=0 else'U',
                       max_joint_probability=float(p.max()),p_visible_exact=float(p[0]),
                       binary_exact=base.call(row,MODEL,'D_vs_ES',probabilities,.5,rules),
                       binary_exact_plus_geometry=base.call(row,MODEL+'+GEOMETRY','D_vs_ES',probabilities,.5,rules))
            for j in range(6):out['joint_p'+str(j)]=float(p[j])
        else:
            out.update(joint_MAP='',confident_joint_or_U=-1,exact_source_support_or_U='U',radius2_source_support_or_U='U',
                       max_joint_probability='',p_visible_exact='',binary_exact='OTHER_UNRESOLVED',
                       binary_exact_plus_geometry=base.call(row,MODEL+'+GEOMETRY','D_vs_ES',probabilities,.5,rules))
            for j in range(6):out['joint_p'+str(j)]=''
        exports.append(out)
    exported={(r['id'],r['native_corner']):r for r in exports}
    for cohort in ['paired_same_query_RGB','same_slot_pipeline']:
        for group in base.GROUPS:
            rr=[r for r in truth if r['group']==group and base.yes(r,cohort+'_D_vs_ES')]
            for state in [base.D,base.E,base.S]:
                z=[r for r in rr if r['human_state']==state]
                for scale in ['exact','radius2']:
                    count=Counter(exported[r['id'],r['native_corner']][scale+'_source_support_or_U']for r in z)
                    distributions.append(dict(cohort=cohort,group=group,human_state=state,scale=scale,N=len(z),
                                              **{name:count[name]for name in OUTPUT_NAMES}))
    provenance=PALLET/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv'
    near=base.readcsv(provenance)
    assert len(near)==461 and all(r['human_state']==base.D and float(r['initial_reference_error_px'])<=3 for r in near)
    manual=[r for r in near if r['annotation_coordinate_source']=='manual_click']
    assert len(manual)==376
    manual_metrics=[]
    for group in base.GROUPS:
        z=[r for r in manual if r['group']==group]
        for model in ['v61','legacy',oldname,oldname+'+GEOMETRY',v65name,v65name+'+GEOMETRY',MODEL,MODEL+'+GEOMETRY']:
            pp,rr=(oldprob,oldrules)if model.startswith(oldname)else(v65prob,v65rules)if model.startswith(v65name)else(probabilities,rules)
            counts=Counter(base.call(r,model,'D_vs_ES',pp,.5,rr)for r in z)
            manual_metrics.append(dict(group=group,scope='MANUAL_COORDINATE_NEAR_D_LE3',model=model,N=len(z),
                pred_visible=counts['VISIBLE'],pred_absent_nonvisible=counts['ABSENT_NONVISIBLE'],pred_U=counts['ABSTAIN'],
                pred_other_unresolved=counts['OTHER_UNRESOLVED'],semantic_D_recall=base.ratio(counts['VISIBLE'],len(z)),
                exact_pixel_GT='UNAVAILABLE; semantic/manual reference evaluation only'))
    base.writecsv(Path(str(prefix)+'_SOURCE_SUPPORT_SIX_STATE_U_EXPORT.csv'),exports)
    base.writecsv(Path(str(prefix)+'_HUMAN_STATE_VS_SOURCE_SUPPORT_COUNTS.csv'),distributions)
    base.writecsv(Path(str(prefix)+'_MANUAL_NEAR_D_LE3.csv'),manual_metrics)
    summary=dict(schema='V66_SEALED_CURRENT_SAVED_OUTPUT_EVALUATION_V1',primary='exact pV>=.5 no confidence-U; invalid input unresolved',
                 model=MODEL,score_audit=score_audit,manual_near_D=manual_metrics,
                 six_state_scope='Source support prediction distribution versus human semantic states only; E_PROXY never equated to EXTERNAL_OCCLUDED, O never relabeled external, no private exact-pixel ownership GT.',
                 geometry='Saved initial-pose conditional cuboid self proxy reported separately; no learned self or PnP.',
                 current_eval_used_for_selection=False,new_model_inference_training_PnP_calls=0,
                 inference_seal_sha256=source_base.sha(a.inference_seal),source_selection_seal_sha256=SOURCE_SEAL_SHA,
                 hashes={str(path):source_base.sha(path)for path in [Path(__file__),a.gate,a.inference_seal,a.scores,manifest,provenance,copied,
                                                                   oldscores,oldfreeze,oldinference,v65scores,v65freeze,v65inference]})
    Path(str(prefix)+'_TRANSFER_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    # One concrete schema for root's report: all original eligibility cohorts,
    # manual<=3 subset, fixed geometry and radius2 kept as distinct rows/tags.
    metrics={};confusion={};error_strata={};geometry_tradeoff={}
    specs=[(a.output_prefix,'PRIMARY_EXACT_BINARY'),(oldprefix,'HISTORICAL_V63_PRIMARY'),(v65prefix,'HISTORICAL_V65_PRIMARY'),
           (radius_prefix,'RADIUS2_SUPPORT_DIAGNOSTIC')]
    for saved_prefix,role in specs:
        for suffix,keys,target in [
            ('CLASSIFICATION_METRICS.csv',['cohort','group','task','model'],metrics),
            ('CONFUSION_BY_HUMAN_STATE.csv',['cohort','group','task','model','human_state'],confusion),
            ('ERROR_STRATA.csv',['cohort','group','human_state','error_bin','model'],error_strata),
            ('GEOMETRY_TRADEOFF.csv',['cohort','group','task','human_state'],geometry_tradeoff)]:
            for row in base.readcsv(HERE/(saved_prefix+'_'+suffix)):
                key=tuple(row[k]for k in keys)
                if suffix=='GEOMETRY_TRADEOFF.csv':key+=(row['raw_model'],)
                label='HISTORICAL_BASELINE'if row.get('model')in['v61','legacy']else role
                row['role']=label
                if key in target:
                    if target[key]!=row:raise AssertionError('Repeated historical baseline mismatch')
                else:target[key]=row
    combined=dict(schema='V66_PRIVATE_PRIMARY_RESULTS_V1',selected_primary=MODEL,
                  selection='Canonical source-only seal; no private model/runtime/threshold/radius selection.',
                  primary_prediction='exact pV>=.5, no confidence U, invalid input unresolved; negative ABSENT_NONVISIBLE',
                  target='Existing human native-corner D/E/S semantics; private exact-pixel support GT unavailable.',
                  metrics=list(metrics.values()),confusion_by_human_state=list(confusion.values()),
                  error_strata=list(error_strata.values()),geometry_tradeoff=list(geometry_tradeoff.values()),
                  manual_near_D_results=manual_metrics,source_support_output_counts=distributions,
                  source_support_caution='Sixstate/U output distributions are not a semantic E/O ground-truth confusion matrix. O is never external by default.',
                  radius2='Separate learned finite-neighborhood support semantic-transfer diagnostic; exact remains primary.',
                  geometry='Saved initial-pose pure cuboid proxy only, raw probabilities preserved, no PnP.',
                  source_selection_seal_sha256=SOURCE_SEAL_SHA,inference_seal_sha256=source_base.sha(a.inference_seal),
                  no_new_model_inference_training_PnP_by_evaluator=True,current_eval_used_for_selection=False,
                  source_bindings=summary['hashes'])
    joint_counts=[]
    for group in base.GROUPS:
        for state in [base.D,base.E]:
            selected=[r for r in truth if r['group']==group and r['human_state']==state and base.yes(r,'paired_same_query_RGB_D_vs_E')]
            count=Counter(exported[r['id'],r['native_corner']]['confident_joint_or_U']for r in selected)
            joint_counts.append(dict(group=group,human_state=state,N=len(selected),
                                     **{name:count[j]for j,name in enumerate(['VV','EV','EE','OV','OE','OO'])},U=count[-1]))
    combined['source_support_six_joint_counts_paired_D_E']=joint_counts
    (HERE/'PRIVATE_PRIMARY_RESULTS.json').write_text(json.dumps(combined,indent=2)+'\n')
    base.writecsv(HERE/'PRIVATE_PRIMARY_RESULTS.csv',list(metrics.values()))
    base.writecsv(HERE/'PRIVATE_PRIMARY_CONFUSION.csv',list(confusion.values()))
    base.writecsv(HERE/'PRIVATE_PRIMARY_ERROR_STRATA.csv',list(error_strata.values()))
    base.writecsv(HERE/'PRIVATE_PRIMARY_GEOMETRY_TRADEOFF.csv',list(geometry_tradeoff.values()))
    base.writecsv(HERE/'PRIVATE_PRIMARY_SIX_JOINT_COUNTS.csv',joint_counts)
    print(json.dumps(dict(output_prefix=str(prefix),manual_near_D=manual_metrics)))


if __name__=='__main__':main()
