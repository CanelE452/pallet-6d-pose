"""SOURCE-only v65 replay, v64 controls, per-object report and fixed release rule.

Executes a byte-identical copy of the v64 NumPy saved-output evaluator. It never
imports a model, opens source RGB/masks, or reads private images/labels/scores.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

import saved_evaluate_frozen_source as base

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
V64=ROOT.parent/'query_visibility_multitarget_20261022_v64'
V63=ROOT.parent/'query_visibility_public_real_20261022_v63'


def validate_global_cache(path):
    """Validate source-global ownership and unchanged native local arrays."""
    with np.load(path,allow_pickle=False)as z:
        n=len(z['sample_id']);t=len(z['global_id'])
        expected={'target_index':(n,),'global_rgb':(t,3,64,64),'global_sam':(t,1,64,64),
                  'global_prompt':(t,1,64,64),'global_joint':(t,64,64),'global_valid':(t,64,64),
                  'global_id':(t,),'global_image_sha256':(t,),'global_box':(t,4),
                  'global_index64':(n,9,9),'global_index8':(n,9,9),'global_query_valid':(n,9,9)}
        for key,shape in expected.items():
            if z[key].shape!=shape:raise AssertionError('Global cache shape mismatch: '+key)
        ti=z['target_index'];ids=z['global_id'];images=z['global_image_sha256']
        if not np.issubdtype(ti.dtype,np.integer)or(ti<0).any()or(ti>=t).any():raise AssertionError('Bad target index')
        if len(set(ids))!=t:raise AssertionError('Duplicate target-global ID')
        if not np.array_equal(ids[ti],z['id']):raise AssertionError('Target ownership mismatch')
        if not np.array_equal(images[ti],z['image_sha256']):raise AssertionError('Target image mismatch')
        if z['global_rgb'].dtype!=np.uint8:raise AssertionError('Source global RGB encoding mismatch')
        if not np.isfinite(z['global_sam']).all()or np.abs(z['global_sam']).max()>4.000001:raise AssertionError('Bad frozen SAM context')
        gv=z['global_valid'].astype(bool);label=z['global_joint']
        if not np.isin(label[gv],np.arange(6)).all():raise AssertionError('Invalid global supervision')
        i64,i8,qv=z['global_index64'],z['global_index8'],z['global_query_valid'].astype(bool)
        if (i64<0).any()or(i64>=4096).any()or(i8<0).any()or(i8>=64).any():raise AssertionError('Global lookup out of bounds')
        if not np.array_equal(i8[qv],((i64//64)//8*8+(i64%64)//8)[qv]):raise AssertionError('Bottleneck phase mismatch')
        if (qv&~z['dense_valid']).any():raise AssertionError('Global mapping valid on invalid native query')
        if (i64[~qv]!=0).any()or(i8[~qv]!=0).any():raise AssertionError('Invalid lookup placeholders changed')
        source=z['source'];target_sources={}
        for i,s in zip(ti,source):target_sources.setdefault(int(i),set()).add(str(s))
        if any(len(s)!=1 for s in target_sources.values()):raise AssertionError('Target spans source domains')
        oldpath=V64/'source_cache'/Path(path).name.replace('_GLOBAL.npz','_WITH_SAM.npz')
        with np.load(oldpath,allow_pickle=False)as old:
            for key in old.files:
                if not np.array_equal(old[key],z[key]):raise AssertionError('Native source array changed: '+key)
        return dict(queries=n,targets=t,local_arrays_unchanged=True,
                    source_query_counts={str(s):int((source==s).sum())for s in sorted(set(source))},
                    unique_targets_with_queries=len(target_sources),
                    global_valid_cells=int(gv.sum()),global_joint_counts=np.bincount(label[gv],minlength=6).tolist(),
                    query_center_global_mapping_valid=int(qv[:,4,4].sum()),
                    original_local_cache=str(oldpath),original_local_cache_sha256=base.sha(oldpath))


def comparative_release(report,rule):
    d=report['results']['FROZEN_HEAD']['YCB']['scales']['exact']['confident_joint_MAP_U']
    cm=np.asarray(d['confusion_truth_rows_prediction_columns_U_last'],dtype=np.int64)
    strict=report['evidence_results']['FROZEN_HEAD']['YCB_strict_external_corroborated_5x5']['scales']['exact']['confident_joint_MAP_U']
    now={'E_precision':(int(cm[1,1]),int(cm[:,1].sum())),
         'strict_E_recall_all_GT':(strict['per_class'][1]['correct'],strict['per_class'][1]['support']),
         'O_recall_all_GT':(int(cm[2,2]),int(cm[2].sum())),
         'overall_confidence_coverage':(int(cm[:,:3].sum()),int(cm.sum()))}
    checks={}
    for name,(numerator,denominator)in now.items():
        frozen=rule['criteria'][name];bn,bd=frozen['numerator'],frozen['denominator']
        passed=denominator>0 and bd>0 and numerator*bd>bn*denominator
        checks[name]=dict(numerator=numerator,denominator=denominator,
                          value=numerator/denominator if denominator else None,
                          frozen_numerator=bn,frozen_denominator=bd,frozen_value=bn/bd,
                          strict_ratio_improvement=bool(passed))
    nll_checks={}
    for source in sorted(report['sources']):
        head=report['results']['FROZEN_HEAD'][source]['joint_NLL']
        prior=report['results']['TRAIN_EMPIRICAL_JOINT_PRIOR'][source]['joint_NLL']
        nll_checks[source]=dict(head_joint_NLL=head,TRAIN_prior_joint_NLL=prior,strictly_better=bool(head<prior))
    passed=all(x['strict_ratio_improvement']for x in checks.values())and all(x['strictly_better']for x in nll_checks.values())
    return dict(RELEASE=passed,HOLD=not passed,criteria=checks,domain_NLL_checks=nll_checks,
                role='Comparative engineering release only; no checkpoint/threshold changes and no claim of solved private visibility')


def verify_frozen_run(freeze_path,rule):
    freeze=json.loads(freeze_path.read_text())
    args=freeze['arguments']
    if Path(args['out']).name not in rule['candidate_budget']:
        raise AssertionError('Run is outside the two preregistered candidates')
    expected={'max_updates':2000,'val_every_updates':100,'batch_size':256,'learning_rate':.001,
              'weight_decay':.0001,'dense_loss_weight':.25,'global_loss_weight':.25,'seed':20261022}
    if any(args[k]!=v for k,v in expected.items())or freeze['optimizer_updates']!=2000:
        raise AssertionError('Run changed the fixed source training recipe')
    protocol_sha=next(b['sha256']for b in rule['bindings']if Path(b['path']).name=='PROTOCOL.json')
    if freeze['protocol']['sha256']!=protocol_sha:
        raise AssertionError('Run protocol differs from preregistered release contract')
    for filename,digest in freeze['code_sha256'].items():
        if base.sha(freeze_path.parent/'code'/filename)!=digest:
            raise AssertionError('Frozen training/model snapshot changed')
    if base.sha(freeze_path.parent/'BEST_GLOBAL_AMODAL_HEAD.pt')!=freeze['checkpoint_sha256']:
        raise AssertionError('Checkpoint changed after source freeze')
    return freeze


def normalize_source_binding(path,outputs,cache,freeze,prefix):
    """Accept native v65 helper metadata or the explicit v64-style binding."""
    if path is None:
        return None
    binding=json.loads(path.read_text())
    if 'output_sha256'in binding:
        if binding['optimizer_updates']!=0 or binding['model_or_threshold_selection']or binding['private_data_read']or binding['PnP_calls']:
            raise AssertionError('Source output helper metadata violated frozen replay rules')
        if binding['model_code_sha256']!=freeze['code_sha256']['query_global_amodal_head.py']:
            raise AssertionError('Source output model snapshot mismatch')
        if binding['eval_denylist_sha256']!=freeze['eval_denylist_sha256']:
            raise AssertionError('Source output exclusion manifest mismatch')
        binding['outputs_sha256']=binding['output_sha256']
        # The helper may bind an immutable per-output snapshot in addition to
        # recording its executed source hash. Verify it whenever supplied.
        if 'inference_snapshot'in binding:
            if base.sha(binding['inference_snapshot'])!=binding['inference_code_sha256']:
                raise AssertionError('Source inference helper snapshot changed')
    if binding['outputs_sha256']!=base.sha(outputs)or binding['cache_sha256']!=base.sha(cache):
        raise AssertionError('Source output/cache binding mismatch')
    if binding['checkpoint_sha256']!=freeze['checkpoint_sha256']:
        raise AssertionError('Source output checkpoint binding mismatch')
    normalized=dict(cache_sha256=binding['cache_sha256'],outputs_sha256=binding['outputs_sha256'],
                    checkpoint_sha256=binding['checkpoint_sha256'],original_binding=str(path.resolve()),
                    original_binding_sha256=base.sha(path))
    prefix.parent.mkdir(parents=True,exist_ok=True)
    destination=Path(str(prefix)+'_SOURCE_OUTPUT_BINDING.json')
    destination.write_text(json.dumps(normalized,indent=2)+'\n')
    return destination


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['train-cache','val-cache','outputs','freeze','prefix']:ap.add_argument('--'+name,required=True,type=Path)
    ap.add_argument('--output-cache-binding',type=Path)
    ap.add_argument('--release-rule',type=Path,default=HERE/'RELEASE_RULE_BEFORE_FIT.json')
    a=ap.parse_args()
    rule=json.loads(a.release_rule.read_text())
    for binding in rule['bindings']:
        if base.sha(binding['path'])!=binding['sha256']:raise AssertionError('Registered release binding changed: '+binding['path'])
    for name,binding in json.loads((HERE/'V64_PURE_EVALUATOR_COPY_BINDINGS.json').read_text()).items():
        copied=HERE/('saved_'+name if name.startswith('evaluate')else name)
        if base.sha(copied)!=binding['sha256']:raise AssertionError('Pure v64 evaluator copy changed')
    freeze=verify_frozen_run(a.freeze,rule)
    output_binding=normalize_source_binding(a.output_cache_binding,a.outputs,a.val_cache,freeze,a.prefix)
    global_audit={split:validate_global_cache(path)for split,path in [('TRAIN',a.train_cache),('VALIDATION',a.val_cache)]}
    evidence=V64/'source_audit/YCB_VALIDATION_QUERY_EVIDENCE.csv'
    queries=[V64/'source_cache'/f'{source}_VALIDATION_QUERIES.csv'for source in ['SYNTH','YCB']]
    command=[sys.executable,str(HERE/'saved_evaluate_frozen_source.py'),'--train-cache',str(a.train_cache),
             '--val-cache',str(a.val_cache),'--outputs',str(a.outputs),'--freeze',str(a.freeze),
             '--prefix',str(a.prefix),'--inputs',str(V63/'sam_source/INPUTS.jsonl'),'--evidence',str(evidence)]
    for path in queries:command+=['--query-csv',str(path)]
    if output_binding:command+=['--output-cache-binding',str(output_binding)]
    subprocess.run(command,check=True)
    report=json.loads(a.prefix.with_suffix('.json').read_text())
    if report['n']!=4942 or report['sources']!=dict(SYNTH=1742,YCB=3200):raise AssertionError('All4942samequery source output required')
    cache=base.load_cache(a.val_cache);index,y,p=base.validate_outputs(a.outputs,cache)
    with np.load(a.outputs,allow_pickle=False)as z:
        for name in ['sample_id','image_sha256']:
            if name not in z.files or not np.array_equal(z[name],cache[name][index]):raise AssertionError('Saved output query identity mismatch')
        output_target_index=z['target_index']
    with np.load(a.val_cache,allow_pickle=False)as z:
        sample=z['sample_id'][index]
        if not np.array_equal(output_target_index,z['target_index'][index]):raise AssertionError('Saved output target identity mismatch')
    ev={row['sample_id']:row for row in base.rows(evidence)}
    with np.load(a.train_cache,allow_pickle=False)as z:train_has_YCB=bool((z['source']=='YCB').any())
    training_objects=set(r['object_id']for r in base.rows(V64/'source_audit/YCB_TRAIN_QUERY_EVIDENCE.csv'))
    object_ids=np.array([ev[str(s)]['object_id']if str(s)in ev else''for s in sample])
    objects={}
    for obj in sorted(set(object_ids)-{''},key=int):
        selected=object_ids==obj
        objects[obj]=dict(n=int(selected.sum()),object_in_public_YCB_TRAIN=obj in training_objects,
                          object_in_this_run_YCB_TRAIN=train_has_YCB and obj in training_objects,
                          metrics=base.summarize(y[selected],p[selected]))
    report['v65_global_cache_audit']=global_audit
    report['YCB_per_object']=objects
    report['comparative_release_vs_frozen_v64']=comparative_release(report,rule)
    report['release_rule_sha256']=base.sha(a.release_rule)
    report['v65_wrapper_sha256']=base.sha(__file__)
    report['source_comparison_selection_caution']='Report both preregistered fits; no private evaluation, retries, decoder changes or source outcome based thresholds.'
    a.prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    a.prefix.parent.mkdir(exist_ok=True,parents=True)
    Path(str(a.prefix)+'_RELEASE.json').write_text(json.dumps(report['comparative_release_vs_frozen_v64'],indent=2)+'\n')
    records=[]
    for obj,value in objects.items():
        for scale,scales in value['metrics']['scales'].items():
            for decoder in ['logical_support_diagnostic','joint_MAP_diagnostic','confident_joint_MAP_U']:
                metric=scales[decoder]
                for cls in metric['per_class']:
                    records.append(dict(object_id=obj,object_in_public_YCB_TRAIN=value['object_in_public_YCB_TRAIN'],
                                        object_in_this_run_YCB_TRAIN=value['object_in_this_run_YCB_TRAIN'],
                                        n=value['n'],scale=scale,decoder=decoder,
                                        class_name=base.CLASS_NAMES[cls['class_index']],coverage=metric['coverage'],**cls))
    base.write_csv(str(a.prefix)+'_YCB_PER_OBJECT.csv',records)
    print(json.dumps(dict(output=str(a.prefix.with_suffix('.json')),release=report['comparative_release_vs_frozen_v64'])))


if __name__=='__main__':main()
