#!/usr/bin/env python3
"""Audit the fixed V67 admission gates from sealed saved source outputs only.

No torch/model imports, checkpoint deserialization, input-feature/RGB reads,
inference, optimizer, or checkpoint selection. A PASS only establishes source
eligibility; a separate immutable root selection/evaluation gate is required.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V66 = ROOT.parent / 'query_visibility_controlled_real_20261022_v66'
PROTOCOL_SHA = '6ea43e9e55e69b442285a2c99e0960a11a491257c376a0264db9c212cf6c3980'
EVALUATOR_SHA = 'ebd7c52eae3d09f36c84488980a0a0a27522c30b7dec48998c878c9f378a1efe'
SOURCES = ('SYNTH', 'ORIGINAL_YCB', 'CONTROLLED_YCB', 'PALLOX')
PAIRS = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))
NAMES = ('V', 'E', 'O', 'U')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound(binding, expected_path=None):
    p = Path(binding['path']).resolve()
    if expected_path is not None:
        require(p == Path(expected_path).resolve(), 'Unexpected binding path: ' + str(p))
    require(sha(p) == binding['sha256'], 'SHA binding mismatch: ' + str(p))
    return p


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def boolean(value):
    if value in (True, 'True', 'true', '1', 1):
        return True
    if value in (False, 'False', 'false', '0', 0):
        return False
    raise ValueError('Expected explicit bool, got ' + repr(value))


def fraction(a, b):
    return a / b if b else None


def known_state(allowed, bits):
    possible = [i for i, bitset in enumerate(bits) if allowed & bitset]
    return possible[0] if len(possible) == 1 else -1


def parse_rows(raw):
    """Validate native truth and the frozen confidence decoder without a model."""
    rows = []
    for saved in raw:
        r = dict(saved)
        r['native_valid'] = boolean(r['native_valid'])
        r['informative'] = boolean(r['informative'])
        for key in ('allowed', 'exact_GT', 'radius2_GT', 'joint_MAP'):
            r[key] = int(r[key])
        for key in ('partial_NLL', 'joint_max'):
            r[key] = float(r[key])
            require(math.isfinite(r[key]), 'Nonfinite saved ' + key)
        a, valid = r['allowed'], r['native_valid']
        require(1 <= a <= 63, 'Allowed mask outside 1..63')
        require(r['source'] in SOURCES, 'Unregistered source')
        require(bool(r['sample_id']) and bool(r['group_id']), 'Missing source identity')
        require(r['informative'] == (valid and a != 63), 'Informative flag mismatch')
        require(valid or a == 63, 'Native-invalid row must have allowed63')
        require(0 <= r['joint_MAP'] < 6 and 0 <= r['joint_max'] <= 1, 'Saved joint range')
        require(r['partial_NLL'] >= 0, 'Negative partial NLL')
        require(a != 63 or r['partial_NLL'] == 0, 'Allowed63 must have exactly zero NLL')
        for k, (support, bits) in enumerate((('exact', (1, 6, 56)), ('radius2', (11, 20, 32)))):
            gt = r[support + '_GT']
            expected = known_state(a, bits) if valid else -1
            require(gt == expected, 'Known truth does not match native compatible set: ' + r['sample_id'])
            expected_pred = NAMES[PAIRS[r['joint_MAP']][k]] if r['joint_max'] >= .5 else 'U'
            require(r[support + '_predicted'] == expected_pred, 'Saved confidence decoder mismatch')
            ps = []
            for cn in ('V', 'E', 'O'):
                key = support + '_p' + cn
                r[key] = float(r[key])
                require(math.isfinite(r[key]) and 0 <= r[key] <= 1, 'Saved marginal probability range')
                ps.append(r[key])
            # Numerical validation only; no tolerance ever changes a hard decision/gate.
            require(abs(sum(ps) - 1) <= 1e-12, 'Saved marginal simplex mismatch')
        require(r['exact_pV'] <= r['radius2_pV'] + 1e-15, 'Visible support monotonicity')
        rows.append(r)
    ids = [r['sample_id'] for r in rows]
    require(len(ids) == len(set(ids)), 'Duplicate saved sample ID')
    require(set(r['source'] for r in rows) == set(SOURCES), 'Four fixed source domains required')
    return rows


def metrics_from_rows(rows):
    out = {}
    for source in SOURCES:
        all_rows = [r for r in rows if r['source'] == source]
        valid = [r for r in all_rows if r['native_valid']]
        known = [r for r in valid if r['exact_GT'] >= 0]
        unknown = [r for r in valid if r['exact_GT'] < 0]
        tp = sum(r['exact_GT'] == 0 and r['exact_pV'] >= .5 for r in known)
        fp = sum(r['exact_GT'] != 0 and r['exact_pV'] >= .5 for r in known)
        fn = sum(r['exact_GT'] == 0 and r['exact_pV'] < .5 for r in known)
        tn = sum(r['exact_GT'] != 0 and r['exact_pV'] < .5 for r in known)
        etp = sum(r['exact_GT'] == 1 and r['exact_predicted'] == 'E' for r in known)
        eden = sum(r['exact_predicted'] == 'E' for r in known)
        esupport = sum(r['exact_GT'] == 1 for r in known)
        out[source] = dict(total_rows=len(all_rows), native_valid=len(valid), known_GT=len(known),
            UNKNOWN_GT=len(unknown), native_invalid=len(all_rows)-len(valid),
            uninformative_allowed63_native_valid=sum(r['allowed'] == 63 for r in valid),
            informative_native_valid=sum(r['allowed'] != 63 for r in valid),
            visible=dict(TP=tp, FP=fp, FN=fn, TN=tn, N=len(known), precision=fraction(tp,tp+fp), recall=fraction(tp,tp+fn)),
            external=dict(TP=etp, predicted_on_known_GT=eden, GT_support=esupport,
                          precision=fraction(etp,eden), recall=fraction(etp,esupport)),
            abstention_on_all_native_valid=sum(r['exact_predicted'] == 'U' for r in valid),
            predictions_on_UNKNOWN=dict(Counter(r['exact_predicted'] for r in unknown)),
            binary_visible_on_UNKNOWN=sum(r['exact_pV'] >= .5 for r in unknown))
    return out


def verify_report_metrics(metrics, calculated):
    for source in SOURCES:
        saved = metrics['per_source'][source]['exact']
        actual = calculated[source]
        for key in ('total_rows', 'native_valid', 'native_invalid', 'known_GT', 'UNKNOWN_GT',
                    'uninformative_allowed63_native_valid', 'informative_native_valid'):
            require(saved[key] == actual[key], 'SOURCE_METRICS count mismatch: ' + source + '/' + key)
        for key, value in actual['visible'].items():
            require(saved['visible_binary_pV_ge_0_5'][key] == value, 'Visible saved metric mismatch: ' + source + '/' + key)
        for key, value in actual['external'].items():
            require(saved['class_metrics']['E'][key] == value, 'External saved metric mismatch: ' + source + '/' + key)
        require(saved['abstention']['on_all_native_valid'] == actual['abstention_on_all_native_valid'], 'Saved abstention mismatch')


def crosscheck_npz(path, rows, deny):
    """Only saved output arrays, never source input tensors or a checkpoint load."""
    keys = ('sample_id','source','group_id','image_sha256','center_native_valid','center_allowed',
            'GT_state','joint_logits','joint_probability','state_probability','predicted_state','partial_NLL')
    with np.load(path, allow_pickle=False) as z:
        require(all(k in z.files for k in keys), 'Incomplete sealed saved outputs')
        d = {k:z[k] for k in keys}
    n = len(rows)
    require(d['joint_probability'].shape == (n,6) and d['GT_state'].shape == (n,2), 'Saved output shape')
    require(d['state_probability'].shape == (n,2,3), 'Saved marginal shape')
    require(not(set(d['image_sha256'].astype(str)) & deny), 'Evaluation image SHA in source output')
    p = d['joint_probability']
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Saved joint probability range')
    require(np.all(np.abs(p.sum(1)-1) <= 1e-12), 'Saved joint simplex')
    z = d['joint_logits'].astype(np.float64)
    require(z.shape == (n,6) and np.isfinite(z).all(), 'Saved logit shape/range')
    e = np.exp(z-z.max(1,keepdims=True)); softmax=e/e.sum(1,keepdims=True)
    require(np.array_equal(p,softmax), 'Saved joint probability does not replay saved logits')
    marginal = np.stack((np.stack((p[:,0],p[:,1]+p[:,2],p[:,3]+p[:,4]+p[:,5]),1),
                         np.stack((p[:,0]+p[:,1]+p[:,3],p[:,2]+p[:,4],p[:,5]),1)),1)
    require(np.array_equal(d['state_probability'],marginal), 'Saved marginal differs from six-state sums')
    allowed=d['center_allowed'].astype(np.uint8)
    permit=(allowed[:,None] & (1<<np.arange(6))[None]) != 0
    nll=np.logaddexp.reduce(z,axis=1)-np.logaddexp.reduce(np.where(permit,z,-np.inf),axis=1)
    nll[allowed == 63]=0.
    require(np.array_equal(nll,d['partial_NLL']), 'Saved compatible-mass NLL does not replay saved logits/truth')
    for i,r in enumerate(rows):
        for key in ('sample_id','source','group_id'):
            require(str(d[key][i]) == r[key], 'CSV/NPZ identity ordering mismatch')
        require(bool(d['center_native_valid'][i]) == r['native_valid'] and int(d['center_allowed'][i]) == r['allowed'], 'CSV/NPZ native truth mismatch')
        require(float(p[i].max()) == r['joint_max'] and int(p[i].argmax()) == r['joint_MAP'], 'CSV/NPZ confidence mismatch')
        require(float(d['partial_NLL'][i]) == r['partial_NLL'], 'CSV/NPZ NLL mismatch')
        for k,support in enumerate(('exact','radius2')):
            require(int(d['GT_state'][i,k]) == r[support+'_GT'], 'CSV/NPZ GT mismatch')
            require(NAMES[int(d['predicted_state'][i,k])] == r[support+'_predicted'], 'CSV/NPZ hard state mismatch')
            for c,cn in enumerate(('V','E','O')):
                require(float(marginal[i,k,c]) == r[support+'_p'+cn], 'CSV/NPZ marginal mismatch')


def strict_cohorts(reference, rows, spec):
    """The frozen V65 reference fixes IDs; the new model cannot alter the cohort."""
    lookup = {r['sample_id']:r for r in rows}
    result = {}
    for source, gate in spec.items():
        original = [r for r in reference if r['model'] == 'V65_BASELINE' and r['source'] == source]
        require(len(original) == 3200, 'Frozen source reference must retain all 3200 queries: '+source)
        require(len({r['sample_id'] for r in original}) == len(original), 'Duplicate reference ID')
        source_ids = {r['sample_id'] for r in rows if r['source'] == source}
        require(source_ids == {r['sample_id'] for r in original}, 'Changed source reference ID universe: '+source)
        selected=[]
        for old in original:
            new=lookup[old['sample_id']]
            require(new['source'] == source, 'Strict reference source changed')
            require(new['native_valid'] == boolean(old['native_valid']) and new['allowed'] == int(old['allowed'])
                    and new['exact_GT'] == int(old['exact_GT']) and new['radius2_GT'] == int(old['radius2_GT']),
                    'Same-ID native validity/partial truth changed: '+old['sample_id'])
            if boolean(old['strict25_V']):
                require(new['native_valid'] and new['exact_GT'] == 0 and new['allowed'] == 1, 'Strict25 requires known native-visible VV truth')
                selected.append((old,new))
        require(len(selected) == gate['N'], 'Strict25 cohort size changed: '+source)
        baseline_tp=sum(float(old['exact_pV']) >= .5 for old,_ in selected)
        require(baseline_tp == gate['TP_at_least'], 'Frozen V65 baseline anchor differs from protocol')
        tp=sum(new['exact_pV'] >= .5 for _,new in selected)
        digest=hashlib.sha256(('\n'.join(sorted(new['sample_id'] for _,new in selected))+'\n').encode()).hexdigest()
        result[source]=dict(N=len(selected),TP=tp,FN=len(selected)-tp,recall=fraction(tp,len(selected)),
            frozen_V65_TP=baseline_tp,sorted_sample_ids_sha256=digest,
            native_valid_known_V_allowed1_checked=True,all3200_same_ID_truth_parity=True)
    return result


def quality_checks(spec, calculated, strict):
    checks=[]
    def add(name,source,metric,actual,minimum,details):
        checks.append(dict(condition=name,source=source,metric=metric,value=actual,minimum=minimum,
                           passed=actual is not None and actual >= minimum,mandatory=True,details=details))
    for source in spec['public_exact_visible']['sources']:
        for metric in ('precision','recall'):
            add('public_exact_visible',source,metric,calculated[source]['visible'][metric],
                spec['public_exact_visible'][metric+'_at_least'],calculated[source]['visible'])
    for source in spec['public_external_confident_joint_MAP']['sources']:
        for metric in ('precision','recall'):
            add('public_external_confident_joint_MAP',source,metric,calculated[source]['external'][metric],
                spec['public_external_confident_joint_MAP'][metric+'_at_least'],calculated[source]['external'])
    for source,rule in spec['strict25_public_visible_nonregression_vs_frozenV65'].items():
        add('strict25_public_visible_nonregression_vs_frozenV65',source,'TP',strict[source]['TP'],rule['TP_at_least'],strict[source])
    for key,kind in (('SYNTH_visible_nonregression_vs_frozenV65','visible'),('SYNTH_external_nonregression_vs_frozenV65','external')):
        for metric in ('precision','recall'):
            add(key,'SYNTH',metric,calculated['SYNTH'][kind][metric],spec[key][metric+'_at_least'],calculated['SYNTH'][kind])
    require(len(checks) == 16, 'Every one of the sixteen frozen conditions must be reported')
    return checks


def load_provenance(metrics_path):
    protocol_path=ROOT/'FIT_PROTOCOL.json'
    require(sha(protocol_path) == PROTOCOL_SHA, 'Frozen V67 protocol changed')
    protocol=read_json(protocol_path)
    require(protocol['thresholds']['visible'] == .5 and protocol['thresholds']['confident_joint_MAP'] == .5, 'Fixed thresholds changed')
    metrics=read_json(metrics_path)
    for key,value in dict(completed=True,source_only=True,current223_RGB_GT_scores_read=False,
                           optimizer_updates=0,new_PnP_calls=0,thresholds_fitted=False,model_or_checkpoint_selected=False).items():
        require(metrics.get(key) is value if isinstance(value,bool) else metrics.get(key) == value, 'Invalid saved source report flag: '+key)
    require(metrics['sources'] == list(SOURCES), 'Source report must retain fixed four domains')
    require(metrics['numerical_threshold_notes']['thresholds_fixed'] == dict(binary_visible=.5,joint_MAP_confident=.5), 'Source output threshold drift')
    b=metrics['bindings']
    freeze_path=bound(b['model_seal'],ROOT/'runs/VISIBLE_BCE1/MODEL_FREEZE.json')
    freeze=read_json(freeze_path)
    for key,value in dict(completed=True,source_only=True,current223_used=False,eval_data_used=False,PnP_used=False).items():
        require(freeze.get(key) is value, 'Invalid frozen model provenance flag: '+key)
    require(freeze['parameters'] == 95508 and freeze['optimizer_updates'] == protocol['updates'], 'Frozen model architecture/budget changed')
    require(freeze['threshold'] == .5 and freeze['visible_bce_weight'] == 1 and freeze['protocol_sha256'] == PROTOCOL_SHA, 'Frozen objective/protocol drift')
    require(sha(ROOT/'runs/VISIBLE_BCE1/PROTOCOL_AT_START.json') == PROTOCOL_SHA, 'Fit-start protocol snapshot changed')
    require(freeze['fitting_sources'] == sorted(SOURCES) and freeze['validation_sources'] == sorted(SOURCES), 'Frozen source domains changed')
    args=freeze['arguments']
    require(args['candidate'] == 'VISIBLE_BCE1' and args['seed'] == protocol['seed'], 'Run candidate/seed drift')
    require(args['updates'] == protocol['updates'] and args['val_every'] == protocol['validation_every'], 'Run budget drift')
    # These are literals in the SHA-bound trainer, not user-configurable CLI fields.
    require(protocol['batch_size'] == 256 and protocol['optimizer'] == dict(type='AdamW',lr=.001,weight_decay=.0001), 'Frozen sampler/optimizer drift')
    require(args['visible_bce_weight'] == 1., 'Run objective coefficient drift')
    checkpoint_path=bound(b['checkpoint'],ROOT/'runs/VISIBLE_BCE1/BEST_PARTIAL_HEAD.pt')
    require(b['checkpoint']['sha256'] == freeze['checkpoint_sha256'], 'Checkpoint does not match model freeze')
    cm=metrics['checkpoint_metadata']
    require(cm['updates'] == freeze['best_update'] and cm['selection_partial_NLL'] == freeze['best_source_partial_NLL'] and cm['threshold'] == .5,
            'Source report is not the unchanged-NLL-selected frozen checkpoint')
    history_path=ROOT/'runs/VISIBLE_BCE1/HISTORY.json';history=read_json(history_path)
    require([r['update'] for r in history] == list(range(protocol['validation_every'],protocol['updates']+1,protocol['validation_every'])),
            'Complete fixed-budget validation history required')
    best=float('inf');best_update=None
    for row in history:
        value=row['VAL_partial_NLL'];require(math.isfinite(value),'Nonfinite NLL history')
        improved=value < best
        require(row['saved_best'] is improved,'History strict minimum marker mismatch')
        if improved:best,best_update=value,row['update']
    require(best == freeze['best_source_partial_NLL'] and best_update == freeze['best_update'],
            'Admission must use the earliest strict minimum NLL checkpoint, not a quality-gate-selected step')
    code_bindings={Path(x['path']).name:x for x in protocol['bindings'] if Path(x['path']).parent == ROOT/'model'}
    for name in ('fit_partial_head.py','query_global_amodal_head.py'):
        require(freeze['code'][name] == code_bindings[name]['sha256'], 'Frozen model/trainer code mismatch')
        bound(code_bindings[name],ROOT/'model'/name)
        require(sha(ROOT/'runs/VISIBLE_BCE1/code'/name) == freeze['code'][name], 'Frozen run code snapshot mismatch')
    bound(b['model_code'],ROOT/'model/query_global_amodal_head.py')
    require(b['model_code']['sha256'] == freeze['code']['query_global_amodal_head.py'], 'Evaluator used different model code')
    lock_path=ROOT/'provenance/FIT_INPUT_LOCK.json'
    require(Path(args['input_lock']).resolve() == lock_path and sha(lock_path) == freeze['input_lock_sha256'], 'Fit input lock drift')
    lock=read_json(lock_path);run=lock['runs']['VISIBLE_BCE1']
    require(run['protocol']['sha256'] == PROTOCOL_SHA, 'Input lock protocol drift')
    require(lock['source_only'] is True and lock['current223_RGB_annotations_scores_inputs_used'] is False, 'Input lock source-only condition')
    # Compare sealed cache digests as metadata. Never open or hash source input tensors.
    for key in ('train','val'):
        require(freeze[key+'_sha256'] == run[key]['sha256'] and Path(args[key]).resolve() == Path(run[key]['path']).resolve(), 'Frozen cache provenance drift')
    require(b['cache']['sha256'] == run['val']['sha256'] and Path(b['cache']['path']).resolve() == Path(run['val']['path']).resolve(), 'Report uses a different validation cache')
    cache_seal_path=bound(b['cache_seal']);cache_seal=read_json(cache_seal_path)
    require(cache_seal.get('complete',cache_seal.get('completed')) is True, 'Incomplete joined source cache seal')
    def absolute_cache_path(p):
        return Path(p).resolve() if Path(p).is_absolute() else (V66/Path(p)).resolve()
    hits=[entry for entry in cache_seal['outputs']
          if absolute_cache_path(entry['path']) == Path(run['val']['path']).resolve()
          and entry['sha256'] == run['val']['sha256'] and entry['split'] == 'VAL']
    require(len(hits) == 1, 'Input lock VAL digest/path not bound exactly once by joined source seal')
    require(b['denylist'] == run['denylist'], 'Evaluation exclusion list drift')
    deny=read_json(bound(b['denylist']))
    gate_path=bound(b['evaluation_gate']);gate=read_json(gate_path)
    require(gate['source_only'] is True and gate['validation_sources'] == list(SOURCES), 'Invalid source evaluation gate')
    require(gate['thresholds'] == dict(binary_visible=.5,confident_joint_MAP=.5), 'Source evaluation gate threshold drift')
    require(Path(gate['output_directory']).resolve() == Path(metrics_path).resolve().parent and gate['name'] == metrics['name'], 'Source evaluation output registration mismatch')
    evaluator=bound(b['evaluator'],V66/'contract/evaluate_partial_source.py')
    require(b['evaluator']['sha256'] == EVALUATOR_SHA and gate['evaluator'] == b['evaluator'], 'Unchanged source evaluator is required')
    for key in ('cache','cache_seal','checkpoint','model_seal','model_code','denylist'):
        require(gate['inputs'][key] == b[key], 'Source evaluation input/gate mismatch: '+key)
    for binding in gate['bindings']:
        # These are source-only metadata/code admission bindings, not model inputs.
        require(Path(binding['path']).suffix in ('.json','.py'), 'Auxiliary source gate may bind only metadata/code, never input feature/RGB reads')
        bound(binding)
    files={}
    for binding in metrics['outputs']:
        p=Path(binding['path']).resolve()
        require(p.parent == Path(metrics_path).resolve().parent, 'Saved source output escapes registered directory')
        require(p.name in ('SOURCE_OUTPUTS.npz','PER_QUERY.csv','PER_SOURCE_METRICS.csv','PALLOX_PAIR_COUNTERFACTUAL.csv'), 'Unregistered source output name')
        require(p.name not in files, 'Duplicate saved output binding')
        files[p.name]=bound(binding)
    require('PER_QUERY.csv' in files and 'SOURCE_OUTPUTS.npz' in files, 'Sealed source CSV and saved probability output required')
    bindings=dict(protocol=ref(protocol_path),source_metrics=ref(metrics_path),model_freeze=ref(freeze_path),
                  checkpoint=ref(checkpoint_path),fit_selection_history=ref(history_path),input_lock=ref(lock_path),source_evaluator=ref(evaluator),
                  evaluation_gate=ref(gate_path),cache_seal=ref(cache_seal_path),admission_code=ref(__file__),saved_source_outputs=metrics['outputs'],
                  fit_train_cache_metadata=run['train'],fit_VAL_cache_metadata=run['val'])
    return protocol,metrics,freeze,files,set(deny['image_sha256']),bindings


def evaluate(metrics_path):
    protocol,metrics,freeze,files,deny,bindings=load_provenance(metrics_path)
    rows=parse_rows(read_csv(files['PER_QUERY.csv']))
    require(len(rows) == metrics['N'] == 8590, 'Common frozen 8590-row source VAL required')
    calculated=metrics_from_rows(rows)
    verify_report_metrics(metrics,calculated)
    crosscheck_npz(files['SOURCE_OUTPUTS.npz'],rows,deny)
    # The checkpoint's selected CUDA statistic and CPU replay can round differently.
    # Validate saved CPU statistic against its own outputs, never use it to reselect.
    nll={s:float(np.mean([r['partial_NLL'] for r in rows if r['source'] == s and r['informative']])) for s in SOURCES}
    for s in SOURCES:
        require(nll[s] == metrics['fixed_selection_statistic']['by_source'][s]['partial_NLL'], 'Source NLL report mismatch')
    require(float(np.mean(list(nll.values()))) == metrics['fixed_selection_statistic']['common4domain_mean_informative_center_partial_NLL'], 'Macro source NLL report mismatch')
    diagnosis=next(b for b in protocol['bindings'] if Path(b['path']).name == 'SOURCE_BOUNDARY_FAILURE_DIAGNOSIS.json')
    diagnosis_path=bound(diagnosis);d=read_json(diagnosis_path)
    require(d['completed'] is True and d['current_RGB_GT_scores_inputs_read'] == 0 and d['new_model_inference_training_PnP_GPU_calls'] == 0, 'Source-only strict reference audit required')
    reference_binding=next(b for b in d['output_bindings'] if Path(b['path']).name == 'SOURCE_BOUNDARY_FAILURE_DIAGNOSIS_QUERY_ROWS.csv')
    reference_path=bound(reference_binding)
    spec=protocol['evaluation']['source_quality_admission']
    strict=strict_cohorts(read_csv(reference_path),rows,spec['strict25_public_visible_nonregression_vs_frozenV65'])
    checks=quality_checks(spec,calculated,strict)
    bindings['strict25_reference_diagnosis']=ref(diagnosis_path)
    bindings['strict25_reference_saved_rows']=ref(reference_path)
    return dict(completed=True,integrity_PASS=True,status='PASS' if all(c['passed'] for c in checks) else 'HOLD',
        source_eligible=all(c['passed'] for c in checks),conditions=checks,per_source_exact=calculated,strict25=strict,
        unchanged_NLL_selected_checkpoint=dict(best_update=freeze['best_update'],best_source_partial_NLL=freeze['best_source_partial_NLL'],
            checkpoint_sha256=freeze['checkpoint_sha256']),bindings=bindings,
        gate_specification=spec,conditional_precision_scope='Native-valid known GT only. UNKNOWN remains separate coverage, never a negative.',
        source_evidence_caveat='Controlled causal cutout facts and PALLOX conditional teacher facts are separate evidence tiers; neither proves semantic pallet transfer.')


def write_result(out,report):
    out=Path(out)
    require(not out.exists(), 'Preserve previous admission output directory')
    out.mkdir(parents=True)
    report.update(schema='V67_FIXED_SOURCE_QUALITY_ADMISSION_SAVED_ONLY_V1',created_unix=time.time(),
        current_inference_allowed=False,separate_root_selection_and_current_gate_required=True,
        solved=False,source_only=True,current223_RGB_GT_scores_inputs_read=False,
        new_model_inference_calls=0,model_forward_calls=0,optimizer_updates=0,PnP_calls=0,
        input_feature_RGB_reads=0,checkpoint_deserialization_calls=0,
        thresholds_changed=False,checkpoint_reselected=False,
        interpretation='PASS is source eligibility only. HOLD forbids current inference. No checkpoint/threshold/radius change or alternate-step reselection is performed.')
    columns=('condition','source','metric','value','minimum','passed','mandatory','details')
    csv_path=out/'SOURCE_ADMISSION.csv'
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for row in report.get('conditions',[]):
            r=dict(row);r['details']=json.dumps(r['details'],sort_keys=True,allow_nan=False);w.writerow(r)
    snapshot=out/'evaluate_source_admission.py';snapshot.write_bytes(Path(__file__).read_bytes())
    report['outputs']=[ref(csv_path),ref(snapshot)]
    path=out/'SOURCE_ADMISSION.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],source_eligible=report['source_eligible'],current_inference_allowed=False,
                          passed=sum(c['passed'] for c in report.get('conditions',[])),conditions=len(report.get('conditions',[])),
                          report=str(path),sha256=sha(path))),flush=True)


def mock(out):
    """Generated scalar probabilities/counts only. No model/cache/weight reads."""
    spec=dict(public_exact_visible=dict(sources=list(SOURCES[1:]),precision_at_least=.8,recall_at_least=.8),
        public_external_confident_joint_MAP=dict(sources=['CONTROLLED_YCB','PALLOX'],precision_at_least=.8,recall_at_least=.6),
        strict25_public_visible_nonregression_vs_frozenV65=dict(ORIGINAL_YCB=dict(TP_at_least=679,N=714),CONTROLLED_YCB=dict(TP_at_least=278,N=296)),
        SYNTH_visible_nonregression_vs_frozenV65=dict(precision_at_least=.8113821138211382,recall_at_least=.7118402282453637),
        SYNTH_external_nonregression_vs_frozenV65=dict(precision_at_least=.573170731707317,recall_at_least=.5053763440860215))
    def row(source,i,gt=0,allowed=1,pv=.8,jmap=0,jmax=.8):
        return dict(source=source,sample_id=source+':MOCK:'+str(i),group_id=source+':MOCK',native_valid=True,
            informative=allowed != 63,allowed=allowed,exact_GT=gt,radius2_GT=gt,partial_NLL=0 if allowed == 63 else .2,
            joint_MAP=jmap,joint_max=jmax,exact_predicted=NAMES[PAIRS[jmap][0]] if jmax>=.5 else 'U',
            radius2_predicted=NAMES[PAIRS[jmap][1]] if jmax>=.5 else 'U',exact_pV=pv,exact_pE=(1-pv)/2,exact_pO=(1-pv)/2,
            radius2_pV=pv,radius2_pE=(1-pv)/2,radius2_pO=(1-pv)/2)
    generated=[];reference=[]
    for s in SOURCES:
        nv=714 if s == 'ORIGINAL_YCB' else 296 if s == 'CONTROLLED_YCB' else 20
        total=3200 if s in SOURCES[1:3] else 40
        for i in range(total):
            r=row(s,i) if i<nv else row(s,i,gt=1,allowed=4,pv=.05,jmap=2,jmax=.8)
            generated.append(r)
            if s in SOURCES[1:3]:
                reference.append(dict(model='V65_BASELINE',source=s,sample_id=r['sample_id'],native_valid=True,
                    allowed=r['allowed'],exact_GT=r['exact_GT'],radius2_GT=r['radius2_GT'],strict25_V=i<nv,
                    exact_pV=.8 if i < spec['strict25_public_visible_nonregression_vs_frozenV65'][s]['TP_at_least'] else .1))
    rows=parse_rows(generated)
    calculated=metrics_from_rows(rows)
    strict=strict_cohorts(reference,rows,spec['strict25_public_visible_nonregression_vs_frozenV65'])
    checks=quality_checks(spec,calculated,strict);require(all(c['passed'] for c in checks),'Generated all-pass case')
    # Every threshold separately fails just below its exact bound, equality passes.
    for c in checks:
        value=c['minimum'];require(value >= c['minimum'],'Equality must pass')
        below=math.nextafter(float(value),-math.inf)
        require(not(below >= c['minimum']),'Just below gate must fail')
        changed_metrics=copy.deepcopy(calculated);changed_strict=copy.deepcopy(strict)
        if c['metric'] == 'TP':changed_strict[c['source']]['TP']=c['minimum']-1
        else:
            kind='external' if 'external' in c['condition'] else 'visible'
            changed_metrics[c['source']][kind][c['metric']]=below
        changed=quality_checks(spec,changed_metrics,changed_strict)
        failed=[x for x in changed if not x['passed']]
        require(len(failed) == 1 and failed[0]['condition'] == c['condition'] and failed[0]['source'] == c['source']
                and failed[0]['metric'] == c['metric'],'Each mandatory condition must independently force HOLD')
        if c['metric'] != 'TP':
            changed_metrics[c['source']][kind][c['metric']]=None
            require(any(not x['passed'] for x in quality_checks(spec,changed_metrics,changed_strict)), 'Undefined precision/recall must fail closed')
    with_unknown=rows+[row('PALLOX','UNKNOWN',gt=-1,allowed=63,pv=.99)]
    before=calculated['PALLOX'];after=metrics_from_rows(with_unknown)['PALLOX']
    require(before['visible'] == after['visible'] and before['external'] == after['external'], 'UNKNOWN must not enter precision/recall denominators')
    require(after['UNKNOWN_GT'] == 1 and after['binary_visible_on_UNKNOWN'] == 1,'UNKNOWN coverage must remain visible')
    for corruption in ('native','known','ID'):
        changed=[dict(r) for r in rows]
        victim=next(r for r in changed if r['source'] == 'ORIGINAL_YCB')
        if corruption == 'native':victim['native_valid']=False
        elif corruption == 'known':victim['allowed']=63
        else:victim['sample_id']='CHANGED_ID'
        try:strict_cohorts(reference,changed,spec['strict25_public_visible_nonregression_vs_frozenV65'])
        except ValueError:pass
        else:raise AssertionError('Corrupt strict reference escaped: '+corruption)
    report=dict(completed=True,PASS=True,generated_only=True,admission_code=ref(__file__),
        fixed_16_conditions_PASS=True,exact_threshold_equality_and_just_below_PASS=True,
        each_individual_mandatory_failure_forces_HOLD_PASS=True,undefined_precision_or_recall_HOLD_PASS=True,
        strict_same_ID_native_valid_known_V_guard_PASS=True,UNKNOWN_excluded_from_negative_denominators_PASS=True,
        all_pass_fixture_conditions=checks,fixture_source_rows=len(rows),source_checkpoint_cache_RGB_reads=0,
        source_quality_data_or_scores_read=0,model_forward_calls=0,optimizer_updates=0,PnP_calls=0)
    path=Path(out);require(not path.exists(),'Preserve previous mock report')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(PASS=True,report=str(path),sha256=sha(path))),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-metrics',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--mock',type=Path,help='Write generated-only math/guard smoke JSON; no source files/models are read')
    args=ap.parse_args()
    if args.mock:
        require(args.source_metrics is None and args.out is None,'Mock cannot be mixed with actual source inputs')
        mock(args.mock);return
    if args.source_metrics is None or args.out is None:
        ap.error('--source-metrics and --out are required')
    require(not args.out.exists(),'Preserve previous admission output')
    try:
        report=evaluate(args.source_metrics)
    except Exception as exc:
        report=dict(completed=False,integrity_PASS=False,status='HOLD',source_eligible=False,
            conditions=[],integrity_error=dict(type=type(exc).__name__,message=str(exc)),
            requested_source_metrics=str(args.source_metrics.resolve()),admission_code=ref(__file__))
    write_result(args.out,report)
    if not report['source_eligible']:
        sys.exit(2)


if __name__ == '__main__':
    main()
