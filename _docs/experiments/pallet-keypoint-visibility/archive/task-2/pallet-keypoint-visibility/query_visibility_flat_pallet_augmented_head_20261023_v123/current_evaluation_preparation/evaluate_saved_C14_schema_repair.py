"""Bound saved-only C14 minimal-score enrichment repair. Never calls infer/head/model."""
from pathlib import Path
import argparse,json,math
import numpy as np
import frozen_head_current as m
ORIGINAL_OPERATOR_SHA='28b4d0b2b56cfd5e19fc6975a7a7935940a50ef2f3d1170d81c670406ee71640'

def validate_repair_entry(path):
 e=m.read(path)
 m.require(e.get('approved')is True and e.get('stage')=='EVALUATE_SAVED','Explicit saved evaluation root entry required')
 m.require(m.same(e['saved_schema_adapter'],m.bind(__file__)),'Unbound saved schema adapter')
 m.require(m.sha(m.__file__)==ORIGINAL_OPERATOR_SHA and m.same(e['code'],m.bind(m.__file__)),'Original operator changed')
 old=m.read(m.check(e['original_evaluation_entry']))
 m.require(all(k in e and e[k]==v for k,v in old.items()),'Original failed evaluation entry fields changed')
 proof=m.read(m.check(e['saved_schema_test']))
 m.require(proof.get('completed')is True and proof.get('PASS')is True and m.same(proof['adapter'],e['saved_schema_adapter']),'Unbound generated schema proof')
 return e

def enrich_C14_minimal(core,prior,minimal):
 """Only restores original geometry metadata; preserves saved C14 values/q/L/availability."""
 m.require(len(prior)==len(minimal)==1784,'Original C14 rowcount')
 z=[];pv=[];available=[]
 for p,r in zip(prior,minimal):
  m.require(core.key(p)==core.key(r),'Original C14 id/native order changed')
  for k in ('query_x','query_y','roi_side'):m.require(core.numeric_same(p[k],r[k]),'Original C14 q/L changed')
  a=core.boolean(r['valid']);zi=float(r['raw_query_aux_hidden_logit']);pi=float(r['raw_p_visible_FP32'])
  m.require(math.isfinite(zi)and math.isfinite(pi)and 0<=pi<=1,'C14 saved scalar schema')
  expected=('D'if zi<0 else'E')if a else'UNKNOWN'
  m.require(r['authoritative_decision']==expected,'C14 original raw0/U decode changed')
  z.append(zi);pv.append(pi);available.append(a)
 pcopy=[dict(p,prior_reason=p.get('prior_reason',p['reason']))for p in prior]
 decoded=core.decode_rows(pcopy,np.asarray(z,np.float32),np.asarray(pv,np.float32),np.asarray(available,bool))
 result=[{k:(''if v is None else str(v))for k,v in r.items()}for r in decoded]
 for r,old in zip(result,minimal):
  m.require(r['authoritative_decision']==old['authoritative_decision']and float(r['raw_query_aux_hidden_logit'])==float(old['raw_query_aux_hidden_logit']),'C14 reconstruction altered output')
 return result

def evaluate_repaired(entry):
    validate_repair_entry(entry)
    e, c11seal = m.authorize(entry, 'EVALUATE_SAVED')
    m.check(e['inference_seal'])
    seal = m.read(e['inference_seal']['path'])
    m.require(seal['completed'] and seal['PASS'] and seal['model_unchanged'] and (seal['slots'] == 1784) and (seal['U'] == 45) and (seal['additional_U'] == 0) and m.same(seal['source_model_freeze'], e['source_freeze']) and m.same(seal['checkpoint'], e['checkpoint']) and (seal['D_rule'] == 'z<0') and (not seal['current_GT_or_baseline_scores_read']), 'Complete frozen new predictions required')
    for d in seal['outputs']:
        m.check(d)
    b = e['bindings']
    m.require(set(b) == m.EVAL_KEYS, 'Exact saved-eval bindings')
    for d in b.values():
        m.check(d)
    m.require(m.contains(c11seal['outputs'], b['baseline_C11_scores']), 'C11 baseline scores unsealed')
    core, extra = m.pure_core()
    base = m.module(m.check(dict(path=str(m.METRIC), sha256=m.METRIC_SHA)), 'c15_frozen_metric')
    truth = core.readcsv(b['evaluation_manifest']['path'])
    manual = core.readcsv(b['manual_provenance']['path'])
    prior = [json.loads(x) for x in Path(b['prior_input_rows']['path']).read_text().splitlines() if x]
    old = core.readcsv(b['baseline_v91_scores']['path'])
    c11rows = core.readcsv(b['baseline_C11_scores']['path'])
    c14rows = core.readcsv(b['baseline_C14_scores']['path'])
    c14seal = m.read(b['baseline_C14_inference_seal']['path'])
    m.require(c14seal['completed'] and c14seal['model_unchanged'] and (c14seal['slots'] == 1784) and (c14seal['U'] == 45) and (c14seal['additional_U'] == 0) and m.contains(c14seal['outputs'], b['baseline_C14_scores']), 'C14 baseline not sealed')
    oldseal = m.read(b['baseline_v91_inference_seal']['path'])
    m.require(oldseal['completed'] and m.contains(oldseal['outputs'], b['baseline_v91_scores']), 'V91 baseline not sealed')
    scorepath = next((d['path'] for d in seal['outputs'] if Path(d['path']).name == 'HEAD_ONLY_SCORES.csv'))
    rawpath = next((d['path'] for d in seal['outputs'] if Path(d['path']).name == 'RAW_HEAD_OUTPUTS.npz'))
    minimal = core.readcsv(scorepath)
    with np.load(rawpath, allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    m.require(len(minimal) == len(prior) == 1784 and a['query_logit'].shape == (1784,) and (a['query_logit'].dtype == np.float32), 'Saved rows/schema')
    for i, (r, p) in enumerate(zip(minimal, prior)):
        m.require(core.key(r) == core.key(p) == (str(a['id'][i]), str(int(a['native_corner'][i]))), 'Original order changed')
        m.require(core.boolean(r['valid']) == bool(a['valid'][i]) and float(r['raw_query_aux_hidden_logit']) == float(a['query_logit'][i]), 'CSV/NPZ mismatch')
        for axis, k in enumerate(('query_x', 'query_y')):
            m.require(core.numeric_same(r[k], p[k]) and core.numeric_same(a['original_query_xy'][i, axis], p[k]), 'Original q changed')
        m.require(core.numeric_same(r['roi_side'], p['roi_side']) and core.numeric_same(a['original_side_L'][i], p['roi_side']), 'Original L changed')
        p['prior_reason'] = p['reason']
    enriched = core.decode_rows(prior, a['query_logit'], a['p_visible_FP32'], a['query_available'])
    newrows = [{k: '' if v is None else str(v) for k, v in r.items()} for r in enriched]
    c14rows = enrich_C14_minimal(core, prior, c14rows)
    tables = m.calculate_tables(core, extra, base, truth, prior, newrows, old, c11rows, manual, c14rows)
    count = core.historical_replay(tables['metrics'], core.readcsv(b['original_baseline_metrics']['path']), {'v61', 'legacy'})
    count += core.historical_replay(tables['metrics'], core.readcsv(b['historical_v91_metrics']['path']), {m.OLD, m.OLD_GEO})
    c11count = core.historical_replay(tables['metrics'], core.readcsv(b['historical_C11_metrics']['path']), {m.BASE, m.BASE_GEO})
    c14count = core.historical_replay(tables['metrics'], core.readcsv(b['historical_C14_metrics']['path']), {m.C14, m.C14_GEO})
    out = Path(e['output_directory'])
    m.require(out.resolve() == (m.OUT / 'evaluation').resolve() and (not out.exists()), 'Fixed new evaluation output')
    out.mkdir()
    for name in ('metrics', 'confusion', 'human_self_binary_response', 'manual_near_D', 'availability_by_group', 'coordinate_error_bins', 'D_vs_external_E_2x3'):
        core.writecsv(out / (name.upper() + '.csv'), tables[name])
    core.writecsv(out / 'SCORES_WITH_ORIGINAL_GEOMETRY_JOIN.csv', enriched)
    m.write(out / 'CURRENT_PRIMARY_RESULTS.json', dict(completed=True, tables=tables, historical_v61_legacy_V91_rows_exact=count, historical_C11_rows_exact=c11count, historical_C14_rows_exact=c14count, primary_model=m.NEW, geometry_routes_separate=[m.NEW_GEO, m.C14_GEO, m.BASE_GEO, m.OLD_GEO], descriptive_evaluation_only=True, model_promotion_authorized=False, original1784_retained=True, rawsign='D iff z<0, else binaryhidden; factualU first', self='Existing initialgeometry route and humanS binary-response diagnostics separate; no learnedS or cause claim.', coordinate='Original query/reference error diagnostics only; coordinate correctness not established.', current_historically_development_exposed=True, current_used_for_choice=False, new_forward_fit_threshold_selection_PnP=0, bindings=e))
    m.write(out / 'EVALUATION_SEAL.json', dict(completed=True, PASS=True, entry=m.bind(entry), inference=e['inference_seal'], source_freeze=e['source_freeze'], files=[m.bind(p) for p in sorted(out.iterdir()) if p.is_file()], new_model_forward_fit_current_selection_PnP=0))
    print(json.dumps(dict(completed=True, evaluation=m.bind(out / 'EVALUATION_SEAL.json'))))

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--eval-entry",type=Path,required=True);a=p.parse_args();evaluate_repaired(a.eval_entry)
