"""Freeze SHA-ID selected source disagreements before RGB inspection."""
from pathlib import Path
import collections,csv,hashlib,json
OUT=Path(__file__).resolve().parent;V70=OUT.parents[1];BASE=V70.parent
V66=BASE/'query_visibility_controlled_real_20261022_v66'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=OUT/'SELECTION_LOCK.json';assert not target.exists()
    saved=V70/'source_evaluation/OWNERSHIP_FROZEN_COMPARISON/PER_QUERY.csv'
    with saved.open() as f:rows=list(csv.DictReader(f))
    scores=collections.defaultdict(dict)
    for r in rows:
        if r['source']=='PALLOX':
            assert r['model'] not in scores[r['sample_id']];scores[r['sample_id']][r['model']]=r
    pool=[]
    for sid,s in scores.items():
        v=s['V70_OWNERSHIP_LOO_PAIR_V1'];m=s['MOBILE_SAM_RAW']
        if v['native_valid']=='True' and v['exact_GT']=='1' and float(v['p_visible'])>=.5 and float(m['p_visible'])<=.5:
            pool.append({'sample_id':sid,'sha256_of_UTF8_sample_id':hashlib.sha256(sid.encode()).hexdigest(),'global_id':v['global_id']})
    pool.sort(key=lambda r:(r['sha256_of_UTF8_sample_id'],r['sample_id']))
    chosen=[];seen=set()
    for r in pool:
        if r['global_id'] in seen:continue
        chosen.append(r);seen.add(r['global_id'])
        if len(chosen)==2:break
    assert len(chosen)==2
    ip=V66/'public_pallet/PAIRED_QUERIES/INPUTS_RGB_ONLY.jsonl';lp=V66/'public_pallet/PAIRED_QUERIES/QUERIES_LABELS.jsonl'
    mp=V66/'public_pallet/SYMMETRIC_ALL50_V2/PAIR_EVIDENCE_MANIFEST.jsonl'
    inputs={r['id']:r for r in map(json.loads,ip.read_text().splitlines())}
    labels={r['sample_id']:r for r in map(json.loads,lp.read_text().splitlines())}
    pairs={r['pair_id']:r for r in map(json.loads,mp.read_text().splitlines())}
    compare=['MOBILE_SAM_RAW','V65_BASELINE','V66_CONTROLLED_PLUS_PALLOX','V70_OWNERSHIP_LOO_PAIR_V1']
    for r in chosen:
        sid=r['sample_id'];label=labels[sid];pid=label['pair_id'];view=label['view'];other='B' if view=='A' else 'A'
        other_id=r['global_id'].rsplit(':',1)[0]+':'+other;other_sid=other_id+':'+str(label['slot'])
        r.update({'selected_label':label,'paired_reference_label':labels[other_sid],
            'selected_input_metadata':inputs[r['global_id']],'paired_input_metadata':inputs[other_id],
            'selected_saved_scores':{m:scores[sid][m] for m in compare},'paired_saved_scores':{m:scores[other_sid][m] for m in compare},'source_pair_evidence_metadata':pairs[pid]})
        assert label['query']==labels[other_sid]['query'] and inputs[r['global_id']]['query']==inputs[other_id]['query']
        assert inputs[r['global_id']]['side']==inputs[other_id]['side']
        assert inputs[r['global_id']]['image_sha256']==scores[sid]['V70_OWNERSHIP_LOO_PAIR_V1']['image_sha256']
    result={'schema':'PALLOX_FIXED_E_DISAGREEMENT_SELECTION_V1','status':'FROZEN_BEFORE_RGB_READ',
        'filter':'PALLOX native-valid exact_GT1, V70pV>=.5 and rawMobileSAMpV<=.5. SHA256(UTF8 sample_id) ascending; greedy first2 distinct global target IDs. No confidence ranking, label/source correction, replacement or model choice.',
        'eligible_count':len(pool),'eligible_ID_hash_order':pool,'chosen_count':2,
        'capture_family_note':'Distinct global target rows/pairs, both1.23m_hall sourceVAL from same recording session; not2 independent physical pallet families.',
        'selection_is_postfit_diagnostic':'Models/outputs already frozen; no fit/threshold/crop/label tuning or current data.',
        'evidence_limit':'Original paired conditional teacher, not human semantic gold or calibrated pixelGT. No corner annotation inferred from these generic query slots.',
        'bindings':[{'path':str(p),'sha256':sha(p)} for p in [saved,ip,lp,mp,V66/'PAIRED_REAL_LABEL_PROTOCOL.json',V66/'PAIRED_QUERY_RECIPE.json',Path(__file__)]],
        'read_scope_before_lock':{'RGB_decoded':0,'source_saved_scores_metadata_and_labels_only':True,'current_model_features_weights_forward_fit_PnP':0},'selected':chosen}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'path':str(target),'sha256':sha(target),'selected':[r['sample_id'] for r in chosen],'eligible_count':len(pool)}))
if __name__=='__main__':main()
