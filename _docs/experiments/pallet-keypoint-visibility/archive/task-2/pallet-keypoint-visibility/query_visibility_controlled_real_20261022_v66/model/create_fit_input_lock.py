"""Bind completed source joins and audits before either fixed-budget fit."""
from pathlib import Path
import hashlib,json,time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def main():
 out=ROOT/'provenance/FIT_INPUT_LOCK.json';assert not out.exists()
 protocol=ROOT/'FIT_PROTOCOL_V2.json';trainer=ROOT/'model/fit_partial_head.py';model=ROOT/'model/query_global_amodal_head.py'
 deny=ROOT.parent/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json'
 data=ROOT/'joined_source_cache';seal=json.loads((data/'SEAL.json').read_text());assert seal['complete']
 for b in seal['outputs']:assert sha(b['path'])==b['sha256']
 audits=[ROOT/'source_audit/JOINED_CACHE_AUDIT.json',ROOT/'source_audit/PALLOX_FEATURE_CACHE_FULL_AUDIT/REPORT.json',
  ROOT/'source_audit/FEATURE_CACHE_FULL_AUDIT/REPORT.json',ROOT/'source_audit/INDEPENDENT_PARTIAL_TRAINER_AUDIT.json',
  ROOT/'source_audit/INDEPENDENT_PARTIAL_CACHE_MERGE_MOCK_AUDIT.json',ROOT/'source_audit/INDEPENDENT_LEGACY_FEATURE_METADATA_MOCK.json',
  ROOT/'source_audit/INDEPENDENT_PARTIAL_CACHE_REPLAY.json',ROOT/'source_audit/INDEPENDENT_FIXED_PAIRED_QUERY_AUDIT.json',
  ROOT/'public_pallet/PAIRED_QUERIES/QUERY_PARITY_AUDIT.json']
 for p in audits:
  a=json.loads(p.read_text())
  assert a.get('PASS')is True,(p,'explicit audit PASS required')
 runs={};paths=[protocol,trainer,model,deny,data/'SEAL.json',ROOT/'provenance/MERGE_INPUT_LOCK.json']+audits
 for c in json.loads(protocol.read_text())['candidates']:
  runs[c]=dict(train=bind(data/(c+'_TRAIN.npz')),val=bind(data/(c+'_VAL.npz')),protocol=bind(protocol),denylist=bind(deny),out=str(ROOT/'runs'/c))
  paths.extend([data/(c+'_TRAIN.npz'),data/(c+'_VAL.npz')])
 result=dict(schema='V66_FIT_INPUT_WHITELIST_BEFORE_BOTH_CANDIDATE_RUNS',created_unix=time.time(),
  trainer_sha256=sha(trainer),model_sha256=sha(model),runs=runs,bindings=[bind(p)for p in dict.fromkeys(paths)],
  source_only=True,current223_RGB_annotations_scores_used=False,PnP_calls=0,optimizer_updates_before_this_lock=0,
  provenance='Controlled new-occlusion and photographic paired-TEACHER remain separate sources; source family direct-user-confirmed distinct. Private metadata used only for exclusion hash denylist.',
  selection='Frozen FIT_PROTOCOL_V2 common4source macroinformativecenterpartialNLL, strictminimum/tieCONTROLLED_ONLY. No private results for selection.')
 out.write_text(json.dumps(result,indent=2)+'\n');print('FIT_INPUT_LOCK',sha(out),flush=True)
if __name__=='__main__':main()
