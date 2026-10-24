"""Apply prospectively frozen source criteria; no model execution or selection."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
def b(p):
 p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 decisions=R/'EXPERIMENT_DECISIONS_BEFORE_FEATURES.json'
 assert b(decisions)['sha256']=='a008fc7d35e9aa1d0d4b7d631a796f238aaba7f0aa052fe7da087080b70115e9'
 run=R/'runs/SOURCE_DINO_V1'
 assert json.loads((run/'FIT_COMPLETE.json').read_text())['completed']
 result=json.loads((run/'SOURCE_RESULTS.json').read_text())
 m=result['results']['natural_and_after']['NATURAL_ORIGINAL|YCB_PUBLIC_REAL|val']['metrics']
 pair=result['results']['paired_anchors']['YCB_PUBLIC_REAL|val']
 checks={
 'public_natural_C_precision_ge_0.50':dict(value=m['covered_precision'],minimum=.5),
 'public_natural_C_recall_ge_0.80':dict(value=m['covered_recall'],minimum=.8),
 'public_natural_V_recall_ge_0.80':dict(value=m['visible_recall'],minimum=.8),
 'public_controlled_both_correct_ge_0.70':dict(value=pair['both_correct_rate'],minimum=.7)}
 for v in checks.values():v['PASS']=v['value']>=v['minimum']
 out=R/'SOURCE_ADMISSION_DECISION.json';assert not out.exists()
 out.write_text(json.dumps(dict(schema='V78_PROSPECTIVE_SOURCE_CANDIDATE_GATE',checks=checks,
 source_candidate_pass=all(x['PASS']for x in checks.values()),
 source_only=True,current_reads=0,current_inference=0,PnP=0,fully_solved=False,
 threshold_or_checkpoint_reselected=False,threshold=.5,
 scope='Repeated-source weakreference-localsupport criteria only. No humanD/E/S/generalization proof. No current223 work inV78stage.',
 decisions=b(decisions),saved_results=b(run/'SOURCE_RESULTS.json'),fit=b(run/'FIT_COMPLETE.json'),code=b(Path(__file__))),ensure_ascii=False,indent=2)+'\n')
 print(out)
if __name__=='__main__':main()
