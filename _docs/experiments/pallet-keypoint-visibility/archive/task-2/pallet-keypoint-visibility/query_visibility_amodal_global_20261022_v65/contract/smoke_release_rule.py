"""Mock-only exact ratio and release-HOLD checks; no source/model access."""
from pathlib import Path
import copy,json
from evaluate_v65_source import comparative_release

rule={'criteria':{k:dict(numerator=n,denominator=d)for k,n,d in [
    ('E_precision',43,66),('strict_E_recall_all_GT',36,176),('O_recall_all_GT',472,1096),('overall_confidence_coverage',2212,3200)]}}
def report(cm,strict):
    return dict(sources=dict(SYNTH=1742,YCB=3200),
                results=dict(FROZEN_HEAD={s:dict(joint_NLL=.8,scales=dict(exact=dict(confident_joint_MAP_U=dict(confusion_truth_rows_prediction_columns_U_last=cm))))for s in ['SYNTH','YCB']},
                             TRAIN_EMPIRICAL_JOINT_PRIOR={s:dict(joint_NLL=1.3)for s in ['SYNTH','YCB']}),
                evidence_results=dict(FROZEN_HEAD=dict(YCB_strict_external_corroborated_5x5=dict(scales=dict(exact=dict(confident_joint_MAP_U=dict(per_class=[{},dict(correct=strict,support=176),{}])))))))
cm=[[1123,3,110,335],[229,43,36,225],[176,20,472,428]]
assert comparative_release(report(cm,36),rule)['HOLD'] # ties fail all4strict improvements
better=copy.deepcopy(cm);better[1][0]-=1;better[1][1]+=1;better[2][3]-=1;better[2][2]+=1
assert comparative_release(report(better,37),rule)['RELEASE']
r=report(better,37);r['results']['FROZEN_HEAD']['YCB']['joint_NLL']=1.3
assert comparative_release(r,rule)['HOLD'] # equalTRAINprior is not better
noE=copy.deepcopy(cm)
for row in noE:row[3]+=row[1];row[1]=0
assert comparative_release(report(noE,37),rule)['HOLD'] # undefined Eprecision cannot pass
result=dict(PASS=True,checks=['all four strict improvement','exact rational tie rejection','NLL equal prior HOLD','undefined Eprecision HOLD'],
            source_private_model_GPU_training_PnP_reads_or_calls=0)
(Path(__file__).parent/'RELEASE_RULE_MOCK_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
