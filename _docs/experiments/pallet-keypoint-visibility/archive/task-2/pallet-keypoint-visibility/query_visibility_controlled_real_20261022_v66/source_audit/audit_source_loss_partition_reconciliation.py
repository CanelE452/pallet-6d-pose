"""Small independent saved-output reconciliation; no evaluator imports/forward."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'source_audit/INDEPENDENT_SOURCE_LOSS_PARTITION_RECONCILIATION.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))

def main():
    assert not OUT.exists()
    src=ROOT/'source_audit/SOURCE_LOSS_TRADEOFF_DIAGNOSIS.json'
    report=json.loads(src.read_text());rows=[];model_macro={};checks=[]
    for model in report['models']:
        p=ROOT/'source_evaluation'/model/'SOURCE_OUTPUTS.npz'
        with np.load(p,allow_pickle=False)as z:
            d={k:z[k]for k in ['joint_logits','source','sample_id','center_allowed','center_native_valid','partial_NLL']}
        a=d['center_allowed'];valid=d['center_native_valid'];info=valid&(a!=63)
        logits=d['joint_logits'].astype(np.float64)
        assert logits.shape==(8590,6)and len(set(d['sample_id']))==8590
        permit=(a[:,None]&(1<<np.arange(6))[None])!=0
        # Recompute directly as log-normalizer difference, independently of
        # the published partition builder and its gradient decomposition.
        loss=np.logaddexp.reduce(logits,axis=1)-np.logaddexp.reduce(np.where(permit,logits,-np.inf),axis=1)
        checks.append(dict(model=model,output=ref(p),max_abs_recomputed_vs_saved_NLL=float(np.max(np.abs(loss-d['partial_NLL'])))))
        cat=np.full(len(a),'UNKNOWN_EXACT_PARTIAL',dtype='<U24')
        cat[a==1]='KNOWN_VV'
        cat[((a&57)==0)]='KNOWN_EXACT_E'
        cat[((a&7)==0)]='KNOWN_EXACT_O'
        means=[]
        for s in sorted(set(d['source'])):
            pick=info&(d['source']==s);n=int(pick.sum());mean=float(loss[pick].mean());means.append(mean)
            total=next(x for x in report['source_totals']if x['model']==model and x['source']==s)
            parts=[x for x in report['exact_GT_partition']if x['model']==model and x['source']==s]
            assert sum(x['N']for x in parts)==n==total['informative_N']
            assert all(x['source_informative_denominator']==n for x in parts)
            for x in parts:
                pp=pick&(cat==x['category']);count=int(pp.sum());loss_sum=float(loss[pp].sum())
                assert count==x['N']
                assert abs(loss_sum-x['loss_sum'])<1e-10
                assert abs(loss_sum/n-x['source_mean_contribution'])<1e-12
                assert abs(loss_sum/(4*n)-x['macro_mean_contribution'])<1e-12
            assert abs(sum(x['source_mean_contribution']for x in parts)-mean)<1e-12
            assert abs(mean-total['source_mean_NLL'])<1e-12
            rows.append(dict(model=model,source=s,informative_N=n,partition_N=sum(x['N']for x in parts),
                independently_recomputed_source_mean_NLL=mean,published_source_mean_NLL=total['source_mean_NLL'],
                partition_loss_sum=float(sum(x['loss_sum']for x in parts)),
                category_counts={x['category']:x['N']for x in parts}))
        macro=float(np.mean(means));published=report['recomputation_audit'][model]['CPU_saved_macro']
        assert abs(macro-published)<1e-12
        model_macro[model]=dict(reconstructed_macro=macro,published_macro=published,absolute_difference=abs(macro-published))
    result=dict(PASS=True,scope='Only exact_GT_partition count/loss contribution and four-domain macro reconciliation; no broad saved-score or gradient reanalysis.',
        source_report=ref(src),code=ref(Path(__file__)),source_output_checks=checks,source_reconstructions=rows,macro_reconstructions=model_macro,
        partition_count_mismatches=0,partition_category_mismatches=0,
        note='Informative partial UNKNOWN remains in its own partition. Native-valid all63 and invalid rows remain excluded from informative NLL denominators.',
        model_feature_weight_current_private_reads=0,new_forward_training_threshold_selection_PnP_GPU_calls=0)
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('SOURCE_PARTITION_RECONCILIATION_PASS',sha(OUT),model_macro)

if __name__=='__main__':main()
