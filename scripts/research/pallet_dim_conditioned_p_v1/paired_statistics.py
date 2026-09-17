"""Locked five contrasts and explanatory damage/subgroups; no model changes."""
import dcp_env as E
from eval_math import contrast,damage,summary
import numpy as np

def main():
    results={};damages={};groups={};replay={}
    partitions=[r['partition'] for r in E.read(E.LINE/'SOURCE_MANIFEST.json')['records']]
    pairs=E.read(E.DOC/'TRAIN_PROTOCOL_LOCK.json')['primary_contrasts']
    for split in ['SYNTH_HELDOUT','REAL_DEV']:
        allrows=E.read(E.RAW/f'{split}_METRICS.json');valid={k:[r for r in v if r['evaluable']] for k,v in allrows.items()};reference=valid['OLD_P_seed1'];ids=[r['id'] for r in reference]
        assert all([r['id'] for r in v]==ids for v in valid.values());clusters=[r['session'] for r in reference]
        results[split]={};damages[split]={};groups[split]={}
        for new,base in [*pairs,['N0_BASE_REPLAY','OLD_P'],['N4_META_SYM','OLD_P']]:
            name=new+'__minus__'+base
            a=[[r['E_sym'] for r in valid[f'{new}_seed{s}']] for s in [1,2,3]];b=[[r['E_sym'] for r in valid[f'{base}_seed{s}']] for s in [1,2,3]]
            c=contrast(a,b,clusters if split=='REAL_DEV' else None)
            dam=[damage(valid[f'{base}_seed{s}'],valid[f'{new}_seed{s}']) for s in [1,2,3]]
            gross=np.mean([summary(valid[f'{new}_seed{s}'])['gross20']-summary(valid[f'{base}_seed{s}'])['gross20'] for s in [1,2,3]])
            safe=sum(r['good5_to_bad10']-r['reverse_good5_to_bad10'] for r in dam)<=0 and gross<=0
            lo,hi=c['CI95'];c.update(verdict='WORSENED' if lo>0 else 'SUPPORTED' if hi<0 and c['improved_seeds']>=2 and safe else 'UNRESOLVED',safety_pass=bool(safe),gross20_delta=float(gross))
            if split=='SYNTH_HELDOUT':c['scenario_cluster_secondary']=contrast(a,b,clusters)
            for seed,d in zip([1,2,3],dam):
                npred=E.read(E.RAW/f'predictions/{split}/{new}_seed{seed}.json')['records'];bpred=E.read(E.RAW/f'predictions/{split}/{base}_seed{seed}.json')['records'];movement=[]
                for np_,bp in zip(npred,bpred):
                    assert np_['id']==bp['id'] and np_['selected_index']==bp['selected_index']
                    idx=np_['selected_index']
                    if idx is not None:movement.extend(np.linalg.norm(np.array(np_['candidates'][idx]['keypoints_xy'])[:8]-np.array(bp['candidates'][idx]['keypoints_xy'])[:8],axis=-1).tolist())
                d['expectation_coordinate_change_px']=dict(mean=float(np.mean(movement)),P90=float(np.quantile(movement,.9)),max=float(np.max(movement)))
                tops=[]
                if split=='SYNTH_HELDOUT':
                    logs=[]
                    for arm in [new,base]:
                        p=E.C.BRAW/f'validation_P{seed}.npz' if arm=='OLD_P' else E.RAW/f'logits/{arm}_seed{seed}.npz';z=np.load(p)
                        mask=np.array([partitions[int(row)]=='heldout' for row in z['rows']]);logs.append(z['logits'][mask].argmax(-1))
                    tops=(logs[0]!=logs[1]).reshape(-1).tolist()
                else:
                    for p in npred:
                        fid=p['id'].replace(':','__');a=E.RAW/f'DEV_logits/{new}_seed{seed}/{fid}.npz';b=E.RAW/f'DEV_logits/{base}_seed{seed}/{fid}.npz'
                        if a.exists() and b.exists():tops.extend((np.load(a)['logits'].argmax(-1)!=np.load(b)['logits'].argmax(-1)).tolist())
                d['candidate_top1_changed_fraction']=float(np.mean(tops)) if tops else None
            results[split][name]=c;damages[split][name]=dam
        for arm,rows in allrows.items():
            groups[split][arm]={field:{str(key):summary([r for r in rows if r[field]==key]) for key in sorted({r[field] for r in rows})} for field in ['group','object','ratio_bin','session']}
        replay[split]=dict(per_seed=[dict(seed=s,absolute_Esym_delta=abs(summary(valid[f'N0_BASE_REPLAY_seed{s}'])['E_sym']-summary(valid[f'OLD_P_seed{s}'])['E_sym']),
          absolute_median_px_delta=abs(summary(valid[f'N0_BASE_REPLAY_seed{s}'])['matched_pooled_corner8_median_px']-summary(valid[f'OLD_P_seed{s}'])['matched_pooled_corner8_median_px'])) for s in [1,2,3]],
          bit_exact_expected=False,reason='CUDA grid_sample backward nondeterminism; unchanged protocol/order/initial P weights, prelocked metric tolerance. Never tune against DEV discrepancy.')
        tolerance=E.read(E.DOC/'TRAIN_PROTOCOL_LOCK.json')['parity_tolerance'];replay[split]['within_prelocked_tolerance']=all(r['absolute_Esym_delta']<=tolerance['oldP_replay_Esym_absolute'] and r['absolute_median_px_delta']<=tolerance['oldP_replay_pooled_median_px_absolute'] for r in replay[split]['per_seed'])
    E.write(E.DOC/'PAIRED_STATISTICS.json',dict(complete=True,contrasts=results,OLD_P_reproduction=replay,no_multiplicity_corrected_claim=True))
    E.write(E.DOC/'DAMAGE_DECOMPOSITION.json',dict(complete=True,contrasts=damages,canonical_GT_identity_aligned=True,subgroups=groups,subgroups_not_causal=True))
    print('PAIRED_STATISTICS_COMPLETE',flush=True)
if __name__=='__main__':main()
