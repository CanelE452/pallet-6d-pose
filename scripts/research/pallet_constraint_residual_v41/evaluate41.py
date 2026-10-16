from context41 import *
METHODS=('anchor','V40_actual')+tuple(s+'_'+f+'_'+m+route for s in SOURCES for f in FAMILIES for m in MODES for route in ('','_gate','_GT_safe'))+tuple(s+'_'+f+'_GT_candidate' for s in SOURCES for f in FAMILIES)

def main():
    assert read(RUN/'TRAINING_COMPLETE.json')['complete'];p=population();part=partition(p);g=p['g'];v=[dict(np.load(RUN/'arrays'/f'seed{k}.npz')) for k in SEEDS];hard=np.stack([q['hard'] for q in v]);E={}
    for method in METHODS:
        pts=np.stack([q['anchor' if method=='anchor' else 'points_'+method] for q in v]) if method!='V40_actual' else np.stack([np.load(V40/'arrays'/f'seed{k}.npz')['points_pred_segment_all'] for k in SEEDS]);E[method]=np.linalg.norm(pts-g['gt'][None],axis=-1)
    results={};pointcomparisons={};direction={};count=0
    for key,M in masks(p,hard):
        if key.startswith('calibration/') or (key.startswith('fit/') and key not in ('fit/all/all_detected','fit/all/hard','fit/all/easy')):continue
        rr={}
        for method in METHODS:
            r=metric(E[method],E['anchor'],M,p['groups']);r['harmed_over1px']=[int(((E[method][i]-E['anchor'][i]>1)&M[i]).sum()) for i in range(3)];r['changed_points']=[int(((np.abs(E[method][i]-E['anchor'][i])>1e-9)&M[i]).sum()) for i in range(3)];rr[method]=r
        results[key]=rr;count+=1
        if key in ('fit/all/all_detected','fit/all/hard','check/all/all_detected','check/all/hard'):
            dd={};pc={}
            for source in SOURCES:
              for family in FAMILIES:
                prefix=source+'_'+family;valid=np.stack([q[prefix+'_valid'] for q in v]);MM=M[...,None]&valid;y=np.stack([q[prefix+'_target'] for q in v]);scale=p['diag'][None,:,None,None];err={mode:np.abs(np.stack([q['mu_'+prefix+'_'+mode] for q in v])-y)*scale for mode in MODES};dd[prefix]={}
                for mode in MODES:
                    r=metric(err[mode],err['direct'],MM,p['groups']);mu=np.stack([q['mu_'+prefix+'_'+mode] for q in v]);sel=MM&(np.abs(y)*scale>=1);r['sign_accuracy']=[float((np.sign(mu[i][sel[i]])==np.sign(y[i][sel[i]])).mean()) for i in range(3)];dd[prefix][mode]=r
                dd[prefix]['residual_vs_analytic']=metric(err['residual'],err['analytic'],MM,p['groups'])
                pc[prefix]=dict(residual_vs_direct=metric(E[prefix+'_residual'],E[prefix+'_direct'],M,p['groups']),residual_vs_analytic=metric(E[prefix+'_residual'],E[prefix+'_analytic'],M,p['groups']))
            direction[key]=dd;pointcomparisons[key]=pc
        if count%10==0:print('EVAL41 strata',count,flush=True)
    def improves(r,percent):return min(r['gain_percent'])>=percent and r['difference']['ci95'][1]<0
    primary='pred_segment_physical3_residual_gate';gates=dict(analytic_residual_direction=improves(direction['check/all/hard']['pred_segment_physical3']['residual'],10),actual_points=improves(results['check/all/all_detected'][primary],1),all_condition_preservation=all(min(r[primary]['gain_percent'])>=-1 for key,r in results.items() if key.startswith('check/')),raw_residual_beats_direct=pointcomparisons['check/all/all_detected']['pred_segment_physical3']['residual_vs_direct']['difference']['ci95'][1]<0);gates['accepted']=all(gates[k] for k in ('analytic_residual_direction','actual_points','all_condition_preservation'))
    write(RUN/'SUMMARY.json',dict(results=results,direction=direction,point_comparisons=pointcomparisons,gates=gates,scope='Reused synthetic diagnostic. NormalMAE validrole populations andnormals differ bysource/family; only withinfamily matchedcomparison causal. GTvariants notdeployable.'))
    write(RUN/'EVALUATION_COMPLETE.json',dict(complete=True,files={str(q.relative_to(ROOT)):sha(q) for q in (RUN/'SUMMARY.json',KIT/'evaluate41.py')}));print('EVAL41 COMPLETE',gates,flush=True)
    print('POINT41',{k:round(r['mean_px'],3) for k,r in results['check/all/all_detected'].items() if 'GT_safe' not in k and 'GT_candidate' not in k},flush=True)
    print('NORMAL41',{k:{m:round(q['mean_px'],3) for m,q in r.items()} for k,r in direction['check/all/hard'].items()},flush=True)
if __name__=='__main__':main()
