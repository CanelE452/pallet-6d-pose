from context41 import *
from observations40 import canonical

def main():
    write(RUN/'PROBE_PROTOCOL.json',dict(exploratory=True,primary_sha256=sha(RUN/'SUMMARY.json'),question='Which selected actual constraints cause damage: missing GTvisible role, mismatched physicalsupport, or close physicalsupport? Does oraclephysicalquality rejection suffice?',definition='For selected pred_segment physical3 proposal: GTrolevisible length>0. close iff meanGTsupport endpoint distance toselectedline<=3px andacuteangle<=10deg; otherwisemismatch. NoGTvisible role is absence ofcachedvisiblephysicalsupport,not proof everysegment is false. Quality_GT_oracle keepsfixedpredictiononlywhenclose,elseanchor. This usesGTline/visibility,notGTpointgain; diagnosticonly. ComparepointGTsafe separately. No newfit/thresholdtuning.',limits='GTline coverage andalignment quality conditioning canexcludehardcases. Comparefinalallpointmetric,butdo notclaimoraclequalitydeployable. ExistingV35supportenvelope canbe disjoint.'))
    p=population();s=read(RUN/'SUMMARY.json');part=partition(p);inc=np.array(INC);g=p['g'];out=[];quality_errors=[];base_errors=[];hard=[];counts=[]
    for seed in SEEDS:
        v=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'));a=dict(np.load(V40/'arrays'/f'matching_seed{seed}.npz'));seg=a['pred_segment'];nr=canonical(seg)[1];rho=(nr*seg[:,:,0]).sum(-1);distance=np.abs((g['support']*nr[:,:,None]).sum(-1)-rho[...,None]).mean(-1);angle=np.degrees(np.arccos(np.clip(np.abs((nr*g['normal']).sum(-1)),0,1)));visible=g['length']>0;quality=visible&(distance<=3)&(angle<=10);base=np.linalg.norm(v['anchor']-g['gt'],axis=-1);stats={}
        for method in ('direct','analytic','residual','residual_gate'):
            name='pred_segment_physical3_'+method;selected=v['selected_'+name];role=inc[np.arange(8)[None],np.maximum(selected,0)];selectedmask=part['check'][:,None]&(selected>=0);has=visible[np.arange(p['N'])[:,None],role];good=quality[np.arange(p['N'])[:,None],role];err=np.linalg.norm(v['points_'+name]-g['gt'],axis=-1);categories={}
            for label,cat in [('no_GT_visible_support',~has),('physical_mismatch',has&~good),('physical_close',good)]:
                mask=selectedmask&cat;categories[label]=dict(selected_points=int(mask.sum()),harmed_over1px=int((mask&(err-base>1)).sum()),improved_over1px=int((mask&(base-err>1)).sum()),sum_error_change_px=float((err-base)[mask].sum()))
            assert sum(q['selected_points'] for q in categories.values())==int(selectedmask.sum());assert sum(q['harmed_over1px'] for q in categories.values())==s['results']['check/all/all_detected'][name]['harmed_over1px'][seed-1];stats[method]=categories
            if method=='residual_gate':
                keep=(selected>=0)&good;pt=np.where(keep[...,None],v['points_'+name],v['anchor']);quality_errors.append(np.linalg.norm(pt-g['gt'],axis=-1));np.savez_compressed(RUN/'arrays'/f'quality_oracle_seed{seed}.npz',points=pt,keep=keep,role=role,quality=quality,distance=distance,angle=angle);counts.append(int((keep&part['check'][:,None]).sum()))
        base_errors.append(base);hard.append(v['hard']);out.append(stats)
    E=np.stack(quality_errors);R=np.stack(base_errors);metrics={}
    for key,M in masks(p,np.stack(hard)):
        if key.startswith('check/'):metrics[key]=metric(E,R,M,p['groups'])
    gtkey='GT_line_physical3_residual';worst=sorted([(min(r[gtkey]['gain_percent']),key) for key,r in s['results'].items() if key.startswith('check/')]);directkey='pred_segment_physical3_direct';worstdirect=sorted([(min(r[directkey]['gain_percent']),key) for key,r in s['results'].items() if key.startswith('check/')]);write(RUN/'PROBE.json',dict(complete=True,exploratory=True,selected_categories=out,GT_quality_oracle=metrics,GT_quality_kept_check=counts,GTphysical_residual_worst_conditions=worst[:5],actualphysical_direct_worst_conditions=worstdirect[:5]));print('PROBE41 COMPLETE',metrics['check/all/all_detected'],out,flush=True)
if __name__=='__main__':main()
