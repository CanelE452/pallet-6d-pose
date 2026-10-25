"""V239 SOURCE 산술만. 파일/특징/모델을 읽지 않는다."""
from collections import Counter
import json

ARMS=('CONTROL_CANONICAL_LOCAL','TREAT_CANONICAL_CONTEXT')

def metrics(rows, arm):
    cm=[[sum(r['target']==y and r[arm+'_prediction']==p for r in rows) for p in ('D','E','U')] for y in (0,1)]
    result=dict(N=len(rows),CM_D_E_by_D_E_U=cm,U=sum(r[2] for r in cm),U_rate=sum(r[2] for r in cm)/len(rows) if rows else None)
    for name,k in (('D',0),('E',1)):
        truth=sum(cm[k]);predicted=cm[0][k]+cm[1][k];tp=cm[k][k]
        result[name]=dict(truth=truth,predicted=predicted,precision=tp/predicted if predicted else None,recall=tp/truth if truth else None)
    recalls=[result[k]['recall'] for k in ('D','E')]
    result['balanced_recall']=sum(recalls)/2 if all(v is not None for v in recalls) else None
    return result

def paired(rows):
    out={}
    for label,subset in [('ALL',rows)]+[(name,[r for r in rows if r['target']==y]) for y,name in enumerate(('D','E'))]:
        counts=Counter()
        for r in subset:
            truth=('D','E')[r['target']];a=r['control_prediction']==truth;b=r['treatment_prediction']==truth
            counts['both_correct' if a and b else 'damaged' if a else 'recovered' if b else 'both_wrong']+=1
            counts['prediction_changed']+=r['control_prediction']!=r['treatment_prediction']
        out[label]=dict(N=len(subset),**{k:counts[k] for k in ('both_correct','both_wrong','recovered','damaged','prediction_changed')},net_correct=counts['recovered']-counts['damaged'])
    return out

def reduce_saved(rows):
    by_family={};syn_distance={};dev_pairs={}
    for family,name in ((0,'SYN_MAIN'),(1,'YCB_AUX')):
        by_family[name]={}
        for role in ('FIT','DEV'):
            rr=[r for r in rows if r['family']==family and r['role']==role]
            by_family[name][role]={arm:metrics(rr,arm) for arm in ('control','treatment')}
        dev_pairs[name]=paired([r for r in rows if r['family']==family and r['role']=='DEV'])
    for role in ('FIT','DEV'):
        syn_distance[role]={}
        for label in ('LE3','GT3','UNKNOWN'):
            rr=[r for r in rows if r['family']==0 and r['role']==role and (('UNKNOWN' if r['q_reference_distance_px'] is None else 'LE3' if r['q_reference_distance_px']<=3 else 'GT3')==label)]
            syn_distance[role][label]={arm:metrics(rr,arm) for arm in ('control','treatment')}
    recalls={a:[by_family[f]['DEV'][a][c]['recall'] for f in ('SYN_MAIN','YCB_AUX') for c in ('D','E')] for a in ('control','treatment')}
    defined=all(v is not None for values in recalls.values() for v in values)
    delta=[b-a for a,b in zip(recalls['control'],recalls['treatment'])] if defined else None
    gain=sum(delta)/4 if defined else None
    gate=dict(defined=defined,macro_DEV_balanced_recall_gain=gain,macro_gain_pp=None if gain is None else gain*100,
              class_order=['SYN_D','SYN_E','YCB_D','YCB_E'],class_recall_deltas=delta,
              required_macro_gain=.03,max_class_recall_decrease=.02,
              numeric_gate_pass=bool(defined and gain>=.03-1e-12 and min(delta)>=-.02-1e-12),
              ancestry='UNKNOWN',current_admission=False,selected_arm=None,model_adoption=False)
    return dict(family_metrics=by_family,SYN_q_reference_distance=syn_distance,DEV_paired=dev_pairs,gate=gate)

def metric_table(summary):
    table=[]
    def add(family,role,stratum,arms):
        for arm,m in arms.items():
            c=m['CM_D_E_by_D_E_U'];table.append(dict(family=family,role=role,stratum=stratum,arm=arm,N=m['N'],DD=c[0][0],DE=c[0][1],DU=c[0][2],ED=c[1][0],EE=c[1][1],EU=c[1][2],D_precision=m['D']['precision'],D_recall=m['D']['recall'],E_precision=m['E']['precision'],E_recall=m['E']['recall'],U_rate=m['U_rate'],balanced_recall=m['balanced_recall']))
    for family,roles in summary['family_metrics'].items():
        for role,arms in roles.items():add(family,role,'ALL',arms)
    for role,strata in summary['SYN_q_reference_distance'].items():
        for name,arms in strata.items():add('SYN_MAIN',role,name,arms)
    return table


def reduce_canonical_source(rows):
    result=reduce_saved(rows);g=result['gate'];mean_class=g['numeric_gate_pass'];g.pop('ancestry',None)
    syn=result['family_metrics']['SYN_MAIN']['DEV']['treatment']
    ycb=result['family_metrics']['YCB_AUX']['DEV']['treatment']
    rd,re,ye=syn['D']['recall'],syn['E']['recall'],ycb['E']['recall']
    syn_d=rd is not None and rd>=.9;syn_e=re is not None and re>=.9;ycb_e=ye is not None and ye>=.5
    g.update(mean_gain_and_class_drop_gate_pass=mean_class,SYN_DEV_D_recall=rd,SYN_DEV_E_recall=re,YCB_DEV_E_recall=ye,
             SYN_DEV_D_recall_floor=.9,SYN_DEV_E_recall_floor=.9,YCB_DEV_E_recall_floor=.5,
             SYN_D_floor_pass=syn_d,SYN_E_floor_pass=syn_e,YCB_E_floor_pass=ycb_e,
             numeric_gate_pass=bool(mean_class and syn_d and syn_e and ycb_e),current_admission=False,
             selected_arm=None,model_adoption=False)
    result.update(arms=list(ARMS),new_CONTROL_optimizer_updates=1000,new_TREAT_optimizer_updates=1000,
                  same_source_exposed_DEV_not_freshblind=True,new_current_inference=0)
    return result


def self_test():
    rows=[]
    for f in (0,1):
        for y in (0,1):
            for k in range(10):
                correct=('D','E')[y];wrong=('E','D')[y]
                control=correct if k<5 and not(f==1 and y==1) else wrong
                good=9 if f==0 and y==1 else 5 if f==1 and y==1 else 10
                treatment=correct if k<good else wrong
                rows.append(dict(row_id=str(len(rows)),family=f,target=y,role='DEV',available=True,q_reference_distance_px=3 if f==0 else None,control_prediction=control,treatment_prediction=treatment))
    g=reduce_canonical_source(rows)['gate']
    assert g['numeric_gate_pass'] and g['SYN_DEV_E_recall']==.9 and g['YCB_DEV_E_recall']==.5
    syn_bad=[dict(r) for r in rows];syn_bad[18]['treatment_prediction']='D'
    g=reduce_canonical_source(syn_bad)['gate'];assert g['mean_gain_and_class_drop_gate_pass'] and not g['SYN_E_floor_pass'] and not g['numeric_gate_pass']
    ycb_bad=[dict(r) for r in rows];ycb_bad[34]['treatment_prediction']='D'
    g=reduce_canonical_source(ycb_bad)['gate'];assert g['mean_gain_and_class_drop_gate_pass'] and not g['YCB_E_floor_pass'] and not g['numeric_gate_pass']
    harm=[dict(r) for r in rows]
    for r in harm[:10]:r['control_prediction']='D'
    harm[9]['treatment_prediction']='E'
    g=reduce_canonical_source(harm)['gate'];assert not g['mean_gain_and_class_drop_gate_pass'] and not g['numeric_gate_pass']
    assert g['current_admission'] is False
    print(json.dumps(dict(completed=True,PASS=True,invented_checks=4,actual_result_or_payload_reads=0)))


if __name__=='__main__':self_test()
