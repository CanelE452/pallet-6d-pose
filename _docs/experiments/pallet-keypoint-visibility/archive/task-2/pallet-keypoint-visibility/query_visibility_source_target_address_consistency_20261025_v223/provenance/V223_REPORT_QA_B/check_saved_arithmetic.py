"""Independent scalar arithmetic on exact V223 saved SOURCE JSON/CSV only."""
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_source_target_address_consistency_20261025_v223'
H=P/'publication_root_preparation/V223_REPORT_QA_B'
A='C68SHARED_POINT_PART_LOGIT'
B='C69SEPARATE_POINT_PART_LOGITS'
READS=[]


def read(p,expected=None):
    p=Path(p)
    assert p.suffix in ('.json','.csv','.md') and p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).lower()
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nextfd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nextfd
        leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        with os.fdopen(leaf,'rb') as f:b=f.read()
    finally:os.close(fd)
    binding=dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))
    if expected:
        assert binding['sha256']==expected['sha256']
        if 'size_bytes' in expected:assert len(b)==expected['size_bytes']
    READS.append(binding)
    return json.loads(b) if p.suffix=='.json' else b.decode()


def near(a,b):
    assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-12),(a,b)


def prediction(r):
    z=r['raw_logit']
    if z is None:return -1
    assert math.isfinite(z)
    return int(z>=0)


def cm(rows):
    result=[[0,0,0],[0,0,0]]
    unknown=0
    for r in rows:
        t=r['target'];p=prediction(r)
        expected=r['prediction_D0_E1_Uminus1'] if 'prediction_D0_E1_Uminus1' in r else r['prediction_D0_E1']
        assert p==expected
        if t not in (0,1):unknown+=1;continue
        result[t][p if p in (0,1) else 2]+=1
    return result,unknown


def verify_metric(rows,m):
    c,u=cm(rows);known=sum(map(sum,c));correct=c[0][0]+c[1][1];pu=c[0][2]+c[1][2]
    assert m['rows']==len(rows) and m['known_D_E']==known and m['truth_UNKNOWN']==u
    assert m['CM_D_E_rows_D_E_U_columns']==c and m['correct']==correct and m['predicted_U_on_known']==pu
    near(m['U_rate_on_known'],pu/known if known else None)
    for k in (0,1):
        x=m['classes'][k];assert x['class_D0_E1']==k and x['truth']==sum(c[k]) and x['correct']==c[k][k]
        denom=c[0][k]+c[1][k]
        near(x['precision'],c[k][k]/denom if denom else None)
        near(x['recall'],c[k][k]/sum(c[k]) if sum(c[k]) else None)
    return c


def paired(left,right,m):
    assert len(left)==len(right)==m['rows']
    counter=Counter();classes={str(k):Counter() for k in (0,1)}
    for l,r in zip(left,right):
        assert l['row_id']==r['row_id'] and l['target']==r['target']
        t=l['target'];lc=prediction(l)==t;rc=prediction(r)==t
        name=('both_correct' if rc else 'damaged') if lc else ('recovered' if rc else 'both_wrong')
        counter[name]+=1
        if t in (0,1):classes[str(t)][name]+=1;classes[str(t)]['truth']+=1
    for key in ('recovered','damaged','both_correct','both_wrong'):assert m[key]==counter[key]
    assert m['net_correct_change']==counter['recovered']-counter['damaged']
    for k,d in classes.items():
        for key in ('truth','recovered','damaged','both_correct','both_wrong'):assert m['classes'][k][key]==d[key]
    return dict(counter)


def filter_role(rows,role):
    return rows if role=='ALL' else [r for r in rows if r['source_role']==role]


def main():
    fit=read(V/'paired_source_fit_v1/COMPLETE.json',dict(sha256='934c74fe7f0e017b0f106b2505ee02fdab2ae60da1adfae5f7601cd2a934792a'))
    done=read(V/'saved_source_comparison_v1/COMPLETE.json')
    summary=read(done['summary']['path'],done['summary'])
    seal=read(fit['all2_freeze']['path'],fit['all2_freeze'])
    fp=read(fit['input_fingerprints']['path'],fit['input_fingerprints'])
    replay=read(P/'publication_root_preparation/V223_ROOT_PREPARATION/SOURCE_SHARED_CONTROL_REPLAY_V222_C67.json')
    assert fit['completed'] and fit['PASS'] and fit['steps_each']==1100 and fit['parameters_each']==121561
    assert fit['from_scratch'] is False and fit['optimizer_state_reset'] is True
    assert set(fit['arms'])=={A,B} and fit['arms'][A]['initial']==fit['arms'][B]['initial']
    for k in ('old_point_draw_sha','naturalweak_draw_sha','CF_draw_sha'):
        assert fit['arms'][A][k]==fit['arms'][B][k]
    assert seal['completed'] and seal['PASS'] and seal['DEV_or_real_output_emitted'] is False
    for arm in (A,B):assert seal['checkpoints'][arm]==fit['arms'][arm]['checkpoint']
    assert fp['all2_freeze']==fit['all2_freeze'] and fp['DEV_after_both_final'] is True
    assert summary['training_complete']['sha256']=='934c74fe7f0e017b0f106b2505ee02fdab2ae60da1adfae5f7601cd2a934792a'
    assert summary['all2_freeze']==fit['all2_freeze']
    families={};score_bindings={};metric_nodes=0;paired_counts={}
    for arm in (A,B):
        out=fit['arms'][arm]
        assert out['use_CF'] is True and out['strong_score_logit']==('PART' if arm==A else 'POINT') and out['weak_CF_score_logit']=='PART'
        for family,key,rowkey in [('CF','CF_all144_outputs','rows144'),('NATURAL_WEAK','weak57_outputs','rows57'),('STRONG_POINT_TRAIN','strong_outputs','rows')]:
            d=read(out[key]['path'],out[key]);rows=d[rowkey]
            assert d['score_logit']==('POINT' if family=='STRONG_POINT_TRAIN' and arm==B else 'PART')
            assert len({r['row_id'] for r in rows})==len(rows)
            families.setdefault(family,{})[arm]=rows
            score_bindings[arm+'/'+family]=out[key]
    metrics={}
    for family,models in summary['families'].items():
        assert family in families
        for arm,roles in models['models'].items():
            for role,m in roles.items():
                rows=filter_role(families[family][arm],role)
                verify_metric(rows,m);metric_nodes+=1
                metrics[family+'/'+arm+'/'+role]=m
                if family=='CF':
                    assert m['components']==sorted({r['component_id'] for r in rows})
                    groups=defaultdict(dict)
                    for r in rows:groups[r['parent_row_id']][r['variant']]=r
                    assert all(set(rr)=={'BEFORE','COVER','SHAM'} for rr in groups.values())
                    expected=dict(parents=len(groups),all_three_correct=0,BEFORE_D=0,COVER_E=0,SHAM_D=0,cover_logit_greater_before=0,cover_logit_greater_sham=0)
                    for rr in groups.values():
                        expected['all_three_correct']+=all(prediction(x)==x['target'] for x in rr.values())
                        for v in ('BEFORE','COVER','SHAM'):expected[v+('_E' if v=='COVER' else '_D')]+=prediction(rr[v])==rr[v]['target']
                        expected['cover_logit_greater_before']+=rr['COVER']['raw_logit']>rr['BEFORE']['raw_logit']
                        expected['cover_logit_greater_sham']+=rr['COVER']['raw_logit']>rr['SHAM']['raw_logit']
                    assert m['triplets']==expected
                    assert set(m['by_native'])=={str(r['native_slot']) for r in rows}
                    for k,nm in m['by_native'].items():verify_metric([r for r in rows if r['native_slot']==int(k)],nm);metric_nodes+=1
        for role,pm in models['paired'].items():
            l=filter_role(families[family][A],role);r=filter_role(families[family][B],role)
            paired_counts[family+'/'+role]=paired(l,r,pm)
    text=read(V/'saved_source_comparison_v1/SOURCE_FAMILIES_SEPARATE_CM_PRECISION_RECALL.csv')
    csvrows=list(csv.DictReader(text.splitlines()))
    assert len(csvrows)==len(metrics)==16
    for x in csvrows:
        m=metrics[x['family']+'/'+x['arm']+'/'+x['role']];c=m['CM_D_E_rows_D_E_U_columns']
        for k,v in {'rows':m['rows'],'D':sum(c[0]),'E':sum(c[1]),'D_D':c[0][0],'D_E':c[0][1],'D_U':c[0][2],'E_D':c[1][0],'E_E':c[1][1],'E_U':c[1][2]}.items():assert int(x[k])==v
        for k,v in {'D_recall':m['classes'][0]['recall'],'D_precision':m['classes'][0]['precision'],'E_recall':m['classes'][1]['recall'],'E_precision':m['classes'][1]['precision'],'U_rate':m['U_rate_on_known']}.items():near(float(x[k]) if x[k] else None,v)
    groups=defaultdict(list)
    for row in fp['rows']:groups[row['input_sha256']].append(row)
    conflicts=[rr for rr in groups.values() if len({r['target'] for r in rr})>1]
    count=dict(rows=len(fp['rows']),distinct_signatures=len(groups),duplicate_unordered_pairs=sum(len(rr)*(len(rr)-1)//2 for rr in groups.values()),opposite_target_groups=len(conflicts))
    stored=summary['actual_decoded_model_input_consistency']
    assert all(stored[k]==v for k,v in count.items()) and not conflicts
    assert Counter(r['family'] for r in fp['rows'])=={'STRONG_POINT':6096,'NATURAL_WEAK':57,'CF_BEFORE':48,'CF_COVER':48,'CF_SHAM':48}
    for row in fp['rows']:
        family='STRONG_POINT_TRAIN' if row['family']=='STRONG_POINT' else ('NATURAL_WEAK' if row['family']=='NATURAL_WEAK' else 'CF')
        x=next(x for x in families[family][A] if x['row_id']==row['row_id'])
        assert row['target']==x['target'] and row['source_role']==x['source_role']
    assert replay['completed'] and sum(r['rows'] for r in replay['rows'])==6297
    assert all(r['logits_exact'] is True and r['max_absolute_logit_difference']==0 and r['prediction_changes']==0 for r in replay['rows'])
    gallery=read(V/'saved_source_comparison_v1/PORTABLE_GALLERY_KO.md')
    assert gallery.count('![](')==9 and 'weak target' in gallery and 'Two frame/components' in gallery
    assert summary['new_protected_model_inference'] is False and summary['new_native_point_gold'] is False
    assert summary['selected_arm'] is None and summary['adopted'] is False and summary['task_solved'] is False
    assert 'eager-parsed held360 metadata' in summary['protected_metadata_exception']
    result=dict(completed=True,PASS=True,input_bindings=READS,metric_nodes=metric_nodes,CSV_rows=16,
                raw_saved_score_rows_each=sum(len(x[A]) for x in families.values()),
                metrics=metrics,paired=paired_counts,input_fingerprint_grouping=count,
                input_fingerprint_scope='Grouping/counts independently recomputed from saved common decoded-input digests, no original bytes re-opened.',
                source_replay=dict(record=replay,old_V222_score_files_reopened_by_B=False,claim='Root provided saved SOURCE exact replay record, not independently replayed against old files in this task.'),
                run_metadata=dict(steps_each=1100,parameters_each=121561,same_init_and_all_four_draw_hashes=True,from_scratch=False,optimizer_reset=True,seconds=fit['seconds']),
                scope=dict(strong_score='C68 PART / C69 POINT actual strong-supervised logit',weak_CF_score='PART both',
                           control_POINT6_unused_expected=True,shared_encoder_remains=True,GT_derived_training_q8=True,
                           weak_not_physical_native_gold=True,DEV_small_correlated_preexposed=True,
                           new_private_forward=False,adopted=False,goal_solved=False,
                           B_images_models_input_arrays_GT_protected_read=0,
                           phase_global_protected_metadata_zero_claim=False,
                           disclosed_incident=summary['protected_metadata_exception'],
                           root_gallery_actual_views_not_replayed_by_B=True),
                final_report_QA_pending=True)
    b=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    p=H/'NUMERIC_QA.json';fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as f:f.write(b)
    print(json.dumps(dict(completed=True,PASS=True,path=str(p),sha256=hashlib.sha256(b).hexdigest(),metric_nodes=metric_nodes,CSV_rows=16,fingerprints=count,paired=paired_counts),ensure_ascii=False))


if __name__=='__main__':main()
