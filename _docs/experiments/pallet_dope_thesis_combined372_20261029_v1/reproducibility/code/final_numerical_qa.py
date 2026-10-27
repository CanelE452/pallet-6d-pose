"""Final372 scalar/observed-error verification only; no NN/PnP/bootstrap draws.

ROOT must enumerate all exact derived inputs in PUBLICATION_INPUTS.json.
PRIVATE caches contain point errors, never GT xy, and cannot enter the payload.
"""
import math
from pathlib import Path
import publication_common as P


def distribution(values):
    import numpy as np
    P.require(all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in values),'Invalid point-error array')
    x=np.asarray(values,float)
    if not len(x):return dict(n=0,mean=None,variance=None,std=None,median=None,q1=None,q3=None)
    q1,med,q3=np.quantile(x,[.25,.5,.75],method='linear')
    return dict(n=len(x),mean=float(x.mean()),variance=float(x.var(ddof=0)),std=float(x.std(ddof=0)),median=float(med),q1=float(q1),q3=float(q3))


def catalogue():
    conditions=('control','full','no_flip','no_confidence','no_ransac','no_loo','no_final_support','confidence_only')
    return {(k,'R0',0) for k in P.EXPECTED}|{(k,c,r) for k in P.EXPECTED for c in conditions for r in (1,2)}


def point_groups(io,cache_role,table_role,expected_groups):
    cache=io.read(io.one(cache_role));table=io.csv(io.one(table_role));rows=cache['frames']
    P.require(cache['complete'] is True and len(table)==expected_groups and len({r['group'] for r in table})==expected_groups,'Point group count differs')
    wanted=catalogue() if expected_groups==51 else {(k,c,2) for k in P.EXPECTED for c in ('A','B','C','D','E')}
    P.require({(s['kind'],s['condition'],int(s['round'])) for s in table}==wanted,'Exact point kind/condition/round catalogue differs')
    if not hasattr(io,'point_frames'):io.point_frames={}
    io.point_frames['ratio' if expected_groups==15 else 'ablation']=rows
    if not hasattr(io,'point_summaries'):io.point_summaries={}
    io.point_summaries['ratio' if expected_groups==15 else 'ablation']=table
    fields=0
    for s in table:
        rr=[r for r in rows if r['group']==s['group']];kind=s['kind'];n,nref=P.EXPECTED[kind]
        P.require(len(rr)==len({r['id'] for r in rr})==n and sum(r['n_reference'] for r in rr)==nref,'Same-kind frame/reference denominator differs')
        P.require(int(s['n_DEV'])==n and int(s['n_reference_points'])==nref,'Summary population differs')
        available=s['available']=='True';P.require(all(r['available']==available for r in rr),'Availability differs')
        if not available:
            P.require(all(s[k]=='' for k in (*P.QUALITY,'n_observed_reference_predictions')),'Failed model was scored as zero')
            continue
        observed=sum(r['n_observed'] for r in rr);errors=[e for r in rr for e in r['errors_px']]
        P.require(observed==len(errors),'Observed error denominator differs')
        for r in rr:
            P.require(r['n_observed']==len(r['errors_px'])<=r['n_reference'],'Frame observation count differs')
            for t in (5,10,20):P.require(r['correct'+str(t)]==sum(e<=t for e in r['errors_px']),'Frame threshold class differs')
        actual={'coverage':observed/nref,'conditional_precision10':sum(r['correct10'] for r in rr)/observed if observed else None,
            'n_observed_reference_predictions':observed}
        for t in (5,10,20):actual['PCK'+str(t)]=sum(r['correct'+str(t)] for r in rr)/nref
        actual.update({'conditional_pixel_error_'+k:v for k,v in distribution(errors).items()})
        for key,value in actual.items():P.same(value,s[key],s['group']+'/'+key);fields+=1
    return {'groups':expected_groups,'fields_checked':fields,'frames_per_kind':P.EXPECTED}


def pose_groups(io):
    summary=io.csv(io.one('table_supplemental_pose'));frames=io.csv(io.one('frames_supplemental_pose'))
    P.require(len(summary)==51 and len({r['group'] for r in summary})==51,'Expected51 supplemental pose groups')
    P.require({(s['kind'],s['condition'],int(s['round'])) for s in summary}==catalogue(),'Exact pose kind/condition/round catalogue differs')
    fields=0
    for s in summary:
        rr=[r for r in frames if r['group']==s['group']];n=P.EXPECTED[s['kind']][0]
        P.require(len(rr)==len({r['id'] for r in rr})==int(s['n_DEV'])==n,'Pose full-population denominator differs')
        P.require(s['scope']=='SUPPLEMENTAL_2D_RECONSTRUCTED_REFERENCE','Pose truth label differs')
        available=s['available']=='True';nref=sum(r['reference_PnP_success']=='True' for r in rr)
        P.same(nref,s['n_reference_PnP_success'],s['group']+'/reference_success')
        for r in rr:
            if r['cross_hypothesis_rotation_NA']=='True':P.require(r['rotation_indexed_deg']==r['rotation_symmetry_deg']=='','Cross-hypothesis rotation cannot be numeric')
            if r['reference_PnP_success']!='True' or r['model_PnP_success']!='True':
                P.require(all(r[m]=='' for m in ('translation_error_m','relative_translation_error','rotation_indexed_deg','rotation_symmetry_deg')),'Failed pose pair must have no error values')
        if not available:
            P.require(s['n_model_PnP_success']=='' and s['PnP_success_rate_full_DEV']=='','Unavailable pose must remain NA');continue
        nmodel=sum(r['model_PnP_success']=='True' for r in rr)
        nboth=sum(r['reference_PnP_success']=='True' and r['model_PnP_success']=='True' for r in rr)
        for key,value in {'n_model_PnP_success':nmodel,'PnP_success_rate_full_DEV':nmodel/n,'n_model_and_reference_PnP_success':nboth,
            'PnP_success_rate_given_reference':nboth/nref if nref else None,
            'n_cross_hypothesis_rotation_NA':sum(r['cross_hypothesis_rotation_NA']=='True' for r in rr)}.items():P.same(value,s[key],s['group']+'/'+key);fields+=1
        for metric in ('translation_error_m','relative_translation_error','rotation_indexed_deg','rotation_symmetry_deg'):
            values=[P.number(r[metric]) for r in rr if r[metric]!='']
            for stat,value in distribution(values).items():P.same(value,s[metric+'_'+stat],s['group']+'/'+metric+'/'+stat);fields+=1
        for stat in ('n','mean','variance','std','median','q1','q3'):
            value=P.number(s['translation_error_m_'+stat]);value=None if value is None else value*(1 if stat=='n' else 10000 if stat=='variance' else 100)
            P.same(value,s['translation_error_cm_'+stat],s['group']+'/cm/'+stat);fields+=1
    return {'groups':51,'fields_checked':fields,'WOOD_full_PnP_denominator':125,'independent_measured_6D':False}


def manual_groups(io,role,source,kind,expected_groups,expected_points):
    table=io.csv(io.one(role));P.require(len(table)==expected_groups,'Manual group count differs');fields=0
    identity=lambda r:(r['kind'],r['condition'],int(r['round']),r['group'])
    wanted={identity(r) for r in io.point_summaries[source] if r['kind']==kind}
    actual=[identity(r) for r in table]
    P.require(len(actual)==len(set(actual))==expected_groups and set(actual)==wanted,'Exact unique manual catalogue differs')
    for s in table:
        rr=[r for r in io.point_frames[source] if r['group']==s['group']]
        P.require(s['kind']==kind and len(rr)==P.EXPECTED[kind][0] and sum(r['manual_n_reference'] for r in rr)==expected_points,'Manual population differs')
        if s['available']!='True':P.require(all(s[k]=='' for k in P.QUALITY),'Unavailable manual model must remain NA');continue
        for r in rr:
            P.require(r['manual_n_observed']==len(r['manual_errors_px'])<=r['manual_n_reference'],'Manual frame observation count differs')
            for t in (5,10,20):P.require(r['manual_correct'+str(t)]==sum(e<=t for e in r['manual_errors_px']),'Manual frame PCK class differs')
        seen=sum(r['manual_n_observed'] for r in rr);ee=[e for r in rr for e in r['manual_errors_px']]
        P.require(seen==len(ee),'Manual observed count differs')
        values={'n_reference_points':expected_points,'n_observed_reference_predictions':seen,'coverage':seen/expected_points,
            'conditional_precision10':sum(r['manual_correct10'] for r in rr)/seen if seen else None}
        for t in (5,10,20):values['PCK'+str(t)]=sum(r['manual_correct'+str(t)] for r in rr)/expected_points
        values.update({'conditional_pixel_error_'+k:v for k,v in distribution(ee).items()})
        for key,value in values.items():P.same(value,s[key],role+'/'+s['group']+'/'+key);fields+=1
    return dict(role=role,groups=expected_groups,reference_points=expected_points,fields_checked=fields)


def ci_catalogue(role):
    """Exact catalogues from frozen MAIN/combined/ratio/main-comparison producers."""
    kinds=('WOOD',) if 'WOOD' in role else ('GREEN',) if 'GREEN' in role else tuple(P.EXPECTED)
    panels=lambda k:('all_annotated',{'WOOD':'WOOD_manual330','GREEN':'GREEN_manual600'}[k]) if 'recording' in role else (
        ({'WOOD':'WOOD_manual330','GREEN':'GREEN_manual600'}[k],) if 'manual' in role else ('all_annotated',))
    unit='paired_recording' if 'recording' in role else 'paired_frame'
    if role.startswith('ci_ratio'):
        pairs=[(c,2,'C',2) for c in ('A','B','D','E')]+[(c,2,'control_R2',2) for c in ('A','B','C','D','E')]
    elif role.startswith('ci_main_comparisons'):
        pairs=[('full',1,'control',1),('full',2,'control',2),('full',1,'no_flip',1),('full',2,'no_flip',2),
            ('full',2,'full',1),('control',2,'control',1)]
    else:
        pairs=[('full',r,c,r) for r in (1,2) for c in ('no_confidence','no_ransac','no_loo','no_flip','no_final_support','confidence_only')]
    return {(k,a,ar,b,br,m,p,unit) for k in kinds for a,ar,b,br in pairs for m in P.QUALITY for p in panels(k)}


def ci_identity(role,row):
    if role.startswith('ci_main_comparisons'):
        pair=(row['candidate_condition'],int(row['candidate_round']),row['comparator_condition'],int(row['comparator_round']))
        P.require(row['candidate_group']==f"{row['kind']}__{pair[0]}__R{pair[1]}"
            and row['comparator_group']==f"{row['kind']}__{pair[2]}__R{pair[3]}",'Main comparison group/condition differs')
        P.require(row['comparison']==f'{pair[0]}_R{pair[1]}_minus_{pair[2]}_R{pair[3]}','Main comparison label differs')
    elif role.startswith('ci_ratio'):pair=(row['candidate'],2,row['comparator'],2)
    else:pair=(row['candidate'],int(row['round']),row['comparator'],int(row['round']))
    unit=row.get('resampling_unit','')
    if not unit:
        P.require(role.startswith('ci_ratio') or role in ('ci_ablation','ci_ablation_GREEN_manual') and row['kind'] in ('PLASTIC','GREEN'),
            'Explicit resampling unit missing from new producer')
        unit='paired_frame'  # Frozen MAIN P/G and ratio producers omit this field.
    return (row['kind'],*pair,row['metric'],row['panel'],unit)


def ci_groups(io):
    reports=[]
    for role,n in (('ci_ablation',180),('ci_ablation_WOOD_manual',60),('ci_ablation_GREEN_manual',60),('ci_ablation_WOOD_recording',120),
        ('ci_ratio',135),('ci_ratio_GREEN_manual',45),('ci_main_comparisons',90),('ci_main_comparisons_WOOD_manual',30),
        ('ci_main_comparisons_GREEN_manual',30),('ci_main_comparisons_WOOD_recording',60)):
        rows=io.csv(io.one(role));P.require(len(rows)==n,'CI row count differs: '+role)
        identities=[ci_identity(role,r) for r in rows];wanted=ci_catalogue(role)
        P.require(len(identities)==len(set(identities))==n and set(identities)==wanted,'Exact unique CI catalogue differs: '+role)
        for r in rows:
            P.require(int(r['replicates'])==10000 and int(r['seed'])==20261026,'CI seed/draw count differs')
            low,high=P.number(r['ci95_low']),P.number(r['ci95_high'])
            if low is not None or high is not None:
                P.require(low is not None and high is not None and low<=high,'Malformed CI interval')
                label=('candidate_higher' if low>0 else 'candidate_lower' if high<0 else 'includes_zero') if role.startswith(('ci_ratio','ci_main_comparisons')) else ('full_higher' if low>0 else 'full_lower' if high<0 else 'includes_zero')
                P.require(r['decision']==label,'CI sign label differs')
            undefined=P.number(r.get('undefined_replicates'))
            if undefined is not None and undefined>0:P.require(low is None and high is None,'Undefined draws were discarded instead of whole-CI NA')
            source=io.point_frames['ratio' if role.startswith('ci_ratio') else 'ablation'];prefix='manual_' if 'manual' in r['panel'] else ''
            if role.startswith('ci_main_comparisons'):
                aa=[x for x in source if x['group']==r['candidate_group']];bb=[x for x in source if x['group']==r['comparator_group']]
            else:
                matched=lambda condition:[x for x in source if x['kind']==r['kind'] and x['condition']==condition
                    and (role.startswith('ci_ratio') or x['round']==int(r['round']))]
                aa,bb=matched(r['candidate']),matched(r['comparator'])
            P.require(aa and bb and [x['id'] for x in aa]==[x['id'] for x in bb],'CI paired population/order differs')
            field='point_difference' if role.startswith(('ci_ratio','ci_main_comparisons')) else 'point_difference_full_minus_ablation'
            if not all(x['available'] for x in aa+bb):P.require(r[field]=='','Unavailable CI point difference must be NA');continue
            numerator='n_observed' if r['metric']=='coverage' else 'correct10' if r['metric']=='conditional_precision10' else 'correct'+r['metric'][3:]
            denominator='n_observed' if r['metric']=='conditional_precision10' else 'n_reference'
            da,db=sum(x[prefix+denominator] for x in aa),sum(x[prefix+denominator] for x in bb)
            value=sum(x[prefix+numerator] for x in aa)/da-sum(x[prefix+numerator] for x in bb)/db if da and db else None
            P.same(value,r[field],role+'/'+r['kind']+'/'+r['metric']+'/difference')
        reports.append({'role':role,'rows':n,'exact_unique_catalogue':True,'bootstrap_draws_recomputed':0})
    return reports


def main():
    io=P.Inputs(P.NUMERICAL_INPUTS);io.gates()
    merged=io.read(io.one('merged_complete'))
    P.require(merged['complete'] is True and merged['passed'] is True and merged['main_groups']==51
        and merged['primary_frames']==372 and merged['primary_reference_points']==2891,'Combined completion contract differs')
    for role in ('table_ablation_2d','private_ablation_errors','ci_ablation','table_WOOD_manual','table_GREEN_manual',
        'ci_ablation_WOOD_manual','ci_ablation_GREEN_manual','ci_ablation_WOOD_recording'):
        path=io.one(role);P.require(merged['output_sha256'][str(path)]==P.sha(io.path(path)),'Merged output receipt differs: '+role)
    checks={'ablation':point_groups(io,'private_ablation_errors','table_ablation_2d',51),
        'ratio':point_groups(io,'private_ratio_errors','table_ratio',15),
        'supplemental_pose':pose_groups(io),'CI':ci_groups(io)}
    checks['manual']=[manual_groups(io,'table_WOOD_manual','ablation','WOOD',17,330),
        manual_groups(io,'table_GREEN_manual','ablation','GREEN',17,600),
        manual_groups(io,'table_ratio_GREEN_manual','ratio','GREEN',5,600)]
    teacher=io.csv(io.one('table_future_teacher'))
    expected_teacher={(k,c,'all_annotated') for k in P.EXPECTED for c in ('A','B','C','D','E')}|{('GREEN',c,'GREEN_manual600') for c in ('A','B','C','D','E')}
    P.require(len(teacher)==20 and {(r['kind'],r['condition'],r['panel']) for r in teacher}==expected_teacher,'Exact15 all-reference plus5 GREEN manual teacher rows required')
    P.require({(r['kind'],r['condition']) for r in teacher}=={(k,c) for k in P.EXPECTED for c in ('A','B','C','D','E')},'Future teacher15 combinations missing')
    P.require(len([r for r in teacher if r['panel']=='all_annotated'])==15,'Future teacher all-reference row count differs')
    for r in teacher:
        count=lambda key:int(r[key]);nref=600 if r['panel']=='GREEN_manual600' else P.EXPECTED[r['kind']][1]
        P.require(r['teacher']=='new_R2_hypothetical_R3_teacher' and count('n_frames')==P.EXPECTED[r['kind']][0]
            and count('n_reference_points')==nref,'Teacher scope/denominator differs')
        seen,selected=count('observed_teacher_points'),count('scorable_accepted_points')
        P.require(0<=selected<=seen<=nref and seen+count('missing_teacher_points')==nref,'Teacher count partition differs')
        P.same(selected/nref,r['selected_reference_coverage'],'Teacher selected-reference coverage')
        P.same(seen/nref,r['teacher_reference_coverage'],'Teacher observed-reference coverage')
        for t in (5,10,20):
            correct,wrong=count('teacher_correct'+str(t)),count('teacher_incorrect'+str(t))
            sc,sw=count('selected_correct'+str(t)),count('selected_incorrect'+str(t))
            P.require(correct+wrong==seen and sc+sw==selected and sc<=correct and sw<=wrong,'Teacher correctness partition differs')
            for key,value in {f'precision{t}':sc/selected if selected else None,f'correct_retention{t}':sc/correct if correct else None,
                f'incorrect_acceptance{t}':sw/wrong if wrong else None}.items():P.same(value,r[key],'Teacher '+key)
    checks['optional_372_diagnostics']=[]
    for role,expected_rows in (('table_R0_components372',35),('table_T3_stages372',165)):
        paths=[p for p,item in io.files.items() if item['role']==role]
        if paths:
            P.require(len(paths)==1,'Unique new372 diagnostic table required');rr=io.csv(paths[0]);P.require(len(rr)==expected_rows,'New372 diagnostic row count differs')
            checks['optional_372_diagnostics'].append(dict(role=role,rows=len(rr),no_old292_relabeling=True))
    io.unchanged();P.write(P.OUT/'preflight/FINAL_NUMERICAL_QA.json',dict(passed=True,complete=True,scope='primary372 point+reconstructed_pose+ratio+future_teacher+CI scalar QA',
        checks=checks,source_sha256=io.bindings,numerical_inputs_sha256=io.manifest_sha,numerical_inputs_path=str(io.manifest_path),
        code_sha256=P.sha(Path(__file__).resolve()),helper_sha256=P.sha(Path(P.__file__).resolve()),
        neural_forwards=0,PnP_fits=0,new_bootstrap_draws=0,raw_RGB_reads=0,raw_annotation_reads=0,
        all15_ratios_required=True,WOOD_inconclusive_CI_does_not_stop_delivery=True))


if __name__=='__main__':main()
