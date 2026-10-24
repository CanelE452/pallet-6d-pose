"""Source-only saved tile-region vs mosaic operator bridge. No learned forward."""
from pathlib import Path
from collections import Counter
import ast
import csv
import hashlib
import json
import math
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUN=ROOT/'runs/PHYSICAL_CUE_AUXILIARY_V1'


def bind(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(p):return json.loads(Path(p).read_text())


def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']


def write(name,d):
    with(HERE/name).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def writecsv(name,records):
    with(HERE/name).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def bilinear(array,x,y):
    x=np.clip(x,0,8);y=np.clip(y,0,8);x0=int(math.floor(x));y0=int(math.floor(y));x1=min(x0+1,8);y1=min(y0+1,8)
    dx=x-x0;dy=y-y0
    return float(array[y0,x0])*(1-dx)*(1-dy)+float(array[y0,x1])*dx*(1-dy)+float(array[y1,x0])*(1-dx)*dy+float(array[y1,x1])*dx*dy


def metric(labels,logits,available,weights):
    pred=np.where(available,np.where(logits<0,0,1),2);cm=np.zeros((2,3),np.float64);np.add.at(cm,(labels,pred),weights)
    ratio=lambda n,d:float(n/d)if d else None
    safe=np.where(available,logits,0).astype(np.float32)
    nll=F.binary_cross_entropy_with_logits(torch.from_numpy(safe),torch.from_numpy(labels.astype(np.float32)),reduction='none').numpy().astype(np.float64)
    byclass={}
    for c in(0,1):
        use=labels==c;ok=use&available;mass=float(weights[use].sum());amass=float(weights[ok].sum())
        byclass[str(c)]=dict(units=int(use.sum()),weight=mass,available_units=int(ok.sum()),available_weight=amass,
            unavailable_units=int((use&~available).sum()),unavailable_weight=mass-amass,
            all_known_NLL_with_fixed_log2_U_penalty=ratio(np.sum(weights[use]*nll[use]),mass),
            available_only_conditional_NLL=ratio(np.sum(weights[ok]*nll[ok]),amass))
    both=[byclass[str(c)]['all_known_NLL_with_fixed_log2_U_penalty']for c in(0,1)]
    return dict(N=len(labels),weight_total=float(weights.sum()),rows_D_E_cols_D_E_U=cm.tolist(),
        D_precision=ratio(cm[0,0],cm[:,0].sum()),D_recall_all_GT=ratio(cm[0,0],cm[0].sum()),
        E_precision=ratio(cm[1,1],cm[:,1].sum()),E_recall_all_GT=ratio(cm[1,1],cm[1].sum()),
        U_rate=ratio(cm[:,2].sum(),cm.sum()),class_NLL=byclass,
        macro_class_NLL=(sum(both)/2 if all(x is not None for x in both)else None),
        prevalence_weighted_NLL=ratio(np.sum(weights*nll),weights.sum()))


def main():
    torch.set_num_threads(1)
    rules=read(HERE/'RULES_BEFORE_SAVED_UNIT_ANALYSIS.json');assert rules['analysis_rules_frozen_before_unit_values']
    fit=read(ROOT/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json');complete=read(RUN/'FIT_COMPLETE.json');check(complete['model_freeze']);check(complete['tilemaps']);check(complete['source_results'])
    freeze=read(complete['model_freeze']['path']);assert freeze['best_update']==1900 and freeze['optimizer_updates']==2000
    b=fit['bindings']
    for name in('teacher_units','teacher_views','source_grid','model_code'):check(b[name])
    nodes=[n for n in ast.parse(Path(b['model_code']['path']).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='region_logit'];assert len(nodes)==1
    env={'torch':torch,'F':F};exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_region_logit_AST_only','exec'),env)
    region=env['region_logit']  # No model class, checkpoint or feature import.
    tiles=[json.loads(s)for s in Path(b['source_grid']['path']).read_text().splitlines()]
    views=[json.loads(s)for s in Path(b['teacher_views']['path']).read_text().splitlines()]
    assert len(tiles)==12900 and len(views)==604
    with np.load(complete['tilemaps']['path'],allow_pickle=False)as z:
        idx=z['tile_index'];ids=z['tile_id'];maps=z['physical_external_logit'];valid=z['map_valid']
    assert np.array_equal(idx,np.arange(12900))and np.array_equal(ids,[t['tile_id']for t in tiles])
    assert maps.shape==valid.shape==(12900,9,9)and maps.dtype==np.float32 and valid.dtype==bool and np.isfinite(maps).all()
    with np.load(b['teacher_units']['path'],allow_pickle=False)as z:
        units={k:z[k]for k in('tile_index','cue_view_index','training_label','loss_admitted','loss_weight','uncertainty9_normalized_xy_safe','full','uncertainty_inside_image','duplicate_weight')}
    splits=np.array([tiles[int(t)]['split']for t in units['tile_index']]);use=(splits=='VAL_HALL')&units['loss_admitted']
    ix=np.flatnonzero(use);assert len(ix)==1264
    ti=units['tile_index'][ix];vi=units['cue_view_index'][ix];y=units['training_label'][ix].astype(np.int64);w=units['loss_weight'][ix].astype(np.float64);coords=units['uncertainty9_normalized_xy_safe'][ix]
    assert np.isin(y,[0,1]).all()and np.all(w>0)and units['full'][ix].all()and units['uncertainty_inside_image'][ix].all()and np.isfinite(coords).all()
    z=[];av=[]
    for begin in range(0,len(ix),128):
        a=slice(begin,begin+128);pooled,available=region(torch.from_numpy(maps[ti[a]]),torch.from_numpy(valid[ti[a]]),torch.from_numpy(coords[a]))
        z.append(pooled.numpy());av.append(available.numpy())
    z=np.concatenate(z);av=np.concatenate(av);assert z.dtype==np.float32 and np.isfinite(z).all()
    # Independent scalar F64 oracle, diagnostic only, never changes FP32 decisions.
    oracle=[];oracle_av=[]
    for t,points in zip(ti,coords):
        values=[];ok=[]
        for nx,ny in points.astype(np.float64):
            x=(nx+1)*9/2-.5;ycoord=(ny+1)*9/2-.5
            den=bilinear(valid[t],x,ycoord);num=bilinear(maps[t]*valid[t],x,ycoord)
            values.append(num/max(den,1e-12));ok.append(den>0)
        oracle.append(np.mean(values));oracle_av.append(all(ok))
    oracle=np.array(oracle);oracle_av=np.array(oracle_av)
    assert np.array_equal(av,oracle_av)
    oracle_sign_diff=int(np.sum(av&((z<0)!=(oracle<0))))
    L=np.array([tiles[int(t)]['roi_side']for t in ti],np.int32)
    centre_inside=(coords[:,:,0].min(1)<=0)&(coords[:,:,0].max(1)>=0)&(coords[:,:,1].min(1)<=0)&(coords[:,:,1].max(1)>=0)
    pred=np.where(av,np.where(z<0,'D','E'),'U')
    records=[]
    for j,u in enumerate(ix):
        v=views[int(vi[j])];assert v['cue_view_index']==int(vi[j])and v['split']=='VAL_HALL'and v['teacher_label']==int(y[j])
        records.append(dict(unit_index=int(u),tile_index=int(ti[j]),tile_id=tiles[int(ti[j])]['tile_id'],cue_view_index=int(vi[j]),cue_id=v['cue_id'],
            view=v['view'],L=int(L[j]),teacher_label=int(y[j]),loss_weight=float(w[j]),region_available=bool(av[j]),region_logit_FP32=float(z[j]),prediction=pred[j],
            qcentre_inside_original_uncertaintybox=bool(centre_inside[j]),centre_label_or_centre_performance_created=False))
    by_view_scale=[];knownVAL=[v for v in views if v['split']=='VAL_HALL'and v['teacher_label']in(0,1)]
    assert len(knownVAL)==170
    for v in knownVAL:
        for side in(128,256):
            sel=(vi==v['cue_view_index'])&(L==side);inside=sel&centre_inside
            mass=float(w[sel].sum());expected=.5*float(v['duplicate_weight'])
            by_view_scale.append(dict(cue_view_index=v['cue_view_index'],cue_id=v['cue_id'],view=v['view'],teacher_label=v['teacher_label'],L=side,
                admitted_units=int(sel.sum()),original_weight_mass=mass,expected_scale_mass_if_covered=expected,
                missing_FULL_scale=not bool(sel.any()),mass_matches_original_rule=abs(mass-expected)<1e-12 if sel.any()else None,
                available_units=int(np.sum(sel&av)),available_mass=float(w[sel&av].sum()),
                qcentre_inside_units=int(inside.sum()),qcentre_inside_weight_mass=float(w[inside].sum()),
                qcentre_inside_subset_coverage_only=True))
    groups=[]
    for c in(0,1):
        for side in(128,256):
            sel=(y==c)&(L==side);inside=sel&centre_inside
            all_v=set(vi[sel]);inside_v=set(vi[inside])
            groups.append(dict(teacher_label=c,L=side,all_admitted_units=int(sel.sum()),all_weight_mass=float(w[sel].sum()),
                qcentre_inside_units=int(inside.sum()),qcentre_inside_weight_mass=float(w[inside].sum()),
                all_known_cue_views=len(all_v),cue_views_with_any_qcentre_inside=len(inside_v),
                qcentre_inside_fraction_units=float(inside.sum()/sel.sum()),qcentre_inside_fraction_mass=float(w[inside].sum()/w[sel].sum()),
                performance_not_computed=True))
    raw=metric(y,z,av,np.ones_like(w));weighted=metric(y,z,av,w)
    byscale={str(side):dict(raw=metric(y[L==side],z[L==side],av[L==side],np.ones_like(w[L==side])),
        original_weighted=metric(y[L==side],z[L==side],av[L==side],w[L==side]))for side in(128,256)}
    summary=read(complete['source_results']['path'])['source_metrics']['VAL_HALL']
    writecsv('ALL1264_VAL_ADMITTED_UNIT_SCORES.csv',records);writecsv('ALL170_VIEW_SCALE_COVERAGE.csv',by_view_scale)
    result=dict(completed=True,PASS=True,rules=bind(HERE/'RULES_BEFORE_SAVED_UNIT_ANALYSIS.json'),code=bind(__file__),
        bindings=dict(fit_protocol=bind(ROOT/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json'),fit_complete=bind(RUN/'FIT_COMPLETE.json'),model_freeze=complete['model_freeze'],
            saved_tilemaps=complete['tilemaps'],source_results=complete['source_results'],teacher_units=b['teacher_units'],teacher_views=b['teacher_views'],source_grid=b['source_grid'],region_logit_code=b['model_code']),
        unit_population=dict(VAL_admitted_units=1264,all_77916_units_order_preserved=True,no_besttile_or_favorable_filter=True,
            distinct_cue_views=len(set(vi)),known_VAL_cue_views=170,known_cue_views_without_any_admitted_FULL_unit=170-len(set(vi)),
            original_effective_weight_total=float(w.sum()),available_units=int(av.sum()),unavailable_units=int((~av).sum()),
            unavailable_zero_gradient_weight_mass=float(w[~av].sum()),all_viewscale_weight_rules_match=all(r['mass_matches_original_rule']is not False for r in by_view_scale),
            viewscale_missing_FULL_N=sum(r['missing_FULL_scale']for r in by_view_scale)),
        single_tile_original_uncertainty9_region=dict(unit='weak physical-cue/view/tile units, not independent physical cues or human native corners',
            raw_unit_counts=raw,original_loss_weighted=weighted,by_L=byscale,decoder='available z<0 D; z>=0 E; unavailableU',
            NLL_U_policy='All known kept, fixed zero-logit BCE ln2 penalty; conditional available-only NLL separate. Training unavailable gradient0 unchanged.'),
        native_mosaic_view_level=dict(unit='known weak cue/view, not tile unit',known_N=170,region=summary['region'],reference_only=summary['reference'],paired_views=summary['paired_views']),
        qcentre_inside_uncertainty_coverage_only=dict(all_units=int(centre_inside.sum()),all_weight_mass=float(w[centre_inside].sum()),
            all_distinct_cue_views=len(set(vi[centre_inside])),all_distinct_view_scale=len({(int(v),int(side))for v,side in zip(vi[centre_inside],L[centre_inside])}),
            groups=groups,box_rule='Closed originalunc9 bounding rectangle around q-origin0, no tolerance or relocation',
            qfar_cue_label_not_assigned_to_centre=True,centre_performance_computed=False),
        independent_numeric_audit=dict(runtime='CPU Torch FP32 frozen region formula, no learned forward',max_abs_F32_vs_F64_logit=float(np.max(np.abs(z-oracle))),
            availability_disagreements=0,threshold_sign_disagreements=oracle_sign_diff,min_abs_available_FP32_logit=float(np.min(np.abs(z[av])))),
        interpretation_KO=[
            '단일 tile 결과는 각 tile의 원래 uncertainty영역을 평가한 값이며 q중심 정답 성능이 아니다. 모든1264 FULL단위를 유지했고 원래 cue/view/scale weight만 적용했다.',
            '170개 view의 native mosaic는 모든129tile을 먼저 고정 tent/valid로 합친 뒤 원래영역을 평가한다. tile단위1264와 view단위170의 raw counts를 합치거나 동일표본의 순수 ablation으로 부를 수 없다.',
            'q중심이 원래 uncertaintybox안인 부분은 coverage만 보고한다. qfar의 cue label을 centre label로 만들지 않았고 GT로tile을골라재추론하거나recenter하지 않았다.',
            'source L128/256·한physicalbin·AI조건부cue라는 한계가 그대로다. 이 source 단일tile bridge가 current native8 의미/target association/다른L 전이를 검증하지는 않는다.',
            'factual support0은 U 및 zero-gradient mass로 남기며 E/D의 정답으로 만들지 않는다. 본 진단은 어떤 모델/threshold/checkpoint/입력 선택도 변경하지 않는다.',
        ],
        new_model_forward_fit_weights_RGB_current_access_PnP=0,outputs=[bind(HERE/'ALL1264_VAL_ADMITTED_UNIT_SCORES.csv'),bind(HERE/'ALL170_VIEW_SCALE_COVERAGE.csv')])
    write('SOURCE_SINGLE_TILE_REGION_BRIDGE.json',result)
    write('FINAL_SEAL.json',dict(completed=True,PASS=True,report=bind(HERE/'SOURCE_SINGLE_TILE_REGION_BRIDGE.json'),files=[bind(p)for p in sorted(HERE.iterdir())if p.is_file()],
        actual_new_model_forward_training_RGB_weights_current_PnP=0,counts_unit_types_not_merged=True))
    print(json.dumps(dict(report=bind(HERE/'SOURCE_SINGLE_TILE_REGION_BRIDGE.json'),weighted=weighted,coverage=result['qcentre_inside_uncertainty_coverage_only'],population=result['unit_population'],numeric=result['independent_numeric_audit']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
