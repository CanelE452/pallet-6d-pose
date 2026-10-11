"""Render frozen recorded proxy results and the six pre-existing RGB cases.

No pose, GT error, moment or CI is recomputed. Fixed case selection precedes
V19 outcomes. Mask labels and model projections do not certify physical ownership.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

NEW=('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS','HYBRID_OBSERVED_FRAME_POINT_ONLY')
OLD='HYBRID_POINT_CONSENSUS_RESTORED'
METHODS=('BASE','N3_SUBPIX','HYBRID_STANDARD_PNP','NATIVE_STANDARD_PNP',OLD,*NEW)
CONTRASTS=((NEW[0],NEW[1]),(NEW[0],'N3_SUBPIX'),(NEW[0],METHODS[3]),(NEW[0],METHODS[2]),(NEW[0],OLD),(NEW[1],OLD))
DISPLAY=('N3_SUBPIX',NEW[1],NEW[0])
LABELS={'BASE':'Base','N3_SUBPIX':'N3→cornerSubPix',METHODS[2]:'Hybrid ordinary',METHODS[3]:'Native ordinary',OLD:'V18 point restored',NEW[0]:'Observed-frame point + line',NEW[1]:'Same-q point only'}
FIELDS=('translation_cm','rotation_deg','ADDsym_cm');UNITS=('cm','degree','cm')
MASKS=('MATCHES_ON_KNOWN','WRONG_ON_KNOWN','NO_KNOWN_VISIBILITY','MASK_NOT_APPLIED')
OUTCOMES=('BOTH_BETTER','BOTH_WORSE','MIXED_OR_UNCHANGED','POSE_UNAVAILABLE')
STATS_SHA='4cad27af50af1711ac1817e3c24cfddf73386b535f1cf8c30950950ef2ab1b0c'
NAMES=('FIGURE_PROTOCOL.json','FIGURES_STARTED.json','FIGURE_BINDINGS.json','FIGURES_FAILURE.json','FIGURE_CASE_ROWS.jsonl.gz','FIGURE_DIAGNOSTIC_COUNTS.json')
LIMITS=dict(new_pose_GT_error_moment_CI_model_training_calls=0,cases_selected_from_V19_results=False,
    reference='saved N3-phase GEOMETRIC_PROXY, not independently measured physical R/t',
    physical_line_ownership='UNKNOWN',model_segments_are_virtual=True,old_full_V16_564_FAIL_remains_FAIL=True,
    mask_state_is_saved_known_visibility_attempt_metadata=True,fallback_preserves_original_N3=True,
    visible_damage_means_saved_proxy_error_increase_not_physical_object_damage=True,ROOT_visual_review_claimed=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_bytes())
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank saved row');yield json.loads(line)
def metadata_helpers(path):
    require(binding(path)['sha256']==STATS_SHA,'immutable statistics metadata source')
    names={'require','read','binding','same','paths','metadata'};nodes=[n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==names,'exact metadata-only AST whitelist')
    env=dict(Path=Path,hashlib=hashlib,json=json,METHODS=METHODS,CONTRASTS=CONTRASTS,NEW=NEW,__file__=str(path))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+':unchanged_metadata_only','exec'),env)
    return env
def inputs(a):
    checker=Path(a.stats_checker).absolute();h=metadata_helpers(checker);f=h['paths'](a);b={k:binding(p) for k,p in f.items()};m=h['metadata'](f,b)
    sp,ss,sc=[Path(a.stats_input)/n for n in ('STATISTICS_CHECK_PROTOCOL.json','STATISTICS_CHECK_STARTED.json','STATISTICS_CHECKS.json')]
    p,s,c=[read(v) for v in (sp,ss,sc)]
    require(p['schema']=='independent_observed_frame_statistics_protocol_v19' and c['schema']=='independent_observed_frame_statistics_checks_v19' and c['complete'] is True and c['passed'] is True and c['error'] is None,'actual independent saved-statistics PASS required')
    require(c['inputs']==p['inputs']==s['inputs']==b and c['old_full_V16_validation_passed'] is False and c['new_production_calls']==s['new_production_calls']==0,'actual same frozen statistics inputs')
    for item in (c['own_protocol'],s['own_protocol']):require(same(binding(sp),item),'actual statistics protocol')
    for item in (p['checker'],s['checker'],c['checker']):require(same(binding(checker),item),'current statistics checker')
    require(all(c['actual_counts'][k]==v for k,v in (('frames',245),('methods',7),('saved_error_rows_streamed',1960),('saved_error_rows_selected',1715),('moment_distributions_checked',675),('moment_scalars_checked',5400),('paired_groups_checked',54),('metric_CI_slots_checked',162))),'full seven-method arithmetic coverage')
    f.update(renderer=Path(__file__),statistics_protocol=sp,statistics_started=ss,statistics_checks=sc,
        case_protocol=Path(a.repo)/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2/VISUAL_CASE_PROTOCOL.json',
        source_render_v17=Path(a.repo)/'scripts/research/pallet_hybrid_standard_controls_20261011_v17/render.py',
        source_render_v18=Path(a.repo)/'scripts/research/pallet_consensus_decision_repair_20261011_v18/render.py',
        posthoc=Path(a.input)/'POSTHOC_ROWS.jsonl.gz')
    require(same(binding(f['posthoc']),m['posthoc_rows']),'actual posthoc490 bytes')
    case,cohort=read(f['case_protocol']),read(f['cohort']);require(same(binding(f['cohort']),case['cohort']),'original six-case cohort bytes')
    cases=case['cases'];by_id={r['id']:r for r in cohort['frames']}
    require(len(cases)==len({r['id'] for r in cases})==6 and len(cohort['ids'])==len(by_id)==245 and Counter(r['label'] for r in cases)==Counter(clean=3,moderate=3),'exact original3clean+3moderate/full245')
    for i,r in enumerate(cases):
        require(r['id'] in by_id and all(r[k]==by_id[r['id']][k] for k in ('session','label','image')),'original case metadata')
        relative=Path(r['image']['path']);require(not relative.is_absolute() and '..' not in relative.parts,'safe existing RGB path')
        f['image:'+str(i)]=Path(a.source_root)/relative;require(same(binding(f['image:'+str(i)]),r['image']),'original fixed-case RGB bytes')
    return f,{k:binding(p) for k,p in f.items()},m,cohort,cases
def point_valid(p):return isinstance(p,list) and len(p)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in p) and p!=[-1,-1]
def display(row):
    solver=row.get('solver') or {};new=row['new_pose_estimated'];fixed=row['method'] not in NEW;g=solver.get('point_line_geometry') or {}
    lines=[dict(edge=b['edge'],endpoints=b['endpoints'],actual_support_xy=b['actual_support_xy']) for b in g.get('edge_blocks',[])] if new else []
    return dict(id=row['id'],method=row['method'],native_points=row['native_points'],input_points=row.get('input_points'),
        pose=row['pose'],new_pose_estimated=new,fallback_used=row['fallback_used'],output_status=row['output_status'],
        applied_H=None if fixed else row.get('hidden_initial'),reprojected_ids=row.get('reprojected_ids',[]),solver_state=solver.get('state'),
        accepted_fit_ids=None if fixed else solver.get('fit_input_ids',[]) if new else [],accepted_fit_edges=None if fixed else solver.get('fit_line_edges',[]) if new else [],
        actual_final_inliers=None if fixed else solver.get('final_inliers',[]) if new else [],selected_corner_ids=row.get('selected_corner_ids'),
        fit_metadata_scope='UNRECORDED_FIXED_CONTROL_METADATA' if fixed else 'ACCEPTED_NEW_ONLY_NOT_INITIAL_FALLBACK_FIT',
        projected_model_points=solver.get('projected') if new else None,approved_model_edge_blocks=lines,
        evaluation_reference=row.get('evaluation_reference',{}),physical_ownership='UNKNOWN')
def saved_data(f,ids,cases):
    groups={m:{} for m in METHODS};views={};wanted={r['id'] for r in cases}
    for key,allowed in (('fixed_scores',METHODS[:2]),('ordinary_scores',METHODS[2:4]),('point_scores',('HYBRID_POINT_CONSENSUS_ORIGINAL',OLD)),('new_scores',NEW)):
        seen=set()
        for row in rows(f[key]):
            identity=(row['method'],row['id']);require(row['method'] in allowed and row['id'] in ids and identity not in seen,'unique source method/ID');seen.add(identity)
            if row['method'] not in groups:continue
            pose=row['pose'];available=pose['available'];values={k:float(pose['ADDsym_m'])*100 if k=='ADDsym_cm' else float(pose[k]) for k in FIELDS} if available else {}
            groups[row['method']][row['id']]=dict(values=values,available=available,new=row['new_pose_estimated'],fallback=row['fallback_used'],status=row['output_status'])
            if row['method'] in DISPLAY and row['id'] in wanted:views[identity]=display(row)
        require(seen=={(m,i) for m in allowed for i in ids},'full paired490 source '+key)
    require(all(set(g)==set(ids) for g in groups.values()) and len(views)==18,'seven full245 groups/18 fixed panels')
    post={m:{} for m in NEW}
    for r in rows(f['posthoc']):
        require(r['method'] in post and r['id'] in ids and r['id'] not in post[r['method']],'unique posthoc490')
        post[r['method']][r['id']]=r
    require(all(set(g)==set(ids) for g in post.values()),'all245 posthoc per new arm')
    return groups,views,post
def diagnostic_counts(post,m,ids):
    result={}
    for method,g in post.items():
        groups={};corners=[c for i in ids for c in g[i]['corners']];direct=[c for c in corners if c['human_state']=='DIRECT_VISIBLE']
        for mask in MASKS:
            for outcome in OUTCOMES:
                chosen=[i for i in ids if g[i]['mask_state']==mask and g[i]['pose_outcome_vs_N3']==outcome]
                value=dict(n=len(chosen),ids=chosen,new_pose=sum(g[i]['new_pose_estimated'] for i in chosen),fallback=sum(g[i]['fallback_used'] for i in chosen))
                require(value==m['strata']['combined']['diagnostics'][method]['mask_and_pose_groups_vs_fixed_N3'][mask+'__'+outcome],'saved mask/outcome ID group')
                groups[mask+'__'+outcome]=value
        require(sum(v['n'] for v in groups.values())==245,'all245 mask/outcome partitions')
        damage=[(i,c['id']) for i in ids for c in g[i]['corners'] if c['directly_visible_damage_5_to_10']]
        require(len(direct)==m['strata']['combined']['diagnostics'][method]['DIRECT_VISIBLE']['corners'] and len(damage)==m['strata']['combined']['diagnostics'][method]['DIRECT_VISIBLE']['damage_5px_to_over10px'],'saved direct-visible damage count')
        result[method]=dict(mask_and_pose_groups_vs_fixed_N3=groups,DIRECT_VISIBLE=dict(corners=len(direct),damage_5px_to_over10px=len(damage),damage_frame_corner_IDs=damage),reference='existing saved proxy labels; ownership UNKNOWN')
    return result
def run(a,f,b,m,cohort,cases,out):
    lock=read(out/NAMES[0]);require(lock['inputs']==b and lock['limits']==LIMITS and lock['case_ids']==[r['id'] for r in cases],'own frozen render inputs/policy/cases')
    for n in NAMES[1:]:require(not(out/n).exists(),'preserve '+n)
    folder=out/'figures';require(not folder.exists(),'preserve existing figures');write(out/NAMES[1],dict(protocol=binding(out/NAMES[0]),inputs=b,configured_PNGs=9,configured_case_panels=18,new_production_calls=0));began=time.monotonic();case_rows=[]
    try:
        data,views,post=saved_data(f,cohort['ids'],cases);diagnostic=diagnostic_counts(post,m,cohort['ids']);ids=cohort['ids'];folder.mkdir()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        fig,axes=plt.subplots(1,3,figsize=(20,6),layout='constrained')
        for ax,field,unit in zip(axes,FIELDS,UNITS):
            for j,method in enumerate(METHODS):
                values=[data[method][i]['values'][field] for i in ids if data[method][i]['available']];saved=m['strata']['combined']['methods'][method];moment=saved['metrics']['operational'][field]
                require(len(values)==moment['n']==saved['operational'],'full operational denominator');ax.scatter(values,[j]*len(values),s=6,alpha=.25)
                if values:ax.plot([moment['median'],moment['P90']],[j,j],linewidth=3);ax.scatter([moment['mean']],[j],marker='D',facecolor='white',edgecolor='black')
            ax.set_yticks(range(7),[LABELS[k]+' (op '+str(m['strata']['combined']['methods'][k]['operational'])+'/245, NEW '+str(m['strata']['combined']['methods'][k]['new_pose'])+')' for k in METHODS]);ax.invert_yaxis();ax.set_xscale('symlog',linthresh=1);ax.set_xlabel(field+' ['+unit+']');ax.grid(alpha=.2)
        fig.suptitle('Full245 operational outputs including fallback · saved GEOMETRIC_PROXY R/t/errors\nDiamond=saved mean; segment=saved median–P90. Numerical NEW is not accuracy success.');fig.savefig(folder/'01_all245_operational.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(18,17),layout='constrained')
        for ax,(candidate,comparator) in zip(axes.flat,CONTRASTS):
            comparison=m['strata']['combined']['contrasts'][candidate+'_minus_'+comparator];common=comparison['common_operational'];pairs=common['pair_ids']
            for j,(field,unit) in enumerate(zip(FIELDS,UNITS)):
                delta=[data[candidate][i]['values'][field]-data[comparator][i]['values'][field] for i in pairs];saved=common['metrics'][field];mom=common['matched_recorded_error_and_delta_moments'][field]['delta']
                require(len(delta)==saved['n']==mom['n'],'actual same-ID delta denominator');ax.scatter(delta,[j]*len(delta),s=8,alpha=.3)
                if delta:ax.plot(saved['CI95'],[j,j],linewidth=4);ax.scatter([saved['mean_delta']],[j],marker='D',edgecolor='black',color='white')
            ax.axvline(0,color='black',linestyle='--');ax.set_xscale('symlog',linthresh=1);ax.set_yticks(range(3),[k+' ['+u+']' for k,u in zip(FIELDS,UNITS)]);ax.grid(alpha=.2)
            ax.set_title(LABELS[candidate]+' − '+LABELS[comparator]+'\ncommon operating n='+str(len(pairs))+'; candidate NEW pairs='+str(comparison['candidate_new_pose']['common_frames'])+'; '+comparison['both_new_pose']['scope']+' n='+str(comparison['both_new_pose']['common_frames']),fontsize=10)
            ax.set_xlabel('saved-error difference; below zero favors candidate; diamond=mean, segment=fixed session-bootstrap CI')
        fig.suptitle('Same frame IDs; six prespecified contrasts · errors use cm / degree as marked\nSaved matched distributions and CIs. Cases and thresholds are not selected from these outcomes.');fig.savefig(folder/'02_same_ID_paired_errors.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(21,6),layout='constrained');colors=('seagreen','firebrick','goldenrod','gray')
        for ax,method in zip(axes[:2],NEW):
            bottom=[0]*4
            for outcome,color in zip(OUTCOMES,colors):
                values=[diagnostic[method]['mask_and_pose_groups_vs_fixed_N3'][mask+'__'+outcome]['n'] for mask in MASKS];ax.bar(range(4),values,bottom=bottom,color=color,label=outcome);bottom=[x+y for x,y in zip(bottom,values)]
            ax.set_xticks(range(4),[k.replace('_','\n') for k in MASKS],fontsize=8);ax.set_ylabel('all245 frames');ax.set_title(LABELS[method]+'\nT and R outcome vs fixed N3');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
        ax=axes[2];damage=[diagnostic[k]['DIRECT_VISIBLE']['damage_5px_to_over10px'] for k in NEW];ax.bar(range(2),damage,color='firebrick');ax.set_xticks(range(2),[LABELS[k] for k in NEW],rotation=15);ax.set_ylabel('saved directly visible corner flags');ax.set_title('Separate proxy point-error increase\nN3 ≤5px and output >10px')
        for j,k in enumerate(NEW):ax.text(j,damage[j],str(damage[j])+' / '+str(diagnostic[k]['DIRECT_VISIBLE']['corners'])+' visible corners',ha='center',fontsize=8)
        fig.suptitle('Saved known visibility labels; mask declaration can be wrong even when pose errors improve\nFallback uses original N3, not an H refit. Visibility/proxy error does not prove physical line ownership.');fig.savefig(folder/'03_mask_and_visible_error.png',dpi=140);plt.close(fig)
        for ci,case in enumerate(cases):
            with Image.open(f['image:'+str(ci)]) as image:rgb=image.convert('RGB').copy()
            fig,axes=plt.subplots(1,3,figsize=(22,9));fig.subplots_adjust(wspace=.04,top=.73)
            for ax,method in zip(axes,DISPLAY):
                r=views[method,case['id']];ax.imshow(rgb);points=r['native_points']
                for k,p in enumerate(points[:8]):
                    if point_valid(p):ax.scatter(*p,s=34,facecolor='none',edgecolor='cyan');ax.text(p[0]+3,p[1]+3,str(k),fontsize=8,color='cyan')
                for k in r['reprojected_ids']:
                    require(r['new_pose_estimated'] and k in (r['applied_H'] or []),'approved H-only display projection');ax.scatter(*points[k],s=80,marker='s',facecolor='none',edgecolor='orange')
                for p in (r['input_points'] or [])[:8]:
                    if point_valid(p):ax.scatter(*p,s=32,marker='+',color='#43ff50')
                ref=r['evaluation_reference'];xy=ref.get('native_points_px',[])
                for k in ref.get('valid_native_ids',[]):
                    if k<len(xy) and point_valid(xy[k]):ax.scatter(*xy[k],s=14,marker='.',color='white')
                for line in r['approved_model_edge_blocks']:
                    support=line['actual_support_xy'];ax.scatter([p[0] for p in support],[p[1] for p in support],s=13,marker='.',color='yellow');q=r['projected_model_points'];u,v=line['endpoints']
                    require(q is not None and point_valid(q[u]) and point_valid(q[v]),'stored approved virtual model edge');ax.plot([q[u][0],q[v][0]],[q[u][1],q[v][1]],'m--',linewidth=1.3)
                if method in post:
                    for c in post[method][case['id']]['corners']:
                        if c['directly_visible_damage_5_to_10']:ax.scatter(*points[c['id']],s=85,marker='x',color='red')
                pose=r['pose'];text=LABELS[method]+'\n'+r['output_status']+' / '+str(r['solver_state'])+'\nT='+str(pose.get('translation_cm'))+'cm; R='+str(pose.get('rotation_deg'))+'deg; ADDsym='+str(pose.get('ADDsym_m'))+'m'
                recorded=lambda v:'unrecorded (fixed metadata)' if v is None else str(v)
                text+='\naccepted NEW fit='+recorded(r['accepted_fit_ids'])+'; final8px='+recorded(r['actual_final_inliers'])+'; fit edges='+recorded(r['accepted_fit_edges'])+'\nH='+recorded(r['applied_H'])+'; adopted='+recorded(r['selected_corner_ids'])
                if method in post:text+='\n'+post[method][case['id']]['mask_state']+'; '+post[method][case['id']]['pose_outcome_vs_N3']
                ax.set_title(text,fontsize=9,pad=8);ax.axis('off');case_rows.append(dict(case_index=ci,id=case['id'],method=method,display=r,posthoc=post[method][case['id']] if method in post else None))
            fig.suptitle(case['label']+' / '+case['id']+' · original fixed-case ID\ncyan=output; green+=supplied input q; orange square=one approved H reprojection; white=saved proxy XY\nyellow=actual line samples; magenta dashed=virtual projected model segment; red×=saved visible proxy-error increase\nNew point/line fits exclude H and center; initial N3 retains its original branch policy. Fallback retains full original N3. Ownership remains UNKNOWN.',fontsize=11)
            fig.savefig(folder/('case_%02d.png'%(ci+1)),dpi=140);plt.close(fig)
        require(len(case_rows)==18 and len(list(folder.glob('*.png')))==9,'actual planned9 PNG/18 panels')
        with (out/NAMES[4]).open('xb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
                for r in case_rows:z.write((json.dumps(r,allow_nan=False)+'\n').encode())
            raw.flush();os.fsync(raw.fileno())
        write(out/NAMES[5],dict(complete=True,methods=diagnostic,limits=LIMITS,saved_moments_or_CIs_recomputed=False));require(inputs(a)[1]==b,'render inputs changed')
        write(out/NAMES[2],dict(schema='observed_frame_results_figure_bindings_v19',complete=True,inputs=b,own_protocol=binding(out/NAMES[0]),renderer=b['renderer'],figures=[binding(p) for p in sorted(folder.glob('*.png'))],case_rows=binding(out/NAMES[4]),diagnostic_counts=binding(out/NAMES[5]),actual_counts=dict(PNG_files=9,case_panels=18,RGB_images_read=6,full_selected_score_rows=1715,source_score_rows=1960,posthoc_rows=490),elapsed_seconds=time.monotonic()-began,elapsed_is_inference_latency=False,limits=LIMITS))
    except BaseException as exc:
        write(out/NAMES[3],dict(complete=False,error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),preserved_PNG_prefix=[binding(p) for p in sorted(folder.glob('*.png'))],completed_case_panels=len(case_rows),automatic_retry=False,limits=LIMITS));raise
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--repo',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--stats-input',type=Path,required=True);p.add_argument('--stats-checker',type=Path,default=Path(__file__).with_name('stats_check.py'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.absolute();require(out!=a.input.absolute() and not any(q.is_symlink() for q in (out,*out.parents)),'distinct nonsymlink output');out.mkdir(parents=True,exist_ok=True)
    for n in NAMES if a.stage=='freeze' else NAMES[1:]:require(not(out/n).exists(),'preserve '+n)
    f,b,m,cohort,cases=inputs(a);require(not any(v.absolute().is_relative_to(out) for v in f.values()),'output overlaps inputs')
    if a.stage=='freeze':write(out/NAMES[0],dict(schema='prespecified_observed_frame_results_figures_v19',inputs=b,case_ids=[r['id'] for r in cases],methods=list(METHODS),display_methods=list(DISPLAY),planned_PNGs=9,planned_case_panels=18,frames=245,full_selected_score_rows=1715,original_layout_sources=['source_render_v17','source_render_v18'],metadata_AST_whitelist=['require','read','binding','same','paths','metadata'],frozen_after_full_geometry_and_statistics_PASS_before_image_decode=True,limits=LIMITS))
    else:run(a,f,b,m,cohort,cases,out)
if __name__=='__main__':main()
