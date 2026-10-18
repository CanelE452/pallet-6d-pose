"""Public reference geometry/conditional masks; original RGB companion private."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.pallet_risk_information_v1.common import read,write,rows,sha
from .b_mask_bounds import bounds_with_canvas

VARIANTS=('Q00_ORIGINAL','Q10_GT_ANCHOR','Q01_GT_DIRECTION','Q11_GT_BOTH')
NAMES=('Original anchor / original direction','GT anchor / original direction','Original anchor / GT direction','GT anchor / GT direction')


def idhash(value):return hashlib.sha256(value.encode()).hexdigest()


def select_cases(querypath,boundspath):
    # Immutable diagnostic selection, never a real-pose performance selection.
    best=[None,None,None];keys=[None,None,None]
    with gzip.open(boundspath,'rt',newline='')as stream:
        maskkeys={(r['id'],r['support_id']) for r in csv.DictReader(stream)
                  if r['mask_kind']=='renderer_visible'and r['family']=='ORIGINAL_WD'and r['phase_index']=='0'and r['query_variant']=='Q00_ORIGINAL'and int(r['whole_minimum'])>=2}
    for row in rows(querypath):
        if not row['original_B_input_present']:continue
        f=next(f for f in row['families']if f['family']=='ORIGINAL_WD'and f['phase_index']==0)
        groups={q['support_id']:{}for q in f['profiles']}
        for q in f['profiles']:groups[q['support_id']][q['query_variant']]=q
        for support,queries in groups.items():
            a,b=queries['Q00_ORIGINAL'],queries['Q11_GT_BOTH'];tie=(idhash(row['id']),support)
            condition=not a['finite_support']and b['finite_support']and not b['query_domain_overrun']and a['geometry_status']in('BEFORE_FINITE_ENDPOINT','AFTER_FINITE_ENDPOINT')
            key=(-f['summaries']['Q00_ORIGINAL']['profiles']+f['summaries']['Q00_ORIGINAL']['finite_profiles'],)+tie
            if condition and(keys[0]is None or key<keys[0]):keys[0]=key;best[0]=(row,f,queries)
            length=b['reference_edge_length_px'];ratio=b['along_raw_px']/length if length>0 else None
            key=(-ratio,)+tie if ratio is not None else None
            if b['query_domain_overrun']and key is not None and(keys[1]is None or key<keys[1]):keys[1]=key;best[1]=(row,f,queries)
            if(row['id'],support)in maskkeys and(keys[2]is None or tie<keys[2]):keys[2]=tie;best[2]=(row,f,queries)
    if any(v is None for v in best):raise ValueError('A prespecified diagnostic case is unavailable; report missing instead of silently changing selection')
    return best


def geometry_panel(case,title,outfile,*,rgb=None):
    row,f,queries=case;q11=queries['Q11_GT_BOTH'];a=np.asarray(q11['reference_anchor_xy']);b=a+np.asarray(q11['tangent_xy'])*q11['reference_edge_length_px']
    points=[a,b]
    for q in queries.values():
        points.extend((q['center_xy'],q['original_observed_anchor_xy']))
        if q['intersection_xy']is not None:points.append(q['intersection_xy'])
    points=np.asarray(points);lower=points.min(0);upper=points.max(0);span=np.maximum(upper-lower,10);lower-=.15*span;upper+=.15*span
    fig,axes=plt.subplots(1,4,figsize=(16,5))
    for ax,variant,name in zip(axes,VARIANTS,NAMES):
        q=queries[variant];center=np.asarray(q['center_xy']);normal=np.asarray(q['outward_normal_xy']);hit=q['intersection_xy'];along=q['along_raw_px']
        if rgb is not None:ax.imshow(rgb,zorder=0)
        ax.axline(a,b,ls='--',color='#acacac',lw=1,label='Infinite edge extension')
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#067d27',lw=3,label='Finite GT cuboid edge')
        ax.scatter([a[0],b[0]],[a[1],b[1]],c='#067d27',s=18)
        radius=max(float(np.linalg.norm(b-a))*.55,10)
        ax.axline(center,center+normal,color='#dc8b00',lw=1.2,label='Infinite normal, not RGB data')
        ax.arrow(*center,*(normal*radius*.25),head_width=radius*.05,color='#dc8b00',length_includes_head=True)
        ax.scatter(*q['original_observed_anchor_xy'],marker='x',c='#3075bc',s=28,label='Original observed anchor')
        ax.scatter(*center,c='#973da9',s=25,zorder=5,label='Query center')
        if hit is not None:ax.scatter(*hit,marker='*',c='#d22e2e',s=80,zorder=6,label='Intersection, unclipped')
        ax.set_xlim(lower[0],upper[0]);ax.set_ylim(upper[1],lower[1]);ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel('Raw canvas x (px)');ax.set_ylabel('Raw canvas y (px)')
        lam=q['lambda_segment'];offset=q['normal_offset_px']
        number=lambda v:'undefined'if v is None else f'{v:.5g}'
        ax.set_title(name+'\n'+q['geometry_status']+'\n'+f'lambda={number(lam)}, normal offset={number(offset)} px\nalong={along:.4g} / edge={q["reference_edge_length_px"]:.4g} px; overrun={q["query_domain_overrun"]}',fontsize=8)
    axes[0].legend(fontsize=6,loc='best')
    fig.suptitle(title+'\n'+row['id']+f' | ORIGINAL_WD phase0 | native {q11["native_corner"]}->{q11["native_neighbor"]} | profile '+q11['support_id'][:12],fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.88),w_pad=1.4)
    footer='Reference-only cuboid diagnostic, not an automatic candidate or material edge label. Original RGB likelihood not transported to changed query.'
    if rgb is not None:footer+=' PRIVATE ORIGINAL SYNTHETIC RGB: not authorized for public distribution.'
    else:footer+=' Geometry only; no original RGB published.'
    fig.text(.02,.02,footer,fontsize=8);fig.savefig(outfile,dpi=150);plt.close(fig)


def mask_panel(bounds,controls,outfile):
    fig,axes=plt.subplots(1,3,figsize=(16,5.5));groups=bounds['groups'];x=np.arange(4)
    specs=[('renderer_visible','Archived renderer values: conditional'),('SAM','Archived predicted SAM membership')]
    classes=[('CONDITIONALLY_EXACT_SINGLE','#1b9e77','Exactly one'),('CONDITIONALLY_MULTIPLE','#d95f02','At least two'),('CONDITIONALLY_EXACT_ZERO','#7570b3','Zero'),('UNRESOLVED_RANGE','#bbbbbb','Unresolved range')]
    for ax,(kind,title)in zip(axes,specs):
        baseline=np.zeros(4)
        for key,color,label in classes:
            values=np.array([groups['all/ORIGINAL_WD/'+v+'/'+kind].get(key,0)for v in VARIANTS])
            ax.bar(x,values,bottom=baseline,color=color,label=label);baseline+=values
        ax.set_title(title+'\nOriginal phase-expanded profiles:17,497 / query variant',fontsize=10)
        ax.set_xticks(x,['Q00','Q10','Q01','Q11']);ax.set_ylabel('Profile count (not image count)');ax.legend(fontsize=8);ax.set_ylim(0,19000)
    labels=[]
    for i,case in enumerate(controls['cases']):
        b=bounds_with_canvas(case['ownership_labels'])['whole_line'];axes[2].plot([b['minimum'],b['maximum']],[i,i],color='#444444',lw=3);axes[2].scatter([b['minimum'],b['maximum']],[i,i],c='#087f35',s=30)
        labels.append(('Openings'if i<2 else'Solid')+(' + occluder'if case['occlusion']else' / no occluder'))
    axes[2].set_yticks(range(4),labels);axes[2].invert_yaxis();axes[2].set_xlim(0,5);axes[2].set_xlabel('Exact possible transition range');axes[2].grid(alpha=.2)
    axes[2].set_title('Own procedural controls:4 fixed renders\nActual instance ownership; mixed AA stays unknown',fontsize=10)
    fig.suptitle('Geometry support and object-membership transitions are different contracts',fontsize=13)
    fig.tight_layout(rect=(0,.16,1,.9),w_pad=2)
    fig.text(.02,.025,'Renderer results are conditional on the unverified archived 0=background /255=target encoding; 1..254 stay unknown.\nOutside samples remain unobserved; full-line ranges do not invent measurements. SAM is predicted membership.\nNo physical edge ownership, new likelihood, automatic correction, or validated performance gain follows from these counts.',fontsize=9)
    fig.savefig(outfile,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('query-dir','mask-dir','bounds-dir','control-dir','output','private-output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--render-GO',action='store_true');p.add_argument('--private-RGB-GO',action='store_true');a=p.parse_args()
    if not a.render_GO:raise PermissionError('Root render GO required')
    inputs=[(a.query_dir,'B_QUERY_FACTORIAL_SEAL.json'),(a.bounds_dir,'DIAGNOSTIC_SEAL.json'),(a.control_dir,'PROCEDURAL_RENDER_SEAL.json')]
    for folder,name in inputs:
        for filename,digest in read(folder/name)['artifacts'].items():
            if sha(folder/filename)!=digest:raise ValueError('Sealed diagnostic input changed')
    a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    cases=select_cases(a.query_dir/'B_QUERY_FACTORIAL_ROWS.jsonl.gz',a.bounds_dir/'B_MASK_TRANSITION_BOUNDS.csv.gz')
    inventory={r['id']:r for r in rows(a.mask_dir/'MASK_INPUT_INVENTORY.jsonl.gz')};records=[]
    titles=['Original finite-support failure / GT query recovery','Unchanged along exceeds GT edge length','Conditional archived-mask multiple-transition example']
    for i,(case,title)in enumerate(zip(cases,titles)):
        filename=f'query_factorial_{i:02d}.png';geometry_panel(case,title,a.output/filename)
        row,f,qs=case
        if a.private_RGB_GO:
            source=Path(inventory[row['id']]['renderer_visible_path']);rgb=source.parent.parent/'rgb'/(source.stem+'_rgb.png')
            with Image.open(rgb)as image:geometry_panel(case,title,a.private_output/(filename[:-4]+'_PRIVATE_RGB.png'),rgb=np.asarray(image.convert('RGB')))
        records.append(dict(id=row['id'],phase_index=0,family='ORIGINAL_WD',support_id=qs['Q00_ORIGINAL']['support_id'],native_corner=qs['Q00_ORIGINAL']['native_corner'],native_neighbor=qs['Q00_ORIGINAL']['native_neighbor'],public_file=filename,
            profiles=[qs[v]for v in VARIANTS],raw_backend_mapping_audit=f['original_raw_backend_mapping_audit']))
    bounds=read(a.bounds_dir/'B_MASK_TRANSITION_BOUNDS.json');controls=read(a.control_dir/'PROCEDURAL_SHAPE_CONTROL.json')
    mask_panel(bounds,controls,a.output/'mask_transition_contracts.png')
    write(a.output/'QUERY_EXAMPLE_RECORDS.json',dict(schema='REFERENCE_ONLY_QUERY_FIGURE_RECORDS_V1',cases=records,selection_rules=[
        'ORIGINAL_WD phase0: Q00 before/after segment and Q11 supported without overrun; greatest Q00 failed profile count, SHA256(frame ID), support ID',
        'ORIGINAL_WD phase0: greatest unchanged along / GT edge length among domain overruns; SHA256(frame ID), support ID',
        'ORIGINAL_WD phase0 Q00 renderer-visible DP minimum>=2 conditional on archived0/255 contract; SHA256(frame ID), support ID'],
        diagnostic_reference_only=True,new_candidate_bank_entries=0,new_solver_calls=0,new_RGB_likelihoods=0,physical_material_edge_labels=False))
    write(a.output/'FIGURE_SEAL.json',dict(schema='P2_REFERENCE_DIAGNOSTIC_FIGURES_V1',status='COMPLETE',source_sha256=sha(__file__),input_seals={name:sha(folder/name)for folder,name in inputs},
        artifacts={f.name:sha(f)for f in a.output.iterdir()if f.is_file()},original_RGB_publication=False,private_original_RGB_companions_created=a.private_RGB_GO,
        human_visual_review=False,physical_ground_truth_certification=False))


if __name__=='__main__':main()
