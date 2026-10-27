"""Fixed primary372 manuscript panels after actual numerical QA and ROOT GO.

Reads four scalar CSVs plus authorization/completion metadata. No private
error caches, images, annotations, checkpoints, NN, PnP or bootstrap draws.
The original poster figures and scientific results remain unchanged.
Import performs no IO; validation-only never creates figures or a marker.
"""
from pathlib import Path
import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
import publication_common as P

SOURCE=Path(__file__).resolve()
GO=P.OUT/'preflight/MANUSCRIPT_PANEL_PLOT_GO.json'
NQA=P.OUT/'preflight/FINAL_NUMERICAL_QA.json'
KINDS=('PLASTIC','WOOD','GREEN')
REMOVED=('no_confidence','no_ransac','no_loo','no_flip','no_final_support','confidence_only')
CONDITIONS=('control','full',*REMOVED)
ORDER=(('R0',0),*((c,r) for c in CONDITIONS for r in (1,2)))
METRICS=('PCK5','PCK10','PCK20','coverage','conditional_precision10')
CURVE_METRICS=('PCK10','coverage','conditional_precision10')
RATIOS=(.2,.5,1,2,4)
RATIO_CONDITIONS=('A','B','C','D','E')
REAL_COUNTS={'PLASTIC':20,'WOOD':550,'GREEN':550}
RATIO_PAIRS=tuple((c,'C') for c in ('A','B','D','E'))+tuple((c,'control_R2') for c in RATIO_CONDITIONS)
ROLES=('table_ablation_2d','ci_ablation','table_ratio','ci_ratio')
LABELS={'R0':'Initial R0','control':'Synthetic control','full':'Full LOO + flip','no_confidence':'No confidence',
    'no_ransac':'No RANSAC','no_loo':'No LOO','no_flip':'No flip','no_final_support':'No final >=6','confidence_only':'Confidence only'}
METRIC_LABELS={'PCK5':'PCK@5 px','PCK10':'PCK@10 px','PCK20':'PCK@20 px','coverage':'Coverage',
    'conditional_precision10':'Conditional precision@10 px'}
COLORS={'PLASTIC':'#246B9A','WOOD':'#BB573E','GREEN':'#248060'}
ROUND_COLORS={1:'#BB573E',2:'#246B9A'}
FONT=9.5
NODE=Path('/home/minjae/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
MARKER=Path('/home/minjae/.codex/plugins/cache/openai-primary-runtime/pdf/26.1017.31632/skills/pdf/container_tools/mark_artifact_operation_started.mjs')


def panel_specs():
    return ([dict(stem='MODEL_METRICS_'+k,role='ablation_figure',kind=k,width=6.5,height=6.0) for k in KINDS]
        +[dict(stem='PCK10_ABLATIONS_'+k,role='ablation_figure',kind=k,width=6.5,height=5.0) for k in KINDS]
        +[dict(stem='RATIO_'+m.upper(),role='ratio_figure',metric=m,width=6.5,height=4.8) for m in CURVE_METRICS]
        +[dict(stem='RATIO_PCK10_COMPARISONS_'+k,role='ratio_figure',kind=k,width=6.5,height=4.8) for k in KINDS])


def contract():
    return dict(ROOT_GO_path=str(GO),numerical_inputs_path=str(P.NUMERICAL_INPUTS),numerical_QA_path=str(NQA),
        GO=dict(schema='primary372_manuscript_panel_plot_go_v1',passed=True,plotting_authorized=True,
            input_manifest_sha256='ACTUAL_NUMERICAL_INPUTS_SHA256',numerical_QA_sha256='ACTUAL_FINAL_NUMERICAL_QA_SHA256',
            plot_source_sha256='CURRENT_PLOT_SOURCE_SHA256',helper_source_sha256='CURRENT_PUBLICATION_COMMON_SHA256'),
        scalar_CSV_roles=list(ROLES),output_directory=str(P.OUT/'figures/manuscript_panels/TAG'),panels=panel_specs(),
        expected_png=12,expected_pdf=12,dpi=300,minimum_font_pt=9,maximum_width_in=6.5,maximum_height_in=7.5,
        visual_QA='Inspect all12 PNG and rendered12 PDF at100 percent; generation checks do not certify appearance',
        private_error_cache_reads=0,raw_image_or_annotation_reads=0,NN_forwards=0,PnP_fits=0,new_bootstrap_draws=0)


def number(value):
    if value in ('',None,'NA','None'):return None
    value=float(value);P.require(math.isfinite(value),'Nonfinite scalar plot value');return value


def integer(value):
    value=number(value);P.require(value is not None and value==int(value),'Integer scalar required');return int(value)


def read_metadata(path,expected,bindings):
    p=Path(path);P.require(p.is_file() and not p.is_symlink() and p.resolve()==p,'Exact regular ROOT metadata required')
    digest=P.sha(p);P.require(expected is None or digest==expected,'Approved ROOT metadata SHA differs')
    value=json.loads(p.read_text());P.no_coordinates(value);P.require(P.sha(p)==digest,'Metadata changed while reading')
    bindings[str(p)]=digest;return value


def authorize():
    bindings={str(SOURCE):P.sha(SOURCE)};go=read_metadata(GO,None,bindings)
    P.require(go['schema']=='primary372_manuscript_panel_plot_go_v1' and go.get('passed') is True
        and go.get('plotting_authorized') is True and go['plot_source_sha256']==bindings[str(SOURCE)]
        and go['helper_source_sha256']==P.sha(Path(P.__file__).resolve()),'Reviewed ROOT panel plotting GO missing/different')
    bindings[str(Path(P.__file__).resolve())]=go['helper_source_sha256']
    io=P.Inputs(P.NUMERICAL_INPUTS);P.require(io.manifest_sha==go['input_manifest_sha256'],'Numerical input contract changed')
    io.gates('numerical');qa=read_metadata(NQA,go['numerical_QA_sha256'],bindings)
    P.require(qa.get('passed') is True and qa.get('complete') is True
        and qa['numerical_inputs_path']==str(P.NUMERICAL_INPUTS) and qa['numerical_inputs_sha256']==io.manifest_sha
        and qa['helper_sha256']==go['helper_source_sha256'],'Actual final numerical QA not authoritative')
    qa_source=P.OUT/'code/final_numerical_qa.py';bindings[str(qa_source)]=P.sha(qa_source)
    P.require(qa['code_sha256']==bindings[str(qa_source)],'Numerical QA source changed')
    for role in (*ROLES,'merged_complete'):
        path=io.one(role);P.require(qa['source_sha256'].get(str(path))==io.files[str(path)]['sha256'],
            'Numerical QA does not bind exact panel input: '+role)
    merged=io.read(io.one('merged_complete'))
    P.require(merged['complete'] is True and merged['passed'] is True and merged['main_groups']==51
        and merged['primary_frames']==372 and merged['primary_reference_points']==2891,'Primary372 completion differs')
    for role in ('table_ablation_2d','ci_ablation'):
        path=io.one(role);P.require(merged['output_sha256'].get(str(path))==io.files[str(path)]['sha256'],'Merged output binding differs')
    # Deliberately do not iterate/re-hash QA.source_sha256: it also lists private error caches.
    return io,bindings


def metric_value(row,metric):return number(row[metric])


def forest_record(row,field):
    point,lo,hi=(number(row[k]) for k in (field,'ci95_low','ci95_high'))
    P.require((lo is None)==(hi is None),'CI must have both endpoints or neither')
    if point is None:
        P.require(lo is None,'Unavailable point cannot have a numeric CI');return None,None,None,'unavailable'
    P.require(-1<=point<=1,'Ratio difference outside[-1,1]')
    if lo is None:return point,None,None,'point_only'
    P.require(-1<=lo<=hi<=1,'Invalid95percent interval');return point,lo,hi,'interval'


def validate_tables(tables):
    summaries={};cis={};ratio={};ratio_ci={}
    for role,dest,wanted in [('table_ablation_2d',summaries,{(k,c,r) for k in KINDS for c,r in ORDER}),
        ('table_ratio',ratio,{(k,c,2) for k in KINDS for c in RATIO_CONDITIONS})]:
        for row in tables[role]:
            key=(row['kind'],row['condition'],integer(row['round']));P.require(key in wanted and key not in dest,'Unexpected/duplicate summary catalogue')
            n,nref=P.EXPECTED[key[0]];P.require(row['panel']=='all_annotated' and integer(row['n_DEV'])==n
                and integer(row['n_reference_points'])==nref and row['available'] in ('True','False'),'Same-kind primary population differs')
            if role=='table_ablation_2d':P.require(row['evaluation_population']=='PRIMARY372','Historical table prohibited')
            else:
                percent=(20,50,100,200,400)[RATIO_CONDITIONS.index(key[1])];real=REAL_COUNTS[key[0]]
                P.require(integer(row['real_unique'])==real and integer(row['synthetic_unique'])==real*percent//100
                    and number(row['synthetic_percent'])==percent,'Fixed A-E curve ratio labels differ from source counts')
            if row['available']=='True':
                observed=integer(row['n_observed_reference_predictions']);P.require(0<=observed<=nref,'Observed denominator differs')
                for metric in METRICS:
                    value=metric_value(row,metric);P.require(value is not None or metric=='conditional_precision10' and observed==0,'Available metric missing')
                    P.require(value is None or 0<=value<=1,'Percentage metric outside[0,1]')
                P.require(number(row['PCK5'])<=number(row['PCK10'])<=number(row['PCK20']),'PCK not monotone')
                P.same(number(row['coverage']),observed/nref,'Plot coverage denominator')
                P.same(number(row['conditional_precision10']),number(row['PCK10'])*nref/observed if observed else None,'Plot conditional denominator')
            else:P.require(all(metric_value(row,m) is None for m in METRICS),'Unavailable model must remain NA')
            dest[key]=row
        P.require(len(tables[role])==len(wanted) and set(dest)==wanted,'Incomplete summary catalogue')
    for role,dest,wanted in [('ci_ablation',cis,{(k,c,r,m) for k in KINDS for c in REMOVED for r in (1,2) for m in METRICS}),
        ('ci_ratio',ratio_ci,{(k,a,b,m) for k in KINDS for a,b in RATIO_PAIRS for m in METRICS})]:
        for row in tables[role]:
            P.require(row['panel']=='all_annotated' and integer(row['seed'])==20261026 and integer(row['replicates'])==10000,'Primary CI recipe differs')
            if role=='ci_ablation':
                key=(row['kind'],row['comparator'],integer(row['round']),row['metric']);field='point_difference_full_minus_ablation'
                P.require(row['candidate']=='full' and row['difference_direction']=='full_minus_removed_condition','Ablation direction differs')
                a=summaries.get((row['kind'],'full',key[2]));b=summaries.get((row['kind'],key[1],key[2]))
            else:
                key=(row['kind'],row['candidate'],row['comparator'],row['metric']);field='point_difference'
                P.require(row['difference_direction']=='candidate_minus_comparator','Ratio direction differs')
                a=ratio.get((key[0],key[1],2));b=ratio.get((key[0],key[2],2)) if key[2]=='C' else summaries.get((key[0],'control',2))
            P.require(key in wanted and key not in dest and a is not None and b is not None,'Unexpected/duplicate CI catalogue')
            unit=row.get('resampling_unit','');P.require(unit in ('','paired_frame'),'Recording/manual CI cannot replace primary frame CI')
            point,lo,hi,state=forest_record(row,field)
            av,bv=metric_value(a,row['metric']),metric_value(b,row['metric'])
            P.same(point,av-bv if av is not None and bv is not None else None,'CSV point difference differs')
            undefined=number(row.get('undefined_replicates'));P.require(undefined is None or undefined>=0,'Invalid undefined draw count')
            if undefined is not None and undefined>0:P.require(lo is None,'Undefined draws cannot be dropped')
            if state=='interval':
                label=('full_higher' if lo>0 else 'full_lower' if hi<0 else 'includes_zero') if role=='ci_ablation' else (
                    'candidate_higher' if lo>0 else 'candidate_lower' if hi<0 else 'includes_zero')
                P.require(row['decision']==label,'CI label differs')
            dest[key]=row
        P.require(len(tables[role])==len(wanted) and set(dest)==wanted,'Incomplete CI catalogue')
    return summaries,cis,ratio,ratio_ci


def footer(fig,lines):
    for index,text in enumerate(reversed(lines)):fig.text(.5,.025+.028*index,text,ha='center',va='bottom',fontsize=9)


def row_label(condition,round_idx):return LABELS[condition]+(' R'+str(round_idx) if round_idx else '')


def heatmap(spec,rows,plt,np):
    kind=spec['kind'];fig,axis=plt.subplots(figsize=(spec['width'],spec['height']))
    values=np.asarray([[math.nan if metric_value(rows[kind,c,r],m) is None else 100*metric_value(rows[kind,c,r],m)
        for m in METRICS] for c,r in ORDER],float)
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#E8E8E8')
    image=axis.imshow(np.ma.masked_invalid(values),aspect='auto',vmin=0,vmax=100,cmap=cmap,interpolation='nearest')
    axis.set_yticks(range(17),[row_label(c,r) for c,r in ORDER],fontsize=9)
    axis.set_xticks(range(5),['PCK@5','PCK@10','PCK@20','Coverage','Precision\n@10'],fontsize=FONT)
    for y in range(17):
        for x in range(5):
            v=values[y,x];axis.text(x,y,'N/A' if not math.isfinite(v) else f'{v:.1f}',ha='center',va='center',
                fontsize=9,color='white' if math.isfinite(v) and v>=65 else '#161616')
    axis.tick_params(length=0);axis.set_title(kind+' - all17 evaluation groups',fontsize=10,pad=10)
    fig.subplots_adjust(left=.34,right=.89,bottom=.16,top=.91)
    bar=fig.colorbar(image,cax=fig.add_axes([.92,.16,.025,.75]));bar.ax.tick_params(labelsize=9);bar.set_label('Percent (%)',fontsize=9)
    n,ref=P.EXPECTED[kind];footer(fig,['PCK: correct / all refs; coverage: observed / all refs.',
        'Precision@10: correct / observed. Missing is failure; N/A stays N/A.',f'{n} images / {ref} reference corners; native-pixel thresholds.'])
    return fig


def forest(spec,records,labels,field,title,xlabel,plt):
    fig,axis=plt.subplots(figsize=(spec['width'],spec['height']));parsed=[forest_record(r,field) for r in records]
    finite=[0.0]+[100*v for record in parsed for v in record[:3] if v is not None]
    lo,hi=min(finite),max(finite);span=max(hi-lo,1.0);axis.set_xlim(lo-.12*span,hi+.35*span)
    axis.axvline(0,color='#777777',linewidth=.8)
    for y,(row,(point,low,high,state)) in enumerate(zip(records,parsed)):
        color=ROUND_COLORS[integer(row['round'])] if 'round' in row else COLORS[spec['kind']]
        if point is not None:axis.plot(100*point,y,'o',color=color,markersize=4)
        if state=='interval':axis.hlines(y,100*low,100*high,color=color,linewidth=1.3)
        else:axis.text(.99,y,'N/A' if state=='unavailable' else 'CI N/A',ha='right',va='center',
            transform=axis.get_yaxis_transform(),fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    axis.set_yticks(range(len(labels)),labels,fontsize=FONT);axis.set_ylim(len(labels)-.5,-.5)
    axis.set_title(title,fontsize=10,pad=10);axis.set_xlabel(xlabel,fontsize=FONT);axis.tick_params(axis='x',labelsize=FONT)
    axis.grid(axis='x',alpha=.2);axis.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(left=.36,right=.97,bottom=.19,top=.90)
    footer(fig,['95% paired-frame CI; zero included means inconclusive.',
        'Point-only estimates retain CI N/A; unavailable models remain N/A.'])
    return fig


def ratio_curve(spec,ratio,base,plt):
    metric=spec['metric'];fig,axis=plt.subplots(figsize=(spec['width'],spec['height']))
    for kind in KINDS:
        values=[metric_value(ratio[kind,c,2],metric) for c in RATIO_CONDITIONS]
        missing=[c for c,v in zip(RATIO_CONDITIONS,values) if v is None]
        label=kind+' R2'+(' (N/A: '+','.join(missing)+')' if missing else '')
        axis.plot(RATIOS,[math.nan if v is None else 100*v for v in values],'-o',color=COLORS[kind],label=label,markersize=4,linewidth=1.2)
        control=metric_value(base[kind,'control',2],metric);label=kind+' control R2'+(' (N/A)' if control is None else '')
        if control is None:axis.plot([],[],':',color=COLORS[kind],label=label)
        else:axis.axhline(100*control,color=COLORS[kind],linestyle=':',linewidth=1.1,label=label)
    axis.set_xscale('log',base=2);axis.set_xticks(RATIOS,['A\n20%','B\n50%','C\n100%','D\n200%','E\n400%'],fontsize=FONT)
    axis.set_ylim(-2,102);axis.set_ylabel(METRIC_LABELS[metric]+' (%)',fontsize=FONT)
    axis.set_xlabel('Unique synthetic / fixed real record count',fontsize=FONT);axis.tick_params(axis='y',labelsize=FONT)
    axis.set_title(METRIC_LABELS[metric]+' - fixed A-E ratios',fontsize=10,pad=10)
    axis.grid(alpha=.2);axis.spines[['top','right']].set_visible(False)
    axis.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=2,fontsize=9,frameon=False,columnspacing=1.1)
    fig.subplots_adjust(left=.14,right=.97,bottom=.39,top=.90)
    footer(fig,['Fixed15 ratio models; dotted lines are same-kind synthetic-control R2.'])
    return fig


def audit_figure(fig,spec):
    from matplotlib.text import Text
    width,height=map(float,fig.get_size_inches());P.require(abs(width-spec['width'])<1e-10 and abs(height-spec['height'])<1e-10,'Panel size differs')
    fonts=sorted({float(a.get_fontsize()) for a in fig.findobj(match=Text) if a.get_visible() and a.get_text()})
    P.require(width<=6.5 and height<=7.5 and fonts and min(fonts)>=9 and max(fonts)<=10,'Physical size/font contract differs')
    return dict(width_in=width,height_in=height,visible_font_sizes_pt=fonts,minimum_font_pt=min(fonts))


def pure_fixtures():
    specs=panel_specs();P.require(len(specs)==12 and len({p['stem'] for p in specs})==12,'Panel catalogue differs')
    P.require(sum(p['role']=='ablation_figure' for p in specs)==sum(p['role']=='ratio_figure' for p in specs)==6,'Panel roles differ')
    P.require(all(p['width']<=6.5 and p['height']<=7.5 for p in specs),'DOCX would downscale a panel')
    for row,state in [({'point_difference':'.12','ci95_low':'.01','ci95_high':'.20'},'interval'),
        ({'point_difference':'.12','ci95_low':'','ci95_high':''},'point_only'),
        ({'point_difference':'','ci95_low':'','ci95_high':''},'unavailable')]:
        P.require(forest_record(row,'point_difference')[3]==state,'NA state collapsed')
    P.require(len(ORDER)==17 and len(REMOVED)*2==12 and len(RATIO_PAIRS)==9,'Fixed row catalogue differs')
    return dict(passed=True,panel_specs=12,NA_states=3,figures_created=0,file_reads=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--describe-contract',action='store_true')
    parser.add_argument('--tag');parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if args.describe_contract:print(json.dumps(contract(),indent=2));return
    P.require(args.tag is not None and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.tag),'Safe ROOT figure tag required')
    io,bindings=authorize();tables={role:io.csv(io.one(role)) for role in ROLES};base,ci,ratio,ratio_ci=validate_tables(tables)
    def unchanged():
        io.unchanged()
        for path,digest in bindings.items():P.require(P.sha(path)==digest,'Approved plotting source/metadata changed')
    unchanged()
    if args.validate_only:print(json.dumps(dict(passed=True,summary_groups=51,ratio_groups=15,ablation_CI_rows=180,ratio_CI_rows=135,files_written=0)));return
    target=P.OUT/'figures/manuscript_panels'/args.tag;P.require(target.resolve()==target and not target.exists(),'Preserve existing/partial or unsafe panel output')
    target.mkdir(parents=True,exist_ok=False)
    try:
        os.environ['MPLCONFIGDIR']=str(target/'_matplotlib_config')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':FONT,'axes.titlesize':10,'axes.labelsize':FONT,
            'xtick.labelsize':FONT,'ytick.labelsize':FONT,'legend.fontsize':9,'figure.titlesize':10,'pdf.fonttype':42,'ps.fonttype':42})
        subprocess.run([str(NODE),str(MARKER),'--operation-kind','create','--expected-output-count','12','--output-format','pdf'],check=True)
        artifacts={};panels=[]
        for spec in panel_specs():
            stem=spec['stem'];kind=spec.get('kind')
            if stem.startswith('MODEL_METRICS_'):fig=heatmap(spec,base,plt,np)
            elif stem.startswith('PCK10_ABLATIONS_'):
                rows=[ci[kind,c,r,'PCK10'] for c in REMOVED for r in (1,2)]
                fig=forest(spec,rows,[row_label(c,r) for c in REMOVED for r in (1,2)],'point_difference_full_minus_ablation',
                    kind+' - all12 fixed ablation contrasts','PCK@10 full minus removed (percentage points)',plt)
            elif stem.startswith('RATIO_PCK10_COMPARISONS_'):
                rows=[ratio_ci[kind,a,b,'PCK10'] for a,b in RATIO_PAIRS]
                fig=forest(spec,rows,[a+' minus '+('control R2' if b=='control_R2' else b) for a,b in RATIO_PAIRS],
                    'point_difference',kind+' - all9 fixed ratio contrasts','PCK@10 candidate minus comparator (percentage points)',plt)
            else:fig=ratio_curve(spec,ratio,base,plt)
            dimensions=audit_figure(fig,spec)
            for extension in ('png','pdf'):
                path=target/(stem+'.'+extension);P.require(not path.exists(),'Preserve existing figure')
                fig.savefig(path,dpi=300,bbox_inches=None,facecolor='white');P.require(path.stat().st_size>0,'Empty panel file')
                artifacts[path.name]=dict(sha256=P.sha(path),bytes=path.stat().st_size,role=spec['role'] if extension=='png' else spec['role']+'_pdf',**dimensions)
            plt.close(fig);panels.append(dict(spec,**dimensions))
        P.require(len(artifacts)==24 and sum(n.endswith('.png') for n in artifacts)==sum(n.endswith('.pdf') for n in artifacts)==12,'Expected12 PNG/PDF pairs')
        unchanged();qa=dict(passed=True,complete=True,figure_generation_passed=True,visual_review_status='PENDING_ROOT_IMAGE_AND_PDF_INSPECTION',
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),primary_frames=372,primary_reference_points=2891,
            panel_count=12,png_count=12,pdf_count=12,dpi=300,minimum_font_pt=9,maximum_width_in=6.5,maximum_height_in=7.5,
            summary_groups=51,heatmap_cells=255,all_ablation_CI_rows_validated=180,plotted_ablation_PCK10_CIs=36,
            ratio_groups=15,ratio_CI_rows_validated=135,plotted_ratio_PCK10_CIs=27,ratio_curve_values=45,control_reference_lines=9,
            source_sha256={**io.bindings,**bindings},artifacts=artifacts,panels=panels,command=sys.argv,
            private_error_cache_reads=0,raw_RGB_reads=0,raw_annotation_reads=0,checkpoint_reads=0,NN_forwards=0,PnP_fits=0,bootstrap_draws=0)
        P.write(target/'PLOT_QA.json',qa)
        P.write(target/'PROVENANCE.json',dict(schema='primary372_fixed_manuscript_panels_v1',complete=True,source_sha256=qa['source_sha256'],
            PLOT_QA_sha256=P.sha(target/'PLOT_QA.json'),artifacts=artifacts,panels=panels,ROOT_GO_sha256=bindings[str(GO)],
            selection='Fixed12 panels; all17 groups/12 ablation contrasts/9 ratio contrasts preserved; no outcome selection',
            old_poster_figures_unchanged=True,all_new_plots_require_visual_QA=True));unchanged()
        print('MANUSCRIPT_PANELS_COMPLETE',12,24,str(target),flush=True)
    except Exception as error:
        if not (target/'FAILURE.json').exists():P.write(target/'FAILURE.json',dict(passed=False,error=repr(error),partial_outputs_preserved=True))
        raise


if __name__=='__main__':main()
