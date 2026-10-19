"""Postseal T-horizontal/R-vertical comparison figures from saved metrics only.

No reference/RGB decoding, pose metric, bootstrap, forward, solve or fit. The
explicit manifest binds completed canonical and independent output bundles.
All192 PRIMARY frames remain in each arm/seed; unavailable poses are counted,
never placed at zero. Full-axis exports retain every available outlier.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import html
import json
import math
from pathlib import Path
from . import f_core_resume as M

SEEDS=(20261018,20261019,20261020)
INDEPENDENT={'INDEPENDENT_ROWS.jsonl.gz','INDEPENDENT_VERIFICATION.json','INDEPENDENT_SUMMARY.json','INDEPENDENT_PAIRED.json'}


def load_rows(path):
    with gzip.open(path,'rt')as stream:return [json.loads(line)for line in stream if line.strip()]


def metric_agrees(a,b):
    if a['success']!=b['success']or a['pose_available']!=b['pose_available']:return False
    for axis in ('translation_cm','rotation_deg'):
        if (a[axis]is None)!=(b[axis]is None):return False
        if a[axis]is not None and(not math.isfinite(a[axis])or not math.isfinite(b[axis])or abs(a[axis]-b[axis])>1e-6):return False
    return True


def authenticate(manifest):
    plan=M.read(manifest);merged={};bindings=[];reference_sha=None
    for source in plan['sources']:
        ev,ind=Path(source['evaluation_dir']),Path(source['independent_dir']);es=M.read(ev/'EVALUATION_SEAL.json')
        if es.get('complete')is not True:raise ValueError('Complete canonical evaluation required')
        if not INDEPENDENT.issubset(source['independent_artifacts']):raise ValueError('Full four-file independent verification bundle required')
        for name,digest in source['independent_artifacts'].items():
            if name not in INDEPENDENT:raise ValueError('Explicit independent bundle filenames required')
            if M.sha(ind/name)!=digest:raise ValueError('Independent metric bundle changed')
        check=M.read(ind/'INDEPENDENT_VERIFICATION.json')
        if check.get('PASS')is not True or check['canonical_evaluation_seal_sha256']!=M.sha(ev/'EVALUATION_SEAL.json'):
            raise ValueError('Independent verification belongs to another canonical score')
        if reference_sha is None:reference_sha=es['reference_sha256']
        elif reference_sha!=es['reference_sha256']:raise ValueError('Canonical reference hash differs across stages')
        if M.sha(ev/'SCORED_ROWS.jsonl.gz')!=es['artifacts']['SCORED_ROWS.jsonl.gz']:raise ValueError('Canonical scored rows changed')
        for row in es['prediction_bindings']:
            path=Path(row['path']);seal=M.read(path.parent/'PREDICTION_SEAL.json')
            if M.sha(path)!=row['sha256']or M.sha(path.parent/'PREDICTION_SEAL.json')!=row['seal_sha256']or seal.get('complete')is not True:
                raise ValueError('Completed prediction input bytes/seal differ')
        rows=load_rows(ev/'SCORED_ROWS.jsonl.gz');ii=load_rows(ind/'INDEPENDENT_ROWS.jsonl.gz');independent={r['id']:r for r in ii}
        if len(rows)!=238 or len(ii)!=238 or len(independent)!=238 or len({r['id']for r in rows})!=238:
            raise ValueError('Exact complete238 saved population required')
        if Counter(r['partition']for r in rows)!=Counter(TEST=49,REMAINDER=143,CAL=46):raise ValueError('Original reporting denominator differs')
        for row in rows:
            other=independent[row['id']]
            if any(other[key]!=row[key]for key in ('partition','session','cluster_id')):raise ValueError('Canonical/independent metadata differs')
            for arm in es['arms']:
                if not metric_agrees(row['metrics'][arm],other['metrics'][arm]):raise ValueError('Saved canonical/independent metric differs')
            if row['id']not in merged:merged[row['id']]={k:row[k]for k in ('id','partition','session','cluster_id')}|dict(metrics={})
            current=merged[row['id']]
            if any(current[key]!=row[key]for key in ('partition','session','cluster_id')):raise ValueError('Cross-stage identity differs')
            for arm in es['arms']:
                if arm in current['metrics']and not metric_agrees(current['metrics'][arm],row['metrics'][arm]):raise ValueError('Repeated baseline saved metrics disagree')
                current['metrics'][arm]=row['metrics'][arm]
        bindings.append(dict(evaluation_seal_sha256=M.sha(ev/'EVALUATION_SEAL.json'),canonical_rows_sha256=M.sha(ev/'SCORED_ROWS.jsonl.gz'),
            independent_artifacts=source['independent_artifacts'],prediction_bindings=es['prediction_bindings']))
    if len(merged)!=238:raise ValueError('All sources must share the same238 IDs')
    return plan,merged,dict(manifest_sha256=M.sha(manifest),reference_sha256=reference_sha,sources=bindings)


def public_metric(metric):
    out={key:metric[key]for key in ('translation_cm','rotation_deg','success','pose_available')}
    for key in ('translation_cm','rotation_deg'):
        if out[key]is not None and(not math.isfinite(out[key])or out[key]<0):raise ValueError('Finite nonnegative saved error required')
    if not out['pose_available']and any(out[k]is not None for k in ('translation_cm','rotation_deg')):raise ValueError('No-pose errors cannot be zero imputed')
    return out


def build_payload(plan,rows):
    primary=sorted((r for r in rows.values()if r['partition']in ('TEST','REMAINDER')),key=lambda r:r['id'])
    if len(primary)!=192 or len({r['cluster_id']for r in primary})!=6:raise ValueError('All192 original PRIMARY frames/six clusters required')
    datasets=[];keys=set()
    for pair in plan['pairs']:
        key=(pair['stage'],pair['name'])
        if key in keys:raise ValueError('Unique stage/pair names required')
        keys.add(key)
        before_seeded=pair.get('before_seeded',True);after_seeded=pair.get('after_seeded',True)
        if type(before_seeded)is not bool or type(after_seeded)is not bool:raise ValueError('Seeded arm flags must be explicit booleans')
        for seed in SEEDS:
            before=pair['before']+('_s'+str(seed)if before_seeded else'');after=pair['after']+('_s'+str(seed)if after_seeded else'')
            if any(before not in r['metrics']or after not in r['metrics']for r in primary):raise ValueError('Pending arms cannot be omitted or zero filled')
            points=[dict(id=r['id'],before=public_metric(r['metrics'][before]),after=public_metric(r['metrics'][after]))for r in primary]
            datasets.append(dict(stage=pair['stage'],pair=pair['name'],seed=seed,before_arm=before,after_arm=after,
                before_seeded=before_seeded,after_seeded=after_seeded,
                static_estimator_note='Unseeded arm is one frozen estimator repeated for head-seed comparisons,not three independent estimator experiments'if not(before_seeded and after_seeded)else None,n=192,points=points))
    rep=plan['representative']
    if rep['seed']!=SEEDS[0]or not any(d['stage']==rep['stage']and d['pair']==rep['pair']and d['seed']==rep['seed']for d in datasets):
        raise ValueError('Predeclared representative pair/seed20261018 required')
    return dict(schema='POSTSEAL_PRIMARY_T_R_SCATTER_V1',datasets=datasets,representative=rep,
        x_axis='saved translation error,cm',y_axis='saved C2 rotation error,deg',PRIMARY_n=192,
        all_outliers_retained=True,available_only_scatter=True,no_pose='counted in192,not plotted or assigned zero',
        GT_pose_or_raw_RGB_included=False,independent_real_test_n=0)


def view_stats(dataset,x_limit=None,y_limit=None):
    stats={};max_x=5.;max_y=5.
    for side in ('before','after'):
        values=[r[side]for r in dataset['points']];available=[m for m in values if m['translation_cm']is not None and m['rotation_deg']is not None]
        max_x=max([max_x]+[m['translation_cm']for m in available]);max_y=max([max_y]+[m['rotation_deg']for m in available])
        stats[side]=dict(full_denominator=len(values),coordinate_points=len(available),unavailable=len(values)-len(available),
            outside_zoom=sum((x_limit is not None and m['translation_cm']>x_limit)or(y_limit is not None and m['rotation_deg']>y_limit)for m in available))
    return dict(sides=stats,full_x_max=max_x*1.05,full_y_max=max_y*1.05)


HTML=r'''<!doctype html><meta charset="utf-8"><title>Saved PRIMARY pose-error scatter</title>
<style>body{font:15px system-ui;margin:20px;color:#172436}label{margin-right:16px}svg{max-width:1100px;width:100%;background:white;border:1px solid #ccd}#info{white-space:pre-wrap}</style>
<h1>Saved pose errors: T horizontal / R vertical</h1><p>All192 PRIMARY development frames per arm/seed. Independent new real test n=0. No raw RGB or reference pose values. Full axes retain all available outliers.</p>
<label>Stage <select id="stage"></select></label><label>Pair <select id="pair"></select></label><label>Seed <select id="seed"></select></label>
<label><input type="checkbox" id="zoom">Explicit zoom</label><label>T max cm <input id="tx" type="number" min="0.01" value="20"></label><label>R max deg <input id="ry" type="number" min="0.01" value="20"></label>
<p id="info"></p><svg id="plot" viewBox="0 0 1000 700" role="img" aria-label="Translation versus rotation error"></svg>
<p>Blue crosses: baseline. Red circles: comparison. Dashed lines:5cm and5deg thresholds. Zoom only changes display; outlying records remain in SCATTER_DATA.json and full-axis PNG/SVG.</p>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.querySelector('#data').textContent),all=data.datasets,$=id=>document.getElementById(id),svg=$('plot'),NS='http://www.w3.org/2000/svg';
function options(el,values,pick){el.replaceChildren(...values.map(v=>{let o=document.createElement('option');o.textContent=v;o.value=v;return o}));if(pick!==undefined)el.value=String(pick)}
function elem(type,attrs,text){let e=document.createElementNS(NS,type);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e}
function pairOptions(){options($('pair'),[...new Set(all.filter(d=>d.stage===$('stage').value).map(d=>d.pair))]);seedOptions()}
function seedOptions(){options($('seed'),all.filter(d=>d.stage===$('stage').value&&d.pair===$('pair').value).map(d=>d.seed),data.representative.seed);draw()}
function draw(){let d=all.find(d=>d.stage===$('stage').value&&d.pair===$('pair').value&&d.seed===Number($('seed').value));if(!d)return;svg.replaceChildren();let stats={};let vals=d.points.flatMap(p=>[p.before,p.after]).filter(p=>p.translation_cm!==null&&p.rotation_deg!==null),fullX=Math.max(5,...vals.map(p=>p.translation_cm))*1.05,fullY=Math.max(5,...vals.map(p=>p.rotation_deg))*1.05;
let z=$('zoom').checked,X=z?Math.max(.01,Number($('tx').value)||20):fullX,Y=z?Math.max(.01,Number($('ry').value)||20):fullY,x=v=>85+v/X*860,y=v=>620-v/Y*570;
elem('line',{x1:85,y1:620,x2:945,y2:620,stroke:'#263447'});elem('line',{x1:85,y1:620,x2:85,y2:50,stroke:'#263447'});
for(let i=0;i<=5;i++){let a=X*i/5,b=Y*i/5;elem('text',{x:x(a),y:643,'text-anchor':'middle','font-size':13},a.toPrecision(3));elem('text',{x:75,y:y(b)+4,'text-anchor':'end','font-size':13},b.toPrecision(3))}
elem('text',{x:515,y:685,'text-anchor':'middle','font-size':18},'Translation error T (cm)');elem('text',{x:20,y:335,transform:'rotate(-90 20 335)','text-anchor':'middle','font-size':18},'Rotation error R (deg)');
if(X>=5)elem('line',{x1:x(5),x2:x(5),y1:50,y2:620,stroke:'#789','stroke-dasharray':'5,5'});if(Y>=5)elem('line',{x1:85,x2:945,y1:y(5),y2:y(5),stroke:'#789','stroke-dasharray':'5,5'});
for(let side of ['before','after']){let missing=0,outside=0,available=0;for(let p of d.points){let m=p[side];if(m.translation_cm===null||m.rotation_deg===null){missing++;continue}available++;if(m.translation_cm>X||m.rotation_deg>Y){outside++;continue}let xx=x(m.translation_cm),yy=y(m.rotation_deg),e;if(side==='before')e=elem('path',{d:`M${xx-3},${yy-3}l6,6M${xx-3},${yy+3}l6,-6`,stroke:'#2171b5','stroke-width':1.4});else e=elem('circle',{cx:xx,cy:yy,r:3,fill:'#d73027','fill-opacity':.65});let title=document.createElementNS(NS,'title');title.textContent=p.id+' '+side+' T='+m.translation_cm+'cm R='+m.rotation_deg+'deg savedSuccess='+m.success;e.append(title)}stats[side]={available,missing,outside}}
$('info').textContent=d.before_arm+' → '+d.after_arm+' | '+d.n+' paired frames\n'+(d.static_estimator_note?d.static_estimator_note+'\n':'')+(z?'EXPLICIT ZOOM; full data retained':'FULL AXES; all available outliers retained')+'\nBefore: '+stats.before.available+' coordinate points, '+stats.before.missing+' unavailable, '+stats.before.outside+' outside displayed view. After: '+stats.after.available+' coordinate points, '+stats.after.missing+' unavailable, '+stats.after.outside+' outside displayed view.'}
options($('stage'),[...new Set(all.map(d=>d.stage))],data.representative.stage);pairOptions();$('pair').value=data.representative.pair;seedOptions();$('stage').onchange=pairOptions;$('pair').onchange=seedOptions;['seed','zoom','tx','ry'].forEach(id=>$(id).onchange=draw);
</script>'''


def render(manifest,output,*,root_GO=False):
    if root_GO is not True:raise PermissionError('Explicit postseal figure resource GO required')
    output=Path(output);plan,rows,bindings=authenticate(manifest);payload=build_payload(plan,rows);source=M.sha(__file__)
    if (output/'FIGURE_MANIFEST.json').exists():
        old=M.read(output/'FIGURE_MANIFEST.json')
        if old['bindings']!=bindings or old['source_sha256']!=source:raise ValueError('Completed figure inputs/source differ')
        for name,value in old['artifacts'].items():
            if M.sha(output/name)!=value:raise ValueError('Completed figure artifact changed')
        return old
    output.mkdir(parents=True,exist_ok=True);M.write(output/'SCATTER_DATA.json',payload)
    embedded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    (output/'index.html').write_text(HTML.replace('__DATA__',embedded))
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='pallet-full-campaign-saved-error-scatter-v1'
    import matplotlib.pyplot as plt
    rep=payload['representative'];dataset=next(d for d in payload['datasets']if all(d[key]==rep[key]for key in ('stage','pair','seed')))
    stats=view_stats(dataset);fig,ax=plt.subplots(figsize=(9,6.5))
    for side,marker,color in (('before','x','#2171b5'),('after','o','#d73027')):
        values=[p[side]for p in dataset['points']if p[side]['translation_cm']is not None and p[side]['rotation_deg']is not None]
        ax.scatter([p['translation_cm']for p in values],[p['rotation_deg']for p in values],s=18,marker=marker,color=color,alpha=.65,label=dataset[side+'_arm'])
    ax.axvline(5,color='#789',ls='--',lw=.8);ax.axhline(5,color='#789',ls='--',lw=.8)
    ax.set(xlim=(0,stats['full_x_max']),ylim=(0,stats['full_y_max']),xlabel='Translation error T (cm)',ylabel='Rotation error R (deg)',
        title=rep['stage']+' / '+rep['pair']+' / '+str(rep['seed'])+'\nAll192 PRIMARY development frames;full axes;independent test n=0')
    ax.legend(fontsize=7);ax.grid(alpha=.15);fig.text(.5,.015,'No-pose coordinate points omitted and counted:before '+str(stats['sides']['before']['unavailable'])+',after '+str(stats['sides']['after']['unavailable'])+'.No tail trimming.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(output/'REPRESENTATIVE_FULL.png',dpi=180);fig.savefig(output/'REPRESENTATIVE_FULL.svg',metadata={'Date':None});plt.close(fig)
    receipt=dict(complete=True,component='saved-metric visualization only',bindings=bindings,source_sha256=source,representative=rep,
        datasets=len(payload['datasets']),PRIMARY_frames_per_dataset=192,all_outliers_retained=True,representative_stats=stats,
        coordinate_axes=dict(horizontal='T(cm)',vertical='R(deg)'),new_metric_evaluations=0,new_reference_decodes=0,new_RGB_reads=0,
        new_forwards=0,new_pose_calls=0,new_fits=0,artifacts={p.name:M.sha(p)for p in output.iterdir()if p.is_file()and p.name!='FIGURE_MANIFEST.json'})
    M.write(output/'FIGURE_MANIFEST.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--root-go',action='store_true')
    a=p.parse_args();print(render(a.manifest,a.output,root_GO=a.root_go))
