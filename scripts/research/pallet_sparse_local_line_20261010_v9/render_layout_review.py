"""Improve only the layout of the original six V9 case illustrations.

Freeze exact existing score/figure/source bytes before importing plotting
libraries. Preserve every original PNG. Reuse the same six declared cases,
saved output/input coordinates, actual unused line support and pose labels.
Copy the two unchanged aggregate plots byte for byte; render six case PNGs
into a new reviewed_figures folder with reserved header and title space.
There is no inference, PnP, local optimization, scoring or bootstrap here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

REPO = Path(__file__).resolve().parents[3]
DOC = REPO/'_docs/experiments/pallet_sparse_local_line_20261010_v9'
METHODS = ('BASE','N3_SUBPIX','ROLE_BOUNDARY_H_ROBUST','ROLE_BOUNDARY_LOCAL_POINT_LINE')
FILES = ('FIGURE_LAYOUT_PROTOCOL.json','FIGURE_LAYOUT_STARTED.json','FIGURE_LAYOUT_CHECKS.json')
LAYOUT = dict(figsize_inches=[16,14],dpi=140,left=.02,right=.98,bottom=.03,top=.79,
    hspace=.65,wspace=.03,panel_title_fontsize=9,panel_title_pad=12,
    header_lines_y=[.98,.956,.932,.915],global_header_fontsize=10)


def require(value,message):
    if not value:raise RuntimeError(message)


def no_symlink(path):
    path=Path(path).absolute()
    require(not any(p.is_symlink() for p in (path,*path.parents)),'symlink: '+str(path))


def read(path):
    with Path(path).open('r',encoding='utf-8') as stream:return json.load(stream)


def binding(path):
    path=Path(path);no_symlink(path);require(path.is_file(),'missing layout input: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    absolute=path.absolute()
    return dict(path=str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else path.name,
        bytes=path.stat().st_size,sha256=h.hexdigest())


def bound(path,expected,label):
    require(all(binding(path)[k]==expected[k] for k in ('bytes','sha256')),'byte binding differs: '+label)


def public_path(item):
    p=Path(item['path'])
    require(item.get('origin')=='public_repository' and not p.is_absolute() and '..' not in p.parts,
        'public repository binding required')
    path=REPO/p;no_symlink(path);return path


def write_new(path,value):
    no_symlink(path)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def guard(args,stage):
    folder,output,source=(p.absolute() for p in (args.input,args.output,args.source_root))
    for p in (folder,output,source):no_symlink(p)
    require(folder.is_dir() and source.is_dir(),'existing evidence/source directories required')
    output.mkdir(parents=True,exist_ok=True)
    for name in FILES if stage=='freeze' else FILES[1:]:
        require(not (output/name).exists(),'preserve existing '+name)
    require(not (output/'reviewed_figures').exists(),'preserve existing reviewed figure prefix')
    if stage=='run':require((output/FILES[0]).is_file(),'own layout freeze required')
    return folder,output,source


def input_paths(folder,source):
    names=('PROTOCOL.json','SCORING_RECEIPT.json','VERIFICATION.json','FIGURE_BINDINGS.json',
        'FIGURE_CASE_ROWS.jsonl.gz','VISUAL_REVIEW_6.json','PREDICTIONS.jsonl.gz',
        'COMPARATOR_PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz')
    paths={n:folder/n for n in names}
    protocol=read(paths['PROTOCOL.json']);figures=read(paths['FIGURE_BINDINGS.json'])
    for name in ('render.py','visual_case_protocol','parent:OBSERVATIONS.jsonl.gz'):
        paths[name]=public_path(protocol['inputs'][name])
    cases=read(paths['visual_case_protocol'])
    require(len(cases['cases'])==6 and len({x['id'] for x in cases['cases']})==6,'original fixed six cases')
    for i,case in enumerate(cases['cases']):
        p=Path(case['image']['path']);require(not p.is_absolute() and '..' not in p.parts,'safe original RGB path')
        paths['image:%02d'%i]=source/p
    for item in figures['figures']:
        paths['original:'+Path(item['path']).name]=public_path(item)
    paths['layout_code']=Path(__file__).resolve()
    return paths


def metadata(paths):
    core,score,verified,figures,visual=(read(paths[n]) for n in
        ('PROTOCOL.json','SCORING_RECEIPT.json','VERIFICATION.json','FIGURE_BINDINGS.json','VISUAL_REVIEW_6.json'))
    require(core['schema']=='fixed_same_observation_local_C2_protocol_v9' and core['frames']==245 and
        core['methods']==[METHODS[-1]],'fixed one-arm V9 geometry required')
    require(score['complete'] is True and score['actual_total_scored_rows']==980 and score['new_local_optimizers']==0,
        'completed fit-free score980 required')
    for key,name in (('protocol','PROTOCOL.json'),('predictions','PREDICTIONS.jsonl.gz'),
        ('comparator_predictions','COMPARATOR_PREDICTIONS.jsonl.gz'),('fixed_predictions','FIXED_PREDICTIONS.jsonl.gz')):
        bound(paths[name],score[key],'score '+key)
    require(verified['complete'] is True and verified['passed'] is True and
        verified['actual_counts']['moment_scalars_checked']==648 and verified['actual_counts']['metric_CI_slots_checked']==81,
        'existing independent statistics PASS required')
    for name in ('PROTOCOL.json','SCORING_RECEIPT.json','PREDICTIONS.jsonl.gz',
        'COMPARATOR_PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz'):
        bound(paths[name],verified['inputs'][name],'verified '+name)
    require(figures['complete'] is True and figures['PNG_files']==8 and figures['panels']==24 and
        figures['selection_uses_new_scores'] is False,'completed original fixed8 figures required')
    for key,name in (('protocol','PROTOCOL.json'),('renderer_code','render.py'),('fixed_case_protocol','visual_case_protocol'),
        ('case_rows','FIGURE_CASE_ROWS.jsonl.gz'),('scored_methods','PREDICTIONS.jsonl.gz'),
        ('scored_comparator','COMPARATOR_PREDICTIONS.jsonl.gz'),('fixed_controls','FIXED_PREDICTIONS.jsonl.gz'),
        ('stored_ROLE_observations','parent:OBSERVATIONS.jsonl.gz')):
        bound(paths[name],figures[key],'original figures '+key)
    require(visual['complete'] is True and visual['actual_images_viewed']==6 and
        visual['all_headers_fully_readable'] is False,'actual original case visual limitation required')
    require(len(figures['figures'])==8,'original8 complete bindings')
    for item in figures['figures']:bound(paths['original:'+Path(item['path']).name],item,'original PNG')
    for name in ('render.py','visual_case_protocol','parent:OBSERVATIONS.jsonl.gz'):
        bound(paths[name],core['inputs'][name],'frozen display input '+name)
    cases=read(paths['visual_case_protocol'])
    for i,case in enumerate(cases['cases']):bound(paths['image:%02d'%i],case['image'],'original case RGB')
    return cases


def freeze(args):
    folder,output,source=guard(args,'freeze');paths=input_paths(folder,source);cases=metadata(paths)
    write_new(output/FILES[0],dict(schema='supplemental_same_six_case_layout_protocol_v9',
        inputs={k:binding(p) for k,p in paths.items()},input_folder=str(folder),source_folder=str(source),
        output_folder=str(output),layout=LAYOUT,case_ids=[c['id'] for c in cases['cases']],
        original_PNG_files_preserved=8,case_PNGs_to_render=6,aggregate_PNGs_to_copy_unchanged=2,
        new_cases_or_score_selection=0,new_model_PnP_local_optimizer_GT_score_bootstrap_calls=0,
        plotting_or_image_libraries_imported_before_own_freeze=False,
        frozen_before_own_layout_rendering=True,no_automatic_retry=True))
    print('V9_FIGURE_LAYOUT_FROZEN',binding(output/FILES[0])['sha256'],flush=True)


def stream_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            require(line.strip(),'blank saved display row');yield json.loads(line)


def compact(row):
    s=row.get('solver') or {};new=row['new_pose_estimated']
    return dict(id=row['id'],method=row['method'],native_points=row['native_points'],input_points=row.get('input_points'),
        pose=row['pose'],output_status=row['output_status'],reprojected_ids=row.get('reprojected_ids',[]),
        evaluation_reference=row.get('evaluation_reference',{}),
        accepted_fit_ids=s.get('fit_input_ids',[]) if new else [],accepted_inlier_ids=s.get('final_inliers',[]) if new else [],
        actual_unused_line_edges=s.get('line_edges',[]),accepted_fit_line_edges=s.get('fit_line_edges',[]) if new else [],
        diagnostic_inlier_support_rank6=s.get('diagnostic_inlier_support_rank6'))


def title(row):
    T,R=row['pose'].get('translation_cm'),row['pose'].get('rotation_deg')
    fmt=lambda v:'None' if v is None else str(round(v,4))
    lines=[row['method'],row['output_status']+' / T='+fmt(T)+'cm / R='+fmt(R)+'deg',
        'fit points='+str(row['accepted_fit_ids'])+'; diagnostic points='+str(row['accepted_inlier_ids'])]
    if row['method']==METHODS[-1]:
        lines+=['unused lines='+str(row['actual_unused_line_edges']),
            'fit lines='+str(row['accepted_fit_line_edges'])+'; 8px support rank6='+str(row['diagnostic_inlier_support_rank6'])]
    return '\n'.join(lines)


def run(args):
    folder,output,source=guard(args,'run');own=read(output/FILES[0]);paths=input_paths(folder,source)
    current={k:binding(p) for k,p in paths.items()}
    require(own['schema']=='supplemental_same_six_case_layout_protocol_v9' and own['inputs']==current and
        own['layout']==LAYOUT and own['input_folder']==str(folder) and own['output_folder']==str(output) and
        own['source_folder']==str(source),'own layout protocol differs')
    cases=metadata(paths)
    write_new(output/FILES[1],dict(protocol=binding(output/FILES[0]),inputs=current,
        actual_case_layout_runs=1,new_model_PnP_local_optimizer_GT_score_bootstrap_calls=0,no_automatic_retry=True))
    began=time.perf_counter();target=output/'reviewed_figures';target.mkdir(exist_ok=False)
    result=dict(schema='supplemental_same_six_case_layout_checks_v9',complete=False,passed=False,
        protocol=binding(output/FILES[0]),inputs=current,original_PNGs_preserved=True,original_core_and_raws_modified=False,
        changed_only_case_layout=True,aggregate_performance_arithmetic_repeated=False,
        new_model_PnP_local_optimizer_GT_score_bootstrap_calls=0,automatic_retry=False,failure_count=0,failures=[])
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        ids=[c['id'] for c in cases['cases']];wanted=set(ids);data={};population=Counter()
        for filename in ('PREDICTIONS.jsonl.gz','COMPARATOR_PREDICTIONS.jsonl.gz','FIXED_PREDICTIONS.jsonl.gz'):
            for row in stream_rows(paths[filename]):
                population[row['method']]+=1
                if row['id'] in wanted:
                    key=(row['method'],row['id']);require(key not in data,'duplicate case display identity')
                    data[key]=compact(row)
        require(population=={m:245 for m in METHODS} and len(data)==24,'complete same980/six-by-four display population')
        stored={(r['method'],r['id']):r for r in stream_rows(paths['FIGURE_CASE_ROWS.jsonl.gz'])}
        require(set(stored)==set(data),'original same24 panel identities')
        for key,row in data.items():
            old=stored[key]
            for field in ('pose','output_status','accepted_fit_ids','accepted_inlier_ids','actual_unused_line_edges',
                'accepted_fit_line_edges','diagnostic_inlier_support_rank6'):
                require(row[field]==old[field],'layout altered original stored panel field '+field)
        copied=[]
        for name in ('01_all_operational.png','02_same_frame_pairs.png'):
            destination=target/name
            with paths['original:'+name].open('rb') as src,destination.open('xb') as dst:
                for block in iter(lambda:src.read(8*1024*1024),b''):dst.write(block)
                dst.flush();os.fsync(dst.fileno())
            bound(destination,current['original:'+name],'copied unchanged aggregate PNG')
            copied.append(name)
        panels=[]
        for index,case in enumerate(cases['cases']):
            with Image.open(paths['image:%02d'%index]) as image:rgb=np.asarray(image.convert('RGB'))
            fig,axes=plt.subplots(2,2,figsize=LAYOUT['figsize_inches'])
            fig.subplots_adjust(**{k:LAYOUT[k] for k in ('left','right','bottom','top','hspace','wspace')})
            headers=[case['label']+' / '+case['id'],
                'cyan=output; green+=sparse points; yellow dashed=stored unused line support; orange square=H projection; white=proxy',
                'LOCAL uses sealed pose starts only; no residual prior; global uniqueness is unproven.',
                'Missing native coordinates are not fit inputs. 8px inlier support and numerical NEW do not certify pose accuracy.']
            for y,text in zip(LAYOUT['header_lines_y'],headers):fig.text(.5,y,text,ha='center',va='top',fontsize=LAYOUT['global_header_fontsize'])
            for axis,method in zip(axes.flat,METHODS):
                key=(method,case['id']);row=data[key];axis.imshow(rgb)
                points=np.asarray(row['native_points'],float)
                valid=np.isfinite(points[:8]).all(1)&(points[:8]!=[-1,-1]).any(1)
                axis.scatter(points[:8][valid,0],points[:8][valid,1],s=32,facecolor='none',edgecolor='#00dce6',linewidth=1.4)
                for k in np.flatnonzero(valid):axis.text(points[k,0]+3,points[k,1]+3,str(k),color='cyan',fontsize=8)
                for k in row['reprojected_ids']:axis.scatter(*points[k],s=68,marker='s',facecolor='none',edgecolor='#ff8c24')
                if method in METHODS[2:]:
                    q=np.asarray(row['input_points'],float);validq=np.isfinite(q[:8]).all(1)&(q[:8]!=[-1,-1]).any(1)
                    axis.scatter(q[:8][validq,0],q[:8][validq,1],s=35,marker='+',color='#42ff50')
                lines=stored[key]['displayed_original_unused_line_support']
                require([x['edge'] for x in lines]==row['actual_unused_line_edges'] if method==METHODS[-1] else not lines,
                    'original displayed unused line-edge identity')
                for line in lines:
                    support=np.asarray(line['support_points'],float)
                    require(support.ndim==2 and support.shape[1]==2 and len(support)>=2 and np.isfinite(support).all(),
                        'same finite actual stored line support')
                    axis.plot(support[:,0],support[:,1],color='#fff340',linestyle='--',linewidth=1.5,marker='.',markersize=4)
                    axis.text(support[0,0]+4,support[0,1]-4,'e'+str(line['edge']),color='#fff340',fontsize=8)
                ref=row['evaluation_reference'];gt=np.asarray(ref.get('native_points_px',[]),float)
                if len(gt):
                    known=ref.get('valid_native_ids',[]);axis.scatter(gt[known,0],gt[known,1],s=19,marker='.',color='white',alpha=.65)
                axis.set_title(title(row),fontsize=LAYOUT['panel_title_fontsize'],pad=LAYOUT['panel_title_pad']);axis.axis('off')
                panels.append(dict(id=case['id'],method=method,original_panel_identity_and_labels_unchanged=True,
                    original_unused_line_support_unchanged=True,native_missing_points_not_fit_inputs=True))
            destination=target/('case_%02d.png'%(index+1));require(not destination.exists(),'exclusive new case PNG')
            fig.savefig(destination,dpi=LAYOUT['dpi']);plt.close(fig)
        require(len(panels)==24 and len(list(target.glob('*.png')))==8,'actual reviewed8/same24')
        require({k:binding(p) for k,p in paths.items()}==current,'original inputs changed during layout-only rendering')
        result.update(complete=True,passed=True,actual_case_PNGs_rendered=6,aggregate_PNGs_byte_copied=copied,
            PNG_files=8,panels=24,case_ids=ids,panel_checks=panels,figures=[binding(p) for p in sorted(target.glob('*.png'))],
            actual_new_source_RGB_decodes=6,physical_truth_or_actual_line_ownership_certified=False,
            independent_visual_readability_requires_actual_viewing=True)
    except Exception as error:
        result.update(failure_count=1,failures=[dict(type=type(error).__name__,message=str(error))],
            preserved_rendered_prefix=[binding(p) for p in sorted(target.glob('*.png'))])
    result['elapsed_seconds']=time.perf_counter()-began
    write_new(output/FILES[2],result)
    print('V9_LAYOUT','PASS' if result['passed'] else 'FAIL',result.get('PNG_files'),flush=True)
    if not result['passed']:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--input',type=Path,default=DOC)
    parser.add_argument('--output',type=Path,default=DOC)
    parser.add_argument('--source-root',type=Path,required=True,help='preserved original checkout containing only requested RGB paths')
    args=parser.parse_args();(freeze if args.stage=='freeze' else run)(args)


if __name__=='__main__':main()
