"""CPU-only posthoc cases; public RGB is restricted to an existing approval chain.

Inputs are frozen metrics/whole poses and the per-arm case associations produced
by controls.py (the new common selector may supply the identical schema).
Unrestricted top cases remain private. Public JSON contains scalar metrics and
bindings, never corner coordinates, camera matrices, or pose vectors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import os
from pathlib import Path
import re

import numpy as np

from . import common as C
from .robustness import select_cases

CATEGORIES = ('both_improved', 'both_worsened', 'largest_final_T', 'largest_final_R')
PAPER = C.ROOT / '_docs/experiments/pallet_selftraining_paper_closure_v1/FIGURE_MANIFEST.json'
RETAINED = C.ROOT / '_docs/experiments/pallet_pose_objective_followup_v2/RETAINED_FIGURE_MANIFEST.json'
SPLIT = C.ROOT / '_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json'
TRUTH = C.ROOT / 'data/pallet/results/pallet_replay_clean19_v1/TRUTH_FOR_DISPLAY_ONLY.json'
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def safe_tag(tag):
    assert re.fullmatch(r'[A-Za-z0-9_-]+', tag), 'Unsafe output tag'
    return tag


def approved_bindings(manifest, split_rows, retained):
    """IDs from original publication; bytes from the exact historical image list."""
    ids = {row['frame_id'] for row in manifest['examples']}
    historic = {row['id']:row['image'] for row in split_rows}
    assert ids <= set(historic)
    result = {fid:historic[fid] for fid in ids}
    # Retained examples provide a second explicit image-hash link where present.
    for example in retained['examples']:
        if example.get('material') == 'PLASTIC' and example.get('status') == 'AVAILABLE':
            assert example['id'] in result
            assert result[example['id']] == example['image']
    return result


def check_public_image(fid, image_binding, approved):
    assert fid in approved, 'Unapproved public RGB ID'
    assert image_binding == approved[fid], 'Public RGB bytes/path differ from historical image'


def choose(before, after, rows, allowed=None):
    """Same locked scalar rules for full99 and separately approved-ID subset."""
    selected_rows = [r for r in rows if allowed is None or r['id'] in allowed]
    cases = select_cases(before, after, selected_rows, count=2)
    occurrences = Counter(r['id'] for key in CATEGORIES for r in cases[key])
    output = {}
    for key in CATEGORIES:
        entries = [dict(row, repeated_across_categories=occurrences[row['id']] > 1)
                   for row in cases[key]]
        output[key] = dict(status='AVAILABLE' if entries else 'NA', examples=entries,
            reason=None if entries else 'No eligible common-valid frame in this category/scope')
    output['failure_frames'] = cases['failure_frames']
    output['eligible_frames'] = len(selected_rows)
    output['ranking_scope'] = 'NATURAL99' if allowed is None else 'PREVIOUSLY_PUBLIC_IDS_INTERSECT_NATURAL99'
    return output


def project_saved_pose(pose, K):
    """Project the saved final whole-pose cuboid, with no refitting or GT use."""
    if not pose['available']:
        return None
    w,h,d = np.asarray(pose['cf_extents'], dtype=float)
    xyz = np.array([[-w/2,-h/2,-d/2],[w/2,-h/2,-d/2],[w/2,h/2,-d/2],[-w/2,h/2,-d/2],
                    [-w/2,-h/2,d/2],[w/2,-h/2,d/2],[w/2,h/2,d/2],[-w/2,h/2,d/2]])
    camera = xyz @ np.asarray(pose['R_cf'], dtype=float).T + np.asarray(pose['centroid'], dtype=float)
    pixels = camera @ np.asarray(K, dtype=float).T
    valid = np.isfinite(pixels).all(1) & (camera[:,2] > 0)
    result = np.full((8,2), np.nan)
    result[valid] = pixels[valid,:2] / pixels[valid,2:]
    return result


def render(path, row, before, after, predictions, poses, metrics, truth):
    """Only call after membership/hash checks; overlays are not generated scenes."""
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/pallet-clean-minimal-cases-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    from scripts.research.pallet_selftraining_paper_closure_v1.report import overlay

    C.verify(row['image'])
    with Image.open(C.ROOT / row['image']['path']) as original:
        image = original.convert('RGB')
    assert list(image.size) == list(reversed(row['hw']))
    fid = row['id']
    gt = {j:xy for j,xy in enumerate(truth[fid]['gt'][:8]) if truth[fid]['valid'][j]}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    panels = []
    for ax, arm in zip(axes, (before,after)):
        m = metrics[arm][fid]
        caption = arm.replace('_S4', '\nS4')
        caption += ('\nT=%.3f cm | R=%.3f deg' % (m['translation_cm'],m['rotation_deg'])
                    if m['available'] else '\nPose unavailable')
        overlay(ax, image, predictions[arm][fid], gt, caption, '#008dba')
        points = project_saved_pose(poses[arm][fid], row['K'])
        if points is not None:
            for a,b in EDGES:
                if np.isfinite(points[[a,b]]).all():
                    ax.plot(points[[a,b],0], points[[a,b],1], color='#ef7b19', ls='--', lw=1.6)
        assert tuple(ax.get_xlim()) == (0.,float(image.width))
        assert tuple(ax.get_ylim()) == (float(image.height),0.)
        panels.append(dict(arm=arm, available=m['available'], translation_cm=m.get('translation_cm'),
                           rotation_deg=m.get('rotation_deg'), selected_hypothesis=poses[arm][fid].get('selected_hypothesis')))
    fig.suptitle(fid + ' | ' + row['severity'] + ' | ' + row['recording'], fontsize=10)
    legend = [Line2D([0],[0],color='#008dba',label='Native 2D prediction'),
              Line2D([0],[0],color='#ef7b19',ls='--',label='Saved selected whole-pose projection'),
              Line2D([0],[0],color='#34ef58',marker='x',ls='none',label='Legacy 2D reference')]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(.5,.035))
    fig.text(.5,.018,'T/R: geometry-derived repeated DEV reference, not independent physical ground truth.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.11,1,.94))
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=110,facecolor='white')
    plt.close(fig)
    with Image.open(path) as output:
        dimensions = list(output.size)
    return dict(file=C.bind(path),pixels=dimensions,image=row['image'],image_dimensions_wh=list(image.size),
                native_image_bounds_clipped=True,panels=panels,
                reference='Legacy2D green; native2D cyan; saved final whole-pose projection orange dashed; no refit or symmetry remapping.')


def run(metric_file, pose_file, binding_file, pairs_file, tag):
    tag = safe_tag(tag)
    destination = C.DOC / ('CASES_' + tag + '.json')
    if destination.exists():
        value=C.read(destination)
        for binding in value['inputs']+value['files']+value['private_artifacts']:
            C.verify(binding)
        for figure in value['figures']:
            C.verify(figure['image'])
        print('CASES_ALREADY_FROZEN',tag)
        return value
    metrics,poses,associations,pairs = [C.read(path) for path in (metric_file,pose_file,binding_file,pairs_file)]
    assert pairs and len(pairs) == len({tuple(p) for p in pairs})
    cache={}; inputs={}
    def read_bound(binding):
        C.verify(binding)
        inputs[binding['path']]=binding
        if binding['path'] not in cache:
            cache[binding['path']]=C.read(C.ROOT/binding['path'])
        return cache[binding['path']]
    predictions={};metadata=None
    arms=sorted({arm for pair in pairs for arm in pair})
    for arm in arms:
        assoc=associations[arm]
        rows=read_bound(assoc['metadata'])
        if metadata is not None:
            assert rows==metadata, 'Different metadata/population between arms'
        metadata=rows
        predictions[arm]=read_bound(assoc['predictions'])[assoc['prediction_arm']]
        for key in ('candidates','decisions'):
            if assoc.get(key) is not None:
                C.verify(assoc[key]);inputs[assoc[key]['path']]=assoc[key]
        assert set(metrics[arm])==set(poses[arm])==set(predictions[arm])=={r['id'] for r in rows}
    assert len(metadata)==128
    natural=[r for r in metadata if r['severity']!='CLEAN']
    assert len(natural)==99
    rowmap={r['id']:r for r in metadata}
    approved=approved_bindings(C.read(PAPER),C.read(SPLIT)['heldout'],C.read(RETAINED))
    truth=C.read(TRUTH)
    private_root=C.RAW/'cases'/tag
    public_root=C.DOC/'cases'/tag
    figures=[];private_files=[];comparisons={}
    html_parts=['<!doctype html><meta charset="utf-8"><title>Private measured cases</title>',
                '<h1>PRIVATE: unrestricted Natural99 top cases</h1>',
                '<p>Measured legacy-reference T/R; selected posthoc, not independent physical GT.</p>']
    report=['# 개선·악화·큰 오류 사례 ('+tag+')','',
        '전체99 선정은 아래 숫자 표와 private gallery에 보존했다. 공개 RGB 그림은 과거 공개된 ID 교집합에서 같은 규칙으로 별도 선정했으며 전체99의 최악 사례를 대신하지 않는다.',
        '','청록 실선/점은 native2D, 주황 점선은 저장된 최종 whole-pose 재투영, 초록 x는 legacy2D 참조다. T/R는 기하 재구성 참조에 대한 별도 값이며 독립 물리 GT가 아니다.','']
    for index,(before,after) in enumerate(pairs):
        name=after+'-minus-'+before
        all_cases=choose(metrics[before],metrics[after],natural)
        public_cases=choose(metrics[before],metrics[after],natural,set(approved))
        comparisons[name]=dict(before=before,after=after,full_NATURAL99=all_cases,approved_subset=public_cases)
        report += ['## '+name,'','| Scope/category | ID | T before→after cm | R before→after ° |','|---|---|---:|---:|']
        html_parts.append('<h2>'+html.escape(name)+'</h2>')
        for scope,cases,folder,is_public in [('full99',all_cases,private_root,False),('approved',public_cases,public_root,True)]:
            rendered={}
            for category in CATEGORIES:
                records=cases[category]['examples']
                if not records:
                    report.append('| '+scope+'/'+category+' | NA: no eligible frame | — | — |')
                for record in records:
                    fid=record['id'];row=rowmap[fid]
                    report.append('| %s/%s | %s | %.3f→%.3f | %.3f→%.3f |' % (scope,category,fid,
                        record['before']['translation_cm'],record['after']['translation_cm'],
                        record['before']['rotation_deg'],record['after']['rotation_deg']))
                    if fid not in rendered:
                        if is_public:
                            check_public_image(fid,row['image'],approved)
                        filename='%02d_%s.png' % (index,hashlib.sha256(fid.encode()).hexdigest()[:12])
                        result=render(folder/filename,row,before,after,predictions,poses,metrics,truth)
                        rendered[fid]=result
                        if is_public:
                            figures.append(dict(result,id=fid,comparison=name,role='APPROVED_SUBSET_POSTHOC_CASE',
                                approval_manifest=C.bind(PAPER),historical_image_list=C.bind(SPLIT)))
                        else:
                            private_files.append(result['file'])
                    record['figure']=rendered[fid]['file']
                    if not is_public:
                        html_parts.append('<h3>'+html.escape(category+' | '+fid)+'</h3><img style="max-width:100%" src="'+
                                          html.escape(Path(rendered[fid]['file']['path']).name)+'">')
            if is_public:
                report.append('')
                for fid,result in rendered.items():
                    relative=Path(result['file']['path']).relative_to(C.DOC.relative_to(C.ROOT))
                    report.append('!['+fid+']('+relative.as_posix()+')\n')
        report += ['', '전체99의 before/after 산출 실패: '+str(len(all_cases['failure_frames']))+'장. 실패 목록은 JSON에 전수 보존했다.','']
    gallery=private_root/'gallery.html';C.save(gallery,'\n'.join(html_parts)+'\n',True);private_files.append(C.bind(gallery))
    report_path=C.DOC/('CASES_'+tag+'.md');C.save(report_path,'\n'.join(report)+'\n',True)
    for path in (metric_file,pose_file,binding_file,pairs_file,PAPER,RETAINED,SPLIT,TRUTH,
                 Path(__file__),Path(__file__).with_name('robustness.py'),C.DOC/'ANALYSIS_LOCK.md',
                 C.ROOT/'scripts/research/pallet_selftraining_paper_closure_v1/report.py'):
        inputs[str(path)]=C.bind(path)
    value=dict(created_at=C.now(),status='COMPLETE',tag=tag,comparisons=comparisons,figures=figures,
        files=[C.bind(report_path)]+[f['file'] for f in figures],private_artifacts=private_files,inputs=list(inputs.values()),
        full_population=99,approved_ID_count=len(set(approved)&{r['id'] for r in natural}),
        private_gallery=C.bind(gallery),new_fits=0,GPU_seconds=0,
        public_payload='Scalar metrics, IDs, source bindings and already-public RGB overlays only; no coordinate/pose vectors.',
        selection='Posthoc same99; top2 joint decreases/increases ranked by deltaT, top2 finalT/finalR; ID tie; failures separately retained.',
        limitations='Existing geometry-derived reused DEV. Approved subset is not representative of full99 worst cases. No AI-generated scenes.')
    C.save(destination,value,True)
    print('CASES_COMPLETE',tag,'public',len(figures),'private',len(private_files)-1)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for option in ('metrics','poses','bindings','pairs'):
        parser.add_argument('--'+option,type=Path,required=True)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    run(args.metrics,args.poses,args.bindings,args.pairs,args.tag)
