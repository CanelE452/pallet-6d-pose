"""Read-only saved-layout and actual headless-browser checks for global selection."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.research.pallet_dht_global_layout_v1 import report as R
from scripts.research.pallet_dht_coupling_v2.visual_qa import Browser, DataScript


def audit(run_dir):
    run_dir=Path(run_dir).resolve();inputs=R.Inputs()
    receipt=inputs.read(run_dir/'REPORT_RENDER.json')
    assert receipt['complete'] and receipt['PASS'] and receipt['experiment_complete'] and receipt['n_real_frames']==319
    for path,digest in receipt['input_sha256'].items():inputs.bind(path,digest)
    page=inputs.bind(run_dir/'index.html',receipt['html_sha256'])
    parser=DataScript();parser.feed(page.read_text());assert not parser.external
    data=json.loads(''.join(parser.chunks));assert len(data['frames'])==319
    pred=inputs.read(run_dir/'PREDICTIONS.json');metric=inputs.read(run_dir/'FRAME_METRICS.json')
    source={f['id']:f for f in pred['records'] if f['population']=='real_dev'}
    metrics={f['id']:f for f in metric['records'] if f['population']=='real_dev'}
    frames={f['id']:f for f in data['frames']}
    assert len(frames)==319 and set(frames)==set(source)==set(metrics)
    assert data['edges']==[list(e) for e in R.EDGES] and data['roles']==list(R.ROLES)
    assert data['required_case']==R.CASE
    old=inputs.read(R.PROBE/'PREDICTIONS.json')
    old={f['id']:f for f in old['records'] if f['population']=='real_dev'}
    overlays=0;hypotheses=0;duplicate_layouts=0;max_score_delta=0.
    for ident,frame in frames.items():
        saved=source[ident];m=metrics[ident]
        assert frame['baseline']==saved['baseline'] and frame['arms']==saved['arms']
        assert saved['baseline']['points']==old[ident]['baseline']['points']
        assert saved['baseline']['point_valid']==old[ident]['baseline']['point_valid']
        assert frame['metrics']==m['arms']
        assert frame['layout_diagnostic']==m.get('layout_diagnostic') and frame['candidate_diagnostic']==m.get('candidate_diagnostic')
        original=inputs.bind(saved['image'],saved['image_sha256'])
        assert hashlib.sha256(base64.b64decode(frame['image'].split(',',1)[1])).hexdigest()==R.sha(original)
        annotation=inputs.read(frame['gt']['annotation'],frame['gt']['annotation_sha256'])['objects'][0]['keypoint_annotations']
        assert frame['gt']['points']==[k['xy'] for k in annotation]==m['gt_points']
        assert frame['gt']['visibility']==[k['visibility'] for k in annotation]
        assert frame['gt']['sources']==[k.get('source','unknown') for k in annotation]
        assert [v>0 for v in frame['gt']['visibility']]==m['gt_supervised']
        for arm in R.ARMS:
            row=frame['baseline'] if arm=='baseline' else frame['arms'][arm]
            assert row['point_valid']==frame['baseline']['point_valid']
            assert row['points'][8]==frame['baseline']['points'][8]
            assert len(row['points'])==len(row['point_valid'])==9
            overlays+=1
            if arm in ('baseline','independent'):continue
            top=row['top_hypotheses'];assert 1<=len(top)<=3
            assert top[0]['points_xy']==row['points']
            if 'index' in top[0]:assert row['selected_hypothesis_index']==top[0]['index']
            assert all(top[i]['total']<=top[i+1]['total']+1e-12 for i in range(len(top)-1))
            seen=[]
            for h in top:
                assert len(h['points_xy'])==9 and len(h['indices'])==8
                assert h['points_xy'][8]==frame['baseline']['points'][8]
                if 'weighted_terms' in h:
                    total=sum(v for v in h['weighted_terms'].values() if v is not None)
                    delta=abs(total-h['total']);assert delta<1e-9
                    max_score_delta=max(max_score_delta,delta)
                if any(h['points_xy']==p for p in seen):duplicate_layouts+=1
                seen.append(h['points_xy']);hypotheses+=1
        evidence=frame['evidence']
        if evidence is not None:
            for key,value in evidence.items():assert value==saved['evidence'][key]
            if 'line_peaks_h_raw' in evidence:
                lines=np.asarray(evidence['line_peaks_h_raw']);valid=np.asarray(evidence['line_peak_valid'],bool)
                assert lines.shape[:2]==valid.shape and lines.shape[0]==12 and lines.shape[-1]==3
                norms=np.linalg.norm(lines[valid,:2],axis=-1)
                assert np.isfinite(lines).all() and (not len(norms) or np.max(abs(norms-1))<1e-6)
    assert overlays==319*4
    directory=run_dir/'actual_visual_qa';directory.mkdir(exist_ok=True)
    b=Browser(directory);shots={}
    try:
        b.call('Page.navigate',{'url':page.as_uri()});b.ready()
        assert b.js('document.title')==R.TITLE and b.js('REPORT_FRAME_COUNT')==319
        assert b.js('REPORT_CURRENT_FRAME')==R.CASE and b.js("$('role').value")=='7'
        def shot(name,element=None):
            if element:
                clip=b.js(f"(()=>{{const q=document.getElementById('{element}').getBoundingClientRect();return {{x:q.x+scrollX,y:q.y+scrollY,width:q.width,height:q.height,scale:1}}}})()")
            else:clip=b.js('({x:0,y:scrollY,width:innerWidth,height:innerHeight,scale:1})')
            path=directory/name;b.screenshot(path,clip);shots[str(path)]=R.sha(path)
        shot('home.png')
        decoded=b.js("(async()=>{let n=0;for(let i=0;i<DATA.frames.length;i+=8)await Promise.all(DATA.frames.slice(i,i+8).map(async f=>{const im=new Image();im.src=f.image;await im.decode();if(im.width!==f.width||im.height!==f.height)throw Error('raw shape');n++}));return n})()")
        assert decoded==319
        for arm in ('global','global_shared_only'):
            b.select('arm',arm)
            for role in range(12):b.select('role',role)
            for mode in ('all','0','1','2','3'):b.select('mode',mode)
            for alt in b.js("Array.from($('alternative').options).map(o=>o.value)"):
                b.select('alternative',alt);assert b.js("$('alternative').value")==alt
        b.select('arm','global');b.select('role',7);b.select('point',4)
        b.select('mode','all')
        shot('problem_case_global_layout.png','gallery');shot('actual_top3_scores.png','scores')
        b.js("$('yaw90').click();true");b.ready();assert b.js("$('alternative').value")=='explicit:baseline_yaw90'
        shot('problem_case_explicit_yaw90.png','gallery')
        shot('explicit_yaw90_scores.png','scores')
        assert '고정 yaw90 후보는 존재합니다' in b.js("$('casefinding').textContent")
        b.js("$('raw').click();true");b.ready();assert b.js('clean') is True
        b.js("$('raw').click();$('zoom').click();$('zoom').click();true");b.ready()
        b.select('rank','worst');shot('gt_ranked_most_harmed.png','gallery')
        b.select('rank','best');assert b.js('REPORT_CURRENT_FRAME') in frames
        b.select('rank','case');assert b.js('REPORT_CURRENT_FRAME')==R.CASE
        if b.js("Boolean($('viewdiagnostic'))"):
            b.js("$('viewdiagnostic').open=true;true");shot('view_cohorts.png','viewdiagnostic')
            b.js("$('viewdiagnostic').open=false;true")
        assert not b.errors and not b.console_errors and not b.external_requests
        browser=dict(PASS=True,raw_images_decoded=decoded,role_options_tested=12,
            joint_arms_tested=2,actual_saved_hypotheses_tested=True,default_required_case=True,
            gt_ranked_sort_is_posthoc=True,js_errors=b.errors,console_errors=b.console_errors,
            external_requests=b.external_requests)
    finally:b.close()
    inputs.verify()
    result=dict(complete=True,PASS=True,html_sha256=R.sha(page),n_real_frames=319,
        n_exact_prediction_overlays=overlays,n_saved_hypotheses_checked=hypotheses,
        n_identical_coordinate_hypotheses=duplicate_layouts,max_weighted_score_sum_delta=max_score_delta,
        GT_source_and_visibility_exact=True,baseline_points_preserved=True,centroid_and_missing_masks_preserved=True,
        browser=browser,input_sha256=inputs.hashes,output_sha256=shots,
        scope='Actual saved-output equality and browser interactions; not physical GT certification or a search optimality proof.',
        no_inference=True,no_gpu=True,no_desktop_open=True,no_notification=True)
    R.write(run_dir/'ACTUAL_VISUAL_QA.json',result)
    print(json.dumps({k:result[k] for k in ('complete','PASS','html_sha256','n_exact_prediction_overlays')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    audit(parser.parse_args().run_dir)
