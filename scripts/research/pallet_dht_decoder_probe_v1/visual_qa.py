"""Actual saved-output and headless-browser QA; never performs model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.research.pallet_dht_decoder_probe_v1 import evaluate as E
from scripts.research.pallet_dht_decoder_probe_v1 import report as R
from scripts.research.pallet_dht_coupling_v2.visual_qa import Browser, DataScript


def audit(run_dir):
    run_dir=Path(run_dir).resolve();inputs=E.Inputs()
    receipt=inputs.read(run_dir/'REPORT_RENDER.json')
    E.require(receipt.get('complete') and receipt.get('PASS') and receipt.get('experiment_complete')
        and receipt['n_real_frames']==319 and receipt['n_new_training_seeds']==1,'Actual completed decoder pilot required')
    for path,digest in receipt['input_sha256'].items():inputs.bind(path,digest)
    page=inputs.bind(run_dir/'index.html',receipt['html_sha256'])
    parser=DataScript();parser.feed(page.read_text());E.require(not parser.external,'Offline rendering dependency found')
    data=json.loads(''.join(parser.chunks));predictions=inputs.read(run_dir/'PREDICTIONS.json')
    metrics=inputs.read(run_dir/'FRAME_METRICS.json')
    source={r['id']:r for r in predictions['records'] if r['population']=='real_dev'}
    by_id={r['id']:r for r in metrics['records'] if r['population']=='real_dev'}
    frames={r['id']:r for r in data['frames']}
    E.require(len(data['frames'])==len(frames)==319 and set(frames)==set(source)==set(by_id),'Original DEV319 IDs differ')
    E.require(data['required_case']==E.CASE and E.CASE in frames,'Required problem image absent')
    overlay_count=0
    for key,frame in frames.items():
        saved=source[key];metric=by_id[key]
        E.require(frame['baseline']==saved['baseline'] and frame['arms']==saved['arms'],'HTML decoder output differs from saved forward')
        E.require(frame['metrics']==metric['arms'],'HTML frame errors differ')
        expected=[p if v else None for p,v in zip(metric['gt_points'],metric['gt_supervised'])]
        E.require(frame['gt']==expected,'GT IDs/supervision mask differs')
        path=inputs.bind(saved['image'],saved['image_sha256'])
        image=cv2.imread(str(path));E.require(image is not None,'Source decode failure')
        E.require(frame['image']==R.H.data_uri(image),'Embedded image differs from canonical source')
        for arm in E.ARMS:
            value=saved['baseline'] if arm=='baseline' else saved['arms'][arm]
            p,valid=E.finite_points(value['points'],value['point_valid'])
            b,bvalid=E.finite_points(saved['baseline']['points'],saved['baseline']['point_valid'])
            E.require(np.array_equal(valid,bvalid) and np.array_equal(p[8],b[8]),'Coverage/centroid changed')
            if arm!='baseline':
                weights=np.asarray(value['decoder']['candidate_weights'])
                gates=np.asarray(value['decoder']['gate'])
                E.require(weights.shape==(8,49) and np.isfinite(weights).all() and (weights>=0).all(),'Malformed actual candidate weights')
                E.require(np.max(np.abs(weights.sum(-1)-1))<2e-6,'Candidate weights not normalized')
                E.require(gates.shape==(8,) and np.isfinite(gates).all() and (np.abs(gates)<=1).all(),'Malformed signed gate')
            overlay_count+=1
    directory=run_dir/'actual_visual_qa';directory.mkdir(exist_ok=True)
    browser=Browser(directory);shots={}
    try:
        browser.call('Page.navigate',{'url':page.as_uri()});browser.ready()
        E.require(browser.js('document.title')==R.TITLE,'Actual title differs')
        E.require(browser.js('REPORT_FRAME_COUNT')==319 and browser.js('REPORT_CURRENT_FRAME')==E.CASE,'Default original case/coverage differs')
        def shot(name):
            browser.js('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))')
            clip=browser.js('({x:0,y:window.scrollY,width:innerWidth,height:innerHeight,scale:1})')
            path=directory/name;browser.screenshot(path,clip=clip);shots[str(path)]=E.sha(path)
        shot('home.png')
        count=browser.js("(async()=>{let n=0;for(let i=0;i<DATA.frames.length;i+=12){await Promise.all(DATA.frames.slice(i,i+12).map(f=>new Promise((r,j)=>{const x=new Image();x.onload=()=>{n++;r()};x.onerror=j;x.src=f.image})));}return n})()")
        E.require(count==319,'Embedded image browser decode count differs')
        for arm in ('line_fusion','wrong_image_line'):
            browser.select('arm',arm)
            for corner in range(8):browser.select('corner',corner)
        browser.select('arm','line_fusion');browser.select('corner',4)
        browser.js("document.getElementById('gallery').scrollIntoView();true")
        shot('problem_case_candidates.png')
        browser.select('rank','worst')
        shot('gt_ranked_most_harmed.png')
        E.require(not browser.errors and not browser.console_errors and not browser.external_requests,'Browser JS/console/external request issue')
        browser_result=dict(PASS=True,images_decoded=319,line_input_modes_checked=2,corner_choices_checked=8,
            default_problem_case=True,gt_ranked_worst_is_posthoc=True,js_errors=browser.errors,
            console_errors=browser.console_errors,external_requests=browser.external_requests)
    finally:
        browser.close()
    inputs.verify()
    final=dict(complete=True,PASS=True,html_sha256=E.sha(page),n_real_frames=319,n_new_training_seeds=1,
        n_exact_prediction_overlays=overlay_count,input_sha256=inputs.hashes,output_sha256=shots,
        browser=browser_result,scope='Independent saved-data equality and actual headless interactions; human screenshot review belongs to root.',
        no_inference=True,no_gpu=True,no_desktop_open=True,no_notification=True)
    E.write(run_dir/'ACTUAL_VISUAL_QA.json',final)
    print(json.dumps({k:final[k] for k in ('complete','PASS','html_sha256','n_exact_prediction_overlays')}))
    return final


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    audit(parser.parse_args().run_dir)
