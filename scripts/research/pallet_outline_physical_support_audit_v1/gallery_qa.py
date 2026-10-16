"""Check saved display coordinates and source pixels without cost evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from html.parser import HTMLParser
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from .bindings import load_cases,load_dense,permission_bindings
from .gallery import sha,write
from .audit_guard import SupportAuditOnly


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if 'id'in values:self.ids.add(values['id'])
        for k in('href','src'):
            if k in values:self.links.append(values[k])


def project(xyz,R,t,K):
    transform=np.eye(4);transform[:3,:3]=R;transform[:3,3]=t
    points=np.column_stack((xyz,np.ones(len(xyz))))
    homogeneous=np.asarray(K)@(transform@points.T)[:3]
    return (homogeneous[:2]/homogeneous[2]).T


def _check(source_root,score_private,state_private,analysis_private,gallery,output,guard):
    start,cpu=time.monotonic(),time.process_time();root,analysis,g=map(Path,(source_root,analysis_private,gallery))
    cases=load_cases(root,score_private,state_private);permission_bindings(root,cases)
    proof=json.loads((g/'MODEL_COMPARISON_PROOF.json').read_text())['cases']
    original=json.loads((g/'INITIAL_GALLERY_PROOF.json').read_text())['cases']
    if [x['id']for x in proof]!=[x['id']for x in cases]:raise ValueError('Six-case order changed')
    checked_panels=0;sample_count=0;parent_count=0;extreme_count=0;projection_max=0.;out_of_canvas=0;zoom_count=0
    for case,p,first in zip(cases,proof,original):
        ident=case['id'];ordinal=case['ordinal'];folder=g/f'case_{ordinal:02d}'
        rgb=np.asarray(Image.open(case['image_path']).convert('RGB'));h,w=rgb.shape[:2]
        roi=p['display_crop_original_xyxy'];x0,y0,x1,y1=roi
        if roi!=first['crop_original_xyxy']or p['parent_crop_original_xyxy']!=case['crop_xyxy']:
            raise ValueError('Original/display crop transform mismatch')
        crop=np.asarray(Image.open(folder/'crop_RGB_unmarked.png').convert('RGB'))
        if not np.array_equal(crop,rgb[y0:y1,x0:x1]):raise ValueError('Display PNG is not exact original crop pixels')
        K=np.asarray(case['K'])
        for panel,source in zip(p['fixed_candidate_panels'],case['panels']):
            if panel!=source:raise ValueError('A saved candidate/display field changed')
            q=project(np.asarray(panel['native_xyz']),panel['R'],panel['t'],K)
            difference=float(np.abs(q-np.asarray(panel['model_projection_xy'])).max());projection_max=max(projection_max,difference)
            if difference>1e-9:raise ValueError('Fixed original model projection mismatch')
            out_of_canvas+=sum(not(0<=xy[0]<w and 0<=xy[1]<h)for xy in q)
            checked_panels+=1
        ref=case['reference'];a,b,c=np.asarray(ref['xyz'])/2
        xyz=np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
        qr=project(xyz,ref['R'],ref['t'],K)
        difference=float(np.abs(qr-np.asarray(p['reference_projection_xy'])).max());projection_max=max(projection_max,difference)
        if difference>1e-9:raise ValueError('Existing reference projection mismatch')
        dense=load_dense(case,score_private)
        obs=[next(o for o in row['per_observation']if o.get('direction')=='observed_to_model')for row in dense]
        original_samples=np.asarray(obs[0]['sample_xy'])
        if not np.array_equal(original_samples,obs[1]['sample_xy'])or not np.array_equal(original_samples,p['original_observed_sample_xy']):
            raise ValueError('Observed sample identity/order changed')
        if not all(x0<=xy[0]<x1 and y0<=xy[1]<y1 for xy in original_samples):raise ValueError('Audit crop omitted an original observed sample')
        delta=np.asarray(obs[1]['weighted_sample_contributions'])-np.asarray(obs[0]['weighted_sample_contributions'])
        if not np.array_equal(delta,p['observed_Delta']):raise ValueError('Stored sample Delta changed')
        if abs(math.fsum(delta)-p['observed_Delta_total'])>1e-9:raise ValueError('Observed Delta direction sum mismatch')
        lineage=json.loads((analysis/p['mask_lineage_path']).read_text())
        sample_count+=len(original_samples);parent_count+=len(lineage['used_segments'])
        original_extremes={r['sample_index']for k in('largest_positive_samples','largest_negative_samples')for r in case['witness'][k]}
        displayed={r['sample_index']for z in p['parent_contexts']for r in z['source_extreme_members']}
        if original_extremes!=displayed:raise ValueError('Original positive/negative witness list was omitted')
        extreme_count+=len(original_extremes);zoom_count+=len(p['parent_contexts'])
        for z in p['parent_contexts']:
            parent=next(x for x in lineage['used_segments']if x['segment_id']==z['segment_id'])
            if z['all_source_sample_indices']!=parent['sample_indices']:raise ValueError('Partial parent selected instead of entire original segment')
            if abs(math.fsum(delta[parent['sample_indices']])-z['segment_signed_Delta'])>1e-9:raise ValueError('Parent signed Delta changed')
        for row,model,panel in zip(dense,p['model_to_observed_candidate_specific'],p['fixed_candidate_panels']):
            source=next(o for o in row['per_observation']if o.get('direction')=='model_to_observed')
            if not np.array_equal(model['sample_xy'],source['sample_xy'])or model['sample_weighted_contributions']!=source['weighted_sample_contributions']or model['normalizer']!=source['normalizer']:
                raise ValueError('Candidate-specific model samples/arc costs changed')
            if not model['sample_indices_not_paired_across_candidates']or not p['model_indices_never_paired']:raise ValueError('False paired model sample indices')
    links_checked=0;deferred=[]
    for filename in('index.html','initial_index.html'):
        parser=Links();parser.feed((g/filename).read_text())
        for link in parser.links:
            u=urlsplit(link)
            if u.scheme:continue
            if not u.path:
                if u.fragment and u.fragment not in parser.ids:raise ValueError('Missing gallery anchor '+link)
            elif u.path=='../REPORT_KO.md'and not(g/u.path).exists():deferred.append(link)
            elif not(g/u.path).is_file():raise ValueError('Missing gallery asset '+link)
            links_checked+=1
    decoded=Counter()
    for path in g.rglob('*'):
        if path.suffix.lower()not in('.jpg','.png'):continue
        with Image.open(path)as im:im.verify()
        decoded[path.suffix.lower()]+=1
    result=dict(schema='OUTLINE_PHYSICAL_GALLERY_COORDINATE_QA_V1',passed=True,cases=6,
        exact_original_RGB_crop_pixel_checks=6,original_candidate_panels_checked=checked_panels,
        separate_existing_reference_panels_checked=6,independent_homogeneous_projection_max_abs_px=projection_max,
        original_observed_samples_preserved=sample_count,parent_segments_preserved=parent_count,
        original_extreme_sample_ids_preserved=extreme_count,entire_parent_context_images=zoom_count,
        model_to_observed_candidates_checked_separately=12,model_sample_indices_never_paired=True,
        projected_model_vertices_outside_original_canvas=out_of_canvas,
        JPEG_byte_equality_to_original_RGB_not_claimed=True,crop_resize_letterbox_offset_added=False,
        local_gallery_links_checked=links_checked,root_report_links_pending_only=deferred,decoded_image_files=dict(decoded),
        RGB_and_mask_prior_publication_permission_checked=True,
        new_cost_evaluations=0,new_pose_solves=0,new_observations=0,new_forward_calls=0,
        elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,runtime_guard=guard.receipt())
    write(output,result);return result


def check(*args):
    with SupportAuditOnly()as guard:return _check(*args,guard)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in('source-root','score-private','state-private','analysis-private','gallery','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(check(a.source_root,a.score_private,a.state_private,a.analysis_private,a.gallery,a.output),ensure_ascii=False))
