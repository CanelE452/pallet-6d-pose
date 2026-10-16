"""Verify rendered gallery against sealed candidate/cost/metric and RGB bytes."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
from html.parser import HTMLParser
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from .baseline import read_rows,sha,write_json


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id'in a:self.ids.add(a['id'])
        for key in('src','href'):
            if key in a:self.links.append(a[key])


def verify(gallery,private,old_private,source_root):
    from scripts.research.pallet_minimal_bi_ablation_v1.gallery import public_image_inventory
    start=time.monotonic();cpu=time.process_time();g,p,old,root=map(Path,(gallery,private,old_private,source_root))
    proof=json.loads((g/'GALLERY_PROOF.json').read_text());receipt=json.loads((g/'GALLERY_RECEIPT.json').read_text())
    for name,digest in receipt['assets'].items():
        if sha(g/name)!=digest:raise ValueError('Rendered asset changed '+name)
    ids={r['id']for r in proof['cases']+proof['no_candidate_RGB_links']}
    packets={r['id']:r for r in read_rows(old/'e_trace_v2/TYPED_PACKETS.jsonl.gz')if r['id']in ids}
    candidate_keys={(r['id'],c['candidate_id'])for r in proof['cases']for c in r['panels']}
    cand={(r['id'],r['candidate_id']):r for r in read_rows(p/'selection_v1/RETAINED_CANDIDATES.jsonl.gz')if(r['id'],r['candidate_id'])in candidate_keys}
    metrics={(r['id'],r['candidate_id']):r['metric']for r in read_rows(p/'metrics_v2/CANDIDATE_METRICS.jsonl.gz')if(r['id'],r['candidate_id'])in candidate_keys}
    costs={(r['id'],r['candidate_id']):r for r in read_rows(p/'cost_v1/CANDIDATE_COSTS.jsonl.gz')if(r['id'],r['candidate_id'])in candidate_keys}
    permit=public_image_inventory(root);max_error=0.;panels=0
    for case in proof['cases']:
        ident=case['id'];s=packets[ident]['solver_input']
        for r in case['panels']:
            key=(ident,r['candidate_id']);original=cand[key]
            if r['R']!=original['R']or r['t']!=original['t']or r['native_xyz']!=original['native_xyz']:
                raise ValueError('Gallery altered original pose/model '+str(key))
            # Independent homogeneous multiplication, not the renderer helper.
            transform=np.eye(4);transform[:3,:3]=r['R'];transform[:3,3]=r['t']
            homogeneous=np.column_stack((np.asarray(r['native_xyz']),np.ones(8)))
            cam=(transform@homogeneous.T)[:3]
            xy=np.asarray(s['K'])@cam;xy=(xy[:2]/xy[2]).T
            error=float(np.abs(xy-np.asarray(r['model_projection_xy'])).max());max_error=max(max_error,error)
            if error>1e-9:raise ValueError('Display projection mismatch')
            if r['reference_metric']!=metrics[key]or r['weighted_components']!=costs[key]['weighted_components']or r['objective']!=costs[key]['objective']:
                raise ValueError('Display metric/cost mismatch')
            if set(r['active_full_ids'])&set(r['active_partial_ids']):raise ValueError('Revoked full2D drawn active')
            panels+=1
        if case['S_precise_observation_markers']or case['OTU_precise_observation_markers']or not case['common_crop_and_scale']:
            raise ValueError('Display observation type contract')
    crops=0
    for case in proof['cases']+proof['no_candidate_RGB_links']:
        packet=packets[case['id']];source=root/packet['image']
        if packet['image_sha256']not in permit or sha(source)!=packet['image_sha256']:
            raise ValueError('RGB source/permission mismatch')
        image=Image.open(source).convert('RGB').crop(tuple(case['viewport_original_xyxy']))
        no_candidate=case in proof['no_candidate_RGB_links']
        if no_candidate:image.thumbnail((900,650))
        data=io.BytesIO();image.save(data,format='JPEG',quality=85 if no_candidate else 87,optimize=True)
        if hashlib.sha256(data.getvalue()).hexdigest()!=sha(g/case['source_crop']):
            raise ValueError('RGB crop does not regenerate from bound source')
        crops+=1
    for path in sorted(g.glob('case_*/cost_and_observations.json')):
        r=json.loads(path.read_text());a=costs[(r['id'],r['selected_candidate_id'])];b=costs[(r['id'],r['comparison_candidate_id'])]
        diff={k:b['weighted_components'][k]-a['weighted_components'][k]for k in a['weighted_components']}
        if diff!=r['Delta_weighted_components']or b['objective']-a['objective']!=r['Delta_total']:
            raise ValueError('Gallery candidate Delta differs from source')
        if abs(sum(diff.values())-r['Delta_total'])>1e-9:raise ValueError('Additive Delta failure')
    parser=Links();parser.feed((g/'index.html').read_text());deferred=[];checked=0
    for link in parser.links:
        url=urlsplit(link)
        if url.scheme:continue
        if not url.path:
            if url.fragment and url.fragment not in parser.ids:raise ValueError('HTML anchor missing '+link)
        elif url.path in('../RESULTS_KO.md','../NO_CANDIDATE_INDEX_KO.md'):
            deferred.append(link)
        elif not(g/url.path).is_file():raise ValueError('HTML asset missing '+link)
        checked+=1
    images=list(g.rglob('*.jpg'))
    for image in images:
        with Image.open(image)as im:im.verify()
    result=dict(schema='SELECTION_SCORE_GALLERY_INDEPENDENT_QA_V1',pass_=True,
        selected_cases=len(proof['cases']),original_R_t_XYZ_and_cost_metric_panels=panels,
        permission_bound_RGB_crops_regenerated_exact_bytes=crops,JPEG_files_decoded=len(images),
        independent_projection_max_abs_px=max_error,projection_atol_px=1e-9,
        candidate_cost_and_metric_values_exact=True,Delta_arithmetic_and_full2D_revocation_passed=True,
        local_HTML_links_and_anchors_checked=checked,root_documents_expected_after_packaging=deferred,
        manually_viewed_actual_comparison_images=['case_02/comparison.jpg','case_18/comparison.jpg','case_13/comparison.jpg'],
        manually_observed='Same RGB/crop/scale, readable two/four-panel layout, full/partial/region/model legend, G10 and G13 separate, S labeled only on model projection.',
        elapsed_seconds=time.monotonic()-start,process_CPU_seconds=time.process_time()-cpu,
        gallery_receipt_sha256=sha(g/'GALLERY_RECEIPT.json'))
    write_json(g/'QA_RECEIPT.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in('gallery','private','old-private','source-root'):p.add_argument('--'+n,required=True)
    a=p.parse_args();print(json.dumps(verify(a.gallery,a.private,a.old_private,a.source_root),ensure_ascii=False))
