"""Six stored-image topology/junction diagnostics; no GT/NN/line/pose refit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from . import diagnose_edge_response_censoring_v4_17 as C
from . import native_edge_topology_junctions_v4_17 as P

CHAIN_INPUT=C.OUTPUT.parent/'image_supported_edge_chains_v4_17b'
OUTPUT=C.OUTPUT.parent/'native_edge_topology_junctions_v4_17'


def figure(image,chain,proposal,path,identifier):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    for ax in axes:ax.imshow(rgb)
    positions=[]
    for ray in chain['edge_records']:
        if not ray.get('supported'):continue
        for actual in ray['chains']:
            observed=np.asarray(actual['observed_points']);good=np.asarray(actual['inlier_mask'],bool)
            xy=observed[good];positions.append(xy)
            outside=actual['mask_context_kind']=='TARGET_BACKGROUND_OUTLINE'
            color='#00bfff' if outside else '#cc55cc'
            axes[1].scatter(xy[:,0],xy[:,1],s=5,c=color,alpha=.55)
    count=0
    for record in proposal['records']:
        points=np.asarray([x['intersection_xy'] for x in record['candidates']])
        if len(points):
            positions.append(points);count+=len(points)
            axes[1].scatter(points[:,0],points[:,1],marker='x',s=42,c='#00d95f',lw=1.2)
            center=points.mean(axis=0)
            axes[1].annotate(f'B{record["native_corner"]}: {len(points)} candidates',center,
                xytext=(4,-12),textcoords='offset points',fontsize=8,color='#006633')
    if positions:
        xy=np.vstack(positions);lo=xy.min(axis=0)-12;hi=xy.max(axis=0)+12
        for ax in axes:ax.set(xlim=(max(0,lo[0]),min(image.shape[1],hi[0])),
                              ylim=(min(image.shape[0],hi[1]),max(0,lo[1])))
    axes[0].set_title(f'{identifier}: original actual RGB')
    axes[1].set_title(f'Actual chains and {count} conditional B junction candidates; full KP adoption=0')
    axes[1].legend(handles=[Line2D([],[],marker='.',ls='',color='#00bfff',label='External TARGET/BACKGROUND chains'),
                           Line2D([],[],marker='.',ls='',color='#cc55cc',label='Internal TARGET/TARGET chains (separate)'),
                           Line2D([],[],marker='x',ls='',color='#00aa55',label='Finite-support + actual-mask topology candidate')],fontsize=8)
    for ax in axes:ax.set_aspect('equal')
    fig.suptitle('All admissible alternatives retained; no reference points, coordinate replacement or pose fit',fontsize=11)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve all earlier outputs')
    inputs=CHAIN_INPUT/'DIAGNOSTIC.json';seal=CHAIN_INPUT/'DIAGNOSTIC_SEAL.json'
    prior=json.loads(inputs.read_text());bindings=json.loads(seal.read_text())
    for item in bindings['source_bindings']+bindings['artifacts']:
        if C.sha(C.ROOT/item['path'])!=item['sha256']:raise ValueError('Frozen actual-chain source/artifact changed')
    ids=[r['id'] for r in prior['rows']]
    predictions=C.REMAINDER/'pose_evaluation_v4_14/PREDICTIONS.jsonl.gz'
    sources=[Path(__file__),Path(P.__file__),Path(P.CHAINS.__file__),Path(P.D.__file__),Path(P.G.__file__)]
    source_bindings=[C.binding(p) for p in sources]
    args.output.mkdir(parents=True)
    protocol=dict(schema='NATIVE_EDGE_TOPOLOGY_DIAGNOSTIC_PROTOCOL_V4_17',ordered_ids=ids,
        sources=source_bindings,inputs=[C.binding(p) for p in [inputs,seal,predictions]],
        recipe='B: two distinct external chains + old sine.25/support11/shift12/canvas + same actual exterior contour arc and foreground wedge. I stays separate.',
        line_refits=0,pose_calls=0,NN_calls=0,reference_coordinate_reads=0,full_point_adoptions=0,
        conditional_bootstrap_or_total_error_calibration_run=False)
    (args.output/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    needed={r['id']:r for r in prior['rows']};rows=[]
    for runtime in C.read_rows(predictions):
        if runtime['id'] not in needed:continue
        source=needed[runtime['id']];chain=source['observation']
        image_path=C.ROOT/source['image']['path'];mask_path=C.ROOT/source['mask']['path']
        if C.sha(image_path)!=source['image']['sha256'] or C.sha(mask_path)!=source['mask']['sha256']:
            raise ValueError('Bound original actual image/mask changed')
        with np.load(mask_path) as data:mask=data['pallet_mask_clean'].copy()
        geometry=runtime['partial_global_outline']['initial_geometry']
        original=runtime['partial_global_outline']['face_observation']['visible_measurements']
        result=P.propose_native_edge_junctions(chain,mask,geometry,runtime['states'],original,
            image_binding=chain['image_binding'],mask_binding=dict(image_sha256=source['image']['sha256'],
                mask_file_binding=source['mask'],mask_pixel_sha256=P.mask_pixel_sha256(mask)))
        stem=runtime['id'].replace(':','__')
        image=cv2.imread(str(image_path));figure(image,chain,result,args.output/(stem+'.png'),runtime['id'])
        rows.append(dict(id=runtime['id'],states=runtime['states'],image_binding=source['image'],mask_binding=source['mask'],
            figure=stem+'.png',proposal=result))
    rows.sort(key=lambda r:ids.index(r['id']))
    if len(rows)!=len(ids):raise ValueError('Incomplete actual-case inventory')
    summary=[]
    for row in rows:
        summary.append(dict(id=row['id'],B_junctions=[dict(native_corner=x['native_corner'],
            external_chains=len(x.get('external_chain_refs',[])),candidates=len(x['candidates']),
            disagreement_px=x.get('candidate_disagreement_px'),unique_3px=x.get('geometrically_unique_within_existing_3px_rule'),
            rejected_pair_reasons={reason:sum(y['reason']==reason for y in x.get('rejected_pairs',[]))
                for reason in sorted({y['reason'] for y in x.get('rejected_pairs',[])})})
            for x in row['proposal']['records'] if x['state']=='B'],
            total_candidates=sum(len(x['candidates']) for x in row['proposal']['records'])))
    report=dict(schema='NATIVE_EDGE_TOPOLOGY_DIAGNOSTIC_V4_17',summary=summary,rows=rows,
        no_full_point_adoption=True,no_pose_or_NN_run=True,no_reference_coordinate_reads=True,
        missing_proposal_does_not_prove_physical_edge_absence=True,
        conditional_mask_topology_is_not_native_identity_or_precision_certificate=True)
    (args.output/'DIAGNOSTIC.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for value in source_bindings+protocol['inputs']:
        if C.sha(C.ROOT/value['path'])!=value['sha256']:raise ValueError('Bound source/input changed during diagnostic')
    artifacts=[C.binding(path) for path in sorted(args.output.iterdir()) if path.is_file()]
    (args.output/'DIAGNOSTIC_SEAL.json').write_text(json.dumps(dict(schema='NATIVE_EDGE_TOPOLOGY_DIAGNOSTIC_SEAL_V4_17',
        source_bindings=source_bindings,artifacts=artifacts),indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),summary=summary),ensure_ascii=False))


if __name__=='__main__':main()
