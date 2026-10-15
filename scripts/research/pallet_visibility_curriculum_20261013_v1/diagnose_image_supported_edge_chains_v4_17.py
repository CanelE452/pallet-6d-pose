"""Small frozen-case actual-image chain diagnostic; no pose or GT reads."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import cv2
import numpy as np

from . import diagnose_edge_response_censoring_v4_17 as C
from . import image_supported_edge_chains_v4_17 as P

OUTPUT = C.OUTPUT.parent/'image_supported_edge_chains_v4_17'
CASE_IDS = ('wood_night_01:031426','wood_night_01:031344','wood_night_01:030743',
            'wood_night_01:030019','wood_night_01:030101','wood_night_01:030219')


def figure(image,old,new,path,identifier):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    for ax in axes:ax.imshow(rgb)
    positions=[]
    for ray in old['edge_records']:
        if not ray.get('supported'):continue
        xy=np.asarray(ray['observed_points']);positions.append(xy)
        axes[1].plot(xy[:,0],xy[:,1],'.-',color='#ff9d00',ms=2,lw=.7,alpha=.75)
    for ray in new['edge_records']:
        for i,chain in enumerate(ray.get('chains',[])):
            xy=np.asarray(chain['observed_points']);good=np.asarray(chain['inlier_mask'],bool)
            positions.append(xy[good]);chosen=i==ray['selected_chain_index']
            axes[1].scatter(xy[good,0],xy[good,1],s=7 if chosen else 4,
                            c='#00d4ff' if chosen else '#cbfbff',alpha=.9)
            if chosen:
                center=xy[good].mean(axis=0)
                axes[1].annotate(f'{ray["native_corner"]}-{ray["native_neighbor"]}',center,
                    xytext=(3,-8),textcoords='offset points',fontsize=8,color='#008eaa')
    if positions:
        xy=np.vstack(positions);lo=xy.min(axis=0)-12;hi=xy.max(axis=0)+12
        for ax in axes:ax.set(xlim=(max(0,lo[0]),min(image.shape[1],hi[0])),
                              ylim=(min(image.shape[0],hi[1]),max(0,lo[1])))
    axes[0].set_title(f'{identifier}: original actual RGB')
    axes[1].set_title('Orange: archived search samples; cyan: actual image NMS chains')
    for ax in axes:ax.set_aspect('equal')
    fig.suptitle('Conditional native-edge candidates only; no precise KP adoption / no pose fit',fontsize=11)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior output')
    observations=C.REMAINDER/'fresh_observations_v4_14/OBSERVATIONS.jsonl.gz'
    predictions=C.REMAINDER/'pose_evaluation_v4_14/PREDICTIONS.jsonl.gz'
    sources=[Path(__file__),Path(P.__file__),Path(C.__file__),Path(P.D.__file__),Path(P.G.__file__),Path(P.F.__file__)]
    bindings=[C.binding(path) for path in sources+[observations,predictions]]
    obs={r['id']:r for r in C.read_rows(observations) if r['id'] in CASE_IDS}
    args.output.mkdir(parents=True)
    protocol=dict(schema='IMAGE_SUPPORTED_EDGE_CHAIN_DIAGNOSTIC_PROTOCOL_V4_17',
        cases=list(CASE_IDS),source_bindings=bindings,configuration=P.DEFAULT_CONFIGURATION,
        sample_selection='Exploratory failure and successful-control examples selected from existing results.',
        production_adoption=False,parameter_fitting=False,pose_calls=0,NN_calls=0,reference_coordinate_reads=0)
    (args.output/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    rows=[]
    for row in C.read_rows(predictions):
        if row['id'] not in CASE_IDS:continue
        original=obs[row['id']];old=row['partial_global_outline']['face_observation']
        image_path=C.ROOT/row['image'];mask_path=C.REMAINDER/'fresh_observations_v4_14/masks'/original['mask_evidence']['mask_npz']
        if C.sha(image_path)!=row['image_sha256'] or C.sha(mask_path)!=original['mask_evidence']['mask_file_sha256']:
            raise ValueError('Frozen RGB/mask binding mismatch')
        image=cv2.imread(str(image_path))
        with np.load(mask_path) as data:mask=data['pallet_mask_clean'].copy()
        result=P.observe_image_supported_edge_chains(image,mask,row['partial_global_outline']['initial_geometry'],
            row['states'],old['visible_measurements'],image_binding=dict(sha256=row['image_sha256'],image_hw=list(image.shape[:2])))
        stem=row['id'].replace(':','__')
        figure(image,old,result,args.output/(stem+'.png'),row['id'])
        output=dict(id=row['id'],states=row['states'],image=C.binding(image_path),mask=C.binding(mask_path),
                    source_face_observation_image_binding=old['image_binding'],figure=stem+'.png',observation=result)
        rows.append(output)
    rows.sort(key=lambda row:CASE_IDS.index(row['id']))
    if len(rows)!=len(CASE_IDS):raise ValueError('Incomplete requested diagnostic cases')
    if bindings!=[C.binding(C.ROOT/x['path']) for x in bindings]:raise RuntimeError('Bound source/input changed')
    summary=[]
    for row in rows:
        rays=row['observation']['edge_records']
        summary.append(dict(id=row['id'],supported_edges=[dict(native_edge=r['native_edge'],
            chain_count=len(r['chains']),selected=r['selected_chain_index'],
            inlier_count=r['chains'][r['selected_chain_index']]['inlier_count'],
            RMS_px=r['chains'][r['selected_chain_index']]['inlier_rms_px'],
            angle_change_deg=r['chains'][r['selected_chain_index']]['angle_change_deg'],
            angle_sigma_deg=r['chains'][r['selected_chain_index']]['conditional_uncertainty']['angle_sigma_deg'])
            for r in rays if r.get('supported')]))
    report=dict(schema='IMAGE_SUPPORTED_EDGE_CHAIN_DIAGNOSTIC_V4_17',summary=summary,rows=rows,
        pose_calls=0,NN_calls=0,reference_coordinate_reads=0,full_corner_adoptions=0,
        production_inputs_changed=0,no_predictive_confidence_or_pose_improvement_claim=True)
    (args.output/'DIAGNOSTIC.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    artifacts=[C.binding(path) for path in sorted(args.output.iterdir()) if path.is_file()]
    (args.output/'DIAGNOSTIC_SEAL.json').write_text(json.dumps(dict(schema='IMAGE_SUPPORTED_EDGE_CHAIN_DIAGNOSTIC_SEAL_V4_17',
        source_bindings=bindings,artifacts=artifacts),indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),summary=summary),ensure_ascii=False))


if __name__=='__main__':main()
