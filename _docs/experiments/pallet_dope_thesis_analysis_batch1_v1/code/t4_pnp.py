"""T4 reference-free PnP availability from all15 existing core prediction caches.

No image/annotation reads, model construction, neural forwards, or training.
Main rows are the21 own-kind/R0/shared-control comparisons;24 cross-kind
adaptation rows are auxiliary. PnP is an availability test, not pose accuracy.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from common import OUT,FLIP,NOFLIP,KINDS,read,write,sha,start,log,csvwrite,mdtable,roles
from t3_adapter import require,clean,execution_gates,geometry_modules,load_geometry_inputs,assert_bindings

def checkpoint_metadata():
    base=read(FLIP/'GREEN/PROTOCOL.json')['base_checkpoint']
    result={'R0_before_selftraining':base['sha256']}
    for r in (1,2):
        receipt=read(FLIP/'SYNTHETIC_CONTROL'/f'synthetic_control_R{r}_TRAIN_COMPLETE.json')
        require(receipt['complete'] is True and receipt['round']==r and receipt['kind']=='SYNTHETIC_CONTROL','Control receipt differs')
        result[f'SYNTHETIC_CONTROL_R{r}']=receipt['checkpoint_sha256']
    for kind in KINDS:
        for folder,tag,stem in ((FLIP,'flip','corrected_LOO_flip'),(NOFLIP,'only','corrected_LOO_only')):
            for r in (1,2):
                receipt=read(folder/kind/f'{stem}_R{r}_TRAIN_COMPLETE.json')
                require(receipt['complete'] is True and receipt['round']==r and receipt['kind']==kind,'Adaptation receipt differs')
                require(Path(receipt['checkpoint_path'])==folder/kind/f'{stem}_R{r}.pth','Adaptation checkpoint mapping differs')
                result[f'{kind}_LOO_{tag}_R{r}']=receipt['checkpoint_sha256']
    return result

def run():
    gates=execution_gates();items,_=load_geometry_inputs();modules=geometry_modules()
    G,config,_=modules['main']
    source=NOFLIP/'PREDICTIONS.json'
    start('T4',[source,OUT/'cache/DEV_GEOMETRY_INPUTS.json'])
    source_sha=sha(source);payload=read(source);checkpoint_shas=checkpoint_metadata()
    require(payload.get('complete') is True,'Existing prediction aggregate is incomplete')
    require(payload['manifest_sha256']==sha(FLIP/'MANIFEST.json'),'Cached prediction manifest differs')
    predictions=payload['predictions']
    expected_names={'R0_before_selftraining','SYNTHETIC_CONTROL_R1','SYNTHETIC_CONTROL_R2'}
    expected_names|={f'{k}_LOO_{tag}_R{r}' for k in KINDS for tag in ('flip','only') for r in (1,2)}
    require(set(predictions)==expected_names and len(predictions)==15,'Expected the15 frozen core models')
    expected_ids=[item['frame']['id'] for item in items]
    frame_rows=[]
    for model,records in predictions.items():
        require([r['id'] for r in records]==expected_ids,'Cached prediction population/order differs')
        for prediction,item in zip(records,items):
            frame=item['frame'];kind=item['kind']
            require(prediction['image_sha256']==frame['image_sha256'],'Cached prediction image differs')
            require(prediction['checkpoint_sha256']==checkpoint_shas[model],'Cached prediction checkpoint mapping differs')
            p=np.asarray(prediction['points'],float);rawvalid=np.asarray(prediction['valid'],bool)
            require(p.shape==(9,2) and rawvalid.shape==(9,),'Cached predictions require nine semantic channels')
            valid=rawvalid[:8] & np.isfinite(p[:8]).all(1)
            bounded=valid & (p[:8,0]>=0)&(p[:8,0]<frame['width'])&(p[:8,1]>=0)&(p[:8,1]<frame['height'])
            n=int(valid.sum());fits=[]
            if n>=6:
                for name,xyz in item['hypotheses']:
                    # _fit_pose uses only these detected indices in EPnP and LM,
                    # then checks positive depth for all8 fixed cuboid corners.
                    fit=G._fit_pose(p[:8],np.flatnonzero(valid),item['K'],xyz,config.min_depth)
                    fits.append(dict(name=name,positive_depth_PnP=fit is not None))
            success=any(fit['positive_depth_PnP'] for fit in fits)
            scope='primary_own_kind_or_shared' if model in set(roles(kind).values()) else 'auxiliary_cross_kind'
            frame_rows.append(dict(model=model,kind=kind,id=frame['id'],scope=scope,
                n_detected_corners=n,detected_ge6=n>=6,n_native_inframe_corners=int(bounded.sum()),
                native_inframe_ge6=int(bounded.sum())>=6,PnP_possible=success,
                successful_hypotheses=[fit['name'] for fit in fits if fit['positive_depth_PnP']]))
    output=[]
    for model in predictions:
        for kind in KINDS:
            rows=[r for r in frame_rows if r['model']==model and r['kind']==kind]
            require(len(rows)=={'PLASTIC':128,'WOOD':45,'GREEN':119}[kind],'T4 type denominator differs')
            output.append(dict(model=model,kind=kind,scope=rows[0]['scope'],n_frames=len(rows),
                frames_detected_ge6=sum(r['detected_ge6'] for r in rows),
                frames_native_inframe_ge6=sum(r['native_inframe_ge6'] for r in rows),
                frames_PnP_possible=sum(r['PnP_possible'] for r in rows),
                detected_ge6_rate=sum(r['detected_ge6'] for r in rows)/len(rows),
                PnP_possible_rate=sum(r['PnP_possible'] for r in rows)/len(rows)))
    require(Counter(r['scope'] for r in output)==Counter(primary_own_kind_or_shared=21,auxiliary_cross_kind=24),'T4 own/cross model mapping differs')
    assert_bindings(gates)
    require(sha(source)==source_sha,'Existing prediction aggregate changed during T4')
    csvwrite(OUT/'tables/T4_PNP_AVAILABILITY.csv',output)
    csvwrite(OUT/'tables/T4_PNP_AVAILABILITY_FRAMES.csv',
        [{**r,'successful_hypotheses':';'.join(r['successful_hypotheses'])} for r in frame_rows])
    contract=dict(detection='Cached canonical decoder valid AND finite, corners0..7. No new confidence threshold or bound gate on the primary rate.',
        native_bounds_diagnostic='Additional count only; does not change primary PnP inputs.',
        pose_test='At least6 detected corners; unchanged original geometry._fit_pose: EPnP then LM on identical detected indices; all8 projected cuboid depths >1e-6.',
        registry='Every root-cached fixed original registry hypothesis; availability iff any yields a finite positive-depth pose. No accuracy/ranking/GT selection.',
        denominator='All128 PLASTIC/45 WOOD/119 GREEN DEV frames, including missing predictions.',
        limitation='PnP numerical feasibility is not independent6D pose accuracy.',
        references_used=False,neural_forwards=0,images_opened=False,annotations_opened=False)
    write(OUT/'tables/T4_PNP_AVAILABILITY.json',clean(dict(complete=True,rows=output,frame_records=frame_rows,
        contract=contract,source=dict(path=str(source),sha256=source_sha),gates=gates,
        geometry_source=dict(path=str(FLIP/'geometry.py'),sha256=sha(FLIP/'geometry.py')),
        code_sha256=sha(Path(__file__)),primary_rows=21,auxiliary_rows=24,neural_forwards=0)))
    headers=['kind','model','n_frames','frames_detected_ge6','frames_PnP_possible','PnP_possible_rate']
    note=('T4 PnP availability uses frozen cached predictions and native K/registered cuboids only. '
          'EPnP and LM use identical detected indices; all eight depths must be positive. '
          'No RANSAC/LOO/flip screening, reference coordinates, or neural forwards. '
          'The rate measures numerical availability, not pose accuracy.\n\n')
    text=note+'Primary own-kind and shared comparisons\n\n'+mdtable(headers,[[r[k] for k in headers] for r in output if r['scope']=='primary_own_kind_or_shared'])
    text+='\nAuxiliary cross-kind comparisons\n\n'+mdtable(headers,[[r[k] for k in headers] for r in output if r['scope']=='auxiliary_cross_kind'])
    (OUT/'tables/T4_PNP_AVAILABILITY.md').write_text(text)
    log('step_complete',task='T4',primary_rows=21,auxiliary_rows=24,neural_forwards=0,annotations_opened=False)
    print('T4_CACHE_ONLY_COMPLETE',len(output),len(frame_rows),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()

if __name__=='__main__':main()
