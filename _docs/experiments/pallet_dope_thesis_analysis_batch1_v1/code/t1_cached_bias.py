"""T1(a): signed native-pixel bias from existing derived caches only.

No original GT, image, checkpoint or model module is opened. T1(b) stays skipped:
identity with the original training-time decoder is not established.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import statistics

import common as C

EXPECTED={'PLASTIC':(128,985),'WOOD':(45,346),'GREEN':(119,943)}
ROLES=('R0','control_R1','control_R2','flip_R1','flip_R2','loo_R1','loo_R2')


def require(condition,message):
    if not condition:raise RuntimeError(message)


def ensure_new(paths):
    for path in paths:
        require(not path.exists(),f'Output already exists; do not overwrite: {path}')


def finite_pair(values):
    return isinstance(values,list) and len(values)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)


def frame_signature(frame,panel='all_annotated_reference'):
    return [(p['index'],p['xy'],p['source'],p.get('verified_manual')) for p in frame[panel]]


def load_core():
    score_path=C.NOFLIP/'SCORES.json';prediction_path=C.NOFLIP/'PREDICTIONS.json';summary_path=C.NOFLIP/'SUMMARY.json'
    scores=C.read(score_path);predictions=C.read(prediction_path);summary=C.read(summary_path)
    require(scores['predictions_sha256']==C.sha(prediction_path)==summary['predictions_sha256'],'Prediction SHA chain differs')
    require(summary['scores_sha256']==C.sha(score_path),'Score SHA chain differs')
    require(scores['manifest_sha256']==predictions['manifest_sha256'],'Prediction/score manifest SHA differs')
    require(len(scores['models'])==len(predictions['predictions'])==15 and set(scores['models'])==set(predictions['predictions']),
            'Expected15 aligned derived models')
    baseline=scores['models']['R0_before_selftraining']['frames']
    ids=[r['id'] for r in baseline]
    require(len(ids)==len(set(ids))==292,'Expected unique292 population')
    for kind,(nf,nr) in EXPECTED.items():
        frames=[r for r in baseline if C.kind_of(r)==kind]
        require(len(frames)==nf and sum(len(r['all_annotated_reference']) for r in frames)==nr,'Type population differs: '+kind)
    for name,report in scores['models'].items():
        rows=report['frames'];pred=predictions['predictions'][name]
        require([r['id'] for r in rows]==ids==[r['id'] for r in pred],'Score/prediction ID order differs: '+name)
        for frame,base in zip(rows,baseline):
            require(frame_signature(frame)==frame_signature(base),'Reference point identity differs by model: '+name)
    return scores,predictions,summary


def analyze(scores,predictions,summary):
    result=[];checked_errors=0
    for kind,(nframes,nrefs) in EXPECTED.items():
        for role in ROLES:
            model=C.roles(kind)[role]
            frames=scores['models'][model]['frames'];pred=predictions['predictions'][model]
            signed_x=[];signed_y=[];radial=[];eligible=0;correct10=0;selected_frames=0
            for frame,row in zip(frames,pred):
                if C.kind_of(frame)!=kind:continue
                selected_frames+=1
                require(len(row['points'])==len(row['valid'])==len(row['confidence'])==9,'Prediction requires9 semantic slots')
                indices=[p['index'] for p in frame['all_annotated_reference']]
                require(len(indices)==len(set(indices)) and all(i in range(8) for i in indices),'Reference corner indices invalid')
                for point in frame['all_annotated_reference']:
                    eligible+=1;index=point['index'];reference=point['xy'];prediction=row['points'][index]
                    require(finite_pair(reference),'Nonfinite reference coordinate')
                    observed=bool(row['valid'][index]) and finite_pair(prediction)
                    error=point['error_px']
                    require(observed==(error is not None),'Stored error availability differs from valid finite prediction')
                    if not observed:continue
                    dx=prediction[0]-reference[0];dy=prediction[1]-reference[1];distance=math.hypot(dx,dy)
                    require(math.isfinite(error) and abs(distance-error)<=1e-9,'Reconstructed native error differs from cached error')
                    checked_errors+=1;signed_x.append(dx);signed_y.append(dy);radial.append(distance)
                    correct10+=int(error<=10)
            require(selected_frames==nframes and eligible==nrefs,'Full reference population changed')
            if kind=='PLASTIC' and role=='R0':
                require(correct10==258 and eligible==985 and correct10/eligible==summary['primary']['PLASTIC']['models']['R0']['PCK10'],
                        'STOP: mandatory PLASTIC R0 258/985 baseline reproduction failed')
            label={'R0':'R0','control_R1':'sharedcontrolR1','control_R2':'sharedcontrolR2','flip_R1':'priorLOO_flip_R1',
                   'flip_R2':'priorLOO_flip_R2','loo_R1':'LOO_only_R1','loo_R2':'LOO_only_R2'}[role]
            value=summary['primary'][kind]['models'][label]
            require(len(radial)==value['n_observed_reference_predictions'] and eligible-len(radial)==value['n_missing_reference_predictions'],
                    'Observation denominator differs from summary')
            median=statistics.median(radial) if radial else None
            require(median is None and value['conditional_median_px'] is None or median is not None and abs(median-value['conditional_median_px'])<=1e-9,
                    'Radial conditional median differs from existing summary')
            result.append(dict(kind=kind,role=role,model=model,n_frames=nframes,n_reference_points=eligible,n_observed=len(radial),
                n_missing=eligible-len(radial),mean_dx_px=statistics.fmean(signed_x) if signed_x else None,
                mean_dy_px=statistics.fmean(signed_y) if signed_y else None,median_euclidean_error_px=median,
                median_abs_dx_px=statistics.median(map(abs,signed_x)) if signed_x else None,
                median_abs_dy_px=statistics.median(map(abs,signed_y)) if signed_y else None,
                cached_PCK10_correct=correct10,cached_PCK10_denominator=eligible,
                population='DEV292 same-type diagonal; all annotated-reference sources'))
    require(len(result)==21,'Expected7roles×3kinds')
    return result,checked_errors


def main():
    csv=C.OUT/'tables/T1_NATIVE_BIAS.csv';md=C.OUT/'tables/T1_NATIVE_BIAS.md';cache=C.OUT/'cache/T1_NATIVE_BIAS.json';skip=C.OUT/'tables/T1B_SKIPPED.md'
    ensure_new((csv,md,cache,skip))
    inputs=[C.NOFLIP/name for name in ('SCORES.json','PREDICTIONS.json','SUMMARY.json')]
    C.start('T1a',inputs)
    scores,predictions,summary=load_core();rows,checked=analyze(scores,predictions,summary)
    C.csvwrite(csv,rows)
    def fmt(v):return 'NA' if v is None else f'{v:.6f}'
    text='# T1(a) 기존 예측의 원본 좌표 부호 오차\n\n'
    text+='모든 평균과 중앙값은 관측된 기준점에서 계산했다. dx=예측x−기준x, dy=예측y−기준y이며 원본 영상 픽셀 단위다. 누락점은 좌표 통계에 포함하지 않고 개수를 함께 기록한다. 기준점 목록의 위치가 아니라 semantic index로 예측9슬롯을 연결했다.\n\n'
    text+=C.mdtable(['종류','모델','전체점','관측점','누락점','평균 dx','평균 dy','중앙값 Euclidean','중앙값 |dx|','중앙값 |dy|'],
        [[r['kind'],r['role'],r['n_reference_points'],r['n_observed'],r['n_missing'],fmt(r['mean_dx_px']),fmt(r['mean_dy_px']),fmt(r['median_euclidean_error_px']),fmt(r['median_abs_dx_px']),fmt(r['median_abs_dy_px'])] for r in rows])
    text+='\n기존 error_px와 재구성한 거리의 차이는 모든 관측점에서 ≤1e−9px여야 한다. PLASTIC R0의258/985와 기존 SUMMARY의 PCK10은 정확히 같아야 한다. 단일 seed 및 반복 DEV이며 출처 미상/PnP 생성 기준점이 포함된다.\n'
    md.write_text(text)
    skip.write_text('# T1(b) 미실행\n\n학습 당시 원래 decoder와 정확히 같은 경로라는 근거가 확정되지 않아 root 사전 고정 결정에 따라 건너뛴다. 유사 경로를 새로 만들거나 R0를 재추론하지 않았다.\n')
    C.write(cache,dict(schema='T1a_cached_native_bias_v1',complete=True,rows=rows,error_coordinates_checked=checked,max_allowed_error_difference_px=1e-9,
        baseline_gate='PLASTIC R0 exactly258/985 and same SUMMARY fraction',T1b='SKIPPED_ORIGINAL_DECODER_IDENTITY_NOT_ESTABLISHED',
        raw_GT_or_image_or_weights_opened=False,new_forward_count=0))
    C.log('step_complete',task='T1a',rows=len(rows),error_coordinate_assertions=checked,T1b='skipped',new_forward_count=0,
          outputs={str(p):C.sha(p) for p in (csv,md,cache,skip)})


if __name__=='__main__':main()
