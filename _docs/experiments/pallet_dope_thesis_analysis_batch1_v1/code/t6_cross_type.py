"""T6: copy every existing model-by-type score; no new prediction or GT read."""
from __future__ import annotations

import json

import common as C
from t1_cached_bias import EXPECTED,ensure_new,require

TYPE_FOR_KIND={'PLASTIC':'plastic_standard_110x130x11','WOOD':'wood_small_80x59x14','GREEN':'plastic_standard_110x110x15'}


def model_identity(name,experiment):
    if name=='R0_before_selftraining':return dict(training_kind='SHARED',method='R0',round=0,diagonal_applicable=False)
    if name.startswith('SYNTHETIC_CONTROL_R'):
        return dict(training_kind='SHARED',method='synthetic_only_control',round=int(name.rsplit('_R',1)[1]),diagonal_applicable=False)
    kind=name.split('_',1)[0];require(kind in C.KINDS,'Unknown cached model kind: '+name)
    round_idx=int(name.rsplit('_R',1)[1]);require(round_idx in (1,2),'Unexpected fixed round')
    if '_LOO_only_R' in name:method='LOO_only'
    elif '_LOO_flip_R' in name or experiment=='LOO_flip':method='LOO_flip'
    else:raise RuntimeError('Unknown model alias: '+name)
    return dict(training_kind=kind,method=method,round=round_idx,diagonal_applicable=True)


def main():
    csv=C.OUT/'tables/T6_CROSS_TYPE.csv';md=C.OUT/'tables/T6_CROSS_TYPE.md';cache=C.OUT/'cache/T6_CROSS_TYPE.json'
    ensure_new((csv,md,cache));inputs=[C.FLIP/'SCORES.json',C.NOFLIP/'SCORES.json']
    C.start('T6',inputs);rows=[]
    numeric=('n_frames','n_frames_with_reference','n_frames_with_any_observed_reference_prediction','n_missing_prediction_rows',
        'n_reference_points','n_observed_reference_predictions','n_missing_reference_predictions','PCK5','PCK10','PCK20','coverage',
        'conditional_median_px','conditional_p90_px','conditional_mean_px')
    for experiment,directory,nmodels in (('LOO_flip',C.FLIP,9),('LOO_only',C.NOFLIP,15)):
        scores=C.read(directory/'SCORES.json');require(len(scores['models'])==nmodels,'Unexpected cached model count')
        for name,report in scores['models'].items():
            identity=model_identity(name,experiment)
            for kind in C.KINDS:
                group='type/'+TYPE_FOR_KIND[kind];value=report['groups'][group]['all_annotated_reference']
                require((value['n_frames'],value['n_reference_points'])==EXPECTED[kind],'Unexpected type count/denominator')
                require(value['n_observed_reference_predictions']+value['n_missing_reference_predictions']==value['n_reference_points'],
                        'Invalid observed/missing denominator')
                row=dict(source_experiment=experiment,source_score_path=str(directory/'SCORES.json'),model=name,**identity,
                    evaluation_kind=kind,diagonal_flag=(identity['training_kind']==kind if identity['diagonal_applicable'] else None),
                    panel='all_annotated_reference',population='DEV292 type group')
                row.update({key:value[key] for key in numeric})
                row.update({metric+'_percent':None if value[metric] is None else 100*value[metric] for metric in ('PCK5','PCK10','PCK20','coverage')})
                row['reference_sources_json']=json.dumps(value['reference_sources'],ensure_ascii=False,sort_keys=True)
                row['source_JSON_pointer']='#/models/'+name+'/groups/type~1'+TYPE_FOR_KIND[kind]+'/all_annotated_reference'
                rows.append(row)
    require(len(rows)==72,'Expected(9+15)×3 existing model-type groups')
    C.csvwrite(csv,rows)
    def fmt(value):return 'NA' if value is None else f'{value:.4f}'
    text='# T6 기존 모델의 교차 종류 점수\n\n'
    text+='두 실험의 SCORES.json에 이미 존재하는 모든 모델×3종류 그룹을 그대로 옮겼다. 새 추론·영상·원본 주석·가중치 읽기는 없다. PCK/coverage 표시는 %, 거리 통계는 px다. 누락 예측은 전체 기준점 분모에서 PCK 실패다.\n\n'
    text+='diagonal은 재료별 자기학습 모델의 학습종류와 평가종류가 같은지 뜻한다. 공통 R0 및 합성 추가학습 대조군은 SHARED이며 diagonal 적용 대상이 아니므로 NA로 표시한다. 같은 캐시가 두 실험에 다시 포함되어도 독립 모델로 합산하지 않는다.\n\n'
    text+=C.mdtable(['실험','모델','학습종류','평가종류','Diagonal','영상','기준점','PCK5 %','PCK10 %','PCK20 %','coverage %','조건부 median px'],
        [[r['source_experiment'],r['model'],r['training_kind'],r['evaluation_kind'],r['diagonal_flag'],r['n_frames'],r['n_reference_points'],
          fmt(r['PCK5_percent']),fmt(r['PCK10_percent']),fmt(r['PCK20_percent']),fmt(r['coverage_percent']),fmt(r['conditional_median_px'])] for r in rows])
    text+='\nCSV에는 관측/누락 개수, 조건부 P90/평균 및 source별 점 수도 포함한다. 대칭 정렬을 하지 않은 의미 채널0–7 기준의 반복 DEV 비교이며, 출처 미상/PnP 생성 기준점이 포함되고 독립 TEST 또는 물리적6D 정확도를 뜻하지 않는다.\n'
    md.write_text(text)
    C.write(cache,dict(schema='T6_existing_cross_type_scores_v1',complete=True,rows=rows,counts=dict(LOO_flip=27,LOO_only=45),
        source_groups_copied_without_recomputing_metrics=True,original_GT_images_weights_opened=False,new_forward_count=0))
    C.log('step_complete',task='T6',rows=len(rows),new_forward_count=0,outputs={str(p):C.sha(p) for p in (csv,md,cache)})


if __name__=='__main__':main()
