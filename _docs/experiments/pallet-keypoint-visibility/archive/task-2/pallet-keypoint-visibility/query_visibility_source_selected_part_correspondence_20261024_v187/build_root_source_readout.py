"""Publishable source correspondence readout, from sealed metadata only."""
from pathlib import Path
import hashlib
import html
import json

V = Path(__file__).absolute().parent


def bind(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(relative):
    assert '..' not in Path(relative).parts and 'v112' not in relative.casefold()
    return json.loads((V / relative).read_text())


def write(relative, value):
    with (V / relative).open('x') as f:
        f.write(value)


def main():
    summary = read('descriptive_source_tables_v1/SUMMARY.json')
    receipt_paths = {
        'selected_instance': 'selected_instance_metadata_diagnostic_v2_nested_R/COMPLETE.json',
        'independent_instance': 'independent_instance_review_b/actual_review_v1/REVIEW.json',
        'nominal_projection': 'nominal_source_native_projection_v1/COMPLETE.json',
        'independent_nominal': 'independent_nominal_review_b/actual_review_v1/REVIEW.json',
        'descriptive_tables': 'descriptive_source_tables_v1/COMPLETE.json',
        'independent_tables': 'source_table_review_b/REVIEW.json',
        'source_visuals': 'fixed_source_nominal_visual_v1/COMPLETE.json',
        'root_visual_QA': 'ROOT_ACTUAL_FIXED_SOURCE_VISUAL_QA.json',
        'whole_image_visual_QA': 'actual_visual_review_a/REVIEW.json',
        'source_coverage_figure': 'source_coverage_figure_v1/COMPLETE.json',
    }
    receipts = {k: read(path) for k, path in receipt_paths.items()}
    assert all(r['completed'] and r['PASS'] for r in receipts.values())
    coverage = receipts['independent_nominal']['nominal_crop_coverage_diagnostic']
    assert coverage == {'original_available_weakobj1_valid_nominal': 924,
                        'inside_original_crop_extent': 650, 'outside_original_crop_extent': 274}
    observations = read('ROOT_ACTUAL_FIXED_SOURCE_VISUAL_QA.json')['observations_KO']
    result = {
        'completed': True, 'PASS': True, 'phase': 'V187', 'phase_kind': 'SOURCE_CORRESPONDENCE_DIAGNOSTIC',
        'phase_status': 'PROGRESS', 'full_goal_solved': False, 'selected_model': None,
        'new_training_calls': 0, 'new_initializer_forwards': 0, 'PnP_or_pose_optimization_calls': 0,
        'new_source_semantic_labels': 0, 'fit_admission': False,
        'new_current223_or_publicheld45_or_syntheticVAL_body_reads': 0,
        'fixed_inputs': 'Original selected detector candidate, first8 native channels, q/L/U and V90 sampler unchanged.',
        'all_source_population': summary,
        'independent_nominal_max_difference': receipts['independent_nominal']['maximum_numerical_differences'],
        'fixed_visual_scope': {'frames': [0, 118, 166, 199], 'all_native_crops': 32, 'inputU': 3,
                              'all_obj1_instance_observations': 5, 'selection': 'V186 predeclared first frame per TRAIN sequence'},
        'visual_observations_not_annotations': observations,
        'authority': {'association': 'Unique positive maximum bbox IoU over ALL supplied objects; weak identity diagnostic only.',
                      'references': 'Original K/R/t + fixed nominal cuboid + unchanged camera-facing native order; not physical material endpoint gold.',
                      'all_new_native_D_E_S_owner_quality_support_labels': 'UNKNOWN; no authority promotion.',
                      'human_semantic_labels': 'Existing human native semantic D/E/S authority remains valid; exact mesh vertex/depth certification is not required for those labels.',
                      'direct_RGB_weak_semantic_witness': 'Allowed with explicit weak authority; not generated or admitted in this phase.'},
        'preserved_failures': ['V1 flat9-only R guard failed on valid nested3x3 original R; V2 validates notation only.',
                               'Root relative-entry invocation rejected before any body read; corrected absolute invocation succeeded.',
                               'Root audit-entry template filename typo failed before diagnostic reads; corrected exact name.',
                               'Invented fixture floating equality failures retained in agent provenance; no geometry/IoU math change.'],
        'next_source_work': 'Bind intended queried-end from real TRAIN RGB with explicit weak authority; construct local end-only COVER/SHAM pairs with unchanged q/L and visible surrounding part, preserving unresolved rows. Broad patch V>0 cannot stand in for native D.',
        'remaining_scope': 'Full WOOD/SQUARE and native0..7 objective remains. This MR185 source diagnostic neither supplies full native D/E/S labels nor establishes transfer or model improvement. Latest model comparison remains V184; latest fit remains V183.',
        'source_receipts': {k: bind(V / path) for k, path in receipt_paths.items()},
        'report_builder': bind(Path(__file__).absolute()),
    }
    write('ROOT_RESULTS_KO.json', json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    scene_header = '|SOURCE 장면|전체 프레임|최대 IoU 대상 obj1|다른 obj|대응 UNKNOWN|\n|---|---:|---:|---:|---:|'
    scene_rows = '\n'.join(f"|{r['scene']}|{r['frames']}|{r['selected_obj1']}|{r['selected_other']}|{r['selected_UNKNOWN']}|" for r in summary['per_scene'])
    native_header = '|native|전체|입력 가능|원 inputU|명목점 crop 안|명목점 crop 밖|입력 가능·obj1 대응 UNKNOWN|\n|---|---:|---:|---:|---:|---:|---:|'
    native_rows = '\n'.join(f"|{k}|{r['total']}|{r['input_available']}|{r['inputU']}|{r['eligible_inside_original_crop_extent']}|{r['eligible_outside_original_crop_extent']}|{r['input_available']-r['eligible_available_weak_nominal']}|" for k, r in summary['per_native'].items())
    intro = ('공개 TRAIN 자료에서 원래 검출 후보를 모든 GT 물체와 대조하고, 고정된 native 순서의 명목 reference를 원래 query crop에 연결했다. '
             '185프레임·1,480 query·194 inputU를 유지했다. 새 학습·키포인트 재추론·PnP·가림 라벨 생성은 0이다. '
             '이번 단계는 감독 신호를 잘못 연결하지 않기 위한 진단이며 모델 성능 개선 결과가 아니다.')
    method = ('검출 후보 번호는 GT instance 번호가 아니다. 원 검출 box와 모든 GT bbox_obj의 IoU를 계산해 양수 최대값이 유일한 경우만 약한 대응으로 남겼다. '
              '177장은 유일한 최대값(팔레트 obj1 137장·다른 물체 40장), 8장은 겹침이 없었다. '
              '다른 물체의 승자를 팔레트로 강제 연결하거나, query와 가까운 GT를 다시 고르지 않았다. '
              '이 수치는 identity 정확도가 아니다.')
    meaning = ('팔레트와 약하게 대응되고 원 입력이 가능한 924 query 중 명목 reference가 원 crop 안에 650개, 밖에 274개였다. '
               '나머지는 입력 가능·obj1 대응 UNKNOWN 362개와 원 inputU 194개이며 모두 표에 남겼다. '
               '명목 cuboid·제공 pose·카메라 기준 역할과 실제 물질 끝점은 다를 수 있으므로 274개를 실제 좌표 오류나 가림 정답으로 해석할 수 없다.')
    verification = ('독립 scalar IoU 검산이 185프레임·823 GT observation·1,480행 전체에서 일치했다. '
                    '별도 scalar 투영·역행렬·native 순서·crop grid 검산도 258×8=2,064 reference와 1,480 lookup 전체를 확인했다. '
                    '최대 좌표 수치 차이는 2.7285e-12이며, 원 q/L/native/U와 제공 K/R/t는 정확히 보존됐다. '
                    '고정 4프레임의 전체 32 crop에서 원 RGB/valid 해시·저장 PNG 동일성을 확인했다. '
                    'constructed crop은 원 sampler를 재실행했고, unconstructed slot은 원 zero placeholder를 유지했다. '
                    'Root는 4 contact sheet·3 U full card·2 overview를 실제 봤고, A는 4 원본·4 overview를 실제 봤다. HTML browser rendering은 검증하지 않았다.')
    limits = ('새 D/E/S·owner·quality·support 정답은 모두 UNKNOWN이며 학습 admission은 없다. '
              '주변에 팔레트 면이 보인다는 사실만으로 해당 native 끝점이 보인다고 라벨링하지 않는다. '
              '기존 사람의 semantic D/E/S 라벨을 물리 vertex 인증 부족으로 무효화하지 않는다. '
              '현 평가 223장·공개 held45·합성 VAL의 새 본문은 읽지 않았고, 이번 단계에서 선택·보정하지 않았다. '
              '전체 WOOD/SQUARE·native0..7 문제와 실제 전이는 아직 해결되지 않았다.')
    md = ['# V187 — 공개 TRAIN의 선택 물체·native 부위·원 query crop 대응', intro,
          '## 무엇을 계산했나', method, scene_header+'\n'+scene_rows,
          '## 전체 query와 원 crop 범위', meaning,
          '![전체 source native8 분모](source_coverage_figure_v1/PUBLIC185_NATIVE8_CROP_COVERAGE.png)',
          native_header+'\n'+native_rows,
          '[장면별 CSV](descriptive_source_tables_v1/PER_SCENE_INSTANCE_MATCH.csv) · [native별 CSV](descriptive_source_tables_v1/PER_NATIVE_ORIGINAL_CROP_COVERAGE.csv) · [전체 집계 JSON](descriptive_source_tables_v1/SUMMARY.json)',
          '## 원본 영상과 전체32 crop', 'V186에서 사전에 고정한 각 TRAIN 장면의 첫 프레임 4개다. 전체 source의 성공·실패 비율을 대표하도록 뽑은 샘플이 아니다. 청록은 q, 주황은 선택된 obj1의 명목 reference이며 원본과 표시 사본을 구분한다.',
          '[전체32 개별 crop 갤러리](fixed_source_nominal_visual_v1/PORTABLE_GALLERY_KO.md) · [offline HTML](fixed_source_nominal_visual_v1/index.html)']
    for obs in observations:
        fi = obs['frame']
        md += [f'### FRAME {fi:03d}', obs['observation'],
               f'![전체 장면과 명목 대응](fixed_source_nominal_visual_v1/FRAME_{fi:03d}/OVERVIEW.png)',
               f'![native0..7 원 crop](fixed_source_nominal_visual_v1/FRAME_{fi:03d}/ALL8_CONTACT.png)']
    md += ['## 검산과 한계', verification, limits, '## 코드·검증·실패 원문']
    md += [f'- [{k}]({path})' for k, path in receipt_paths.items()]
    md += ['- [instance 대응 코드](selected_instance_preparation_c/associate_frozen_source_instances_v2_nested_R.py)',
           '- [명목 투영 코드](native_projection_preparation_a/project_nominal_source_native.py)',
           '- [V1 실패 보존](selected_instance_metadata_diagnostic_v1/FAILED_EXECUTION_PRESERVED.json)',
           '- [R 표기만 수정한 범위](ROOT_NESTED_R_NOTATION_ONLY_REPAIR_SCOPE.json)',
           '- [원 185/1480 대응 결과 폴더](selected_instance_metadata_diagnostic_v2_nested_R/)',
           '- [전체 258/2064/1480 명목 결과 폴더](nominal_source_native_projection_v1/)',
           '- [정의·fixture·counterfactual 준비](witness_definition_b/)',
           '- [집계 재현 코드](root_describe_source_correspondence.py)',
           '- [Root 전체 결과](ROOT_RESULTS_KO.json)',
           '## 다음 작업', result['next_source_work'], '이전 학습은 V183, 저장 모델 비교는 V184다. 이번 대응 계산으로 새 모델 채택이나 전체 목표 달성을 주장하지 않는다.']
    markdown = '\n\n'.join(md) + '\n'
    write('PORTABLE_REPORT_KO.md', markdown)
    css = 'body{font:17px/1.7 system-ui;margin:0;background:#101827;color:#e6edf5}main{max-width:1300px;margin:auto;padding:28px}a{color:#7dd3fc}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:16px}td,th{padding:8px;border:1px solid #42526b}section{margin:30px 0}h1,h2{line-height:1.4}code{word-break:break-all}.note{color:#bac8dc}'
    def table(header, rows):
        return '<table><tr>'+''.join('<th>'+html.escape(s)+'</th>' for s in header)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in rows)+'</table>'
    scene_table = table(['장면','전체','obj1','다른 obj','UNKNOWN'], [[r[k] for k in ['scene','frames','selected_obj1','selected_other','selected_UNKNOWN']] for r in summary['per_scene']])
    native_table = table(['native','전체','가능','inputU','crop 안','crop 밖','가능·obj1 UNKNOWN'], [[k,r['total'],r['input_available'],r['inputU'],r['eligible_inside_original_crop_extent'],r['eligible_outside_original_crop_extent'],r['input_available']-r['eligible_available_weak_nominal']] for k,r in summary['per_native'].items()])
    body = [f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>V187 공개 source 대응 진단</title><style>{css}</style><main><h1>V187 공개 source 대응 진단</h1>', '<p>'+html.escape(intro)+'</p>', '<p><a href="PORTABLE_REPORT_KO.md">GitHub용 보고서</a> · <a href="fixed_source_nominal_visual_v1/index.html">전체32 crop</a> · <a href="ROOT_RESULTS_KO.json">검증 결과</a></p>', '<h2>선택 물체와 원 query</h2><p>'+html.escape(method)+'</p>'+scene_table, '<h2>원 crop 범위</h2><p>'+html.escape(meaning)+'</p><img src="source_coverage_figure_v1/PUBLIC185_NATIVE8_CROP_COVERAGE.png" alt="source native8 전체 분모">'+native_table]
    for obs in observations:
        fi=obs['frame']
        body += [f'<section><h2>FRAME {fi:03d}</h2><p>{html.escape(obs["observation"])}</p><img src="fixed_source_nominal_visual_v1/FRAME_{fi:03d}/OVERVIEW.png" alt="전체 장면"><img src="fixed_source_nominal_visual_v1/FRAME_{fi:03d}/ALL8_CONTACT.png" alt="전체 native8 crop"></section>']
    body += ['<h2>검증</h2><p>'+html.escape(verification)+'</p><h2>한계와 다음 작업</h2><p>'+html.escape(limits)+'</p><p>'+html.escape(result['next_source_work'])+'</p></main></html>']
    write('ROOT_READOUT_KO.html','\n'.join(body)+'\n')
    complete = {'completed':True,'PASS':True,'builder':bind(Path(__file__).absolute()),'outputs':{n:bind(V/n) for n in ['ROOT_RESULTS_KO.json','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html']},'source_receipts':result['source_receipts'],'HTML_browser_rendered':False,'model_adoption_or_goal_completion':False}
    write('ROOT_READOUT_BUILD_COMPLETE.json',json.dumps(complete,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(complete))


if __name__ == '__main__':
    main()
