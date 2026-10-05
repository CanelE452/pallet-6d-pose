"""Publish a local readable review of already evaluated fixed PnP annotations.

Copies the twelve bound overlays and score rows into the documentation tree.
Does not run a model, change an annotation, or approve a scientific reference.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def copy_bound(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or sha(source) != sha(target):
        shutil.copyfile(source, target)
    if sha(source) != sha(target):
        raise ValueError('Documentation copy changed the source artifact')
    return dict(source=str(source), source_sha256=sha(source),
                target=str(target.relative_to(DOC)), target_sha256=sha(target))


def main():
    started = time.perf_counter()
    result_path = DOC / 'RESULT.json'
    result = read(result_path)
    if result['status'] != 'EVALUATED_EXPLORATORY_ASSISTED_GEOMETRY_12_FRAMES':
        raise ValueError('No completed evaluation to present')
    for binding in result['sources'].values():
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Result source changed; validate current inputs before reporting')
    metrics = result['metrics']
    if metrics['frame_count'] != 12 or metrics['reference_point_count'] != 96:
        raise ValueError('Fixed twelve-image geometry panel changed')
    output = Path(result['output_dir'])
    copies = [copy_bound(output / name, DOC / name)
              for name in ('POINT_ERRORS.csv', 'FRAME_RESULTS.json')]
    with (DOC / 'POINT_ERRORS.csv').open(encoding='utf-8', newline='') as handle:
        scores = list(csv.DictReader(handle))
    if len(scores) != 96 or len({(r['frame_id'], r['point_id']) for r in scores}) != 96:
        raise ValueError('Point-score evidence does not have 96 unique rows')
    images = []
    for overlay in result['overlays']:
        if sha(overlay['path']) != overlay['sha256']:
            raise ValueError('Bound evaluation overlay changed')
        target = DOC / 'images' / Path(overlay['path']).name
        copies.append(copy_bound(overlay['path'], target))
        images.append((overlay['frame_id'], str(target.relative_to(DOC))))
    if len(images) != 12:
        raise ValueError('Show all fixed images, without selecting by performance')

    receipt_path = DOC / 'PAPER_INTEGRATION_RECEIPT.json'
    integration = read(receipt_path) if receipt_path.exists() else None
    inserted = bool(integration and integration.get('status') == 'COMPLETE'
                    and integration.get('source_result_sha256') == sha(result_path))
    if inserted:
        validation = read(integration['validation_path'])
        if (validation.get('status') != 'PASS'
                or len(validation.get('cell_checks', [])) != integration['numeric_cells']
                or not all(row['pass'] for row in validation['cell_checks'])
                or not validation.get('all_12_images_hash_match')
                or sha(integration['patch_path']) != integration['patch_sha256']):
            raise ValueError('Manuscript insertion has unresolved evidence checks')
    table = ['| 지표 | Base | N3 | 변화 |', '|---|---:|---:|---|']
    base, n3 = (metrics['methods'][method] for method in ('Base', 'N3'))
    bm, nm = base['conditional_error'], n3['conditional_error']
    table += [f"| 코너 오차 중앙값(px) | {bm['median_px']:.3f} | {nm['median_px']:.3f} | {nm['median_px']-bm['median_px']:+.3f} px |",
              f"| 코너 오차 P90(px) | {bm['p90_px']:.3f} | {nm['p90_px']:.3f} | {nm['p90_px']-bm['p90_px']:+.3f} px |",
              f"| 오차 10px 이내(전체 96점) | {base['pck10_hit_count']}/96 ({base['pck10_full_reference_percent']:.3f}%) | {n3['pck10_hit_count']}/96 ({n3['pck10_full_reference_percent']:.3f}%) | {n3['pck10_full_reference_percent']-base['pck10_full_reference_percent']:+.3f} %p |",
              f"| 결측·대상 불일치 점 | {base['failed_reference_points']} | {n3['failed_reference_points']} | 전체 분모 유지 |"]
    paired = metrics['paired']
    lines = ['# 리프터 기존 PnP 주석 연결 결과', '',
             '**이번 12장 대상 확인은 완료했습니다. 점을 더 찍거나 같은 사진을 다시 확인할 필요가 없습니다.**', '',
             '기존에 직접 점을 찍고 PnP가 맞는지 확인한 뒤 G로 저장했던 8코너 전체를 참조로 사용했습니다. 네 촬영 세션에서 고정한 12장, 총 96점입니다. 사람의 실제 저장 기록은 12장 모두 같은 파렛트였으며, Base와 N3는 같은 대상과 같은 코너 결측 마스크를 유지했습니다.', '',
             *table, '',
             '결과는 혼합되어 있습니다. N3는 큰 오차 쪽의 P90과 10픽셀 이내 비율을 개선했지만 중앙 오차는 조금 커졌습니다. 전반적인 리프터 정확도 향상이 입증됐다고 쓰지 않습니다.', '',
             f"점별로는 {paired['improved_points']}점 개선, {paired['worsened_points']}점 악화이며, 프레임 중앙값은 5장 개선·7장 악화입니다. 중앙값의 차이는 {paired['median_after_minus_median_before_px']:+.6f}px, 점별 차이의 중앙값은 {paired['median_of_paired_after_minus_before_px']:+.6f}px로 서로 다른 통계입니다.", '',
             '## 원고와 근거', '',
             ('별도 원고 복사본의 LaTeX와 Markdown에 이번 표·설명을 실제 반영했고, 이전 복사본과의 patch를 만들었습니다.' if inserted else '오차 계산과 그림 생성은 완료했습니다. 별도 원고 복사본 반영은 아직 완료 기록이 없습니다.'), '',
             '- [수치와 출처 JSON](RESULT.json)',
             '- [96점별 실제 오차 CSV](POINT_ERRORS.csv)',
             '- [12장별 결과 JSON](FRAME_RESULTS.json)',
             '- [실행 상태와 실제 계산 비용](EVALUATION_STATUS.json)',
             '- [평가기 검증 기록](TEST_VALIDATION.json)', '']
    if inserted:
        lines += ['- [수정 원고 Markdown](paper_updated/manuscript_ko.md)',
                  '- [보충 자료 Markdown](paper_updated/supplement_ko.md)',
                  '- [본문 LaTeX 수정](paper_updated/sections/06_case_study.tex)',
                  '- [원고 변경 patch](paper_patch/ASSISTED_PAPER.patch)',
                  '- [원고 반영 및 원본 보존 검산](PAPER_INTEGRATION_RECEIPT.json)', '']
    lines += ['이 패널은 YOLO 고정 Base와 N3 seed 1을 비교한 소규모 기하 참조 평가입니다. 정적 319장·세 기반·세 seed 결과와 분모를 섞지 않았습니다. 저장된 원본 66개 직접 입력점과 PnP 보완 30점을 모두 그대로 사용했고, 이후 추가 수동 입력 72점 버전으로 참조를 바꾸지 않았습니다.', '',
              '원래 120장/반복 24장 계획의 가시 코너 표와 이번 96점 표는 별개입니다. 독립 실측 위치·회전 및 사람이 확인한 정지 구간은 없으므로 해당 물리 정확도·정지 잡음은 x로 남습니다. 이 값을 채우기 위해 현재 입력을 다시 하라는 뜻은 아닙니다.', '',
              '저장 시 실제 답변에는 이전 예측을 보지 않았다는 기록이 있습니다. 최초 G 주석 작성 시점의 독립적인 노출 기록은 없으므로 블라인드 참조라는 주장은 하지 않습니다. 이번 노란 선택 네모를 본 이력은 현재 대상 확인 기록으로 분리했습니다.', '',
              '새 학습·optimizer update·새 모델 추론·실제 장비 제어는 0회입니다. 기존 8,910프레임 출력에서 해당 12장만 재집계했습니다. 원본과 이전 결과를 보존했고 PDF 생성·push는 하지 않았습니다.', '',
              '## 고정 12장 전체 비교', '',
              '왼쪽은 기존 G 저장 참조, 가운데는 Base, 오른쪽은 N3입니다. 원은 참조, 십자는 모델 예측입니다. 모든 12장을 표시했습니다.', '']
    for fid, image_path in images:
        frame = next(row for row in result['frame_results'] if row['frame_id'] == fid)
        before, after = (frame['methods'][method]['conditional_error']['median_px'] for method in ('Base', 'N3'))
        lines += [f'### {fid}', '', f'이 사진의 코너 중앙값: {before:.3f} → {after:.3f}px.', '',
                  f'![참조·Base·N3 비교 {fid}]({image_path})', '']
    markdown = '\n'.join(lines) + '\n'
    (DOC / 'CLOSEOUT_KO.md').write_text(markdown, encoding='utf-8')

    def esc(value):
        return html.escape(str(value))
    body = ['<h1>기존 PnP 주석 연결 결과</h1>',
            '<p class="done">12장 확인 완료 · 추가 클릭 없이 계산 완료</p>',
            '<p>기존 G 저장 주석 96점을 그대로 사용했습니다. 아래는 고정 12장에 대한 결과입니다.</p>',
            '<table><tr><th>지표</th><th>Base</th><th>N3</th><th>변화</th></tr>']
    for row in table[2:]:
        body.append('<tr>' + ''.join('<td>' + esc(v.strip()) + '</td>' for v in row.strip('|').split('|')) + '</tr>')
    body += ['</table><p>큰 오차와 10px 이내 비율은 개선됐지만 중앙값은 조금 악화됐습니다. 이 결과를 그대로 원고에 기록합니다.</p>',
             '<p><a href="CLOSEOUT_KO.md">상세 Markdown</a> · <a href="POINT_ERRORS.csv">96점 오차 CSV</a> · <a href="RESULT.json">수치·출처 JSON</a></p>']
    if inserted:
        body += ['<p>별도 원고 복사본 반영 완료 · <a href="paper_updated/manuscript_ko.md">원고</a> · <a href="paper_updated/supplement_ko.md">보충 자료</a></p>']
    body += ['<p>왼쪽: 기존 참조 · 가운데: Base · 오른쪽: N3. 아래 12장은 모두 표시합니다. 이미지는 클릭하면 확대됩니다.</p>']
    for fid, image_path in images:
        body += [f'<figure><figcaption>{esc(fid)}</figcaption><a href="{esc(image_path)}" target="_blank"><img loading="lazy" src="{esc(image_path)}" alt="참조 Base N3 비교 {esc(fid)}"></a></figure>']
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>팔레트 · PnP 주석 연결 결과</title><style>body{max-width:1320px;margin:24px auto;padding:0 20px;font:18px/1.7 sans-serif;color:#172d3a;background:#f5f8fa}h1{font-size:30px}.done{padding:14px;background:#dff3e8;font-weight:bold}table{border-collapse:collapse;background:white;width:100%}th,td{padding:12px;border:1px solid #c6d2d9;text-align:left}figure{margin:24px 0;background:white;padding:10px}img{width:100%;height:auto}figcaption{font-weight:bold}a{color:#075a9e}</style><body>' + ''.join(body) + '</body></html>'
    (DOC / 'REVIEW.html').write_text(page, encoding='utf-8')
    receipt = dict(status='COMPLETE', source_result_sha256=sha(result_path),
                   generated_at=datetime.now(timezone.utc).isoformat(),
                   actual_cpu_wall_seconds=time.perf_counter()-started,
                   paper_inserted=inserted, human_decisions=metrics['decision_counts'],
                   source_hash_checks=len(result['sources']), copied_artifacts=copies,
                   fixed_images=12, fixed_points=96, training_runs=0, optimizer_updates=0,
                   new_model_forward_frames=0, hardware_control_calls=0,
                   output_hashes={name:sha(DOC/name) for name in ('CLOSEOUT_KO.md','REVIEW.html')})
    receipt_target = DOC / 'READABLE_REPORT_RECEIPT.json'
    if receipt_target.exists():
        with receipt_target.with_suffix('.history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(read(receipt_target), ensure_ascii=False) + '\n')
    receipt_target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'], paper_inserted=inserted,
                         images=12, points=96, cpu_wall_seconds=receipt['actual_cpu_wall_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
