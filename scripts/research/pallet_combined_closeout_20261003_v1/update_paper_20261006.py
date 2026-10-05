"""Apply verified closeout artifacts to a separate manuscript copy, without TeX/PDF."""
from __future__ import annotations

import csv
import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1'
OUT = BASE / 'closeout_20261006_v1'
OLD = BASE / 'paper_updated'
PAPER = OUT / 'paper_updated'
PATCH = OUT / 'paper_patch'
CELLS = []
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    SOURCES[path.relative_to(ROOT).as_posix()] = dict(sha256=sha(path), bytes=path.stat().st_size)
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def display(value, digits=3):
    if value is None or value == 'x':
        return r'\notrun'
    if value == 'NA':
        return 'NA'
    return f'{float(value):,.0f}' if digits == 0 else f'{float(value):.{digits}f}'


def cell(value, source, pointer, table, *, digits=3, scale=1., unit='', denominator=None, seeds=None):
    formatted = display(value * scale if isinstance(value, (int, float)) else value, digits)
    CELLS.append(dict(target=table, source=source.relative_to(ROOT).as_posix(),
        source_sha256=sha(source), json_pointer=pointer, raw_value=value,
        display=formatted, unit=unit, denominator=denominator, seeds=seeds,
        scale=scale, status=('X_REFERENCE_OR_CONTRACT_MISSING' if value in (None, 'x') else
            'READY_TO_INSERT' if table.startswith('aux:') else 'INSERTED_IN_COPY')))
    return formatted


def table(path, label, caption, columns, header, rows, note, *, wide=True):
    env = 'table*' if wide else 'table'
    width = r'\textwidth' if wide else r'\columnwidth'
    text = '\\begin{' + env + '}[t]\n\\centering\\footnotesize\n'
    text += r'\setlength{\tabcolsep}{3pt}' + '\n'
    text += r'\caption{' + caption + r'}\label{' + label + '}\n'
    text += r'\begin{tabularx}{' + width + '}{' + columns + '}\n\\toprule\n'
    text += header + r' \\' + '\n\\midrule\n'
    text += '\n'.join(' & '.join(row) + r' \\' if isinstance(row, list) else row for row in rows)
    text += '\n\\bottomrule\n\\end{tabularx}\n'
    text += r'\tabnote{' + note + '}\n\\end{' + env + '}\n'
    (PAPER / path).parent.mkdir(parents=True, exist_ok=True)
    (PAPER / path).write_text(text)


def replace_block(path, start, end, new):
    file = PAPER / path
    text = file.read_text()
    left, right = text.index(start), text.index(end, text.index(start))
    file.write_text(text[:left] + new + text[right:])


def static_tables(data):
    source = OUT / 'static/STATIC_REAGGREGATION.json'
    names = dict(all='전체', clean='없음', moderate='중간', severe='어려움')
    common_note = ('등급은 최신 사람 입력을 원본 RGB 해시와 frame ID에 연결한 것이다. '
        '판정 기준의 사람 확인은 아직 없어 외부 물체 가림의 검수 완료로 부르지 않는다. '
        '자체 가림·화면 밖을 포함했는지와 사전 예측 노출을 통제한 독립 시험이 아니며, '
        '보조 사후 분석으로 제시한다. 전체 319장의 원래 2D·6D 분모와 수치는 유지했다.')

    def rows_for(methods, backbone='yolo'):
        rows = []
        for group in ('all', 'clean', 'moderate', 'severe'):
            for method in methods:
                record = data['backbones'][backbone][method]
                result = record['result'][group]
                pointer = f'/backbones/{backbone}/{method}/result/{group}/'
                target = 'tab:occlusion_results' if methods == ('Base', 'N3') else 'sup:occlusion_results'
                seeds = [1, 2, 3] if method != 'Base' else None
                den = dict(frames=result['frames'], reference_corners=result['full_supervised_corners'],
                    finite_corners=result['observed_corners'], pose_frames=result['pose_available_frames'])

                def num(key, digits=3, unit='', scale=1.):
                    return cell(result[key], source, pointer + key, target, digits=digits,
                        unit=unit, scale=scale, denominator=den, seeds=seeds)

                rows.append([names[group], method, num('frames', 0, 'frame'),
                    num('corner_median_px', unit='px'), num('corner_P90_px', unit='px'),
                    num('PCK10_fraction', unit='%', scale=100.),
                    num('translation_median_cm', unit='cm') + ' / ' + num('translation_P90_cm', unit='cm'),
                    num('rotation_median_deg', unit='deg') + ' / ' + num('rotation_P90_deg', unit='deg'),
                    num('full_supervised_corners', 0, 'corner') + ' / ' + num('observed_corners', 0, 'corner')])
        return rows

    header = (r'사람 입력 등급 & 방법 & 영상 & \shortstack{코너 중앙값\\(px)} & '
        r'\shortstack{코너 P90\\(px)} & \shortstack{\pckten\\(\%)} & '
        r'\shortstack{위치 중앙/P90\\(cm)} & \shortstack{회전 중앙/P90\\(도)} & '
        r'\shortstack{참조/유효\\예측 코너}')
    table('tables/occlusion_results.tex', 'tab:occlusion_results',
        '319장의 사람 입력 등급(기준 미확인)에 따른 YOLO Base와 N3 보조 분석.',
        'Ylrrrrrrr', header, rows_for(('Base', 'N3')), common_note)
    start = len(CELLS)
    all_arm_rows = rows_for(('Base', 'P', 'N2', 'N3'))
    for entry in CELLS[start:]:
        entry.update(target='aux:yolo_all_arms_current', status='READY_TO_INSERT')
    table('analysis_tables/yolo_all_arms_current.tex', 'aux:yolo_all_arms_current',
        '사람 입력 등급(기준 미확인)의 YOLO Base·P·N2·N3 전체 행: 별도 보존 조각.',
        'Ylrrrrrrr', header, all_arm_rows, common_note)
    paired_rows = []
    for backbone, name in (('yolo', 'YOLO'), ('dope', 'DOPE'), ('resnet18', 'ResNet-18')):
        for group in ('clean', 'moderate', 'severe'):
            before = data['backbones'][backbone]['Base']['result'][group]
            after = data['backbones'][backbone]['N3']['result'][group]
            def paired(key, unit='', digits=3, scale=1.):
                values = []
                for method, result in (('Base', before), ('N3', after)):
                    values.append(cell(result[key], source,
                        f'/backbones/{backbone}/{method}/result/{group}/{key}', 'sup:occlusion_results',
                        unit=unit, digits=digits, scale=scale,
                        denominator={k: result[k] for k in ('frames','full_supervised_corners','observed_corners','pose_available_frames')},
                        seeds=[1, 2, 3] if method == 'N3' else None))
                return r' $\to$ '.join(values)
            paired_rows.append([name, names[group], cell(before['frames'], source,
                f'/backbones/{backbone}/Base/result/{group}/frames', 'sup:occlusion_results', digits=0, unit='frame'),
                paired('corner_median_px','px'), paired('corner_P90_px','px'),
                paired('PCK10_fraction','%',scale=100.), paired('translation_median_cm','cm'),
                paired('rotation_median_deg','deg'),
                display(before['full_supervised_corners'],0) + '/' + display(before['observed_corners'],0),
                paired('pose_available_frames','frame',digits=0)])
    table('supplement_tables/occlusion_results.tex', 'sup:occlusion_results',
        '세 기반의 사람 입력 등급(기준 미확인)별 Base $\\to$ N3 보조 결과.',
        'llrrrrrrrr', r'기반 & 등급 & 영상 & 중앙값(px) & P90(px) & \pckten(\%) & 위치(cm) & 회전(도) & 참조/유효 & 자세 산출',
        paired_rows, '각 쌍은 Base와N3의 통계이며 N3는 세seed 통계 평균이다. 중앙값·P90은 유효예측 또는 산출자세의 조건부 통계이다. '
        'PCK는 전체참조, 자세 산출수는 영상 분모를 유지한다. 참조/유효 코너 수는 두방법이 같다. '
        '판정 기준 미확인인 사후집단으로 외부 가림 복원에 대한 독립확증이 아니다. '
        'DOPE와ResNet의 위치·회전 악화도 그대로 표시한다. YOLO의 P·N2 및 각seed 상세는 별도CSV·TeX 조각으로 보존한다.')

    counts = data['label_counts']
    square = read(OUT / 'static/SQUARE_REAGGREGATION.json')
    square_counts = square.get('label_counts', {})
    all_rows = []
    for material, name in (('plastic', '직사각형 / 플라스틱'), ('wood', '직사각형 / 목재')):
        material_result = data['backbones']['yolo']['Base']['result']['material_x_severity']
        nums = [material_result.get(material + '::' + group, {}).get('frames', 0)
                for group in ('clean', 'moderate', 'severe')]
        all_rows.append([name] + [cell(n, source,
            f'/backbones/yolo/Base/result/material_x_severity/{material}::{group}/frames',
            'tab:composition', digits=0, unit='frame') for n, group in zip(nums, ('clean', 'moderate', 'severe'))]
            + ['0', display(sum(nums), 0)])
    all_rows.append(['직사각형 합계'] + [cell(counts[k], source, '/label_counts/' + k,
        'tab:composition', digits=0, unit='frame') for k in ('clean', 'moderate', 'severe')]
                    + ['0', '319'])
    if not square_counts:
        square_counts = square.get('classification_counts', square.get('severity_counts', {}))
    if not all(k in square_counts for k in ('clean', 'moderate', 'severe')):
        raise ValueError('Square label counts are required; do not assume 119 are unknown')
    all_rows.append(['정사각형119'] + [cell(square_counts[k], OUT / 'static/SQUARE_REAGGREGATION.json',
        '/label_counts/' + k, 'tab:composition', digits=0, unit='frame') for k in ('clean', 'moderate', 'severe')]
                    + ['0', '119'])
    table('tables/composition.tex', 'tab:composition', '최신 사람 입력 등급(기준 미확인)의 구성.',
        'Yrrrrr', r'집단 & 없음 & 중간 & 어려움 & 미입력 & 전체', all_rows,
        '숫자는 실제 입력된 등급의 개수이며 판정 기준 승인과 구분한다. 과거 외부 가림 128장 레이블과 '
        '혼합하지 않는다. 과거 정사각형 150장은 별도 7세션 자료로 현 119장 성능 분모에 넣지 않았다.', wide=False)
    # Keep all-backbone, material and severity cells available outside the
    # established main/supplement layout, with their raw JSON as authority.
    detailed = []
    for backbone, methods in data['backbones'].items():
        for method, record in methods.items():
            for group, result in record['result'].items():
                if group == 'material_x_severity':
                    groups = result.items()
                else:
                    groups = ((group, result),)
                for identity, values in groups:
                    if isinstance(values, dict) and 'frames' in values:
                        detailed.append(dict(backbone=backbone, method=method, group=identity,
                            seed_aggregation=record['aggregation'], **values))
    write_csv(PAPER / 'evidence/closeout_tables/STATIC_ALL_BACKBONE_GROUPS.csv', detailed)
    return counts, square_counts


def static_text(data, counts, square_counts, visibility_counts):
    material = data['backbones']['yolo']['Base']['result']['material_x_severity']
    plastic = '/'.join(str(material.get('plastic::' + k, {}).get('frames', 0))
                       for k in ('clean', 'moderate', 'severe'))
    wood = '/'.join(str(material.get('wood::' + k, {}).get('frames', 0))
                    for k in ('clean', 'moderate', 'severe'))
    count_text = f'없음 {counts["clean"]}장, 중간 {counts["moderate"]}장, 어려움 {counts["severe"]}장'
    replace_block('sections/04_setup.tex', '\\subsection{가림 등급과 코너 가시성}',
        '\\subsection{비교 방법과 추가 학습}',
        '\\subsection{사람 입력 등급과 코너 가시성}\n'
        '표~\\ref{tab:composition}은 원본 RGB 해시와 frame ID에 연결한 최신 사람 입력 등급의 구성이다. '
        f'직사각형 319장은 {count_text}이며 플라스틱은 {plastic}장, 목재는 {wood}장이다. '
        f'정사각형119도 없음 {square_counts["clean"]}장·중간 {square_counts["moderate"]}장·'
        f'어려움 {square_counts["severe"]}장으로 입력되었다. '
        '이는 등급 입력 완료를 뜻하며 판정 기준의 사람 확인과는 구분된다. '
        '현재 판정 기준은 미확인 상태이므로 외부 가림 검수 완료로 표현하지 않고 보조 사후 분석으로 제시한다.\n\n'
        '외부 물체 가림·팔레트 자체 가림·화면 밖 잘림은 서로 다른 속성이다. '
        '최신 사람 입력의 중간·어려움 경계와 이 속성의 포함 여부를 CLI가 추정해 확정하지 않았다. '
        '과거 외부 가림 128장 패널은 기존29/20/79와 더 오래된29/21/78 버전 차이를 포함한 '
        '원래 출처와 숫자로 별도 보존하며 최신319장 등급과 같은 계약으로 합치지 않는다. '
        '임의로 개수를 맞추거나 오차에 따라 집단을 재선택하지 않았다.\n\n'
        '영상 단위 등급과 코너 단위 가시성도 구분한다. 기존 잠금71점과 새 입력3030점을 '
        '중복 없이 연결해 직사각형2499점·정사각형602점의 총3101개 참조 코너 상태를 보존하였다. '
        '오차 계산에는 원래 평가 참조 좌표와 같은 전체 객체 대응을 사용하며, 새로운 재클릭이나 '
        'PnP 생성점으로 정답 좌표를 교체하지 않았다. 가시성 입력과 좌표의 생성 경로·'
        '독립 블라인드 검수 여부는 서로 다른 정보이다. 상세 결과와 한계는 보충자료에 둔다.\n'
        '\\input{tables/composition}\n\n')
    severe = {m: data['backbones']['yolo'][m]['result']['severe'] for m in ('Base', 'N3')}
    def extra(backbone, group, method, key, unit):
        result = data['backbones'][backbone][method]['result'][group]
        return cell(result[key], OUT / 'static/STATIC_REAGGREGATION.json',
            f'/backbones/{backbone}/{method}/result/{group}/{key}', 'text:05_results',
            unit=unit, denominator={'frames': result['frames']},
            seeds=[1, 2, 3] if method == 'N3' else None)
    additional_negative = (
        '다른 기반에서도 악화를 보존하였다. DOPE의 중간 집단 위치 중앙값은 '
        + extra('dope','moderate','Base','translation_median_cm','cm') + '에서 '
        + extra('dope','moderate','N3','translation_median_cm','cm') + 'cm, 어려움 집단은 '
        + extra('dope','severe','Base','translation_median_cm','cm') + '에서 '
        + extra('dope','severe','N3','translation_median_cm','cm') + 'cm로 높아졌다. '
        'ResNet 어려움 집단의 위치 중앙값은 '
        + extra('resnet18','severe','Base','translation_median_cm','cm') + '에서 '
        + extra('resnet18','severe','N3','translation_median_cm','cm') + 'cm, 회전 중앙값은 '
        + extra('resnet18','severe','Base','rotation_median_deg','deg') + '에서 '
        + extra('resnet18','severe','N3','rotation_median_deg','deg') + '도로 악화하였다. ')
    replace_block('sections/05_results.tex', '\\subsection{가림 조건과 큰 오류의 한계}',
        '초기 코너 오차를',
        '\\subsection{사람 입력 등급의 보조 분석과 큰 오류의 한계}\n'
        f'표~\\ref{{tab:occlusion_results}}는 {count_text}의 사람 입력 등급(기준 미확인)에 '
        '대한 사후 분석이다. 없음·중간에서는 코너 중앙값·P90와 위치·회전 중앙값이 Base보다 N3에서 낮았다. '
        f'어려움에서는 코너 중앙값이 {severe["Base"]["corner_median_px"]:.3f}에서 '
        f'{severe["N3"]["corner_median_px"]:.3f}픽셀로 낮아졌지만, 코너 P90은 '
        f'{severe["Base"]["corner_P90_px"]:.3f}에서 {severe["N3"]["corner_P90_px"]:.3f}픽셀로 높아졌다. '
        f'위치 중앙값도 {severe["Base"]["translation_median_cm"]:.3f}에서 '
        f'{severe["N3"]["translation_median_cm"]:.3f}cm로 악화했다. '
        f'회전 중앙값의 {severe["Base"]["rotation_median_deg"]:.3f}에서 '
        f'{severe["N3"]["rotation_median_deg"]:.3f}도로의 감소가 모든 프레임의 개선을 뜻하지는 않는다. '
        '중앙 순위와 자세 가설 변화가 함께 포함된 결과이며 외부 가림 복원 능력의 독립 확증으로 해석하지 않는다.\n'
        '\\input{tables/occlusion_results}\n\n'
        '현재 등급은 플라스틱194장·목재125장을 모두 포함한다. 등급을 최신 버전으로 연결해 '
        '집단별 통계를 다시 계산했지만 전체319장의 기존2D·6D 결과와 실패 분모는 그대로 유지되었다. '
        '코너 가시성도 원래 참조 좌표에 상태만 연결한 보조 결과이며, 직접 보이는 점과 '
        '기하로 보완된 참조를 동일한 독립 수동 정답으로 간주하지 않는다. '
        '정사각형119의 가시성 분석은 전체수동602점과 영상내600점 모드를 별도로 유지한다. '
        + additional_negative + '\n\n')
    file = PAPER / 'sections/07_discussion.tex'
    text = file.read_text().replace('심한 가림에서 위치 오차가 악화하고',
        '사람 입력의 어려움 집단에서 위치 오차가 악화하고')
    text = text.replace('직사각형 가림 등급은 사람 완료 제출본과 연결했지만, 정확한 판정 경계·예측 노출·표본 모집 이력이 모두 독립적으로 통제된 것은 아니다.',
        '직사각형·정사각형의 사람 입력 등급과 참조 코너 가시성은 확보했으나, 등급 판정 기준의 사람 확인은 아직 없고 예측 노출·표본 모집 이력이 독립적으로 통제된 것은 아니다.')
    text = text.replace('정사각형 119장의 가림 등급과 대부분의 코너 가시성도 사람 검수가 남아 있다.',
        '등급 입력 완료를 외부 가림 기준의 승인으로 바꾸지 않았으며, 과거 정사각형150장의7세션 자료는 현119장 평가에 합산하지 않았다.')
    file.write_text(text)
    file = PAPER / 'sections/05_results.tex'
    text = file.read_text().replace('목재는 재질 단위 평가가 완료된 것이며, 아직 확인되지 않은 가림 등급별 성능을 뜻하지 않는다.',
        '재질별 통계와 최신 사람 입력 등급별 보조 통계는 서로 다른 분류이며, 등급 판정 기준은 아직 미확인이다.')
    file.write_text(text)
    file = PAPER / 'sections/08_conclusion.tex'
    text = file.read_text().replace('일부 P90과 severe 가림의 위치 오차는 악화되었다.',
        '일부 P90과 사람 입력의 어려움 집단의 위치 오차는 악화되었다.')
    text = text.replace('앞으로의 판단에는 정사각형 가림·코너 가시성 검수, 리프터의 사람 참조와 객체 대응, 참조 품질 확인 및 독립 관측에서의 평가가 필요하다.',
        '정사각형을 포함한 사람 입력 등급과 3,101개 참조 코너의 가시성 상태는 확보하였다. '
        '남은 확인은 등급 판정 기준과 참조 품질이며, 리프터 정확도에는 평가용 사람 참조·객체 대응과 독립 물리 참조가 필요하다. '
        '이미 입력된 등급과 가시성의 재작업을 전제로 하지 않으며, 추가 일반화는 독립 관측에서 평가해야 한다.')
    file.write_text(text)


def visibility_and_square_tables():
    source = OUT / 'visibility_square/VISIBILITY_RESULTS.json'
    data = read(source)
    names = dict(DIRECT_VISIBLE='직접 가시', EXTERNAL_OCCLUDED='외부 가림',
        SELF_OCCLUDED='자체 가림', OUT_OF_FRAME='화면 밖')
    rows = []
    for category in names:
        for method in ('Base', 'N3'):
            index, result = next((i, row) for i, row in enumerate(data['family_rows'])
                if row['population'] == 'DEV319' and row['backbone'] == 'yolo'
                and row['category'] == category and row['method'] == method)
            pointer = f'/family_rows/{index}/'
            denominator = {k: result[k] for k in ('frames', 'reference_corners', 'observed_corners')}

            def num(key, digits=3, unit=''):
                return cell(result[key], source, pointer + key, 'sup:visibility', digits=digits,
                    unit=unit, denominator=denominator, seeds=[1, 2, 3] if method == 'N3' else None)

            rows.append([names[category], method, num('reference_corners', 0, 'corner') + '/' +
                num('observed_corners', 0, 'corner'), num('median_px', unit='px'),
                num('P90_px', unit='px'), num('PCK10_percent', unit='%')])
    table('supplement_tables/visibility_results.tex', 'sup:visibility',
        '최신 사람 가시성 상태에 따른 YOLO Base와 N3 보조 분석. 원래2499개 참조 좌표를 유지한다.',
        'Ylrrrr', r'상태 & 방법 & \shortstack{참조/유효\\예측 코너} & 중앙값(px) & P90(px) & \pckten(\%)',
        rows, '직접 가시1776·외부 가림218·자체 가림462·화면 밖43점으로 합계2499점이다. '
        '결측·객체 매칭 실패는 전체 참조 PCK 분모에 유지했다. 조건부 중앙값·P90와 전체 분모를 구분한다. '
        '기존 잠금 상태71점은 덮어쓰지 않았고, 가시성 상태를 입력해도 원래 참조 생성 경로가 바뀌지는 않는다. '
        'P·N2 및 DOPE·ResNet·정사각형 두 모드의 모든 행은 별도CSV에 보존한다.')

    source = OUT / 'visibility_square/SQUARE119_RESULTS.json'
    square = read(source)
    for include_p, target_file, label in ((False, 'tables/square_results.tex', 'tab:square_results'),
            (True, 'supplement_tables/square_results.tex', 'sup:square_results')):
        table_rows = []
        for mode, size in (('manual_declared', 602), ('manual_in_frame', 600)):
            table_rows.append(r'\multicolumn{9}{l}{\textbf{' +
                ('(a) 전체수동 참조602점' if mode == 'manual_declared' else '(b) 영상내 참조600점') + r'}} \\')
            for index, result in enumerate(square['family_rows']):
                if result['mode'] != mode or (result['method'] == 'P' and not include_p):
                    continue
                if result['reference_corners'] != size:
                    raise ValueError('Square modes have a mixed denominator')
                ptr = f'/family_rows/{index}/'
                den = {k: result[k] for k in ('frames', 'reference_corners', 'observed_corners')}

                def num(key, digits=3, unit=''):
                    return cell(result[key], source, ptr + key, label, digits=digits,
                        unit=unit, denominator=den, seeds=[1, 2, 3] if result['method'] != 'Base' else None)

                table_rows.append([dict(yolo='YOLO', dope='DOPE', resnet18='ResNet-18')[result['backbone']],
                    result['method'], num('median_px', unit='px'), num('P90_px', unit='px'),
                    num('PCK10_percent', unit='%'), num('observed_corners', 0, 'corner'),
                    num('reference_corners', 0, 'corner'), num('translation_cm', unit='cm'),
                    num('rotation_deg', unit='deg')])
            table_rows.append(r'\midrule')
        table(target_file, label, '정사각형119장의 동일 고정 예측을602/600점 참조 모드로 각각 평가.',
            'lYrrrrrrr', r'기반 & 방법 & 중앙값(px) & P90(px) & \pckten(\%) & 유효 & 참조 & 위치(cm) & 회전(도)',
            table_rows[:-1], '한 촬영세션·등록 치수[1.1,1.1,0.15]m. N3는 세seed 통계의 평균이며 '
            '앙상블이 아니다. 두 모드는 화면 밖 참조2점 포함 여부만 다르며 분모를 혼합하지 않는다. '
            '독립6D 참조 부재로 위치·회전은x, 세션간 구간은NA다. N3대Base와 N3대N2는 별도 비교다. '
            'ResNet은10-epoch CONSTANT-fold RGB 기준 모델이다.')
    dest = PAPER / 'evidence/closeout_tables'
    for name in ('VISIBILITY_RESULTS', 'VISIBILITY_PER_SEED', 'SQUARE119_RESULTS',
            'SQUARE119_PER_SEED', 'SQUARE119_COMPARISONS', 'HISTORICAL_SQUARE150', 'HISTORICAL_SQUARE150_PER_SEED'):
        for suffix in ('.csv', '.md'):
            source_file = OUT / 'visibility_square' / (name + suffix)
            if source_file.is_file():
                shutil.copy2(source_file, dest / source_file.name)
    return data, square


def review_figure(counts, square_counts, merged):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.3), constrained_layout=True)
    for offset, category, label, color in ((-.24, 'clean', 'None', '#26867c'),
            (0., 'moderate', 'Medium', '#e5a234'), (.24, 'severe', 'Difficult', '#b45163')):
        values = [counts[category], square_counts[category]]
        bars = axes[0].bar(np.arange(2) + offset, values, width=.23, label=label, color=color)
        axes[0].bar_label(bars, padding=3)
    axes[0].set_xticks([0, 1], ['Rectangular 319', 'Square 119'])
    axes[0].set_ylabel('Images')
    axes[0].set_title('Human-entered grade (criterion unconfirmed)')
    axes[0].legend(frameon=False, fontsize=8)
    pop = merged['population_counts']
    colors = ['#26867c', '#b45163', '#738ec0', '#e5a234']
    categories = ['DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED', 'OUT_OF_FRAME']
    bottom = np.zeros(2)
    for category, label, color in zip(categories, ['Visible', 'External', 'Self', 'Outside'], colors):
        values = np.array([pop['DEV319'].get(category, 0), pop['GREEN0918'].get(category, 0)])
        bars = axes[1].bar([0, 1], values, bottom=bottom, label=label, color=color, width=.55)
        for bar, value, base_value in zip(bars, values, bottom):
            if value:
                if value < 10:
                    axes[1].annotate('Outside: ' + str(value),
                        xy=(bar.get_x()+bar.get_width()/2, base_value+value/2),
                        xytext=(bar.get_x()+bar.get_width()/2+.42, base_value+105),
                        ha='center', fontsize=8, arrowprops=dict(arrowstyle='-', color='#555'))
                else:
                    axes[1].text(bar.get_x()+bar.get_width()/2, base_value+value/2, str(value),
                        ha='center', va='center', fontsize=8)
        bottom += values
    axes[1].set_xticks([0, 1], ['Rectangular: 2499', 'Square: 602'])
    axes[1].set_ylabel('Reference corner states')
    axes[1].set_title('Visibility labels; original coordinates preserved')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    for suffix in ('png', 'svg'):
        fig.savefig(PAPER / 'figures' / ('current_review_summary.' + suffix), dpi=220)
    plt.close(fig)
    (PAPER / 'figures/current_review_summary.tex').write_text(
        '\\begin{figure*}[t]\\centering\n'
        '\\includegraphics[width=.91\\textwidth]{figures/current_review_summary.png}\n'
        '\\caption{최신 사람 입력의 자료 구성. 왼쪽 등급은 판정 기준 미확인 상태의 보조 분석이며, '
        '오른쪽은 원래 참조 좌표를 유지한 가시성 상태다. 그림의 입력 개수는 모델 정확도나 독립 참조의 수가 아니다.}'
        '\\label{sup:current_review_summary}\n\\end{figure*}\n')
def supplementary_text(counts):
    file = PAPER / 'supplement.tex'
    text = file.read_text()
    text = text.replace('완료된 사람 분류는 목재를 포함하지만, 집단 크기가 불균형하고 severe 목재가 0장인 개발자료의 사후 분석이다.',
        '최신 사람 입력 등급은 목재를 포함하되 판정 기준은 미확인이고 집단 크기도 불균형하다. 재질별 전체 결과는 기존194/125장 분모로 유지한다.')
    start = text.index('표~\\ref{sup:occlusion_results}')
    end = text.index('\\section{큰 오류', start)
    text = text[:start] + (
        '표~\\ref{sup:occlusion_results}는 사람 입력 등급(기준 미확인)에 따른 세 기반의 Base와N3 쌍이다. '
        f'없음{counts["clean"]}장·중간{counts["moderate"]}장·어려움{counts["severe"]}장의 '
        '사후 집단만 갱신했으며 전체319장의 원래 코너·자세·실패 통계는 유지했다. '
        '조건부 코너·자세 오차와 전체 참조·자세 산출 분모를 함께 보고한다. '
        'YOLO의 P·N2 전체 행, 각seed 및 과거 외부 가림128장의 원시 레이블과 결과는 별도CSV·분석 조각에 보존한다.\n'
        '\\input{supplement_tables/occlusion_results}\n\n'
        '표~\\ref{sup:visibility}는 잠금71점과 새3030개 상태를 중복 없이 연결한 직사각형2499점의 '
        '가시성별 결과이다. 원래 참조 좌표를 유지했고, 없는 좌표를 생성하거나 기계 제안을 '
        '사람 승인으로 승격하지 않았다. 정사각형602/600점 모드와 모든 기반의 상세 행은 '
        '동봉 CSV·JSON에 별도 보존한다. 노출 이력과 참조 생성 경로의 제한 때문에 독립 블라인드 '
        '가림 복원 정확도를 주장하지 않는다.\n'
        '\\input{supplement_tables/visibility_results}\n'
        '\\input{figures/current_review_summary}\n\n') + text[end:]
    file.write_text(text)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)


def lifter_text(audit_source):
    # These are the actual staging inputs; no additional human action or
    # finalized evaluator receipt is fabricated by document preparation.
    audit = read(audit_source)
    assert audit['classification']['states_entered'] == 96
    assert audit['classification']['manual_coordinates'] == 67
    assert audit['classification']['visible_without_coordinates_count'] == 5
    assert audit['formal_reference']['approved_frames'] == 0
    assert audit['formal_reference']['target_match_reviewed_frames'] == 0
    for key, unit in (('frames','frame'), ('states_entered','state'), ('manual_coordinates','corner'),
                      ('visible_without_coordinates_count','corner')):
        cell(audit['classification'][key], audit_source, '/classification/' + key,
            'text:06_case_study', digits=0, unit=unit, denominator={'frames': 12, 'corner_states': 96})
    cell(audit['formal_reference']['visible_corner_accuracy'], audit_source,
        '/formal_reference/visible_corner_accuracy', 'tab:case', unit='px',
        denominator={'stored_frames': 8910, 'formal_references': 0, 'human_target_matches': 0})
    for key, unit in (('translation','cm'), ('rotation','deg')):
        cell(audit['independent_physical_pose_reference'][key], audit_source,
            '/independent_physical_pose_reference/' + key, 'tab:case', unit=unit,
            denominator={'stored_frames': 8910, 'independent_physical_references': 0})
    path = PAPER / 'sections/06_case_study.tex'
    text = path.read_text()
    old = '사람 코너 참조·예측 객체 대응·확인된 정지 구간·독립 물리 참조가 아직 없어 해당 정확도는 \\notrun 으로 유지한다.'
    new = ('별도 12장의 가시성 입력 96개는 보존되었으며, 실제 수동 클릭 좌표 67점·'
        '좌표 없는 직접 가시 판정 5점·자체 가림 24점으로 구분된다. '
        '5개의 가시 판정에 PnP 생성 좌표를 수동 정답처럼 채우지 않았다. '
        '다만 평가용 최종 승인 기록과 예측 객체 대 사람 참조 대응은 각각 0건이며, '
        '확인된 정지 구간·독립 물리 참조도 없어 가시 코너 정확도와 물리 정확도는 \\notrun 으로 유지한다.')
    if old not in text:
        raise ValueError('Expected old lifter limitations sentence was not found')
    path.write_text(text.replace(old, new))
    text = (PAPER / 'supplement.tex').read_text()
    old = '사람 코너·대상 대응·정지·독립 물리 참조가 필요한 값은 x로 보존한다.'
    new = ('12장 96개 상태 중 실제 클릭 좌표 67점·좌표 없는 가시 판정 5점·자체 가림 24점을 '
        '보존했지만 최종 평가 승인과 예측 객체 대응은 아직 0건이다. '
        '사람 입력 확보와 평가 게이트 성립을 구분하며, 코너 오차·정지·독립 물리 참조가 필요한 값은 x로 보존한다.')
    if old not in text:
        raise ValueError('Expected old supplementary lifter limitations sentence was not found')
    (PAPER / 'supplement.tex').write_text(text.replace(old, new))


def patches():
    mutations = []
    for path in sorted(PAPER.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(PAPER)
        original = OLD / rel
        if path.suffix not in ('.tex', '.md'):
            continue
        if original.is_file() and path.read_bytes() == original.read_bytes():
            continue
        text = ''.join(difflib.unified_diff(original.read_text().splitlines(keepends=True) if original.is_file() else [],
            path.read_text().splitlines(keepends=True), fromfile='a/' + rel.as_posix() if original.is_file() else '/dev/null',
            tofile='b/' + rel.as_posix()))
        mutations.append((rel.as_posix(), text))
    PATCH.mkdir(exist_ok=True)
    integrated = ''.join(text for _, text in mutations)
    (PATCH / 'INTEGRATED.patch').write_text(integrated)
    (PATCH / 'STATIC.patch').write_text(''.join(text for name, text in mutations
        if name != 'sections/06_case_study.tex' and name != 'supplement.tex'))
    (PATCH / 'LIFTER.patch').write_text(''.join(text for name, text in mutations
        if name in ('sections/06_case_study.tex', 'supplement.tex')))
    validation_copy = OUT / 'patch_validation_copy'
    if validation_copy.exists():
        shutil.rmtree(validation_copy)
    shutil.copytree(OLD, validation_copy)
    result = subprocess.run(['git', 'apply', '--unsafe-paths', str(PATCH / 'INTEGRATED.patch')],
        cwd=validation_copy, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Patch did not apply to a fresh original copy: ' + result.stderr)
    for name, _ in mutations:
        if (validation_copy / name).read_bytes() != (PAPER / name).read_bytes():
            raise ValueError('Applied patch mismatch: ' + name)
    payload = []
    for path in PAPER.rglob('*'):
        if not path.is_file() or path.suffix in ('.tex', '.md'):
            continue
        destination = validation_copy / path.relative_to(PAPER)
        if not destination.is_file() or destination.read_bytes() != path.read_bytes():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            payload.append(path.relative_to(PAPER).as_posix())
    expected_files = {p.relative_to(PAPER).as_posix(): sha(p) for p in PAPER.rglob('*') if p.is_file()}
    applied_files = {p.relative_to(validation_copy).as_posix(): sha(p)
                     for p in validation_copy.rglob('*') if p.is_file()}
    if expected_files != applied_files:
        raise ValueError('Applied text patch plus supplied assets do not reproduce the updated copy')
    return dict(command=['git', 'apply', '--unsafe-paths', str(PATCH / 'INTEGRATED.patch')],
        exit_code=result.returncode, fresh_original_copy=str(validation_copy.relative_to(ROOT)),
        matching_modified_files=len(mutations), includes_new_text_artifacts=True,
        supplied_nontext_payload_files=payload, full_applied_copy_sha256_equal=True,
        note='Text patch applied; supplied CSV/JSON/PNG/SVG payload copied; complete resulting tree equals paper_updated.')


def preserve_historical_panels():
    dest = PAPER / 'evidence/closeout_tables'
    dest.mkdir(parents=True, exist_ok=True)
    original_csv = OLD / 'evidence/github_tables/tab_occlusion_results.csv'
    shutil.copy2(original_csv, dest / 'HISTORICAL_EXTERNAL128_SOURCE.csv')
    with original_csv.open() as file:
        rows = list(csv.DictReader(file))
    cells = []
    for row in rows:
        if row['Group'] not in ('clean', 'moderate', 'severe') or row['Method'] not in ('R0', 'N3 dimensions + symmetry'):
            continue
        cells.append([row['Group'], 'Base' if row['Method'] == 'R0' else 'N3',
            display(float(row['Frames']), 0), display(float(row['Median px'])),
            display(float(row['P90 px'])), display(float(row['PCK10 %']))])
    table('analysis_tables/historical_external128.tex', 'aux:historical_external128',
        '과거 외부 가림128장 패널: 원래29/20/79 레이블 버전 결과를 별도 보존.', 'Ylrrrr',
        r'과거 등급 & 방법 & 영상 & 중앙값(px) & P90(px) & \pckten(\%)', cells,
        '새319장 사람 입력 등급과 합치지 않는다. 이전29/21/78 버전과 한frame ID가 달랐던 '
        '기록도 원래 감사 자료에 보존한다. 이 조각은 실제main/supplement에 추가삽입하지 않았다.')
    # Preserve the distinct historical result and its provisional contract,
    # without promoting it to the present N3 square119 comparison.
    historical = read(OUT / 'visibility_square/HISTORICAL_SQUARE150.json')
    write_json(PAPER / 'evidence/closeout_tables/HISTORICAL_SQUARE150_SOURCE.json', historical)
    (PAPER / 'analysis_tables/README_KO.md').write_text(
        '# 별도 분석 조각\n\n'
        '이 폴더의 표는 원고에 추가 삽입한 표와 구분합니다. '
        '과거 외부 가림128장 원본CSV와 표를 보존했습니다. 과거 정사각형150장·7세션·'
        '등급103/44/3은 현119장과 별도 자료입니다. 저장된 R0/N0/N2의 제한된2D 결과와 '
        '참조 검수 미완료 계약은 별도CSV/JSON에 보존하며 N3·DOPE·ResNet 및 독립6D 정확도는x입니다.\n\n'
        '모든 기반·재질×등급 상세 결과는 `../evidence/closeout_tables/STATIC_ALL_BACKBONE_GROUPS.csv`, '
        '정사각형 가시성·602/600·seed 행은 같은 폴더의 명시된CSV를 사용합니다. '
        '같은 원고 표에 쓰인 값만 실제 삽입으로 기록합니다.\n')


def readable_manuscript():
    def plain(text):
        # Comments are source lines. Do not strip literal percent signs from
        # already-expanded Markdown tables or escaped TeX percentages.
        text = re.sub(r'(?m)^\s*%[^\n]*', '', text)
        text = re.sub(r'\\label\{[^}]*\}', '', text)
        text = re.sub(r'\\section\{([^}]+)\}', r'\n# \1\n', text)
        text = re.sub(r'\\subsection\{([^}]+)\}', r'\n## \1\n', text)
        text = re.sub(r'\\ref\{([^}]+)\}', r'`\1`', text)
        text = re.sub(r'\\cite\{([^}]+)\}', r' [\1]', text)
        text = text.replace(r'\pckten', 'PCK10').replace(r'\notrun', 'x').replace(r'\%', '%')
        text = re.sub(r'\\(?:textbf|emph|texttt)\{([^{}]*)\}', r'\1', text)
        return text.strip()

    def expand(path):
        text = path.read_text()
        if path.name == 'supplement.tex':
            text = text[text.index(r'\section{'):text.rindex(r'\FloatBarrier')]
        def insert(match):
            relative = match.group(1)
            file = PAPER / (relative + '.tex')
            if relative.startswith(('tables/', 'supplement_tables/')):
                raw = file.read_text()
                caption = re.search(r'\\caption\{([^\n]+?)\}(?:\\label|\n)', raw)
                label = re.search(r'\\label\{([^}]+)\}', raw)
                title = plain(caption.group(1)) if caption else relative
                rows = []
                in_table = False
                for line in raw.splitlines():
                    if line.startswith(r'\begin{tabularx}'):
                        in_table = True
                        continue
                    if line.startswith(r'\end{tabularx}'):
                        in_table = False
                    if in_table and line.startswith(r'\multicolumn'):
                        marker = re.search(r'\\textbf\{([^}]+)\}', line)
                        if marker and rows:
                            rows.append(['**' + plain(marker.group(1)) + '**'] + [''] * (len(rows[0]) - 1))
                        continue
                    if not in_table or '&' not in line:
                        continue
                    line = line.removesuffix(r' \\').strip()
                    # Embedded linebreaks in headings are purely typographic.
                    line = re.sub(r'\\shortstack(?:\[[^]]*\])?\{([^}]+)\}',
                        lambda m: m.group(1).replace('\\\\', ' '), line)
                    rows.append([plain(col).replace('|', '\\|') for col in line.split('&')])
                body = '\n\n**' + title + '**'
                if label:
                    body += ' (`' + label.group(1) + '`)'
                if rows:
                    body += '\n\n|' + '|'.join(rows[0]) + '|\n|'
                    body += '|'.join('---' for _ in rows[0]) + '|\n'
                    body += '\n'.join('|' + '|'.join(row) + '|' for row in rows[1:])
                note = re.search(r'\\tabnote\{(.*)\}', raw)
                if note:
                    body += '\n\n' + plain(note.group(1))
                return body + '\n\n'
            if relative == 'figures/current_review_summary':
                return '\n\n![사람 입력 등급과 코너 가시성 구성](figures/current_review_summary.png)\n\n'
            if relative == 'figures/examples_panel':
                return '\n\n![보정 전후 실제 사례](figures/example_1_frame.png)\n\n'
            return expand(file)
        return plain(re.sub(r'\\input\{([^}]+)\}', insert, text))

    previous = PAPER / 'editorial/archive/manuscript_before_closeout_20261006.md'
    if not previous.exists():
        shutil.copy2(OLD / 'manuscript_ko.md', previous)
    sections = [expand(path) for path in sorted((PAPER / 'sections').glob('*.tex'))]
    (PAPER / 'manuscript_ko.md').write_text(
        '# 영상 특징과 치수 정보를 활용한 단안 팔레트 자세 추정용 국소 키포인트 보정\n\n'
        '> 2026-10-06 실제 수정LaTeX에서 만든 읽기용 원고입니다. PDF를 생성·컴파일하지 않았습니다. '
        '같은 표의 출처·분모·seed는 `../paper_patch/PAPER_CELL_MAP.json`으로 연결됩니다.\n\n'
        + '\n\n'.join(sections) + '\n')
    (PAPER / 'supplement_ko.md').write_text('# 보충자료 읽기용 변환본\n\n'
        '> 실제 수정 supplement.tex의 표와 설명입니다. PDF는 생성하지 않았습니다.\n\n'
        + expand(PAPER / 'supplement.tex') + '\n')


def gap_matrix():
    text = '''# 원고 셀 연결과 남은 x (2026-10-06)

| 원고 위치 | 확보 결과 | 실제 원고 반영 | 상태·제한 | 원시→지표 근거 |
|---|---|---|---|---|
| tab:composition | 319장153/92/74,119장3/85/31 | 반영 | 사람입력등급(기준 미확인); 보조분석 | static/LABEL_PROVENANCE_AUDIT → STATIC/SQUARE_REAGGREGATION |
| tab:occlusion_results | YOLO 최신등급 Base/N3 | 반영 | 전체319 headline·실패 분모 유지 | static/STATIC_REAGGREGATION |
| sup:occlusion_results | 세기반×3등급 Base→N3 9행 | 반영 | seed통계평균; 실패분모·악화보존; 등급 기준미확인 | 같은JSON/per_seed |
| aux:yolo_all_arms_current | YOLO Base/P/N2/N3 최신등급 전체행 | 별도TeX/CSV 완성 자료 | AVAILABLE_AS_SUPPORTING_EVIDENCE; 추가삽입은 선택 | 같은JSON |
| sup:visibility | 2499점 전체 가시성 Base/N3 | 반영 | 원래참조 유지; 독립블라인드 아님 | STATIC_VISIBILITY_MERGE_AUDIT → VISIBILITY_RESULTS |
| tab:square_results / sup:square_results | 119장602/600,세기반 | 반영 | 단일session; N3대Base/N2 구분 | visibility_square/SQUARE119_RESULTS |
| 정사각형119 T/R | x | x 유지 | 독립6D 참조없음 | SQUARE119_RESULTS.reference_6D |
| 정사각형 세션간 CI | NA | NA 설명반영 | 단일session으로 정의되지않음 | 같은JSON.intervals |
| 세기반·재질×등급·각seed·정사각형 등급 세부행 | 계산·서식 완료 | 상세CSV/54행 검토MD·JSON | AVAILABLE_AS_SUPPORTING_EVIDENCE; 필수 미완료 아님, 추가삽입은 선택 | final_review/SUBGROUP_REVIEW + STATIC_ALL_BACKBONE_GROUPS |
| 과거외부가림128 | 기존숫자 보존 | 별도조각만 | 최신319등급과혼합안함 | HISTORICAL_EXTERNAL128_SOURCE.csv |
| 과거정사각형150 | 별도103/44/3; 이전R0/N0/N2 2D | 별도CSV/계약만 | 현재N3·DOPE·ResNet/6D x; 참조검수미완료 | HISTORICAL_SQUARE150_SOURCE.json |
| tab:case frame/fresh/no-pose/인접변화 |8910/8772/138 및8737쌍 | 기존숫자보존 | 모델연속출력; 정확도아님 | 기존 LIFTER_CONTINUITY_SUMMARY + lifter/AUDIT |
| 리프터입력 |12장96상태=67좌표+5가시무좌표+24자체 | 본문설명반영 | 공식참조0·객체대응0 | lifter/AUDIT.classification/formal_reference |
| 리프터 가시코너오차 |x|x 유지| 공식참조·대상대응미확인,5점좌표없음 | lifter/AUDIT.formal_reference |
| 리프터정지변동 |x|x 유지| 사람확인정지구간없음 | lifter/AUDIT |
| 리프터물리 T/R/방향오차 |x|x 유지| 독립물리참조없음 | lifter/AUDIT.independent_physical_pose_reference |
| tab:backbones/절제/비교군/128학생/비용 | 기존동일계약결과검산 | 기존표보존 | 319와128혼합없음; 환경패널 유지 | static회귀검산·기존고정CSV |

`x`는 근거·계약 부족, `NA`는 정의되지 않는 통계, `0`은 실제 측정된 영입니다. 이번원고마감에는 새학습0·새클릭요구0회입니다. 리프터정확도를 별도로 완성할 때에만 현재67점재사용과좌표없는5점·이력·대상대응이 필요합니다. 현재추론8910과사람가시성분류96개는 완료상태로 보존합니다.
'''
    (PATCH / 'PAPER_GAP_MATRIX.md').write_text(text)


def main():
    started = time.perf_counter()
    expected = read(PATCH / 'ORIGINAL_SOURCE_HASHES.json')['files']
    for name, value in expected.items():
        if sha(OLD / name) != value:
            raise ValueError('Original manuscript changed: ' + name)
    # Re-running this document-only builder resets only its owned output copy.
    # Existing immutable source and all annotation/evaluation inputs stay intact.
    shutil.copytree(OLD, PAPER, dirs_exist_ok=True)
    data = read(OUT / 'static/STATIC_REAGGREGATION.json')
    regression = read(OUT / 'static/STATIC_INVARIANCE_CHECK.json')
    read(OUT / 'static/SOURCE_BINDINGS.json')
    read(OUT / 'visibility_square/EXECUTION_VALIDATION.json')
    merged = read(OUT / 'visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
    assert merged['reference_corner_total'] == 3101 and merged['missing_states'] == 0
    counts, square_counts = static_tables(data)
    visibility, square = visibility_and_square_tables()
    static_text(data, counts, square_counts, merged['population_counts'])
    supplementary_text(counts)
    lifter_text(OUT / 'lifter/AUDIT.json')
    review_figure(counts, square_counts, merged)
    preserve_historical_panels()
    readable_manuscript()
    gap_matrix()
    (PAPER / 'COMBINED_UPDATE_README_KO.md').write_text(
        '# 2026-10-06 실제 반영 원고 복사본\n\n'
        '현재LaTeX 입력은 main.tex와 supplement.tex이며 원본paper_updated를 보존한 별도복사본입니다. '
        '새학습·optimizer update·PDF생성·컴파일·push는0회입니다. 이전13쪽+5쪽은 배치의 기준이며 '
        '이번수정본의 실제쪽수는 컴파일하지않아확인하지않았습니다.\n\n'
        '사람입력등급319/119,가시성2499,정사각형602/600을 실제표·본문에 반영했습니다. '
        '등급판정기준은 미확인으로 표시하며 리프터정확도는x를 유지했습니다. '
        '리프터12장96상태입력완료와공식평가승인0·대상대응0을 구분합니다.\n\n'
        '[읽기용최신원고](manuscript_ko.md), [출처셀map](../paper_patch/PAPER_CELL_MAP.json), '
        '[남은x](../paper_patch/PAPER_GAP_MATRIX.md), [검증](../paper_patch/CLOSEOUT_VALIDATION.json)을 확인하세요. '
        '과거13+5쪽 컴파일기록과원래evidence는 역사기록이며이번PDF검증으로사용하지않습니다. '
        'scripts/render_tables.py는과거고정CSV만의변환기로이번최신표를덮어쓸수있으니현재작업에는사용하지않습니다.\n\n'
        '![최신사람입력구성](figures/current_review_summary.png)\n')
    (PAPER / 'README_FIRST_KO.md').write_text((PAPER / 'COMBINED_UPDATE_README_KO.md').read_text())
    (PAPER / 'editorial/PENDING_DATA_AND_EXPERIMENT_CHECKS_KO.md').write_text(
        (PATCH / 'PAPER_GAP_MATRIX.md').read_text())
    copied_csv = PAPER / 'evidence/closeout_tables'
    for source_file in (OUT / 'static').glob('*.csv'):
        shutil.copy2(source_file, copied_csv / source_file.name)
    pointer_checks = []
    loaded = {}
    for entry in CELLS:
        source = ROOT / entry['source']
        if str(source) not in loaded:
            loaded[str(source)] = json.loads(source.read_text())
        value = loaded[str(source)]
        for part in entry['json_pointer'].lstrip('/').split('/'):
            part = part.replace('~1', '/').replace('~0', '~')
            value = value[int(part)] if isinstance(value, list) else value[part]
        if value != entry['raw_value'] or sha(source) != entry['source_sha256']:
            raise ValueError('Source pointer/hash does not match paper value: ' + entry['json_pointer'])
        pointer_checks.append(True)
    write_json(PATCH / 'PAPER_CELL_MAP.json', dict(schema='paper_closeout_cells_20261006_v1',
        cells=CELLS, source_bindings=SOURCES,
        unchanged_tables='Inherited SHA-locked evidence and source CSV retained; headline/cost tables not rewritten.',
        figures=dict(current_review_summary=dict(source=['static/STATIC_REAGGREGATION.json',
            'static/SQUARE_REAGGREGATION.json', 'visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json'],
            kind='descriptive human-input counts; not accuracy'))))
    write_json(PAPER / 'evidence/CLOSEOUT_SOURCE_MANIFEST.json', SOURCES)
    patch_validation = patches()
    unresolved = []
    for file in PAPER.rglob('*.tex'):
        if 'editorial/archive' in file.as_posix():
            continue
        for relative in re.findall(r'\\input\{([^}]+)\}', file.read_text()):
            if not (PAPER / (relative + '.tex')).is_file():
                unresolved.append(dict(file=file.relative_to(PAPER).as_posix(), input=relative))
        for relative in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', file.read_text()):
            if not (PAPER / relative).exists():
                unresolved.append(dict(file=file.relative_to(PAPER).as_posix(), image=relative))
    for name, value in expected.items():
        if sha(OLD / name) != value:
            raise ValueError('Original manuscript changed during closeout: ' + name)
    validation = dict(schema='paper_closeout_validation_20261006_v1',
        original_files_sha256_unchanged=len(expected), new_training=0, optimizer_updates=0,
        new_model_forward_passes=0, pdf_generated=0, tex_compiles=0, pushes=0,
        previous_layout_pages=dict(main=13, supplement=5), current_pages='NA_NOT_COMPILED',
        replaced_numeric_cells_main=sum(c['target'].startswith('tab:') and c['status']=='INSERTED_IN_COPY' for c in CELLS),
        replaced_numeric_cells_supplement=sum(c['target'].startswith('sup:') and c['status']=='INSERTED_IN_COPY' for c in CELLS),
        numeric_cells_standalone_ready=sum(c['status']=='READY_TO_INSERT' for c in CELLS),
        x_cells_preserved=sum(c['status']=='X_REFERENCE_OR_CONTRACT_MISSING' for c in CELLS),
        detailed_fragments='AVAILABLE_AS_SUPPORTING_EVIDENCE; optional additional insertion only; required three-backbone grade claims are in the actual 9-row supplement table',
        source_json_pointer_checks=len(pointer_checks), source_json_pointer_failures=0,
        inserted_text_numeric_bindings=sum(c['target'].startswith('text:') for c in CELLS),
        unresolved_inputs_images=unresolved,
        patch_actual_application=patch_validation, static_regression_receipt=regression,
        figures_created=['figures/current_review_summary.png', 'figures/current_review_summary.svg'],
        status='PASS' if not unresolved else 'FAIL_INPUT_LINKS')
    write_json(PATCH / 'CLOSEOUT_VALIDATION.json', validation)
    previous_receipt_path = OUT / 'final_review/PRE_FINAL_RECEIPTS/EXECUTION_RECEIPT.json'
    if previous_receipt_path.is_file():
        previous = json.loads(previous_receipt_path.read_text())
        elapsed = time.perf_counter() - started
        phase = dict(phase='final_editorial_three_backbone_subgroup_insertion',
            command=[sys.executable, str(Path(__file__).relative_to(ROOT))], wall_seconds=elapsed,
            wall_time_source='time.perf_counter inside executed source builder',exit_code=0)
        receipt = dict(previous)
        receipt.update(generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_script_sha256=sha(Path(__file__)), phases=previous['phases'] + [phase],
            previous_receipt_preserved=str(previous_receipt_path.relative_to(ROOT)),
            previous_receipt_sha256=sha(previous_receipt_path),
            total_executed_script_wall_seconds=previous['total_executed_script_wall_seconds'] + elapsed)
        write_json(PATCH / 'EXECUTION_RECEIPT.json', receipt)
    print(json.dumps({k: v for k,v in validation.items() if k not in ('static_regression_receipt',)},ensure_ascii=False))


if __name__ == '__main__':
    main()
