from context40 import *
from report40 import TITLE, ci_text
from deliver37 import Tables
from urllib.parse import urlsplit, unquote
import re, subprocess, time


def main():
    assert read(RUN/'AUDIT.json')['status'] == 'PASS'
    assert read(RUN/'PROBE_AUDIT.json')['status'] == 'PASS'
    for key, name in [('summary', 'SUMMARY.json'), ('probe', 'PROBE.json')]:
        assert read(RUN/'REPORT_COMPLETE.json')[key+'_sha256'] == sha(RUN/name)
    for name, digest in read(RUN/'PRIOR_BINDINGS.json')['files'].items():
        assert sha(ROOT/name) == digest, name
    url = (RUN/'index.html').as_uri()
    dom = RUN/'rendered_dom.html'
    with dom.open('w') as out, Path('/tmp/pallet-v40-browser.log').open('w') as err:
        result = subprocess.run(['google-chrome', '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--no-default-browser-check', '--user-data-dir=/tmp/pallet-v40-profile', '--window-size=1500,1250', '--screenshot='+str(RUN/'browser_preview.png'), '--dump-dom', url], stdout=out, stderr=err, timeout=40)
    assert result.returncode == 0
    tables = Tables(); tables.feed(dom.read_text())
    assert len(tables.tables) == 3
    summary = read(RUN/'SUMMARY.json')
    data = read(RUN/'FIGURE_DATA.json')
    expected = ['입력 정보', '큰 오류 보정 MAE: 전체 학습', '큰 오류 보정 MAE: 전용 학습']
    for row in data['table']:
        source=row['source']
        assert abs(row['normal_all']-summary['direction']['check/all/hard']['all'][source]['mean_px'])<1e-12
        assert abs(row['normal_hard']-summary['direction']['check/all/hard']['hard'][source]['mean_px'])<1e-12
        expected += [row['label'],f'{row["normal_all"]:.3f}',f'{row["normal_hard"]:.3f}']
    assert tables.tables[0] == expected
    expected = ['입력 정보','전체 점 오차px','시드별 개선율%','기준 대비 차이95%구간px']
    for row in data['table']:
        method='standard' if row['source']=='baseline' else row['source']+'_all'
        original=summary['results']['check/all/all_detected'][method]
        assert row['point']==original['mean_px'] and row['gains']==original['gain_percent'] and row['ci']==original['difference']['ci95']
        expected += [row['label'],f'{row["point"]:.3f}',', '.join(f'{x:+.2f}' for x in row['gains']),ci_text(row['ci'])]
    assert tables.tables[1] == expected
    expected = ['조건','기존 점','실제 영상 선분','정확한 물리 직선 [진단]','정확한 가시 구간 [진단]']
    for row in data['conditions']:
        original=summary['results']['check/'+row['condition']]
        for key,method in [('anchor','anchor'),('pred','pred_segment_all'),('GTline','GT_line_all'),('GTsegment','GT_segment_all')]:assert row[key]==original[method]['mean_px']
        expected += [row['condition']]+[f'{row[k]:.3f}' for k in ('anchor','pred','GTline','GTsegment')]
    assert tables.tables[2] == expected
    for base, refs in [(RUN, tables.refs), (NOTE.parent, re.findall(r'\]\(([^)]+)\)', NOTE.read_text()))]:
        for ref in refs:
            u = urlsplit(ref)
            if not u.scheme and u.path: assert (base/unquote(u.path)).exists(), ref
    subprocess.run(['google-chrome', '--new-window', url], capture_output=True, text=True, timeout=20)
    windows = []
    for _ in range(10):
        clients = subprocess.run(['xprop', '-root', '_NET_CLIENT_LIST'], capture_output=True, text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+', clients):
            title = subprocess.run(['xprop', '-id', wid, '_NET_WM_NAME'], capture_output=True, text=True).stdout
            if TITLE in title:
                info = subprocess.run(['xwininfo', '-id', wid], capture_output=True, text=True).stdout
                windows.append(dict(id=wid, title=title.strip(), viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows): break
        time.sleep(.5)
    visible = any(w['viewable'] for w in windows)
    write(RUN/'DELIVERY.json', dict(status='PASS' if visible else 'NOT_VISIBLE', html_open_confirmed=visible, windows=windows, table_cells=sum(map(len, tables.tables)), local_figures=len(tables.refs)))
    assert visible, 'Chrome report window not viewable'
    files = [p for p in RUN.rglob('*') if p.is_file() and p.suffix != '.log' and p.name not in ('COMPLETION.json', 'DISCORD_NOTIFICATION.json')] + list(KIT.glob('*.py')) + list(KIT.glob('*.md')) + [NOTE, HANDOFF]
    write(RUN/'COMPLETION.json', dict(complete=True, new_hgb_models=30, boosting_rounds=1800, audit_refits=3, neural_updates=0, html_open_confirmed=visible, accepted=summary['gates']['accepted'], gates=summary['gates'], scope='Reused synthetic V36/V34 frozen features, 12fit/4calibration/8check groups, same2CAD. Actual image segment/endpoint feature additions fail stable improvement. GTphysical line provides partial information; visible extent not added benefit. No new real evaluation or manualGT edits.', files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DELIVERY40', visible, flush=True)

if __name__ == '__main__': main()
