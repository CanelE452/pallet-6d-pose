from context41 import *
from report41 import TITLE, ci_text
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
    with dom.open('w') as out, Path('/tmp/pallet-v41-browser.log').open('w') as err:
        result = subprocess.run(['google-chrome', '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--no-default-browser-check', '--user-data-dir=/tmp/pallet-v41-profile', '--window-size=1500,1250', '--screenshot='+str(RUN/'browser_preview.png'), '--dump-dom', url], stdout=out, stderr=err, timeout=40)
    assert result.returncode == 0
    tables = Tables(); tables.feed(dom.read_text())
    assert len(tables.tables) == 3
    summary = read(RUN/'SUMMARY.json')
    data = read(RUN/'FIGURE_DATA.json')
    expected = ['입력/방향','직접 회귀','계산만','계산+잔차','계산+잔차+기존 선택기']
    for row in data['table']:
        prefix=row['source']+'_'+row['family'];original=summary['results']['check/all/all_detected']
        for key,method in [('direct','direct'),('analytic','analytic'),('residual','residual'),('gate','residual_gate')]:assert row[key]==original[prefix+'_'+method]['mean_px']
        expected += [row['label']]+[f'{row[k]:.3f}' for k in ('direct','analytic','residual','gate')]
    assert tables.tables[0] == expected
    expected = ['물리선 입력','큰 오류 역할 수: 시드별','직접 회귀','계산만','계산+잔차']
    for row in data['normal']:
        original=summary['direction']['check/all/hard'][row['source']+'_physical3']
        for key in ('direct','analytic','residual'):assert row[key]==original[key]['mean_px']
        assert row['counts']==original['direct']['per_seed_points']
        expected += [row['label'],str(row['counts'])]+[f'{row[k]:.3f}' for k in ('direct','analytic','residual')]
    assert tables.tables[1] == expected
    expected = ['조건','기존 점','실제 직접 회귀','실제 잔차+선택','GT선 잔차 [진단]']
    for row in data['conditions']:
        original=summary['results']['check/'+row['condition']]
        for key,method in [('anchor','anchor'),('direct','pred_segment_physical3_direct'),('gate','pred_segment_physical3_residual_gate'),('GT','GT_line_physical3_residual')]:assert row[key]==original[method]['mean_px']
        expected += [row['condition']]+[f'{row[k]:.3f}' for k in ('anchor','direct','gate','GT')]
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
    write(RUN/'COMPLETION.json', dict(complete=True, new_hgb_models=27, boosting_rounds=1620, audit_refits=3, neural_updates=0, html_open_confirmed=visible, accepted=summary['gates']['accepted'], gates=summary['gates'], scope='Reused synthetic V36/V34 frozen features, 12fit/4calibration/8check groups, same2CAD. Physical-constraint residual succeeds conditionally with GT lines, fails actual input primary. Actualphysical direct has small consistent reusedcheck gains, not an independent accepted architecture. No new real evaluation or manualGT edits.', files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DELIVERY41', visible, flush=True)

if __name__ == '__main__': main()
