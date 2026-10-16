from context43 import *
from report43 import TITLE
from deliver37 import Tables
from urllib.parse import urlsplit, unquote
import re, subprocess, time


def main():
    assert read(RUN/'AUDIT.json')['status'] == 'PASS'
    assert read(RUN/'DIAGNOSTIC_AUDIT.json')['status'] == 'PASS'
    for key, name in [('summary', 'SUMMARY.json'), ('curves', 'CURVES.json')]:
        assert read(RUN/'REPORT_COMPLETE.json')[key+'_sha256'] == sha(RUN/name)
    for name, digest in read(RUN/'PRIOR_BINDINGS.json')['files'].items():
        assert sha(ROOT/name) == digest, name
    url = (RUN/'index.html').as_uri()
    dom = RUN/'rendered_dom.html'
    with dom.open('w') as out, Path('/tmp/pallet-v43-browser.log').open('w') as err:
        result = subprocess.run(['google-chrome', '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--no-default-browser-check', '--user-data-dir=/tmp/pallet-v43-profile', '--window-size=1500,1250', '--screenshot='+str(RUN/'browser_preview.png'), '--dump-dom', url], stdout=out, stderr=err, timeout=40)
    assert result.returncode == 0
    tables = Tables(); tables.feed(dom.read_text())
    assert len(tables.tables) == 3
    summary = read(RUN/'SUMMARY.json')
    data = read(RUN/'FIGURE_DATA.json')
    curves = read(RUN/'CURVES.json')
    results = summary['results']['check/all/all_detected']
    expected = ['학습 목표·입력','Fit AP: 시드별','Calibration AP: 시드별','Check AP: 시드별']
    for row in data['classification']:
        key=row['arm'] if row['arm']=='V42' else row['arm']+'_probability'
        for split in ('fit','calibration','check'):
            assert row[split] == [summary['diagnostics'][str(seed)][key][split]['useful_AP'] for seed in SEEDS]
        expected += [row['label']] + [', '.join(f'{x:.3f}' for x in row[k]) for k in ('fit','calibration','check')]
    assert tables.tables[0] == expected
    expected = ['학습 목표·입력','기존 잔차 순위: px','분류 점수 순위: px']
    for row in data['policies']:
        for route in ROUTES:assert row[route] == results[row['arm']+'_'+route]['mean_px']
        expected += [row['label']]+[f'{row[k]:.4f}' for k in ROUTES]
    assert tables.tables[1] == expected
    expected = ['고정 문턱 [진단]','평균 오차px','개선 기회 보존: 시드별','1px 초과 개선: 시드별','1px 초과 악화: 시드별']
    for row in data['fixed']:
        raw=[next(x for x in curves['rows'][str(seed)][PRIMARY]['check'] if x['threshold']==row['threshold']) for seed in SEEDS]
        assert row['mean_px']==float(np.mean([x['mean_px'] for x in raw]))
        for key,source in [('covered','covered_points'),('help','help_over1px'),('harm','harm_over1px')]:assert row[key]==[x[source] for x in raw]
        expected += [f'{row["threshold"]:.2f}',f'{row["mean_px"]:.4f}',str(row['covered']),str(row['help']),str(row['harm'])]
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
    write(RUN/'COMPLETION.json', dict(complete=True, new_hgb_models=15, boosting_rounds=900, audit_refits=2, neural_updates=0, html_open_confirmed=visible, accepted=summary['gates']['accepted'], gates=summary['gates'], scope='Reused synthetic V41fixed3proposals/directfallback. 12fit4calibration8check same2CAD, upstreamold16 reuse. Useful supervision improves candidate utility ranking but registered primary abstains in all3seeds. Fixedgrid diagnosis shows coverage versus harmful correction tradeoff; no bestcheck selection or adopted architecture. No newreal/neural/render/manualGT updates.', files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DELIVERY43', visible, flush=True)

if __name__ == '__main__': main()
