from context42 import *
from report42 import TITLE
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
    with dom.open('w') as out, Path('/tmp/pallet-v42-browser.log').open('w') as err:
        result = subprocess.run(['google-chrome', '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--no-default-browser-check', '--user-data-dir=/tmp/pallet-v42-profile', '--window-size=1500,1250', '--screenshot='+str(RUN/'browser_preview.png'), '--dump-dom', url], stdout=out, stderr=err, timeout=40)
    assert result.returncode == 0
    tables = Tables(); tables.feed(dom.read_text())
    assert len(tables.tables) == 3
    summary = read(RUN/'SUMMARY.json')
    data = read(RUN/'FIGURE_DATA.json')
    probe = read(RUN/'PROBE.json')
    results = summary['results']['check/all/all_detected']
    expected = ['영상+기하 분류 목표','AUROC: 시드별','Precision: 시드별','Recall: 시드별']
    for row in data['classification']:
        for key in ('auroc','precision','recall'):
            assert row[key] == [summary['classification'][str(k)][row['target']+'_visual']['check'][key] for k in SEEDS]
        expected += [row['label'], ', '.join(f'{x:.3f}' for x in row['auroc']), ', '.join(f'{100*x:.1f}%' for x in row['precision']), ', '.join(f'{100*x:.1f}%' for x in row['recall'])]
    assert tables.tables[0] == expected
    expected = ['유효성 입력·목표','후보 전 선별','후보 전 선별+점 어려움','기존 선택 후 거절+점 어려움']
    for row in data['table']:
        for key in ROUTES:
            assert row[key] == results[row['method']+'_'+key]['mean_px']
        expected += [row['label']] + [f'{row[k]:.3f}' for k in ROUTES]
    assert tables.tables[1] == expected
    expected = ['대조','전체 점 오차px']
    values = dict(fallback=results['fallback']['mean_px'], primary=results['constraint_visual_pre_all']['mean_px'], GT_labels=results['GT_constraint_pre_all']['mean_px'], GT_valid_best=probe['oracle_results']['constraint']['check/all/all_detected']['mean_px'], accepted_best=probe['oracle_results']['accepted']['check/all/all_detected']['mean_px'], all_best=results['GT_candidate']['mean_px'])
    for row in data['oracle']:
        assert row['value'] == values[row['key']]
        expected += [row['label'],f'{row["value"]:.3f}']
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
    write(RUN/'COMPLETION.json', dict(complete=True, new_hgb_models=18, boosting_rounds=1080, audit_refits=3, neural_updates=0, html_open_confirmed=visible, accepted=summary['gates']['accepted'], gates=summary['gates'], scope='Reused synthetic frozen V41 proposals/fallback/residual and V39 hardness. 12fit/4calibration/8check groups, same2CAD, upstream/offset old16 reuse. Point-specific constraint precision transfers in this diagnostic, but incremental point gain fails. Post-primary candidate coverage/ranking probe has no new fits or policy tuning. No new neural training, real evaluation, renders or manualGT edits.', files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DELIVERY42', visible, flush=True)

if __name__ == '__main__': main()
