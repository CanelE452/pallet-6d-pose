from context39 import *
from report39 import TITLE, ci_text
from deliver37 import Tables
from urllib.parse import urlsplit, unquote
import re, subprocess, time


def main():
    assert read(RUN/'AUDIT.json')['status'] == 'PASS'
    for key, name in [('summary', 'SUMMARY.json'), ('phenotypes', 'PHENOTYPES.json')]:
        assert read(RUN/'REPORT_COMPLETE.json')[key+'_sha256'] == sha(RUN/name)
    for name, digest in read(RUN/'PRIOR_BINDINGS.json')['files'].items():
        assert sha(ROOT/name) == digest, name
    url = (RUN/'index.html').as_uri()
    dom = RUN/'rendered_dom.html'
    with dom.open('w') as out, Path('/tmp/pallet-v39-browser.log').open('w') as err:
        result = subprocess.run(['google-chrome', '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--no-default-browser-check', '--user-data-dir=/tmp/pallet-v39-profile', '--window-size=1500,1250', '--screenshot='+str(RUN/'browser_preview.png'), '--dump-dom', url], stdout=out, stderr=err, timeout=40)
    assert result.returncode == 0
    tables = Tables(); tables.feed(dom.read_text())
    assert len(tables.tables) == 3
    summary = read(RUN/'SUMMARY.json')
    hg = [summary['hardness'][f'both_seed{s}']['check']['line'] for s in SEEDS]
    expected = ['점+선 특징 분류기', '시드1', '시드2', '시드3']
    for label, field, fmt in [('실제 큰 오류 점 수 /1040', 'hard_points', lambda x: str(x)), ('AUROC', 'auroc', lambda x: f'{x:.3f}'), ('Precision', 'precision', lambda x: f'{100*x:.1f}%'), ('Recall', 'recall', lambda x: f'{100*x:.1f}%')]:
        expected += [label]+[fmt(q[field]) for q in hg]
    assert tables.tables[0] == expected
    expected = ['법선 보정량 예측', '학습 큰 오류 MAEpx', '확인 큰 오류 MAEpx']
    for key, label in [('standard', '기존 V38 중심'), ('balanced', '큰 오류 비중 증가'), ('hard_geometry', '큰 오류 전용·영상+기하'), ('hard_appearance', '큰 오류 전용·영상+법선')]:
        expected += [label]+[f'{summary["direction"][split+"/all/hard"]["both"][key]["mean_px"]:.3f}' for split in ('fit', 'check')]
    assert tables.tables[1] == expected
    expected = ['방법', '전체1040점 오차px', '시드별 개선율%', '기준 대비 차이95%구간px']
    for row in read(RUN/'FIGURE_DATA.json')['table']:
        expected += [row['label'], f'{row["mean_px"]:.3f}', ', '.join(f'{x:+.2f}' for x in row['gains']), ci_text(row['ci'])]
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
    write(RUN/'COMPLETION.json', dict(complete=True, new_hgb_models=48, boosting_rounds=2880, audit_refits=3, neural_updates=0, html_open_confirmed=visible, accepted=summary['gates']['accepted'], gates=summary['gates'], scope='Reused synthetic V36/V34 frozen features, 12fit/4calibration/8check groups, same2CAD. Hardness ranking transfers partly; hard correction direction does not. No new real evaluation or manualGT edits.', files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DELIVERY39', visible, flush=True)

if __name__ == '__main__': main()
