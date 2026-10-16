from features44 import *
from report44 import TITLE
from deliver37 import Tables
from urllib.parse import urlsplit, unquote
import re, subprocess, time


def main():
    assert read(RUN/'AUDIT.json')['status'] == 'PASS'
    for key, name in [('summary', 'SUMMARY.json')]:
        assert read(RUN/'REPORT_COMPLETE.json')[key+'_sha256'] == sha(RUN/name)
    for name, digest in read(RUN/'PRIOR_BINDINGS.json')['files'].items():
        assert sha(ROOT/name) == digest, name
    url = (RUN/'index.html').as_uri()
    dom = RUN/'rendered_dom.html'
    with dom.open('w') as out, Path('/tmp/pallet-v44-browser.log').open('w') as err:
        result = subprocess.run(['google-chrome', '--headless', '--no-sandbox', '--disable-gpu', '--disable-background-networking', '--no-first-run', '--no-default-browser-check', '--user-data-dir=/tmp/pallet-v44-profile', '--window-size=1500,1250', '--screenshot='+str(RUN/'browser_preview.png'), '--dump-dom', url], stdout=out, stderr=err, timeout=40)
    assert result.returncode == 0
    tables = Tables(); tables.feed(dom.read_text())
    assert len(tables.tables) == 2
    summary = read(RUN/'SUMMARY.json')
    data = read(RUN/'FIGURE_DATA.json')
    results = summary['results']['check/all/all_detected']
    expected = ['모델','평균px','시드별px','V34 대비 개선율%','차이95%구간px']
    for row in data['rows']:
        original=results[row['method']]
        assert row['mean_px']==original['mean_px'] and row['per_seed']==original['per_seed_px'] and row['gain']==original['gain_percent'] and row['ci']==original['difference']['ci95']
        expected += [row['label'],f'{row["mean_px"]:.3f}',', '.join(f'{x:.3f}' for x in row['per_seed']),', '.join(f'{x:+.2f}' for x in row['gain']),', '.join(f'{x:+.3f}' for x in row['ci'])]
    assert tables.tables[0] == expected
    expected = ['새 평가 조건','V34','고정 V41','새 점 특징','새 점+선 특징']
    for row in data['conditions']:
        for m,z in row['values'].items():assert z==summary['results'][row['key']][m]['mean_px']
        expected += [row['key'].removeprefix('check/')]+[f'{row["values"][m]:.3f}' for m in ('anchor','frozen_v41','fresh_point','fresh_line')]
    assert tables.tables[1] == expected
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
    write(RUN/'COMPLETION.json', dict(complete=True, new_hgb_models=24, boosting_rounds=1440, audit_refits=4, neural_updates=0, html_open_confirmed=visible, accepted=summary['gates']['parent_unseen']['passed'], decision=summary['decision'], gates=summary['gates'], scope='New24scene groups same2CAD,12parentunseenfit4cal8finalcheck;600requested432detected,final200requested137detected1096points. FixedV41smallmean effect reproduced butfullstabilityfails. Newline correction improvesanchor overall butincrementovermatchedpoint CI includeszero andeasycrop harm persists. Close currentfrozenfeaturecorrection researchfamily. Newsource supervision isnot same-dataOOF, origin comparison confoundedbynewscene distribution. No neuraltraining/newreal/manualGTchanges;processlocalGPUlibrary workaround only.', files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('DELIVERY44', visible, flush=True)

if __name__ == '__main__': main()
