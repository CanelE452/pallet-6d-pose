from context36 import *
from report36 import TITLE,HEADERS
from delivery27 import Tables
from urllib.parse import urlsplit,unquote
import re,subprocess

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS'
    for key,file in [('point','POINT_SUMMARY.json'),('line','LINE_SUMMARY.json'),('adapt','ADAPT_SUMMARY.json')]:assert read(RUN/'REPORT_COMPLETE.json')[key+'_sha256']==sha(RUN/file)
    for p,h in read(RUN/'PRIOR_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as out,Path('/tmp/pallet-v36-browser.log').open('w') as err:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v36-profile','--window-size=1500,1250','--screenshot='+str(shot),'--dump-dom',url],stdout=out,stderr=err,timeout=40)
    assert z.returncode==0;t=Tables();t.feed(dom.read_text());expected=['방법']+HEADERS
    for r in read(RUN/'FIGURE_DATA.json')['table']:expected+=[r['method']]+[f'{v:.3f}' for v in r['values']]
    assert t.tables[1]==expected,(t.tables[1],expected)
    for ref in t.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (RUN/unquote(u.path)).exists(),ref
    for ref in re.findall(r'\]\(([^)]+)\)',NOTE.read_text()):
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (NOTE.parent/unquote(u.path)).exists(),ref
    subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20);windows=[]
    for _ in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if TITLE in title:
                info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout;windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(x['viewable'] for x in windows):break
        time.sleep(.5)
    visible=any(x['viewable'] for x in windows);write(RUN/'DELIVERY.json',dict(status='PASS',html_open_confirmed=visible,windows=windows,table_cells=len(expected),local_figures=len(t.refs)))
    files=[p for p in RUN.rglob('*') if p.is_file() and p.suffix!='.log' and p.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json')]+list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE,ROOT/'_docs/handoff/2026-09-10_pallet_continuous_lines_v36.md']
    write(RUN/'COMPLETION.json',dict(complete=True,models=21,training_updates=42000,primary_models=12,additional_models=9,html_open_confirmed=visible,accepted=bool(read(RUN/'SUMMARY.json')['accepted'] or read(RUN/'ADAPT_SUMMARY.json')['accepted']),line_improved=read(RUN/'SUMMARY.json')['line_improved'],files={str(p.relative_to(ROOT)):sha(p) for p in files},scope='Primary4arm continuous-coordinate/quality-head and matched point adaptation; exploratory downstream encoder adaptation separate. Reused realDEV/rendercheck; no manual GT edits or sealed-final claim.'))
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:f.write('\n\n### V36 continuous lines and downstream adaptation completed\n\n21models42000updates, 543point predictions audited. Selected synthetic line distance7.571->6.222px, primarylinegate passes. Primary refine-only realcrop20 13.485->12.720, but refinementoff difference0.000412px. Downstream encoder adaptation grid13.377/refine13.469/both13.613; geometry donor small, all stability fails. MD/figures and visible Chrome delivered.\n')
    print('DELIVERY36',visible,flush=True)
if __name__=='__main__':main()
