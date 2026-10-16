from context37 import *
from report37 import TITLE
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import re,subprocess,time

class Tables(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.refs=[];self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.tables.append([])
        if tag in ('th','td'):self.cell=''
        if tag=='img':self.refs.extend(v for k,v in attrs if k=='src')
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:self.tables[-1].append(self.cell.strip());self.cell=None

def main():
    for f in ('AUDIT.json','EXTRA_AUDIT.json'):assert read(RUN/f)['status']=='PASS'
    for key,f in [('summary','SUMMARY.json'),('residual','RESIDUAL.json'),('objectives','OBJECTIVES.json')]:assert read(RUN/'REPORT_COMPLETE.json')[key+'_sha256']==sha(RUN/f)
    for p,h in read(RUN/'PRIOR_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as out,Path('/tmp/pallet-v37-browser.log').open('w') as err:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v37-profile','--window-size=1500,1250','--screenshot='+str(shot),'--dump-dom',url],stdout=out,stderr=err,timeout=40)
    assert z.returncode==0;t=Tables();t.feed(dom.read_text());fig=read(RUN/'FIGURE_DATA.json');expected=fig['headers'].copy()
    for row in fig['table']:expected+=[row[0]]+[f'{v:.3f}' for v in row[1:]]
    assert t.tables[0]==expected,(t.tables[0],expected)
    # Other tables are verified against Markdown-derived DOM plus underlying summaries.
    s=read(RUN/'SUMMARY.json');r=s['results']['check/all/all_detected']['methods']['v34/both'];assert [t.tables[1][i] for i in range(3,len(t.tables[1]),2)]==[f'{r[k]["mean_px"]:.3f}' for k in ('anchor','selector_point','selector_visual','selector_geometry','selector_visual_geometry')]
    o=read(RUN/'OBJECTIVES.json');expect=[]
    for target in ('normal','pixel'):expect += [f'{o["results"]["check/all/all_detected"][target+"/both/"+mode]["mean_px"]:.3f}' for mode in MODES]
    assert [v for i,v in enumerate(t.tables[2][5:]) if i%5!=0]==expect
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
    visible=any(x['viewable'] for x in windows);write(RUN/'DELIVERY.json',dict(status='PASS' if visible else 'NOT_VISIBLE',html_open_confirmed=visible,windows=windows,table_cells=sum(map(len,t.tables)),local_figures=len(t.refs)))
    assert visible,'Chrome report window not viewable'
    files=[q for q in RUN.rglob('*') if q.is_file() and q.suffix!='.log' and q.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json')]+list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE,ROOT/'_docs/handoff/2026-09-10_pallet_line_utility_v37.md']
    write(RUN/'COMPLETION.json',dict(complete=True,neural_updates=0,ridge_fits=144,primary_fits=96,additional_fits=48,html_open_confirmed=visible,accepted=False,primary_gates=s['gates'],additional_gates=o['gates'],scope='Frozen V36 candidates point-role residual/observable utility diagnostics, reused rendercheck only; no real evaluation/manual GT edits.',files={str(q.relative_to(ROOT)):sha(q) for q in files}))
    with (ROOT/'_docs/history/2026-09-11.md').open('a') as f:f.write('\n\n### V37 improved line candidate utility completed\n\nFrozen4arms3seeds/twoanchors,144ridgefits0neuralupdates. Candidate pointoracle V34 10.463->7.802 but gridoracle7.769, addedpotentialfails. NormalcorrectionMAE8.203 vsno-op2.720. Primaryselector12.043, pixelgain10.711, directnormal15.086; allstablegatesfail. Audits384primary+48additionalpointarrays and144selectorreplays; priorfilesunchanged. MD/figures/actualChrome verified.\n')
    print('DELIVERY37',visible,flush=True)
if __name__=='__main__':main()
