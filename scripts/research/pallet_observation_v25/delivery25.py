"""Check rendered local report and confirm a visible desktop report window."""
from context25 import *
from report25 import NAMES,mean
from prepare25 import ARMS
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import subprocess,re

class Tables(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.current=None;self.cell=None;self.refs=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='table':self.current=[]
        if tag in ('th','td') and self.current is not None:self.cell=''
        for k in ('href','src'):
            if k in a:self.refs.append(a[k])
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('th','td') and self.cell is not None:self.current.append(self.cell);self.cell=None
        if tag=='table' and self.current is not None:self.tables.append(self.current);self.current=None

def main():
    assert read(RUN/'REPORT_COMPLETE.json')['complete'];assert read(RUN/'PROVENANCE_CHECK.json')['status']=='PASS'
    url=(RUN/'index.html').as_uri();dom=Path('/tmp/pallet-v25-dom.html');log=Path('/tmp/pallet-v25-browser.log');shot='/tmp/pallet-v25-report.png'
    with dom.open('w') as f,log.open('w') as e:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v25-render-profile','--window-size=1500,1200','--screenshot='+shot,'--dump-dom',url],stdout=f,stderr=e,timeout=40)
    assert z.returncode==0;p=Tables();p.feed(dom.read_text());s=read(RUN/'SUMMARY.json');expect=['방법','코너 가림','잘림20%','잘림40%','전체 안정성']
    for arm in ('stock','base','line_vector')+ARMS:expect.extend([NAMES[arm],f'{mean(s,arm,"corner20","hidden"):.3f}',f'{mean(s,arm,"crop20","outside"):.3f}',f'{mean(s,arm,"crop40","outside"):.3f}','통과' if arm in s['stable_arms'] else '미통과' if arm in ARMS else '대조군'])
    assert p.tables[0]==expect,(p.tables[0],expect)
    for ref in p.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (RUN/unquote(u.path)).exists(),ref
    import importlib.util
    spec=importlib.util.spec_from_file_location('notes_paths',ROOT/'scripts/docs/organize_research_notes.py');paths=importlib.util.module_from_spec(spec);spec.loader.exec_module(paths)
    documents=[ROOT/'_docs/README.md',*sorted((ROOT/'_docs/notes').rglob('*.md')),*sorted(KIT.glob('*.md'))];links=0
    for document in documents:
        for ref in paths.targets(document.read_text()):
            target=paths.local_target(ref,document.parent)
            if target is not None:assert target.exists(),(document,ref);links+=1
    write(RUN/'DOCS_CHECK.json',dict(status='PASS',markdown_files=len(documents),local_links=links,broken=[]))
    write(RUN/'DELIVERY_CHECK.json',dict(status='PASS',html_sha256=sha(RUN/'index.html'),rendered_table_cells=len(expect),tables=len(p.tables),local_links_checked=len(p.refs),screenshot=shot))
    z=subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20);windows=[]
    for attempt in range(10):
        text0=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',text0):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if 'V25 관측 조건과 보정 방향' not in title:continue
            info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout;windows.append(dict(window_id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    state='OPEN_CONFIRMED' if any(w['viewable'] for w in windows) else 'OPEN_NOT_CONFIRMED';write(RUN/'BROWSER_OPEN.json',dict(status=state,windows=windows,open_returncode=z.returncode,html_sha256=sha(RUN/'index.html')));print('DELIVERY25',state,flush=True)

if __name__=='__main__':main()
