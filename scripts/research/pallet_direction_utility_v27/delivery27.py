"""Verify rendered table values/local links, then open visible report window."""
from context27 import *
from report27 import NAMES
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import subprocess,re

class Tables(HTMLParser):
    def __init__(self):super().__init__();self.tables=[];self.current=None;self.cell=None;self.refs=[]
    def handle_starttag(self,tag,attrs):
        if tag=='table':self.current=[]
        if tag in ('th','td') and self.current is not None:self.cell=''
        for k,v in attrs:
            if k in ('href','src'):self.refs.append(v)
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in ('th','td') and self.cell is not None:self.current.append(self.cell);self.cell=None
        if tag=='table' and self.current is not None:self.tables.append(self.current);self.current=None

def main():
    assert read(RUN/'REPORT_COMPLETE.json')['complete'];url=(RUN/'index.html').as_uri();dom=Path('/tmp/pallet-v27-dom.html');shot=RUN/'browser_preview.png'
    with dom.open('w') as f,Path('/tmp/pallet-v27-browser.log').open('w') as e:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v27-render-profile','--window-size=1500,1300','--screenshot='+str(shot),'--dump-dom',url],stdout=f,stderr=e,timeout=40)
    assert z.returncode==0;p=Tables();p.feed(dom.read_text());d=read(RUN/'FIGURE_DATA.json');expected=['방법','clean','코너 가림','잘림10','잘림20','잘림30','잘림40']
    for row in d['table']:expected+=[NAMES[row['arm']]]+[f'{v:.3f}' for v in row['values']]
    assert p.tables[1]==expected,(p.tables[1],expected)
    for ref in p.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (RUN/unquote(u.path)).exists(),ref
    import importlib.util
    spec=importlib.util.spec_from_file_location('notes_paths',ROOT/'scripts/docs/organize_research_notes.py');paths=importlib.util.module_from_spec(spec);spec.loader.exec_module(paths);documents=[ROOT/'_docs/README.md',*sorted((ROOT/'_docs/notes').rglob('*.md')),*sorted(KIT.glob('*.md'))];links=0
    for document in documents:
        for ref in paths.targets(document.read_text()):
            target=paths.local_target(ref,document.parent)
            if target is not None:assert target.exists(),(document,ref);links+=1
    write(RUN/'DOCS_CHECK.json',dict(status='PASS',markdown_files=len(documents),local_links=links,broken=[]));write(RUN/'DELIVERY_CHECK.json',dict(status='PASS',html_sha256=sha(RUN/'index.html'),performance_table_cells=len(expected),tables=len(p.tables),local_links_checked=len(p.refs),screenshot=str(shot.relative_to(ROOT))))
    z=subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20);windows=[]
    for attempt in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if 'V27 새 구조 학습·안정성 검증' not in title:continue
            info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout;windows.append(dict(window_id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    state='OPEN_CONFIRMED' if any(w['viewable'] for w in windows) else 'OPEN_NOT_CONFIRMED';write(RUN/'BROWSER_OPEN.json',dict(status=state,windows=windows,open_returncode=z.returncode,html_sha256=sha(RUN/'index.html')));print('DELIVERY27',state,flush=True)

if __name__=='__main__':main()
