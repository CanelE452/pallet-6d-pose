"""Verify the generated report, open its visible desktop window, bind artifacts."""
from common import *
import subprocess,re,time
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote

class Parser(HTMLParser):
    def __init__(self):super().__init__();self.active=False;self.buf='';self.cells=[];self.refs=[];self.options=0
    def handle_starttag(self,tag,attrs):
        if tag in ['td','th']:self.active=True;self.buf=''
        if tag=='option':self.options+=1
        for k,v in attrs:
            if k in ['src','href']:self.refs.append(v)
    def handle_data(self,data):
        if self.active:self.buf+=data
    def handle_endtag(self,tag):
        if tag in ['td','th']:self.cells.append(self.buf.strip());self.active=False

def main():
    assert read(OUT/'FINAL_AUDIT.json')['PASS'];report=read(OUT/'REPORT_COMPLETE.json')
    assert report['summary_sha256']==sha(OUT/'foreground_matched'/'SUMMARY.json')
    url=(OUT/'index.html').as_uri()
    with (OUT/'browser.log').open('w') as err,(OUT/'rendered_dom.html').open('w') as dom:
        p=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking',
            '--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-unseen-report-chrome',
            '--window-size=1500,1350','--screenshot='+str(OUT/'browser_preview.png'),'--dump-dom',url],stdout=dom,stderr=err,timeout=40)
    assert p.returncode==0
    parser=Parser();parser.feed((OUT/'rendered_dom.html').read_text())
    assert parser.cells[6:36]==report['table_cells'];assert parser.options==3
    for ref in parser.refs:
        u=urlsplit(ref)
        if u.scheme=='file':assert Path(unquote(u.path)).exists()
        elif not u.scheme and u.path:assert (OUT/unquote(u.path)).exists()
    subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20)
    windows=[]
    for _ in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if report['title'] in title:
                info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout
                windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    visible=any(w['viewable'] for w in windows)
    write(OUT/'DELIVERY.json',dict(html_open_confirmed=visible,windows=windows,table_verified=True,gallery_options=parser.options))
    assert visible,'Report exists, desktop visibility not confirmed'
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['COMPLETION.json','DISCORD_NOTIFICATION.json'] and p.suffix!='.log']
    files+=list(KIT.glob('*.py'))+[KIT/'README.md',ROOT/report['note']]
    write(OUT/'COMPLETION.json',dict(complete=True,models=21,optimizer_steps=25200,observations=1728,
        independent_replay_pass=True,html_open_confirmed=True,decision=read(OUT/'foreground_matched'/'SUMMARY.json')['decision'],
        real_evaluation=False,semantic_rotation_evaluation=False,unfrozen_backbone=False,
        files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('Report table and gallery verified; Chrome visible; completion bound')
if __name__=='__main__':main()
