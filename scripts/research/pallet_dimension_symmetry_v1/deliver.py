"""Verify browser rendering, open the requested report, and bind completion evidence."""
import re,subprocess,time
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from core import *

class Parser(HTMLParser):
    def __init__(self):super().__init__();self.cells=[];self.refs=[];self.active=False;self.buf=''
    def handle_starttag(self,tag,attrs):
        if tag in ['td','th']:self.active=True;self.buf=''
        for k,v in attrs:
            if k in ['src','href']:self.refs.append(v)
    def handle_data(self,data):
        if self.active:self.buf+=data
    def handle_endtag(self,tag):
        if tag in ['td','th']:self.cells.append(self.buf.strip());self.active=False

def main():
    assert read(OUT/'AUDIT.json')['PASS']
    report=read(OUT/'REPORT_COMPLETE.json');assert sha(OUT/'SUMMARY.json')==report['summary_sha256']
    url=(OUT/'index.html').as_uri()
    with (OUT/'browser.log').open('w') as err,(OUT/'rendered_dom.html').open('w') as dom:
        p=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking',
            '--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-dimsym-report-chrome',
            '--window-size=1500,1250','--screenshot='+str(OUT/'browser_preview.png'),'--dump-dom',url],
            stdout=dom,stderr=err,timeout=40)
    assert p.returncode==0
    parser=Parser();parser.feed((OUT/'rendered_dom.html').read_text())
    assert parser.cells==['방법','평균 px','P90 px','주 지표','통과']+report['table_cells']
    for ref in parser.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (OUT/unquote(u.path)).exists(),ref
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
    write(OUT/'DELIVERY.json',dict(html_open_confirmed=visible,windows=windows,verified_table_cells=len(parser.cells)))
    assert visible,'Report created, but desktop visibility could not be confirmed'
    files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.log' and p.name not in ['COMPLETION.json','DISCORD_NOTIFICATION.json']]
    files+=list(KIT.glob('*.py'))+[ROOT/report['note']]
    write(OUT/'COMPLETION.json',dict(complete=True,decision=read(OUT/'SUMMARY.json')['decision'],
        models=12,optimizer_steps=12000,html_open_confirmed=True,
        gpu_recovered_without_reboot=True,full_network_training=False,real_evaluation=False,
        files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('Verified report table, visible Chrome window and completion artifacts')

if __name__=='__main__':main()
