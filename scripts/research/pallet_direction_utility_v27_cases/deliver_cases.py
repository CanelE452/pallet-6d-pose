"""Render and open the requested point/Hough case gallery."""
from show_cases import *
from delivery27 import Tables
import subprocess,re
from urllib.parse import urlsplit,unquote

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as f,Path('/tmp/pallet-v27-cases-browser.log').open('w') as e:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v27-cases-profile','--window-size=1500,1200','--screenshot='+str(shot),'--dump-dom',url],stdout=f,stderr=e,timeout=40)
    assert z.returncode==0;p=Tables();p.feed(dom.read_text());expected=['사례·대상점','기존 점','실제 V27','GT oracle','oracle 후보']
    for c in read(RUN/'CASES.json')['cases']:expected += [f'{c["title"]} · ID{c["point"]}',f'{c["base_error_px"]:.2f}',f'{c["actual_error_px"]:.2f}',f'{c["oracle_error_px"]:.2f}',c['candidate_name']]
    assert p.tables[0]==expected
    for ref in p.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (RUN/unquote(u.path)).exists(),ref
    z=subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20);windows=[]
    import time
    for attempt in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if 'V27 점·Hough 실제 사례' not in title:continue
            info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout;windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    confirmed=any(w['viewable'] for w in windows);write(RUN/'DELIVERY.json',dict(status='PASS',table_cells_checked=len(expected),local_refs=len(p.refs),html_sha256=sha(RUN/'index.html'),open_confirmed=confirmed,windows=windows));files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE]+list((RUN/'figures').glob('*.png'))+[RUN/p for p in ('CASES.json','AUDIT.json','DELIVERY.json','REPORT_KO.md','index.html','browser_preview.png')];write(RUN/'COMPLETION.json',dict(complete=True,case_gallery_only=True,no_new_training=True,html_open_confirmed=confirmed,files={str(p.relative_to(ROOT)):sha(p) for p in files}));print('CASES_DELIVERY',confirmed,flush=True)

if __name__=='__main__':main()
