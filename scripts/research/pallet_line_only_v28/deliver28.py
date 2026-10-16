"""Verify and open V28 local report."""
from report28 import *
from delivery27 import Tables
import subprocess,re,time
from urllib.parse import urlsplit,unquote

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as f,Path('/tmp/pallet-v28-browser.log').open('w') as e:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v28-profile','--window-size=1500,1200','--screenshot='+str(shot),'--dump-dom',url],stdout=f,stderr=e,timeout=40)
    assert z.returncode==0;p=Tables();p.feed(dom.read_text());s=read(RUN/'SUMMARY.json');expected=['3번 점 ID0의 방법','오차px']
    for a in METHODS:expected += [NAMES[a],f'{s["case"]["errors"][a]:.3f}']
    assert p.tables[0]==expected
    for ref in p.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (RUN/unquote(u.path)).exists(),ref
    z=subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20);windows=[]
    for attempt in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if 'V28 선만 적용한 결과' not in title:continue
            info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout;windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    confirmed=any(w['viewable'] for w in windows);write(RUN/'DELIVERY.json',dict(status='PASS',table_cells=len(expected),html_sha256=sha(RUN/'index.html'),open_confirmed=confirmed,windows=windows));files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE]+list((RUN/'figures').glob('*.png'))+[RUN/n for n in ('PROTOCOL.json','SUMMARY.json','DHT_ONLY.json','FIGURE_DATA.json','AUDIT.json','DELIVERY.json','REPORT_KO.md','index.html','case_intervention.npz','browser_preview.png')];write(RUN/'COMPLETION.json',dict(complete=True,new_training=False,case_line_only_improved=True,whole_population_stable_improvement_demonstrated=False,point_free_branch_scope='DHT gate bypass only, no point-free pose decoding',html_open_confirmed=confirmed,files={str(p.relative_to(ROOT)):sha(p) for p in files}));print('DELIVERY28',confirmed,flush=True)

if __name__=='__main__':main()
