"""Render, verify and open the final report, then bind all result artifacts."""
from context30 import *
from report30 import NAMES,LABELS
from delivery27 import Tables
import subprocess,re
from urllib.parse import urlsplit,unquote

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';assert read(RUN/'REPORT_COMPLETE.json')['complete'];url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as f,Path('/tmp/pallet-v30-browser.log').open('w') as err:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v30-profile','--window-size=1500,1250','--screenshot='+str(shot),'--dump-dom',url],stdout=f,stderr=err,timeout=40)
    assert z.returncode==0;p=Tables();p.feed(dom.read_text());expected=['방법']+LABELS
    for r in read(RUN/'FIGURE_DATA.json')['table']:expected+=[NAMES[r['arm']]]+[f'{v:.3f}' for v in r['values']]
    assert p.tables[1]==expected
    for ref in p.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (RUN/unquote(u.path)).exists(),ref
    z=subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20);windows=[]
    for _ in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if 'V30 점별 방향 적응형 실험 결과' not in title:continue
            info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout;windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    opened=any(w['viewable'] for w in windows);write(RUN/'DELIVERY.json',dict(status='PASS',table_cells=len(expected),local_refs=len(p.refs),html_open_confirmed=opened,windows=windows))
    files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE]+list((RUN/'figures').glob('*.png'))+list(RUN.glob('*.json'))+[RUN/'REPORT_KO.md',RUN/'index.html',shot]+list((RUN/'models').glob('*/*'))
    files=[p for p in files if p.is_file() and p.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json')]
    write(RUN/'COMPLETION.json',dict(complete=True,models=12,training_updates=24000,accepted=read(RUN/'SUMMARY.json')['accepted'],html_open_confirmed=opened,files={str(p.relative_to(ROOT)):sha(p) for p in files},prediction_hashes='EVALUATION_BINDINGS.json',scope='State-free signed directional amount/gain training, fixed strong point parents, full evaluation and diagnosis. No unconditional improvement or deployment claim.'))
    record('V30 실제 방향별 적응 학습/평가/원인분해/시각화 완료. 안정성+선이득 통과='+str(read(RUN/'SUMMARY.json')['accepted'])+', HTML표시='+str(opened)+'.');print('DELIVERY30',opened,flush=True)

if __name__=='__main__':main()
