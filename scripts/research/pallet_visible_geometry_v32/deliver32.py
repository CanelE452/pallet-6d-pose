"""Verify and open the actual HTML report, then bind the completed artifacts."""
from context32 import *
from report32 import NAMES,LABELS,TITLE
from delivery27 import Tables
import subprocess,re
from urllib.parse import urlsplit,unquote

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';assert read(RUN/'REPORT_COMPLETE.json')['summary_sha256']==sha(RUN/'SUMMARY.json')
    for p,h in read(RUN/'PRIOR_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    for p,h in read(RUN/'EVALUATION_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    # Bind renderer output and the exact cached observations/frozen parent inputs.
    inputs=list((RUN/'renders').glob('*'))+list((RUN/'cache').glob('*'))+list((RUN/'parents').glob('*'))+list(RUN.glob('physical_segments*.npy'))
    write(RUN/'INPUT_BINDINGS.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in inputs if p.is_file()}))
    url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as f,Path('/tmp/pallet-v32-browser.log').open('w') as err:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v32-profile','--window-size=1500,1250','--screenshot='+str(shot),'--dump-dom',url],stdout=f,stderr=err,timeout=40)
    assert z.returncode==0;p=Tables();p.feed(dom.read_text());expected=['방법']+LABELS
    for r in read(RUN/'FIGURE_DATA.json')['table']:expected+=[NAMES[r['arm']]]+[f'{v:.3f}' for v in r['values']]
    assert p.tables[0]==expected,(p.tables[0],expected)
    for ref in p.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (RUN/unquote(u.path)).exists(),ref
    for ref in re.findall(r'\]\(([^)]+)\)',NOTE.read_text()):
        if not urlsplit(ref).scheme:assert (NOTE.parent/unquote(ref)).exists(),ref
    subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20);windows=[]
    for _ in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if TITLE not in title:continue
            info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout;windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    opened=any(w['viewable'] for w in windows);write(RUN/'DELIVERY.json',dict(status='PASS',table_cells=len(expected),local_refs=len(p.refs),html_open_confirmed=opened,windows=windows))
    handoff=ROOT/'_docs/handoff/2026-09-10_pallet_visible_geometry_v32.md'
    files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE,handoff]+list((RUN/'figures').glob('*.png'))+list(RUN.glob('*.json'))+[RUN/'REPORT_KO.md',RUN/'index.html',shot,RUN/'model32_training_source.py.txt']+list((RUN/'models').glob('*/*'))
    files=[p for p in files if p.is_file() and p.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json')]
    write(RUN/'COMPLETION.json',dict(complete=True,models=15,training_updates=30000,accepted=read(RUN/'SUMMARY.json')['accepted'],html_open_confirmed=opened,files={str(p.relative_to(ROOT)):sha(p) for p in files},prediction_hashes='EVALUATION_BINDINGS.json',input_hashes='INPUT_BINDINGS.json',scope='Rule-defined visible CAD crease/endpoint supervision with joint point repair. Four primary arms plus exploratory aux-only control; post-hoc token/donor interventions. Partial benefit, stability failed; no deployment or sealed-evaluation claim.'))
    record('V32 실제 가시 경계/끝점 감독15모델·30,000업데이트 및 GT/660예측 감사·시각화 완료. 잘림20 새 점16.331→융합14.343px, 보조 감독만15.297px. Hough만 끄기14.359/끝점만 끄기15.537. 전체 안정성 미통과. HTML표시='+str(opened)+'.');print('DELIVERY32',opened,flush=True)

if __name__=='__main__':main()
