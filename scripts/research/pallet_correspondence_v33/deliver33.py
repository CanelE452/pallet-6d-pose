"""Verify and open the actual HTML report, then bind the completed artifacts."""
from context33 import *
from report33 import NAMES,LABELS,TITLE
from delivery27 import Tables
import subprocess,re
from urllib.parse import urlsplit,unquote

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';assert read(RUN/'REPORT_COMPLETE.json')['summary_sha256']==sha(RUN/'SUMMARY.json')
    for p,h in read(RUN/'PRIOR_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    for p,h in read(RUN/'EVALUATION_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    write(RUN/'INPUT_REFERENCE.json',dict(source=str((V32.RUN/'INPUT_BINDINGS.json').relative_to(ROOT)),sha256=sha(V32.RUN/'INPUT_BINDINGS.json')))
    url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as f,Path('/tmp/pallet-v33-browser.log').open('w') as err:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v33-profile','--window-size=1500,1250','--screenshot='+str(shot),'--dump-dom',url],stdout=f,stderr=err,timeout=40)
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
    handoff=ROOT/'_docs/handoff/2026-09-10_pallet_correspondence_v33.md'
    files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE,handoff]+list((RUN/'figures').glob('*.png'))+list(RUN.glob('*.json'))+list(RUN.glob('*.md'))+[RUN/'index.html',shot]+list((RUN/'models').glob('*/*'))
    files=[p for p in files if p.is_file() and p.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json')]
    write(RUN/'COMPLETION.json',dict(complete=True,models=18,training_updates=36000,accepted=read(RUN/'SUMMARY.json')['accepted'],html_open_confirmed=opened,files={str(p.relative_to(ROOT)):sha(p) for p in files},prediction_hashes='EVALUATION_BINDINGS.json',input_hashes='INPUT_REFERENCE.json',scope='Point-specific incident-role Hough support and endpoint attention with matched pooled/geometry/line-only/end-only/aux-only controls. Fixed data and budget. Acceptance is in SUMMARY.json; reused DEV, not sealed evaluation.'))
    record('V33 점별 선 support/끝점 대응18모델·36,000업데이트·GT/예측 감사·시각화 완료. 전체 통과='+str(read(RUN/'SUMMARY.json')['accepted'])+', HTML표시='+str(opened)+'.');print('DELIVERY33',opened,flush=True)


if __name__=='__main__':main()
