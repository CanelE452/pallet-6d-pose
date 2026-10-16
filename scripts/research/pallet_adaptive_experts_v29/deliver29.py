"""Check rendered numbers and local links, confirm a visible browser, seal artifacts."""
from context29 import *
from report29 import NAMES,LABELS
from delivery27 import Tables
import subprocess,re
from urllib.parse import urlsplit,unquote

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';assert read(RUN/'REPORT_COMPLETE.json')['complete']
    url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as f,Path('/tmp/pallet-v29-browser.log').open('w') as err:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v29-profile','--window-size=1500,1250','--screenshot='+str(shot),'--dump-dom',url],stdout=f,stderr=err,timeout=40)
    assert z.returncode==0;p=Tables();p.feed(dom.read_text());expected=['방법']+LABELS
    for row in read(RUN/'FIGURE_DATA.json')['table']:expected+=[NAMES[row['arm']]]+[f'{x:.3f}' for x in row['values']]
    assert p.tables[0]==expected
    for ref in p.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (RUN/unquote(u.path)).exists(),ref
    z=subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20);windows=[]
    for attempt in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if 'V29 점과 선 적응형 실험 결과' not in title:continue
            info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout;windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    opened=any(w['viewable'] for w in windows);write(RUN/'DELIVERY.json',dict(status='PASS',table_cells=len(expected),local_refs=len(p.refs),html_open_confirmed=opened,windows=windows))
    files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE]+list((RUN/'figures').glob('*.png'))+list(RUN.glob('*.json'))+[RUN/'REPORT_KO.md',RUN/'index.html',shot]
    files+=[p for p in (RUN/'models').glob('*/*') if p.is_file()]
    files=[p for p in files if p.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json')]
    write(RUN/'COMPLETION.json',dict(complete=True,training_updates=25200,trained_models=24,accepted=read(RUN/'SUMMARY.json')['accepted'],html_open_confirmed=opened,files={str(p.relative_to(ROOT)):sha(p) for p in files},prediction_hashes='EVALUATION_BINDINGS.json',scope='Expert training, per-point adaptive controls, full stress scoring, case images and failure diagnosis complete; no stable-improvement claim or deployment replacement.'))
    record('V29 완료: 독립 선 사례3 6.18px, 전체 조건 고정 배정/적응 선택은 안정성 미달. 합성/실사 전문가 순위 차이, 가림 상태 오류, 큰 잘림의 선 후보 한계 분석. HTML 창 확인='+str(opened)+'.')
    print('DELIVERY29',opened,flush=True)

if __name__=='__main__':main()
