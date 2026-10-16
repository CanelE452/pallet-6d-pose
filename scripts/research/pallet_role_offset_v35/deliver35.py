from context35 import *
from report35 import TITLE
from delivery27 import Tables
from urllib.parse import urlsplit,unquote
import re,subprocess

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS' and read(RUN/'GRID_AUDIT.json')['status']=='PASS'
    for key,file in [('summary','SUMMARY.json'),('followup','FOLLOWUP.json'),('quantization','QUANTIZATION.json')]:assert read(RUN/'REPORT_COMPLETE.json')[key+'_sha256']==sha(RUN/file)
    for p,h in read(RUN/'PRIOR_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as out,Path('/tmp/pallet-v35-browser.log').open('w') as err:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v35-profile','--window-size=1500,1250','--screenshot='+str(shot),'--dump-dom',url],stdout=out,stderr=err,timeout=40)
    assert z.returncode==0;t=Tables();t.feed(dom.read_text());expected=['방법','평균 점 오차px','기준점 대비 차이95%구간']
    for r in read(RUN/'FIGURE_DATA.json')['table']:expected+=[r['method'],f'{r["value"]:.3f}',str([round(x,3) for x in r['ci']])]
    assert t.tables[0]==expected,(t.tables[0],expected)
    for ref in t.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (RUN/unquote(u.path)).exists(),ref
    for ref in re.findall(r'\]\(([^)]+)\)',NOTE.read_text()):
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (NOTE.parent/unquote(u.path)).exists(),ref
    subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20);windows=[]
    for _ in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if TITLE in title:
                info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout;windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(x['viewable'] for x in windows):break
        time.sleep(.5)
    visible=any(x['viewable'] for x in windows);write(RUN/'DELIVERY.json',dict(status='PASS',html_open_confirmed=visible,windows=windows,table_cells=len(expected),local_figures=len(t.refs)))
    files=[p for p in RUN.rglob('*') if p.is_file() and p.suffix!='.log' and p.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json')]+list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE,ROOT/'_docs/handoff/2026-09-10_pallet_role_offset_v35.md']
    write(RUN/'COMPLETION.json',dict(complete=True,neural_training_updates=0,frozen_model_seeds=3,offset_calibrations=24,html_open_confirmed=visible,primary_gates=read(RUN/'SUMMARY.json')['gates'],files={str(p.relative_to(ROOT)):sha(p) for p in files},scope='Existing synthetic render diagnostic. GT correspondence and oracle lines are not deployment performance. Exploratory bounds/joint/grid controls separate. No new real evaluation or manual GT edits.'))
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:f.write('\n\n### V35 actual boundary / virtual corner verification\n\n600render observations,436detected; check200/130,820supported points. Frozen3seeds,neural updates0,24offsetcalibrations. GTphysical+median11.499px vsbase12.373; predicted16.903; best1152gridGT13.126. Role/offset partly explains but continuous precision/candidate errors remain; no stable deployed fusion claim. Report and browser delivered.\n')
    print('DELIVERY35',visible,flush=True)
if __name__=='__main__':main()
