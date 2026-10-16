from context38 import *
from report38 import TITLE,fmtci
from deliver37 import Tables
from urllib.parse import urlsplit,unquote
import re,subprocess,time

def main():
    for f in ('AUDIT.json','PROBE_AUDIT.json'):assert read(RUN/f)['status']=='PASS'
    for key,f in [('summary','SUMMARY.json'),('probes','PROBES.json'),('reliability','RELIABILITY.json')]:assert read(RUN/'REPORT_COMPLETE.json')[key+'_sha256']==sha(RUN/f)
    for p,h in read(RUN/'PRIOR_BINDINGS.json')['files'].items():assert sha(ROOT/p)==h,p
    url=(RUN/'index.html').as_uri();dom=RUN/'rendered_dom.html';shot=RUN/'browser_preview.png'
    with dom.open('w') as out,Path('/tmp/pallet-v38-browser.log').open('w') as err:
        z=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/pallet-v38-profile','--window-size=1500,1250','--screenshot='+str(shot),'--dump-dom',url],stdout=out,stderr=err,timeout=40)
    assert z.returncode==0;t=Tables();t.feed(dom.read_text());fig=read(RUN/'FIGURE_DATA.json');expected=['방법 / V36 both 후보','전체1040점 평균 오차px','시드별 보정 점 수','기준 대비 차이95%구간px']
    for row in fig['table']:expected += [row['label'],f'{row["mean_px"]:.3f}',str(row['selected']),fmtci(row['ci'])]
    assert t.tables[1]==expected,(t.tables[1],expected)
    summary=read(RUN/'SUMMARY.json');u=summary['uncertainty']['check/all/all_detected']['both'];exp=[]
    for method,field,percent in [('geometry_point','scale_spearman',False),('geometry_point','simultaneous_point_coverage',True),('geometry_point','mean_halfwidth_px',False),('constant_point','mean_halfwidth_px',False)]:
        exp += [f'{100*r[field]:.2f}%' if percent else f'{r[field]:.3f}' for r in u[method]]
    assert [x for i,x in enumerate(t.tables[0][4:]) if i%4!=0]==exp
    expected=[]
    for c,st in [('clean','all_detected'),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside')]:
        key='check/'+c+'/'+st
        if key in summary['results']:
            q=summary['results'][key];r=q['methods']['both'];expected += [c+'/'+st,str(q['points']),f'{r["anchor"]["mean_px"]:.3f}',f'{r["geometry_point"]["mean_px"]:.3f}']
    assert t.tables[2][4:]==expected
    probes=read(RUN/'PROBES.json');oracle=probes['oracle_summary']['both'];assert t.tables[3][4::3]==[f'{oracle[k]["mean_px"]:.3f}' for k in ('oracle_mean','oracle_radius','oracle_both')]
    expected=[]
    for key in ('possible_over1px','possible_with_positive_predicted_gain','possible_with_positive_lower_gain'):expected += [str(probes['oracles'][f'both_seed{s}']['loss_funnel'][key]) for s in SEEDS]
    assert [x for i,x in enumerate(t.tables[4][4:]) if i%4!=0]==expected
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
    visible=any(x['viewable'] for x in windows);write(RUN/'DELIVERY.json',dict(status='PASS' if visible else 'NOT_VISIBLE',html_open_confirmed=visible,windows=windows,table_cells=sum(map(len,t.tables)),local_figures=len(t.refs)));assert visible,'Chrome report window not viewable'
    files=[p for p in RUN.rglob('*') if p.is_file() and p.suffix!='.log' and p.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json')]+list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE,HANDOFF]
    write(RUN/'COMPLETION.json',dict(complete=True,hgb_fits=36,boosting_rounds=2160,audit_refits=3,neural_updates=0,html_open_confirmed=visible,accepted=summary['gates']['accepted'],gates=summary['gates'],scope='Frozen V36/V34 synthetic12fit4calibration8check uncertainty prototype. Partialranking relation; no stabletransfer. No newreal/manualGTediting.',files={str(p.relative_to(ROOT)):sha(p) for p in files}))
    with (ROOT/'_docs/history/2026-09-11.md').open('a') as f:f.write('\n\n### V38 point-conditioned uncertainty completed\n\n36HGBfits2160rounds+3auditrefits,0neuralupdates. Split12fit4calibration8check,upstreamold16reuseexplicit. Primary both geometrypoint rankcorr.525/.542/.568 but pointcoverage.871/.850/.845; only1/3/0actions,10.463->10.487px. All4selectedoutsidepredictedinterval,3harm>1px. Oraclemean9.360/radius10.161/both7.802. Improvementopportunities297/332/303->positivepredictedgain27/51/49->positiveconservative0/0/0. Model/metric/hash/probe auditsPASS; MD,3caseimages,plots andvisibleChrome delivered. No architecture replacement.\n')
    print('DELIVERY38',visible,flush=True)
if __name__=='__main__':main()
