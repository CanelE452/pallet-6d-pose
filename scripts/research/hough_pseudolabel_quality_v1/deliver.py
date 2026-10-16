from run import ROOT, read, write, sha
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location("hough_pl_quality_report",Path(__file__).with_name("report.py"))
report_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(report_module)
TITLE,LABELS=report_module.TITLE,report_module.LABELS
import argparse,re,subprocess,time
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
class Parser(HTMLParser):
    def __init__(self):super().__init__();self.cells=[];self.refs=[];self.active=False;self.buf=''
    def handle_starttag(self,tag,attrs):
        if tag in ['th','td']:self.active=True;self.buf=''
        for k,v in attrs:
            if k in ['src','href']:self.refs.append(v)
    def handle_data(self,data):
        if self.active:self.buf+=data
    def handle_endtag(self,tag):
        if tag in ['th','td']:self.cells.append(self.buf.strip());self.active=False

def main(out):
    a=read(out/'AUDIT.json');assert a['status']=='PASS'
    for p,h in a['source_sha256'].items():assert sha(ROOT/p)==h,p
    rc=read(out/'REPORT_COMPLETE.json');assert rc['summary_sha256']==sha(out/'SUMMARY.json')
    url=(out/'index.html').as_uri()
    with (out/'browser.log').open('w') as err,(out/'rendered_dom.html').open('w') as dom:
        proc=subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check','--user-data-dir=/tmp/hough-pl-quality-chrome','--window-size=1500,1250','--screenshot='+str(out/'browser_preview.png'),'--dump-dom',url],stdout=dom,stderr=err,timeout=40)
    assert proc.returncode==0
    parser=Parser();parser.feed((out/'rendered_dom.html').read_text())
    expected=['기존 필터 통과 35장·307점','평균 오차 px','중앙값 px','P90 px','20px 초과 오류가 있는 장수']
    s=read(out/'SUMMARY.json')
    for arm,label in LABELS.items():
        z=s['results']['F4_accepted'][arm];expected += [label,f"{z['mean_px']:.3f}",f"{z['median_px']:.3f}",f"{z['p90_px']:.3f}",'/'.join(str(v['bad_frames']['20']) for v in z['per_seed'])]
    assert parser.cells==expected,(parser.cells,expected)
    for ref in parser.refs:
        u=urlsplit(ref)
        if not u.scheme and u.path:assert (out/unquote(u.path)).exists(),ref
    subprocess.run(['google-chrome','--new-window',url],capture_output=True,text=True,timeout=20)
    windows=[]
    for _ in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if TITLE in title:
                info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout
                windows.append(dict(id=wid,title=title.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    visible=any(w['viewable'] for w in windows)
    write(out/'DELIVERY.json',dict(status='PASS' if visible else 'NOT_VISIBLE',html_open_confirmed=visible,windows=windows,verified_table_cells=len(expected),figures=4))
    assert visible
    note=ROOT/rc['note']
    paths=[p for p in out.rglob('*') if p.is_file() and p.suffix!='.log' and p.name not in ['COMPLETION.json','DISCORD_NOTIFICATION.json']]+list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.md'))+[note]
    write(out/'COMPLETION.json',dict(complete=True,decision=s['decision'],quality_criteria_passed=s['passed'],n_evaluated_frames=40,n_accepted_frames=35,n_accepted_points=307,student_training_updates=0,new_inference=0,html_open_confirmed=True,files={str(p.relative_to(ROOT)):sha(p) for p in paths}))
    print('Report delivery and table verification passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'data/pallet/results/hough_pseudolabel_quality_v1');main(p.parse_args().out)
