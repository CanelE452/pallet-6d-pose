"""Render local report, verify selectable tables, then open desktop Chrome."""
from context23 import *
from html.parser import HTMLParser
import subprocess
import urllib.parse
import re

class Table(HTMLParser):
    def __init__(self):super().__init__();self.active=False;self.cells=[];self.cell=None;self.refs=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='table' and a.get('id')=='table':self.active=True
        if self.active and tag in ('th','td'):self.cell=''
        for k in ('href','src'):
            if k in a:self.refs.append(a[k])
    def handle_endtag(self,tag):
        if self.active and tag in ('th','td') and self.cell is not None:self.cells.append(self.cell);self.cell=None
        if tag=='table':self.active=False
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data

def main():
    from report23 import NAMES,LABELS
    s=read(RUN/'SUMMARY.json')['results'];receipts=[];base='file://'+str(RUN/'index.html')
    cases=[('real','mean','frame_mean_px'),('synth_val','2','outside'),('real','3','hidden'),('real','1','full_normalized_mean'),('synth_val','mean','pck5'),('new_check','mean','outside'),('prospective','mean','outside')]
    for i,(split,seed,metric) in enumerate(cases):
        url=base+'?'+urllib.parse.urlencode(dict(split=split,seed=seed,metric=metric));dom=Path(f'/tmp/pallet-v23-dom-{i}.html');err=Path(f'/tmp/pallet-v23-chrome-{i}.log')
        cmd=['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run','--no-default-browser-check',f'--user-data-dir=/tmp/pallet-v23-render-profile-{i}','--window-size=1600,1200','--dump-dom']
        if i==0:cmd.append('--screenshot=/tmp/pallet-v23-report.png')
        with dom.open('w') as f,err.open('w') as e:result=subprocess.run(cmd+[url],stdout=f,stderr=e,timeout=40)
        assert result.returncode==0;parser=Table();parser.feed(dom.read_text());keys=[c for c in LABELS if split+'/'+c in s];expected=['방법']+[LABELS[c] for c in keys]
        for a,name in NAMES.items():
            expected.append(name)
            for condition in keys:
                r=s[split+'/'+condition];rs=[r['stock']] if a=='stock' else [r[f'{a}_seed{x}'] for x in (SEEDS if seed=='mean' else [int(seed)])]
                values=[x['strata'][metric]['mean_px'] if metric in ('outside','hidden') else x['strata']['all'][metric] if metric.startswith('pck') else x[metric] for x in rs]
                expected.append('—' if any(v is None for v in values) else f'{np.mean(values):.{6 if metric=="full_normalized_mean" else 3}f}')
        assert parser.cells==expected,(i,parser.cells[:15],expected[:15])
        local=[x for x in parser.refs if not x.startswith(('http:','https:','#','data:'))];assert all((RUN/x).exists() for x in local)
        receipts.append(dict(split=split,seed=seed,metric=metric,table_cells=len(expected),local_links_valid=True))
    write(RUN/'DELIVERY_CHECK.json',dict(status='PASS',html_sha256=sha(RUN/'index.html'),rendered_cases=receipts,screenshot='/tmp/pallet-v23-report.png'))
    result=subprocess.run(['google-chrome','--new-window',base],capture_output=True,text=True,timeout=20)
    windows=[]
    # Poll for the mapped desktop window; no web access or user data inspection.
    for attempt in range(10):
        listing=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',listing):
            title=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout.strip()
            if 'V23 실제 합성 관측과 정확도 균형' not in title:continue
            info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout
            windows.append(dict(window_id=wid,title=title,viewable='IsViewable' in info))
        if any(x['viewable'] for x in windows):break
        time.sleep(.5)
    status='OPEN_CONFIRMED' if any(x['viewable'] for x in windows) else 'OPEN_NOT_CONFIRMED'
    write(RUN/'BROWSER_OPEN.json',dict(status=status,html_sha256=sha(RUN/'index.html'),windows=windows,open_returncode=result.returncode))
    print('DELIVERY',len(receipts),'rendered cases',status,flush=True)

if __name__=='__main__':main()
