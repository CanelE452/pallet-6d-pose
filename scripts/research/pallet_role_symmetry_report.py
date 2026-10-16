"""Render the explanatory note, validate local links and show it in Chrome."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import time
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
import markdown

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/pallet/results/pallet_role_symmetry_audit'
NOTE = ROOT/'_docs/notes/pose_geometry/corner-id-versus-spatial-error.md'
TITLE = '코너 번호와 실제 위치 — 대칭 재분석'

class Check(HTMLParser):
    def __init__(self):
        super().__init__();self.images=0;self.cells=0
    def handle_starttag(self,tag,attrs):
        self.images += tag=='img'; self.cells += tag=='td'
        for key,value in attrs:
            if key not in ['href','src']:continue
            u=urlparse(value)
            if u.scheme=='file': assert Path(unquote(u.path)).exists(),value

def main():
    assert json.loads((OUT/'AUDIT.json').read_text())['PASS']
    md=NOTE.read_text()
    # Resolve Markdown references relative to the source note, not the report.
    def link(match):
        ref=match[1]
        if '://' in ref or ref.startswith('#'):return match[0]
        target=(NOTE.parent/ref).resolve()
        assert target.exists() or target==OUT/'index.html',target
        return ']('+target.as_uri()+')'
    md=re.sub(r'\]\(([^)]+)\)',link,md)
    body=markdown.markdown(md,extensions=['tables','fenced_code'])
    css='''body{font:17px/1.8 system-ui,sans-serif;color:#18283a;background:#edf1f7;margin:0}
    main{max-width:1240px;margin:28px auto;padding:34px 44px;background:white;border-radius:18px}
    h1{font-size:30px;line-height:1.35}h2{margin-top:46px;border-bottom:2px solid #dce6f2;padding-bottom:10px}
    a{color:#1261a0}img{display:block;max-width:100%;height:auto;margin:22px auto;border:1px solid #ddd}
    table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}
    th{background:#e9f1fa}strong{color:#154f70}code{background:#edf2f8;padding:2px 5px;border-radius:4px;font-size:.85em}
    @media(max-width:700px){main{padding:18px;margin:0}body{font-size:15px}table{font-size:12px}td,th{padding:5px}}'''
    page=f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{TITLE}</title><style>{css}</style><main>{body}</main></html>'
    (OUT/'index.html').write_text(page)
    parser=Check();parser.feed(page)
    assert parser.images==3 and parser.cells==15
    url=(OUT/'index.html').as_uri()
    with (OUT/'browser.log').open('w') as log, (OUT/'rendered_dom.html').open('w') as dom:
        subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu','--disable-background-networking','--no-first-run',
            '--user-data-dir=/tmp/pallet-role-symmetry-chrome','--window-size=1500,1500','--screenshot='+str(OUT/'browser_preview.png'),
            '--dump-dom',url],stdout=dom,stderr=log,timeout=40,check=True)
    rendered=Check();rendered.feed((OUT/'rendered_dom.html').read_text())
    assert rendered.images==3 and rendered.cells==15
    subprocess.run(['google-chrome','--new-window',url],capture_output=True,timeout=20,check=True)
    windows=[]
    for _ in range(10):
        clients=subprocess.run(['xprop','-root','_NET_CLIENT_LIST'],capture_output=True,text=True).stdout
        for wid in re.findall(r'0x[0-9a-fA-F]+',clients):
            name=subprocess.run(['xprop','-id',wid,'_NET_WM_NAME'],capture_output=True,text=True).stdout
            if TITLE in name:
                info=subprocess.run(['xwininfo','-id',wid],capture_output=True,text=True).stdout
                windows.append(dict(id=wid,title=name.strip(),viewable='IsViewable' in info))
        if any(w['viewable'] for w in windows):break
        time.sleep(.5)
    visible=any(w['viewable'] for w in windows)
    (OUT/'DELIVERY.json').write_text(json.dumps(dict(html_open_confirmed=visible,windows=windows,images=3,table_cells=15),ensure_ascii=False,indent=2))
    assert visible,'Report generated, visible desktop not confirmed'
    bound=[p for p in OUT.iterdir() if p.is_file() and p.name not in ['COMPLETION.json','DISCORD_NOTIFICATION.json'] and p.suffix!='.log']
    bound += [NOTE,Path(__file__),ROOT/'scripts/research/pallet_role_symmetry_audit.py']
    (OUT/'COMPLETION.json').write_text(json.dumps(dict(complete=True,training_steps=0,post_hoc=True,html_open_confirmed=True,
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bound}),ensure_ascii=False,indent=2))
    print('Report links, table, image references and visible Chrome window verified.')

if __name__=='__main__':
    main()
