from pathlib import Path
import base64,json,time,urllib.request,websocket,hashlib
H=Path(__file__).resolve().parent;V=H.parent
pages=json.load(urllib.request.urlopen('http://127.0.0.1:9228/json'));page=next(x for x in pages if x['type']=='page')
w=websocket.create_connection(page['webSocketDebuggerUrl'],origin='http://localhost:9228');seq=0

def call(method,params=None):
 global seq
 seq+=1;w.send(json.dumps(dict(id=seq,method=method,params=params or{})))
 while True:
  r=json.loads(w.recv())
  if r.get('id')==seq:
   if'error'in r:raise RuntimeError(r['error'])
   return r.get('result',{})
call('Page.enable');call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1200,deviceScaleFactor=1,mobile=False));url=(V/'ROOT_READOUT_KO.html').as_uri();call('Page.navigate',dict(url=url))
for _ in range(30):
 r=call('Runtime.evaluate',dict(expression='({url:location.href,ready:document.readyState})',returnByValue=True))['result']['value']
 if r['url']==url and r['ready']=='complete':break
 time.sleep(.1)
else:raise RuntimeError('Page not ready')
call('Runtime.evaluate',dict(expression='document.fonts.ready',awaitPromise=True))
q=call('Runtime.evaluate',dict(expression="({height:document.documentElement.scrollHeight,images:[...document.images].map(x=>({src:x.src,width:x.naturalWidth,height:x.naturalHeight,complete:x.complete})),cueY:document.getElementById('fixed-crops').getBoundingClientRect().top+scrollY,bodyWidth:document.body.scrollWidth,viewportWidth:innerWidth})",returnByValue=True))['result']['value']
assert len(q['images'])==6 and all(x['width']==96 and x['height']==96 and x['complete']for x in q['images'])and q['bodyWidth']<=q['viewportWidth']
for name,y,h in [('QA_TOP_CDP.png',0,1400),('QA_FIXED_CROPS_CDP.png',max(0,q['cueY']-20),1900),('QA_ALL_PAGE_CDP.png',0,q['height'])]:
 result=call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=True,clip=dict(x=0,y=y,width=1440,height=h,scale=1)))
 (H/name).write_bytes(base64.b64decode(result['data']))
w.close();(H/'BROWSER_LAYOUT_QA.json').write_text(json.dumps(dict(completed=True,method='Isolated headlessChromeCDP, actual savedHTML with6 existing sourceRGB96PNG',layout=q,blank_anchor_CLI_attempt_preserved='QA_FIXED_CROPS.png was blank dueCLI anchor screenshot; CDP actualclip replaces it for QA evidence, HTMLunchanged'),indent=2)+'\n');print(json.dumps(q))
