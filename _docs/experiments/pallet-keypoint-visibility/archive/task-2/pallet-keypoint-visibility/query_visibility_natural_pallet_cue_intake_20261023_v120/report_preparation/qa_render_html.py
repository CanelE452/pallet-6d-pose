"""Local headless Chrome QA of completed report; no model/data input access."""
from pathlib import Path
import base64,json,subprocess,tempfile,time,urllib.request
import websocket
H=Path(__file__).resolve().parent;V=H.parent;Q=H/'qa';Q.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='v120-report-chrome-')as td:
 p=subprocess.Popen(['/usr/bin/google-chrome','--headless','--no-sandbox','--disable-dev-shm-usage','--hide-scrollbars','--remote-debugging-port=0','--remote-allow-origins=*','--window-size=1440,1200','--user-data-dir='+td,'about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:
  portfile=Path(td)/'DevToolsActivePort';start=time.monotonic()
  while not portfile.exists():
   assert time.monotonic()-start<15;time.sleep(.05)
  port=int(portfile.read_text().splitlines()[0]);targets=json.load(urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/json'));tab=next(t for t in targets if t['type']=='page')
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],timeout=20);counter=0
  def c(method,params=None):
   global counter
   counter+=1;ws.send(json.dumps(dict(id=counter,method=method,params=params or{})))
   while True:
    r=json.loads(ws.recv())
    if r.get('id')==counter:
     assert'error'not in r,r
     return r.get('result',{})
  c('Page.enable');c('Page.navigate',dict(url=(V/'ROOT_READOUT_KO.html').as_uri()))
  ready="""new Promise(resolve=>{let t=setInterval(async()=>{if(document.readyState==='complete'){clearInterval(t);await document.fonts.ready;await Promise.all([...document.images].map(i=>i.decode().catch(()=>null)));resolve({title:document.title,images:[...document.images].map(i=>({src:i.getAttribute('src'),width:i.naturalWidth,height:i.naturalHeight})),bodyWidth:document.body.scrollWidth,viewportWidth:innerWidth});}},20);})"""
  facts=c('Runtime.evaluate',dict(expression=ready,awaitPromise=True,returnByValue=True))['result']['value'];facts['screenshots']=[]
  for section,name in[(None,'HTML_TOP.png'),(1,'HTML_CURRENT.png'),(2,'HTML_SOURCE.png'),(4,'HTML_NEXT.png')]:
   expr="window.scrollTo(0,0)"if section is None else"document.querySelectorAll('h2')["+str(section-1)+"].scrollIntoView()"
   c('Runtime.evaluate',dict(expression=expr));c('Runtime.evaluate',dict(expression='new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))',awaitPromise=True))
   image=c('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data'];(Q/name).write_bytes(base64.b64decode(image));facts['screenshots'].append(str(Q/name))
  (Q/'CHROME_LAYOUT_FACTS.json').write_text(json.dumps(facts,indent=2)+'\n');print(json.dumps(facts,indent=2));ws.close()
 finally:
  p.terminate()
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:p.kill();p.wait()
