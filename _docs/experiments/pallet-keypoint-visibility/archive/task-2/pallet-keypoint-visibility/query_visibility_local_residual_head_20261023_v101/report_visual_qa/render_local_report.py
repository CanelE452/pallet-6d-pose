"""Render the saved local report only; no model/data inference or network reads."""
from pathlib import Path
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
import websocket

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / 'ROOT_READOUT_KO.html'


def main():
    assert REPORT.is_file()
    with tempfile.TemporaryDirectory(prefix='pallet-v101-local-report-', ignore_cleanup_errors=True) as profile:
        process = subprocess.Popen([
            '/usr/bin/google-chrome', '--headless=new', '--no-sandbox',
            '--disable-gpu', '--disable-dev-shm-usage', '--disable-background-networking',
            '--remote-debugging-port=0', '--remote-allow-origins=*',
            '--user-data-dir=' + profile, 'about:blank'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ws = None
        try:
            portfile = Path(profile) / 'DevToolsActivePort'
            for _ in range(200):
                if portfile.exists():
                    break
                time.sleep(.05)
            port = int(portfile.read_text().splitlines()[0])
            tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list'))
            target = next(t for t in tabs if t['type'] == 'page')
            ws = websocket.create_connection(target['webSocketDebuggerUrl'], timeout=15)
            serial = 0

            def call(method, params=None):
                nonlocal serial
                serial += 1
                ws.send(json.dumps({'id': serial, 'method': method, 'params': params or {}}))
                while True:
                    result = json.loads(ws.recv())
                    if result.get('id') == serial:
                        assert 'error' not in result, result
                        return result['result']

            call('Page.enable')
            call('Network.enable')
            call('Network.setBlockedURLs', {'urls': ['http://*', 'https://*']})
            call('Emulation.setDeviceMetricsOverride', dict(
                width=1400, height=1100, deviceScaleFactor=1, mobile=False))
            call('Page.navigate', {'url': REPORT.as_uri()})
            call('Runtime.evaluate', {'expression': "document.querySelectorAll('details').forEach(i=>i.open=true);document.querySelectorAll('img').forEach(i=>i.loading='eager')"})
            ready = False
            for _ in range(200):
                call('Runtime.evaluate', {'expression': "document.querySelectorAll('details').forEach(i=>i.open=true);document.querySelectorAll('img').forEach(i=>{i.loading='eager';i.fetchPriority='high'})"})
                ready = call('Runtime.evaluate', {
                    'expression': "document.readyState==='complete'&&document.fonts.status==='loaded'&&Array.from(document.images).every(i=>i.complete)",
                    'returnByValue': True})['result'].get('value')
                if ready:
                    break
                time.sleep(.05)
            assert ready, 'Local report, fonts, or images did not finish loading'
            call('Runtime.evaluate', {'expression': 'new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))', 'awaitPromise': True})
            dom = call('Runtime.evaluate', {'expression': "JSON.stringify({title:document.title,width:innerWidth,scrollWidth:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight,fontsStatus:document.fonts.status,fontFamily:getComputedStyle(document.body).fontFamily,fontFaces:Array.from(document.fonts).map(f=>({family:f.family,status:f.status})),images:Array.from(document.images).map(i=>({src:i.getAttribute('src'),loaded:i.complete&&i.naturalWidth>0,width:i.naturalWidth,height:i.naturalHeight})),headings:Array.from(document.querySelectorAll('h1,h2')).map(i=>({tag:i.tagName,text:i.innerText,y:i.getBoundingClientRect().top+scrollY}))})", 'returnByValue': True})
            info = json.loads(dom['result']['value'])
            (HERE / 'ACTUAL_BROWSER_DOM_RENDER.json').write_text(
                json.dumps(info, ensure_ascii=False, indent=2) + '\n')
            assert info['fontsStatus'] == 'loaded', info
            assert all(f['status'] != 'error' for f in info['fontFaces']), info
            assert all(i['loaded'] for i in info['images']), info
            assert info['scrollWidth'] <= info['width'], 'Horizontal report overflow'
            height = int(info['height'])
            headings = [h for h in info['headings'] if h['tag'] == 'H2']

            def position(keywords, fallback):
                # Optional generic H2 matching; no V94-specific/self heading.
                scored = [(sum(word.lower() in h['text'].lower() for word in keywords), h)
                          for h in headings]
                best = max(scored, key=lambda item: item[0]) if scored else None
                heading = best[1] if best and best[0] else (
                    headings[min(fallback, len(headings)-1)] if headings else None)
                y = max(0, int(heading['y'])-20) if heading else 0
                return min(y, max(0, height-1)), None if heading is None else heading['text']

            targets = [
                ('REPORT_SOURCE_METRICS.png', ['같은 소스 입력에서의 D/E 정량'], 0),
                ('REPORT_TRIPLETS.png', ['BEFORE·COVER·SHAM'], 1),
                ('REPORT_CROPS.png', ['고정 source 그림'], 3),
                ('REPORT_LIMITS.png', ['해석의 한계'], 4)]
            offsets = [('REPORT_TOP.png', 0)]
            info['captureSections'] = [{'file': 'REPORT_TOP.png', 'heading': None, 'y': 0}]
            for name, keywords, fallback in targets:
                y, heading = position(keywords, max(0, fallback))
                offsets.append((name, y))
                info['captureSections'].append({'file': name, 'heading': heading, 'y': y})
            for name, y in offsets:
                if (HERE / name).exists():
                    continue
                result = call('Page.captureScreenshot', {
                    'format': 'png', 'captureBeyondViewport': True,
                    'clip': {'x': 0, 'y': y, 'width': 1400,
                             'height': min(1100, height-y), 'scale': 1}})
                (HERE / name).write_bytes(base64.b64decode(result['data']))
            (HERE / 'ACTUAL_BROWSER_DOM_RENDER.json').write_text(
                json.dumps(info, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({'rendered': True, 'fonts_loaded': True, 'horizontal_overflow': False, 'height': height,
                              'loaded_images': len(info['images']),
                              'screenshots': [n for n, _ in offsets]}))
        finally:
            if ws is not None:
                try:
                    call('Browser.close')
                except Exception:
                    pass
                ws.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
