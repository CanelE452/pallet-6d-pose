#!/usr/bin/env python3
"""Literal HTML img -> Markdown. No image decoder, JS, browser, or model."""
import argparse
import base64
import collections
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import posixpath
import re
import stat
import tempfile
from urllib.parse import quote, unquote, unquote_to_bytes, urljoin, urlsplit

DENY=re.compile(r'(?:^|_)v112(?:_|$)',re.I)
MIMES={'image/png':'.png','image/jpeg':'.jpg','image/gif':'.gif','image/webp':'.webp','image/svg+xml':'.svg','image/avif':'.avif','image/bmp':'.bmp'}

def forbidden(value):
    return any(DENY.search(p) for p in re.split(r'[/\\]',unquote(str(value))))

def safe_status(path):
    """No stat until all names are checked, and no following any symlink."""
    path=Path(os.path.abspath(path))
    if forbidden(path):return 'EXCLUDED_V112'
    cur=Path(path.anchor)
    for part in path.parts[1:]:
        cur=cur/part
        try:info=os.lstat(cur)
        except FileNotFoundError:return 'MISSING'
        if stat.S_ISLNK(info.st_mode):return 'SYMLINK_EXCLUDED'
    return 'EXISTS'

def sha(data):return hashlib.sha256(data).hexdigest()

class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes=[{'tag':'root','attrs':{},'parent':None,'children':[],'data':[]}]
        self.stack=[0];self.images=[];self.script_elements=0;self.base=None;self.canonical=None;self.og_url=None
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs);n=len(self.nodes);parent=self.stack[-1]
        self.nodes.append({'tag':tag,'attrs':attrs,'parent':parent,'children':[],'data':[]})
        self.nodes[parent]['children'].append(n)
        if tag=='img':self.images.append({'ordinal':len(self.images)+1,'node':n,'src':attrs.get('src'),'alt':attrs.get('alt'),'title':attrs.get('title')})
        if tag=='script':self.script_elements+=1
        if tag=='base' and self.base is None:self.base=attrs.get('href')
        if tag=='link' and attrs.get('rel')=='canonical':self.canonical=attrs.get('href')
        if tag=='meta' and attrs.get('property')=='og:url':self.og_url=attrs.get('content')
        if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:self.stack.append(n)
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if self.nodes[self.stack[-1]]['tag']==tag:self.stack.pop()
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.nodes[self.stack[i]]['tag']==tag:
                self.stack=self.stack[:i];break
    def handle_data(self,data):self.nodes[self.stack[-1]]['data'].append((len(self.nodes[self.stack[-1]]['children']),data))
    def text(self,n):
        node=self.nodes[n]
        if node['tag'] in {'script','style'}:return ''
        values=collections.defaultdict(list)
        for i,data in node['data']:values[i].append(data)
        result=[]
        for i,c in enumerate(node['children']):result+=values[i];result.append(self.text(c))
        result+=values[len(node['children'])]
        return ''.join(result)
    def descendants(self,n):
        for c in self.nodes[n]['children']:
            yield c;yield from self.descendants(c)
    def captions(self,img):
        ancestor=self.nodes[img['node']]['parent']
        while ancestor not in (None,0):
            ds=list(self.descendants(ancestor));node=self.nodes[ancestor]
            image_count=sum(self.nodes[x]['tag']=='img' for x in ds)
            if image_count==1 or node['tag']=='figure':
                caps=[]
                for x in ds:
                    nn=self.nodes[x];classes=(nn['attrs'].get('class') or '').split()
                    if nn['tag']=='figcaption' or any(c in ('caption','figcaption','image-caption','img-caption','caption-text','cap') for c in classes):
                        t=self.text(x).strip()
                        if t and t not in caps:caps.append(t)
                if caps:return caps
            ancestor=node['parent']
        return []
    def heading(self,img):
        previous=[i for i in range(img['node']) if self.nodes[i]['tag'] in {'h1','h2','h3','h4','h5','h6'}]
        return self.text(previous[-1]).strip() if previous else None
    def title(self):
        for i,n in enumerate(self.nodes):
            if n['tag']=='title' and self.text(i).strip():return self.text(i).strip()
        return None

def caption_block(label,value):
    if value is None:return []
    runs=re.findall(r'`+',value);fence='`'*max(3,max([len(x)+1 for x in runs],default=3))
    return [label,'',fence+'text',value,fence,'']

def reference(src,doc,parser,lookup,roots,explicit_origin=None):
    if src is None or src=='':return {'status':'NO_LITERAL_SRC'}
    if forbidden(src):return {'status':'EXCLUDED_V112','original_src':src}
    if src.startswith('data:'):return {'status':'DATA_URI'}
    effective=urljoin(parser.base,src) if parser.base else src
    if forbidden(effective):return {'status':'EXCLUDED_V112','original_src':src}
    u=urlsplit(effective)
    if u.scheme in ('http','https') or u.netloc:
        return {'status':'REMOTE_LINK_ONLY','url':'https:'+effective if effective.startswith('//') else effective,'explicit_base':parser.base}
    if u.scheme not in ('','file'):return {'status':'UNSUPPORTED_SCHEME','scheme':u.scheme}
    path=unquote(u.path)
    if not path:return {'status':'EMPTY_IMAGE_PATH'}
    absolute=os.path.abspath(path if os.path.isabs(path) else str(doc.parent/path))
    if forbidden(absolute):return {'status':'EXCLUDED_V112','original_src':src}
    item=lookup.get(absolute)
    if item:
        target=item['published_relative']
        compressed=bool(item.get('gzipchunks'))
        return {'status':'LOCAL_INCLUDED','source_absolute':absolute,'source_relative_path':item['source_relative_path'],
                'source_root_id':item['source_root_id'],'published_logical_target':target,
                'published_target':target+'.parts.json' if compressed else target,'descriptor_link_only':compressed,
                'source_image_sha256':item.get('source_sha256')}
    origin=parser.canonical or parser.og_url or (explicit_origin or {}).get('url')
    def official_remote():
        if origin and urlsplit(origin).scheme in ('http','https'):
            final_url=urljoin(origin,src)
            if forbidden(final_url):return {'status':'EXCLUDED_V112','original_src':src}
            return {'status':'REMOTE_LINK_ONLY','url':final_url,'original_document_origin':origin,
                    'origin_authority':'literal canonical/og:url' if parser.canonical or parser.og_url else explicit_origin,
                    'source_file_not_locally_available':True}
        return None
    # Unknown paths are inspected only under the explicitly inventoried source roots.
    root_id=None;relative=None
    for rid,rp in roots.items():
        try:relative=Path(absolute).relative_to(rp).as_posix();root_id=rid;break
        except ValueError:pass
    if root_id is None:return official_remote() or {'status':'OUTSIDE_INVENTORIED_ROOTS','source_absolute':absolute}
    state=safe_status(absolute)
    if state=='EXISTS':
        prefix='archive/task-2/'+roots[root_id].name
        return {'status':'LOCAL_IMAGE_OVERRIDE_NEEDED','source_absolute':absolute,'source_root_id':root_id,
                'source_relative_path':relative,'published_target':prefix+'/'+relative,'descriptor_link_only':False}
    # Archived HTML can move below provenance while retaining its original relative src.
    # Rebase only when precisely one exact ancestor+literal-path target is inventoried.
    if state=='MISSING' and not os.path.isabs(path):
        candidates={}
        for ancestor in doc.parents:
            if ancestor==roots[root_id].parent:break
            candidate=os.path.abspath(str(ancestor/path))
            if forbidden(candidate):continue
            if candidate in lookup:candidates[candidate]=lookup[candidate]
        if len(candidates)==1:
            candidate,item=next(iter(candidates.items()));target=item['published_relative'];compressed=bool(item.get('gzipchunks'))
            return {'status':'LOCAL_INCLUDED','source_absolute':candidate,'source_root_id':item['source_root_id'],
                    'source_relative_path':item['source_relative_path'],'published_logical_target':target,
                    'published_target':target+'.parts.json' if compressed else target,'descriptor_link_only':compressed,
                    'resolution':'UNIQUE_EXACT_ANCESTOR_LITERAL_PATH_REBASE','original_unresolved_path':absolute,
                    'candidate_count':1,'source_image_sha256':item.get('source_sha256')}
        if len(candidates)>1:return {'status':'AMBIGUOUS_EXISTING_BASES','candidates':[r['published_relative'] for r in candidates.values()],'source_absolute':absolute}
    if state=='MISSING' and official_remote():return official_remote()
    return {'status':state,'source_absolute':absolute,'source_root_id':root_id,'source_relative_path':relative}

def extract_data_uri(src):
    header,payload=src.split(',',1)
    fields=header[5:].split(';');mime=fields[0].lower()
    if mime not in MIMES:raise ValueError('Unsupported image MIME:'+mime)
    data=base64.b64decode(payload,validate=True) if 'base64' in fields[1:] else unquote_to_bytes(payload)
    return mime,data

def rel_link(target,page):return quote(posixpath.relpath(target,posixpath.dirname(page)),safe='/.:@-_%')

def run(manifest_path,summary_path,out,allow_bytes,origins_path=None):
    rows=[json.loads(l) for l in manifest_path.open() if l.strip()]
    roots={k:Path(v) for k,v in json.loads(summary_path.read_text())['source_roots'].items()}
    origins={r['source_absolute']:r for r in json.loads(origins_path.read_text())} if origins_path else {}
    lookup={os.path.abspath(r['source_absolute']):r for r in rows if r['action']=='INCLUDE' and not forbidden(r['source_relative_path'])}
    sources=sorted([r for r in rows if r['action']=='INCLUDE' and r.get('suffix','').lower()=='.html' and not forbidden(r['source_relative_path'])],key=lambda r:r['published_relative'])
    out.mkdir(parents=True,exist_ok=False)
    galleries=[];observations=[];assets=[];overrides={};stats=collections.Counter();derived=[]
    index=['# 전체 literal HTML 이미지 갤러리','',
           '원 HTML의 실제 `<img>` 요소를 출현 순서 그대로 연결한 Markdown 파생본입니다. 실패·UNKNOWN을 제거하지 않았고 모델 결과로 사례를 다시 고르지 않았습니다. 각 HTML마다 최대100개 이미지 요소씩 나눴습니다. 원문 HTML은 수정하지 않았습니다.','',
           '원 caption/alt/title와 직전 heading은 텍스트로 보존했습니다. heading은 그림과의 인과 설명이 아니라 원 문서의 문맥입니다. HTML의 JS를 실행하거나 동적으로 생길 이미지를 추측하지 않았습니다. 외부 이미지는 원 URL 링크만 제공하며 재다운로드하지 않습니다.','',
           '로컬 경로는 archive mapping으로 연결합니다. Base64 data URI는 픽셀 디코딩 없이 원 파일 bytes로만 추출한 별도 자산입니다. 이미지의 의미·가림 원인·새 모델 성공을 이 변환이 추가로 검증하지 않습니다.','',
           '| 원 HTML | literal img 수 | Markdown pages |','|---|---|---|']
    for source in sources:
        doc=Path(source['source_absolute'])
        if safe_status(doc)!='EXISTS':raise RuntimeError('Source HTML missing or symlink:'+str(doc))
        raw=doc.read_bytes()
        if sha(raw)!=source['source_sha256']:raise RuntimeError('Source HTML drift:'+str(doc))
        parser=Parser();parser.feed(raw.decode('utf-8-sig'))
        stats['HTML_total']+=1;stats['script_elements_not_executed']+=parser.script_elements
        if not parser.images:continue
        stats['HTML_with_literal_images']+=1
        html_id=sha(source['published_relative'].encode())[:16]
        directory='galleries/HTML_'+html_id
        pages=[]
        for start in range(0,len(parser.images),100):
            page=directory+'/PORTABLE_GALLERY_'+str(start//100+1).zfill(3)+'.md';pages.append(page)
            lines=['# '+(parser.title() or source['source_relative_path']), '',
                   '원문: ['+source['source_relative_path']+']('+rel_link(source['published_relative'],page)+')', '',
                   '원 HTML SHA256: `'+source['source_sha256']+'`', '',
                   f'원 img ordinal {start+1}–{min(start+100,len(parser.images))} / {len(parser.images)}. 원 순서 유지; 이미지 pixel decode/모델/평가 재실행 없음.', '']
            if start>0:lines+=['[이전 page]('+Path(pages[-2]).name+')','']
            for image in parser.images[start:start+100]:
                src=image['src'];mapped=reference(src,doc,parser,lookup,roots,origins.get(str(doc)))
                record={'source_HTML':source['source_relative_path'],'source_root_id':source['source_root_id'],
                        'source_HTML_sha256':source['source_sha256'],'published_source_HTML':source['published_relative'],
                        'ordinal':image['ordinal'],'original_alt':image['alt'],'original_title':image['title'],
                        'original_caption_texts':parser.captions(image),'preceding_heading':parser.heading(image),
                        'original_src':src if mapped['status']!='DATA_URI' else 'data URI recorded by MIME/byteSHA and sourceHTML+ordinal',
                        'page':page,**mapped}
                if mapped['status']=='DATA_URI':
                    if not allow_bytes:raise RuntimeError('Data URI extraction requires explicit byte-only flag')
                    mime,data=extract_data_uri(src);h=sha(data);target='gallery_assets/'+h+MIMES[mime]
                    f=out/target;f.parent.mkdir(exist_ok=True)
                    if f.exists():assert f.read_bytes()==data
                    else:f.write_bytes(data)
                    record.update({'status':'DATA_URI_BYTE_ASSET','mime_type':mime,'byte_sha256':h,'bytes':len(data),'published_target':target,'image_pixel_decoded':False})
                    assets.append({k:record[k] for k in ['source_HTML','source_HTML_sha256','ordinal','mime_type','byte_sha256','bytes','published_target','image_pixel_decoded']})
                if mapped['status']=='LOCAL_IMAGE_OVERRIDE_NEEDED':
                    key=mapped['source_absolute'];o=overrides.setdefault(key,{k:mapped[k] for k in ['source_absolute','source_root_id','source_relative_path','published_target']})
                    o.setdefault('referenced_by',[]).append({'source_HTML':source['source_relative_path'],'ordinal':image['ordinal']})
                stats[record['status']]+=1;observations.append(record)
                lines+=['## 원 이미지 '+str(image['ordinal']), '']
                if record.get('preceding_heading'):lines+=caption_block('원문 직전 heading',record['preceding_heading'])
                lines+=caption_block('원 alt',image['alt']);lines+=caption_block('원 title',image['title'])
                for caption in record['original_caption_texts']:lines+=caption_block('원 HTML caption',caption)
                if record.get('resolution')=='UNIQUE_EXACT_ANCESTOR_LITERAL_PATH_REBASE':
                    lines+=['경로 대응 참고: 보존 HTML의 상대 src를 원 상위 폴더들에 대입했을 때 C manifest에서 유일하게 일치한 파일입니다. 원 HTML은 바꾸지 않았으며, 과거 HTML 작성 시점의 별도 이미지 bytes까지 인증했다는 뜻은 아닙니다. 원 경로·대응 근거·현재 source SHA는 추가 manifest에 남겼습니다.','']
                if record['status'] in {'LOCAL_INCLUDED','LOCAL_IMAGE_OVERRIDE_NEEDED','DATA_URI_BYTE_ASSET'}:
                    target=record['published_target'];link=rel_link(target,page)
                    if record.get('descriptor_link_only'):lines+=['[원 이미지 lossless descriptor]('+link+')','']
                    else:lines+=['![원 이미지 '+str(image['ordinal'])+']('+link+')','']
                    if record['status']=='LOCAL_IMAGE_OVERRIDE_NEEDED':lines+=['게시 포함 여부를 최종 archive manifest에서 확인할 별도 직접참조 예외입니다.','']
                elif record['status']=='REMOTE_LINK_ONLY':lines+=['[원 외부 이미지 URL — 재다운로드하지 않음]('+record['url']+')','']
                else:lines+=['원 이미지 요소 유지: `'+record['status']+'`. 원문 HTML/provenance를 확인하십시오. 새 파일을 추측하지 않았습니다.','']
            if start+100<len(parser.images):lines+=['[다음 page](PORTABLE_GALLERY_'+str(start//100+2).zfill(3)+'.md)','']
            f=out/page;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('\n'.join(lines))
            derived.append({'source':str(f),'published_relative_path':page,'sha256':sha(f.read_bytes())})
        gallery={'source_root_id':source['source_root_id'],'source_HTML':source['source_relative_path'],'source_HTML_sha256':source['source_sha256'],
                 'published_source_HTML':source['published_relative'],'literal_images':len(parser.images),'pages':pages,'script_elements_not_executed':parser.script_elements}
        galleries.append(gallery)
        label=source['source_relative_path'].replace('|','\\|')
        index.append('| ['+label+']('+quote(source['published_relative'],safe='/.:@-_%')+') | '+str(len(parser.images))+' | '+' · '.join('['+str(i+1)+']('+p+')' for i,p in enumerate(pages))+' |')
    (out/'GALLERY_INDEX_KO.md').write_text('\n'.join(index)+'\n')
    derived.append({'source':str(out/'GALLERY_INDEX_KO.md'),'published_relative_path':'GALLERY_INDEX_KO.md','sha256':sha((out/'GALLERY_INDEX_KO.md').read_bytes())})
    for target in sorted({a['published_target'] for a in assets}):derived.append({'source':str(out/target),'published_relative_path':target,'sha256':sha((out/target).read_bytes()),'image_pixel_decoded':False})
    for name,value in [('HTML_GALLERY_MANIFEST.json',galleries),('LITERAL_IMAGE_OCCURRENCES.json',observations),('DATA_URI_BYTE_ASSET_MANIFEST.json',assets),('LOCAL_IMAGE_OVERRIDES_NEEDED.json',list(overrides.values())),('DERIVED_ADDITION_FILE_MANIFEST.json',derived)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    summary={'completed':True,'literal_only':True,'counts':dict(stats),'galleries':len(galleries),'Markdown_pages':sum(len(g['pages']) for g in galleries),
             'dataURI_occurrences':len(assets),'unique_extracted_byte_assets':len({a['published_target'] for a in assets}),
             'local_image_override_unique':len(overrides),'all_source_HTML_sha_match':True,'all_source_order_ordinals_retained':True,
             'image_pixel_decodes':0,'external_image_downloads':0,'JS_executions_or_dynamic_image_inference':0,
             'raw_evaluation_score_model_reads':0,'original_HTML_or_prior126_documents_modified':0,'V112_stat_read_hash_search':0,
             'actual_staged_Git_or_remote_link_check':False,'final_candidate_manifest_sha256':sha(manifest_path.read_bytes())}
    (out/'COMPLETE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary

def selftest():
    p=Parser();p.feed('<html><title>Title</title><body><h2>UNKNOWN</h2><figure><img src="a.png" alt="A &amp; B"><figcaption>failed <b>case</b></figcaption></figure><img src="b.png"><script>new Image()</script></body></html>')
    assert [x['ordinal'] for x in p.images]==[1,2] and p.images[0]['alt']=='A & B'
    assert p.captions(p.images[0])==['failed case'] and p.captions(p.images[1])==[]
    assert p.heading(p.images[1])=='UNKNOWN' and p.script_elements==1
    assert extract_data_uri('data:image/png;base64,AAEC')[1]==b'\x00\x01\x02'
    assert rel_link('archive/task-2/P/a.png','galleries/G/PORTABLE_GALLERY_001.md')=='../../archive/task-2/P/a.png'
    assert forbidden('https://example.org/query_x_v112/a.png')
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/'x').symlink_to('/tmp')
        assert safe_status(root/'x'/'anything')=='SYMLINK_EXCLUDED'
        old=os.lstat
        def no(_):raise AssertionError('forbidden stat')
        os.lstat=no
        try:assert safe_status(root/'q_v112'/'x')=='EXCLUDED_V112'
        finally:os.lstat=old
        fake={'source_root_id':'P','source_relative_path':'experiment/a.png','published_relative':'archive/P/experiment/a.png'}
        lookup={str(root/'experiment/a.png'):fake};doc=root/'experiment/provenance/old.html'
        r=reference('a.png',doc,Parser(),lookup,{'P':root})
        assert r['status']=='LOCAL_INCLUDED' and r['resolution']=='UNIQUE_EXACT_ANCESTOR_LITERAL_PATH_REBASE'
        lookup[str(root/'a.png')]={**fake,'published_relative':'archive/P/a.png'}
        assert reference('a.png',doc,Parser(),lookup,{'P':root})['status']=='AMBIGUOUS_EXISTING_BASES'
        pa=Parser();pa.feed('<meta property="og:url" content="https://official.example/page">')
        assert reference('/assets/a.png',doc,pa,{}, {'P':root})['url']=='https://official.example/assets/a.png'
    return {'PASS':True,'invented_groups':11,'image_pixel_decodes':0,'checks':['literal order','caption/alt entities','separate heading','JS not executed','base64 originalbytes','published relative target','V112 URL preprune','symlink and instrumented V112 no-stat','unique ancestor literal-path rebase','ambiguous rebase retained','explicit original document origin']}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path);ap.add_argument('--summary',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--official-document-origins',type=Path);ap.add_argument('--allow-base64-byte-extraction',action='store_true');ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test:print(json.dumps(selftest()))
    else:
        if not all([a.manifest,a.summary,a.out]):ap.error('manifest/summary/out required')
        print(json.dumps(run(a.manifest,a.summary,a.out,a.allow_base64_byte_extraction,a.official_document_origins),ensure_ascii=False))
