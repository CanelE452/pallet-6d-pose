"""Split the existing catalog CSV only; no publication/image/source inputs."""
from pathlib import Path
import csv,hashlib,io,json,shutil

H=Path(__file__).resolve().parent;V1=H/'review_image_catalog';OUT=H/'review_image_catalog_v2'
def sha(data):return hashlib.sha256(data).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p.read_bytes()))
def write_json(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    data=(V1/'REVIEW_IMAGE_CATALOG.csv').read_bytes();lines=data.splitlines(keepends=True)
    rows=list(csv.reader(io.StringIO(data.decode('utf-8'))));assert len(rows)==len(lines)==16728
    assert all(len(row)==5 for row in rows) and len(lines[1:])==16727
    OUT.mkdir(exist_ok=False);provenance=OUT/'provenance/BEFORE_CSV_PARTS_V1';provenance.mkdir(parents=True)
    for source in (H/'build_review_image_catalog.py',V1/'REVIEW_IMAGE_CATALOG_KO.md',V1/'CATALOG_COMPLETE.json'):
        shutil.copyfile(source,provenance/source.name)
    shutil.copyfile(V1/'REVIEW_IMAGE_CATALOG.csv',OUT/'REVIEW_IMAGE_CATALOG.csv')
    partsdir=OUT/'csv_parts';partsdir.mkdir();header=lines[0];parts=[];start=1
    while start<len(lines):
        stop=start;size=len(header)
        while stop<len(lines) and stop-start<500 and size+len(lines[stop])<300000:
            size+=len(lines[stop]);stop+=1
        assert stop>start
        part=header+b''.join(lines[start:stop]);name=f'REVIEW_IMAGE_CATALOG_part{len(parts)+1:03d}.csv'
        p=partsdir/name;p.write_bytes(part)
        assert len(part)<300000 and stop-start<=500
        parts.append(dict(filename=name,rows=stop-start,first_data_row=start,last_data_row=stop-1,bytes=len(part),sha256=sha(part),sourcepath=str(p),publishedrelative='publication/review_image_catalog_parts/'+name))
        start=stop
    reconstructed=header+b''.join((partsdir/p['filename']).read_bytes()[len(header):] for p in parts)
    assert reconstructed==data and sum(p['rows'] for p in parts)==16727
    base=(V1/'REVIEW_IMAGE_CATALOG_KO.md').read_text()
    extra=['','## 작은 CSV로 전체 파일 확인','',
           'GitHub에서 표를 열어 보기 쉽도록 전체 CSV의 원래 순서를 유지해 아래 조각을 추가했습니다. 각 조각은 같은 열 제목과 500행 이하, 300,000 bytes 미만입니다. 행 범위는 전체 CSV의 데이터 행 번호(제목 제외, 1부터)입니다. 전체 CSV도 그대로 보존합니다.','',
           '| 조각 | 전체 데이터 행 범위 | 행 수 | bytes |','|---|---:|---:|---:|']
    for p in parts:extra.append(f'| [{p["filename"]}]({p["publishedrelative"]}) | {p["first_data_row"]:,}–{p["last_data_row"]:,} | {p["rows"]:,} | {p["bytes"]:,} |')
    extra+=['',f'분할 합계 **16,727행**. 각 조각의 제목을 한 번만 남기고 순서대로 연결한 bytes의 SHA256은 원 전체 CSV와 동일한 `{sha(data)}`입니다. 새 이미지 열람·선택·평가·원 manifest 변경은 없습니다.','']
    md=OUT/'REVIEW_IMAGE_CATALOG_KO.md';md.write_text(base+'\n'.join(extra),encoding='utf-8')
    qa=dict(completed=True,PASS=True,rows16727=16727,parts=len(parts),each_rows_le500=all(p['rows']<=500 for p in parts),each_bytes_lt300000=all(p['bytes']<300000 for p in parts),
        total_rows=sum(p['rows'] for p in parts),original_csv_sha256=sha(data),reconstructed_sha256=sha(reconstructed),exact_original_bytes=True,original_row_order_preserved=True,
        new_publisher_manifest_image_or_source_reads=0,inputs=[bind(V1/'REVIEW_IMAGE_CATALOG.csv')],v1_preserved=True,code=bind(Path(__file__)),parts_metadata=parts)
    write_json(OUT/'CATALOG_PARTS_QA.json',qa)
    copies=[dict(sourcepath=str(md),sha256=sha(md.read_bytes()),publishedrelative='REVIEW_IMAGE_CATALOG_KO.md'),
            dict(sourcepath=str(OUT/'REVIEW_IMAGE_CATALOG.csv'),sha256=sha(data),publishedrelative='publication/REVIEW_IMAGE_CATALOG.csv'),
            dict(sourcepath=str(OUT/'CATALOG_PARTS_QA.json'),sha256=sha((OUT/'CATALOG_PARTS_QA.json').read_bytes()),publishedrelative='publication/REVIEW_IMAGE_CATALOG_PARTS_QA.json'),
            dict(sourcepath=str(Path(__file__)),sha256=sha(Path(__file__).read_bytes()),publishedrelative='publication/split_review_image_catalog_v2.py')]
    copies.extend(dict(sourcepath=p['sourcepath'],sha256=p['sha256'],publishedrelative=p['publishedrelative']) for p in parts)
    write_json(OUT/'FINAL_COPY_ENTRIES.json',dict(completed=True,PASS=True,entries=copies))
    print(json.dumps(dict(completed=True,PASS=True,parts=len(parts),rows=16727,max_part_bytes=max(p['bytes'] for p in parts),qa=bind(OUT/'CATALOG_PARTS_QA.json'),copy_entries=bind(OUT/'FINAL_COPY_ENTRIES.json'),MD=bind(md)),ensure_ascii=False))

if __name__=='__main__':main()
