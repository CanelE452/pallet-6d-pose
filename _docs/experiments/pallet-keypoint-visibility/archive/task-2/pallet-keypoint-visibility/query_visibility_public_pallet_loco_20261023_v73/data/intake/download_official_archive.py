"""Bounded official LOCO archive intake. No full extraction or RGB decode."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,time,urllib.request,zipfile

OUT=Path(__file__).resolve().parent
URL='https://go.mytum.de/239870'
EXPECTED=769055500

def write(name,obj):
    p=OUT/name;assert not p.exists(),p;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def fetch_metadata(url,name):
    p=OUT/name;assert not p.exists(),p
    with urllib.request.urlopen(url,timeout=30) as r:
        data=r.read();status=r.status;resolved=r.url
    assert len(data)<1000000
    p.write_bytes(data)
    return dict(path=str(p),url=url,resolved_URL=resolved,status=status,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def main():
    assert shutil.disk_usage(OUT).free>2_000_000_000
    research=OUT.parents[2]/'query_visibility_cue_occlusion_20261023_v72/source_audit/PUBLIC_PALLET_DATA_NEXTSTEP_RESEARCH.json'
    source_copy=OUT/'PRIOR_PUBLIC_PALLET_RESEARCH_IMMUTABLE_COPY.json'
    assert not source_copy.exists();source_copy.write_bytes(research.read_bytes())
    meta=[fetch_metadata('https://raw.githubusercontent.com/tum-fml/loco/main/LICENSE','LICENSE_CC0_OFFICIAL.txt'),
          fetch_metadata('https://raw.githubusercontent.com/tum-fml/loco/main/README.md','README_OFFICIAL.md'),
          fetch_metadata('https://api.github.com/repos/tum-fml/loco/git/trees/main?recursive=1','OFFICIAL_REPO_TREE.json')]
    write('INTAKE_PROTOCOL.json',dict(schema='V73_OFFICIAL_LOCO_ACTUAL_INTAKE_PROTOCOL_V1',frozen_UTC=datetime.now(timezone.utc).isoformat(),
      official_archive_URL=URL,expected_archive_bytes=EXPECTED,expected_size_evidence='OfficialserverHTTPHEADContentLength observed; not a publisherchecksum.',
      license='OfficialdatasetrepositoryCC0-1.0',metadata_files=meta,prior_research_copy_sha256=hashlib.sha256(source_copy.read_bytes()).hexdigest(),
      budget_download_bytes=1_200_000_000,full_archive_extraction=False,
      planned_selection='After officialCOCO metadata inspection, freeze <=20 pallet RGB by SHA(imageid+filename), stratified across the five official source subset groups if supported. Freeze before any RGBdecode/view; no model/cue/current-data selection.',
      labels='Inspect actualbbox/category/keypoint/occlusion keys. Missingsemanticnative8/E/S staysmissing.',
      training_admission='HOLD_PENDING_NATIVE8_CUE_SCENE_INDEPENDENT_CONTRACT',current_RGB_GT_scores_weights_forward_fit_PnP=0))
    part=OUT/'dataset.zip.part';final=OUT/'dataset.zip';assert not final.exists()
    offset=part.stat().st_size if part.exists() else 0
    assert offset<EXPECTED
    req=urllib.request.Request(URL,headers={'Range':f'bytes={offset}-'} if offset else {})
    started=time.monotonic();last_report=started;last_bytes=offset
    with urllib.request.urlopen(req,timeout=60) as response:
        if offset:
            assert response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-'),dict(response.headers)
        else:assert response.status==200
        write('ARCHIVE_HTTP_RECEIPT.json',dict(requested_URL=URL,resolved_URL=response.url,status=response.status,headers=dict(response.headers),resume_offset=offset,body_download_expected=EXPECTED-offset))
        with part.open('ab' if offset else 'wb') as f:
            while True:
                block=response.read(4*1024*1024)
                if not block:break
                f.write(block);offset+=len(block);assert offset<=EXPECTED
                now=time.monotonic()
                if offset-last_bytes>=64*1024*1024 or now-last_report>=25:
                    print(json.dumps(dict(bytes=offset,total=EXPECTED,MiB=round(offset/1048576,1),elapsed_s=round(now-started,1))),flush=True);last_bytes=offset;last_report=now
    assert part.stat().st_size==EXPECTED
    sha=hashlib.sha256();md5=hashlib.md5()
    with part.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):sha.update(block);md5.update(block)
    part.rename(final)
    with zipfile.ZipFile(final) as z:
        inventory=[dict(name=i.filename,compressed_bytes=i.compress_size,uncompressed_bytes=i.file_size,CRC32=f'{i.CRC:08x}',method=i.compress_type,is_directory=i.is_dir()) for i in z.infolist()]
        assert all(not i['name'].startswith('/') and '..' not in Path(i['name']).parts for i in inventory)
    write('ARCHIVE_INVENTORY.json',dict(schema='V73_ACTUAL_ZIP_CENTRAL_DIRECTORY',files=inventory,full_extraction=False))
    write('ARCHIVE_DOWNLOAD_RECEIPT.json',dict(completed=True,path=str(final),bytes=final.stat().st_size,expected_bytes=EXPECTED,size_match=True,sha256=sha.hexdigest(),MD5_observed=md5.hexdigest(),publisher_checksum_verified=False,
      checksum_meaning='Locallyobservedfilehashes, noauthenticityclaimbeyondofficialHTTPSsource/size. Selected ZIPmembers will be CRC-checked on read; wholearchiveCRCscan notyetperformed.',metadata_bytes=sum(x['bytes'] for x in meta),download_budget_bytes=1_200_000_000,
      archive_entries=len(inventory),total_uncompressed_bytes=sum(i['uncompressed_bytes'] for i in inventory),seconds=time.monotonic()-started,RGB_decoded=0,model_current_fit_PnP=0))
    print(json.dumps(dict(completed=True,archive_bytes=EXPECTED,sha256=sha.hexdigest(),entries=len(inventory),receipt=str(OUT/'ARCHIVE_DOWNLOAD_RECEIPT.json'))),flush=True)
if __name__=='__main__':main()
