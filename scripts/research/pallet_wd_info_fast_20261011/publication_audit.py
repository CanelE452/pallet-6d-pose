"""Check public artifacts and prepare an explicit new-file publication list."""
import hashlib
import re
import subprocess
import time
from pathlib import Path
from PIL import Image
from . import common as C


def run():
    required=['RESULTS_KO.md','REPRODUCE.md','C_B1_REPRODUCE.md','EXPERIMENT_STATUS.csv',
        'PLAN_LOCK.json','METRICS.json','PAIRED.json','COMPUTE_RECEIPT.json',
        'CHANGED_SELECTIONS.csv','SOURCE_PRESERVATION.json',
        'FINAL_AGGREGATE_VERIFICATION_APPROVED.json','C_B1_VERIFICATION.json']
    assert all((C.DOC/name).is_file() for name in required)
    approved=C.read(C.DOC/'FINAL_AGGREGATE_VERIFICATION_APPROVED.json')
    assert approved['status']=='PASS',approved['status']
    links=[]
    for f in C.DOC.rglob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',f.read_text()):
            if target.startswith(('https:','http:','mailto:')):continue
            target=target.split('#')[0].strip('<>')
            if not target:continue
            path=(f.parent/target).resolve()
            exists=path.exists()
            if not exists and path.is_relative_to(C.ROOT):
                found=subprocess.run(['git','-C',str(C.ROOT),'cat-file','-e',
                    'HEAD:'+str(path.relative_to(C.ROOT))],capture_output=True)
                exists=found.returncode==0
            assert exists,(f.name,target)
            links.append(dict(document=f.name,target=target,exists=True))
    figures=[]
    for f in sorted((C.DOC/'FIGURES').glob('*.png')):
        with Image.open(f) as im:
            im.verify()
        with Image.open(f) as im:
            figures.append(dict(path=str(f.relative_to(C.ROOT)),size=im.size,
                DPI=im.info.get('dpi'),sha256=C.sha(f),bytes=f.stat().st_size))
    assert len(figures)==9
    public=[]
    for root in [C.DOC,C.ROOT/'scripts/research/pallet_wd_info_fast_20261011']:
        for f in sorted(root.rglob('*')):
            if not f.is_file():continue
            assert f.suffix not in ('.pt','.pth','.npz','.jpeg','.jpg','.pdf','.tex','.pyc'),f.name
            assert '__pycache__' not in f.parts
            assert f.stat().st_size<16*1024*1024,f.name
            if f.suffix in ('.md','.json','.csv','.py','.mplstyle','.patch'):
                prefixes=['/'+'home'+'/', '/'+'dev'+'/'+'shm'+'/', '/'+'tmp'+'/']
                assert not any(token in f.read_text() for token in prefixes),f.name
            public.append(C.binding(f))
    original_status=subprocess.check_output(['git','-C',str(C.SOURCE),'status','--porcelain=v1','-z'])
    original_diff=subprocess.check_output(['git','-C',str(C.SOURCE),'diff','--binary','HEAD'])
    start=C.read(C.PRIVATE/'START.json')
    assert hashlib.sha256(original_status).hexdigest()==start['status_sha256']
    assert hashlib.sha256(original_diff).hexdigest()==start['tracked_diff_sha256']
    elapsed=(time.time()-start['start_unix'])/60
    assert elapsed<60,('Project conservative wall budget exceeded',elapsed)
    C.write(C.DOC/'PREPUBLICATION_AUDIT.json',dict(status='PASS',
        required_files=required,local_markdown_links=links,figures=figures,
        scientific_verifier=C.binding(C.DOC/'FINAL_AGGREGATE_VERIFICATION_APPROVED.json'),
        original449_changes_preserved=True,original_diff_and_status_identical=True,
        current_project_wall_minutes=elapsed,public_file_bindings=public,
        manifest_self_hash_excluded=True,privacy_paths_or_private_checkpoints_found=0))
    paths=[str(C.ROOT/x['path']) for x in public]
    paths.append(str(C.DOC/'PREPUBLICATION_AUDIT.json'))
    (C.PRIVATE/'PUBLICATION_PATHS.txt').write_text('\n'.join(paths)+'\n')
    print('PREPUBLICATION_PASS',len(paths),'new files;',len(links),'links;',len(figures),'PNGs')


if __name__=='__main__':run()
