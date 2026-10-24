#!/usr/bin/env python3
"""Root-authorized single-literal HTML wording repair; no science execution."""
import hashlib
import json
import os
import stat

D='/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_part_query_support_contract_20261025_v216'
H=D+'/report_wording_repair_a_v2'
BEFORE='새 SOURCE label/admission/paired controls/fit은 하지 않았습니다.'.encode()
AFTER='새 SOURCE weak 관찰은 기록했지만, 새 학습 타깃 채택·paired controls·fit은 하지 않았습니다.'.encode()
EXPECTED_OLD_READY='b7ffbb92717384f0dd33184a423d59c0d9da63a2d8920c33b4a852b2880b97d3'
EXPECTED_OLD_HTML='cf27937f562d5585201ca3b1ec2e7db037455f963615b780e632fbe6f2284dbe'

def opened(p,flags=os.O_RDONLY):
    if 'v112' in p.lower():raise ValueError('Forbidden lexical path before filesystem access')
    if not os.path.isabs(p) or os.path.normpath(p)!=p:raise ValueError('normalized absolute path required')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for c in p.strip('/').split('/')[:-1]:
            n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        r=os.open(p.rsplit('/',1)[1],flags|os.O_NOFOLLOW,0o644,dir_fd=fd)
        if not stat.S_ISREG(os.fstat(r).st_mode):os.close(r);raise ValueError('regular file required')
        return r
    finally:os.close(fd)

def read(p):
    with os.fdopen(opened(p),'rb') as f:return f.read()

def digest(b):return hashlib.sha256(b).hexdigest()
def binding(p,b):return {'path':p,'sha256':digest(b),'size_bytes':len(b)}
def exclusive(p,b):
    with os.fdopen(opened(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL),'wb') as f:f.write(b)
    assert read(p)==b
    return binding(p,b)

def main():
    ready_raw=read(D+'/report_preparation_a/FINAL_READY.json')
    assert digest(ready_raw)==EXPECTED_OLD_READY
    old=json.loads(ready_raw)
    reports={name:read(b['path']) for name,b in old['outputs'].items()}
    assert all(digest(reports[k])==b['sha256'] for k,b in old['outputs'].items())
    original_html=reports['ROOT_READOUT_KO.html']
    assert digest(original_html)==EXPECTED_OLD_HTML and original_html.count(BEFORE)==1
    generator=read(old['builder']['path'])
    assert digest(generator)==old['builder']['sha256']
    saved={
      'original_HTML':exclusive(H+'/ROOT_READOUT_KO_BEFORE_WORDING_REPAIR.html',original_html),
      'original_READY':exclusive(H+'/FINAL_READY_BEFORE_WORDING_REPAIR.json',ready_raw),
      'original_generator':exclusive(H+'/build_saved_v216_reports_BEFORE_WORDING_REPAIR.py',generator),
    }
    current_html=original_html.replace(BEFORE,AFTER,1)
    assert current_html.count(BEFORE)==0 and current_html.count(AFTER)==1
    assert current_html.replace(AFTER,BEFORE,1)==original_html
    target=D+'/ROOT_READOUT_KO.html'
    assert read(target)==original_html
    with os.fdopen(opened(target,os.O_WRONLY|os.O_TRUNC),'wb') as f:f.write(current_html)
    assert read(target)==current_html
    current={}
    for name,b in old['outputs'].items():
        actual=read(b['path'])
        assert actual==(current_html if name=='ROOT_READOUT_KO.html' else reports[name])
        current[name]=binding(b['path'],actual)
    assert read(D+'/report_preparation_a/FINAL_READY.json')==ready_raw
    assert read(old['builder']['path'])==generator
    selfpath=os.path.abspath(__file__)
    code=binding(selfpath,read(selfpath))
    receipt={
      'completed':True,'PASS':True,'repair_type':'ONE_LITERAL_ONE_OCCURRENCE_HTML_WORDING_ONLY',
      'root_authorization':'Root explicitly authorized preserving original artifacts and one HTML sentence replacement before whole-phase freeze.',
      'before_literal':BEFORE.decode(),'after_literal':AFTER.decode(),
      'before_HTML':old['outputs']['ROOT_READOUT_KO.html'],'after_HTML':current['ROOT_READOUT_KO.html'],
      'original_preserved_byte_exact':saved,'repair_code':code,
      'historical_original_READY':binding(D+'/report_preparation_a/FINAL_READY.json',ready_raw),
      'historical_original_generator':old['builder'],
      'other_three_ROOT_files_byte_unchanged':{k:v for k,v in current.items() if k!='ROOT_READOUT_KO.html'},
      'checks':{'exactly_one_literal_replaced':True,'reverse_replacement_recovers_original_bytes':True,'original_HTML_READY_generator_preserved_before_HTML_write':True,'original_READY_generator_immutable':True,'other_three_ROOT_files_unchanged':True},
      'annotation_join_generator_science_reruns':0,'GT_arrays_models_scores_PNG_reads':0,'Git_mutations':0,
      'future_exposure_note_receipt_only':'SOURCE_DEV was historically exposed and is not fresh blind. Warm-starting from an older all-SOURCE head would not automatically establish independent DEV evaluation; future training must explicitly account for initialization exposure.',
      'no_whole_phase_freeze':True,
    }
    receipt_binding=exclusive(H+'/WORDING_REPAIR_RECEIPT.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode())
    final={
      'completed':True,'PASS':True,'FINAL_READY_FROZEN':True,'current_report_bindings':current,'outputs':current,
      'repair_receipt':receipt_binding,'repair_code':code,'original_preservation':saved,
      'old_READY_is_historical_original_build_only':True,
      'historical_original_READY':receipt['historical_original_READY'],
      'original_report_builder_is_historical_pre_wording_repair':old['builder'],
      'original_build_all_link_and_numeric_checks_retained':old['checks'],
      'numeric_counts_other_three_reports_and_link_targets_changed':False,
      'phase_science_annotation_join_reruns':0,'no_whole_phase_freeze':True,'own_repair_files_frozen_no_further_writes':True,
    }
    final_binding=exclusive(H+'/FINAL_READY.json',(json.dumps(final,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps({'completed':True,'PASS':True,'FINAL_READY':final_binding,'receipt':receipt_binding,'current_reports':current,'repair_code':code},ensure_ascii=False))

if __name__=='__main__':main()
