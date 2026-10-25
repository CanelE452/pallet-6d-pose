"""Final text/link checks, reusing sealed numeric QA without data/model replay."""
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from urllib.parse import unquote,urlsplit

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_source_target_address_consistency_20261025_v223'
H=P/'publication_root_preparation/V223_REPORT_QA_B'


def raw(p):
    p=Path(p);assert p.is_absolute() and 'v112' not in str(p).lower()
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nextfd=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nextfd
        leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        with os.fdopen(leaf,'rb') as f:b=f.read()
    finally:os.close(fd)
    return b


def bound(p,sha):
    b=raw(p);assert hashlib.sha256(b).hexdigest()==sha
    return b,dict(path=str(p),sha256=sha,size_bytes=len(b))


def write(name,b):
    fd=os.open(H/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as f:f.write(b)
    return dict(path=str(H/name),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))


def main():
    nb,numb=bound(H/'NUMERIC_QA.json','e6775bf1bd6c56de46d9232c57f352fdfd27bb5d731134ac25f8c049d4173285')
    num=json.loads(nb);assert num['completed'] and num['PASS']
    bindings={}
    reports=[]
    for n in ('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md'):
        b,bindings[n]=bound(V/n,'10d7adfadd1f5e38b5d13a91a5cfe7d48b45ff1b1c563c66ef8ef6bb6b10a906');reports.append(b)
    assert reports[0]==reports[1]
    md=reports[0].decode()
    jb,bindings['ROOT_RESULT.json']=bound(V/'ROOT_RESULT.json','0ca2bc4050a573d2cf061b2e21a7c8748d4e3f3185df894d4a3f43d6cc096a71')
    result=json.loads(jb)
    assert result['completed'] and result['actual_final_updates']==2200 and result['steps_each']==1100 and result['parameters_each']==121561 and result['discarded_probe_updates']==0
    assert result['adopted'] is False and result['selected_arm'] is None and result['fully_solved'] is False and result['goal_status']=='active'
    assert result['new_current223_inference'] is False and result['new_PnP']==0 and result['latest_real_benchmark_unchanged']=='V220'
    assert result['protected_metadata_exception_rows']==360 and result['exception_values_used_for_design_or_fit'] is False
    for key in ('training_complete','source_comparison'):
        assert any(b['path']==result[key]['path'] and b['sha256']==result[key]['sha256'] for b in num['input_bindings'])
    assert result['report']==bindings['PORTABLE_REPORT_KO.md']
    table={}
    for line in md.splitlines():
        if line.startswith('| '):
            cells=[x.strip() for x in line.strip().strip('|').split('|')]
            if len(cells)==3:table[cells[0]]=cells[1:]
    A='C68SHARED_POINT_PART_LOGIT';B='C69SEPARATE_POINT_PART_LOGITS'
    for label,family,role in [('CF FIT117 CM','CF','SOURCE_FIT'),('CF DEV27 CM','CF','SOURCE_DEV'),('natural DEV12 CM','NATURAL_WEAK','SOURCE_DEV')]:
        expected=[json.dumps(num['metrics'][family+'/'+arm+'/'+role]['CM_D_E_rows_D_E_U_columns']).replace(' ','') for arm in (A,B)]
        assert [x.replace(' ','') for x in table[label]]==expected
    for label,family in [('CF DEV','CF'),('natural DEV','NATURAL_WEAK')]:
        for k,cls in [('D',0),('E',1)]:
            expected=[]
            for arm in (A,B):
                m=num['metrics'][family+'/'+arm+'/SOURCE_DEV']['classes'][cls]
                expected.append(f"{100*m['precision']:.2f}% / {100*m['recall']:.2f}%")
            assert table[label+' '+k+' precision / recall']==expected
    assert table['CF DEV 전체 정답']==['17/27','17/27']
    assert table['natural DEV 전체 정답']==['9/12','9/12']
    assert table['natural FIT 전체 정답']==['45/45','45/45']
    assert table['strong TRAIN 실제 routed logit 정답']==['6096/6096','6096/6096']
    assert table['CF DEV 세 상태 모두 정답']==['2/9','1/9']
    assert table['CF DEV BEFORE D / COVER E / SHAM D 정답']==['7/9 ·3/9 ·7/9','7/9 ·4/9 ·6/9']
    for text in ('z>=0','C68 PART/C69 POINT','PART3202','6tensors','121561','252','6297','3638','6132','2 frame/component','GT-derived','새 blind test가 아니다','phase 전체 보호 metadata 접근0/PASS라고 하지 않는다','그림4장','전9장은 갤러리에','키포인트 GT 좌표오차≤3px와 다른 지표'):
        if text=='PART3202':assert 'POINT3202' in md
        else:assert text in md,text
    assert '강한 point gold' not in md
    pending={
        'provenance/V223_ACTUAL_MODEL_QA_A/actual_review_v1/REVIEW.json',
        'provenance/V223_ROOT_PREPARATION/ROOT_SCOPE_EXCEPTION_AND_CONTAINMENT.json',
        'provenance/V223_PUBLIC_SOURCE_COVERAGE_C/READINESS.json',
        'provenance/ARCHIVE_MANIFEST.json'}
    linkchecks=[]
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',md):
        if urlsplit(link).scheme:
            assert urlsplit(link).scheme=='https';linkchecks.append(dict(link=link,external_not_fetched=True));continue
        p=Path(os.path.normpath(str(V/unquote(link.split('#')[0]))));assert 'v112' not in str(p).lower()
        try:
            for x in reversed([p,*p.parents]):
                st=x.lstat();assert not stat.S_ISLNK(st.st_mode)
            assert stat.S_ISREG(st.st_mode);linkchecks.append(dict(link=link,exists=True,no_symlink=True))
        except FileNotFoundError:
            assert link in pending,link
            linkchecks.append(dict(link=link,exists=False,expected_pending_provenance_copy=True))
    assert len(linkchecks)==16
    ko='''# V223 저장 SOURCE 산술·보고서 QA

PASS. 두 모델의 저장 SOURCE 점수 6개를 원행에서 다시 계산하여 CM·D/E precision/recall·U, native별 표, paired 회복/손상과 triplet 판정이 모두 일치했다. JSON metric 40개와 CSV16행을 검산했다.

- CF DEV: C68 [[14,4,0],[6,3,0]], C69 [[13,5,0],[5,4,0]]. E1 회복/D1 손상, 전체17/27로 같고 triplet2/9→1/9다. natural DEV12는 판정 전행 동일9/12다.
- CF FIT117·natural FIT45·strong TRAIN6096은 양쪽 모두 정답이다. strong 점수는 C68 PART/C69 POINT라는 실제 감독 경로를 사용하며, 약감독 primary PART와 골드 권한을 합치지 않는다.
- 저장 canonical input6297개에서 signature3638개·중복 unordered6132쌍·반대 target 그룹0을 독립 집계했다. 이번 검사는 저장 digest 메타데이터를 재집계했으며 원 RGB/NPZ를 재해시하지 않았다. 모든 데이터의 무모순이나 목표 혼용 무해를 입증한 결과가 아니다.
- C68=V222 C67 logit6297개 bit-exact는 Root 제공 SOURCE replay 기록을 인용했다. B가 이번 범위에서 과거 점수 파일까지 다시 열어 비교한 결과는 아니다.

최종 ROOT/PORTABLE Markdown 바이트 일치, ROOT_RESULT 바인딩·예산·분모·z>=0 decoder·표 수치·scope 설명이 정합하다. 보고서는 warm start/공유 encoder/작고 노출된2-component DEV, GT-derived q8, weak finite-part≠physical native gold, U0≠confidence, 채택·새 private 추론·PnP 없음, 목표 미해결을 명시한다. ROI centroid의 RAW96 raster 거리와 사용자 GT3px 지표를 구분한다.

A 모델/로그/CF CPU replay 검산과 Root4장 실제 뷰는 보고서 및 Root가 제공한 증거로 인용한다. B의 실제 이미지 뷰·model/NPZ/PT/원GT/보호 payload 열람·새 forward는0이다. A의 원본 tensor/로그 재감사, ROI 통계·공개 teacher 역사 원자료, 외부 HTTP/브라우저 render는 이 QA에서 재검산하지 않았다. phase 전체 protected metadata 접근0을 주장하지 않으며 C의 held360 eager-parse 예외 공개를 확인했다.

상대링크16개 중 이미 존재하는 로컬10개는 regular/no-symlink이며 외부2개는 미접속이다. provenance4개는 Root가 예고한 보존 복사/manifest 생성 대기다. 이것을 모두 해결된 링크라고 주장하지 않는다. 최종 공개 전에 Root가 생성할 항목이다. 갤러리9장 PNG는 이 QA에서 열지 않았다.

[전수 산술 결과](NUMERIC_QA.json) · [최종 QA](REVIEW.json) · [산술 코드](check_saved_arithmetic.py) · [최종 text/link 코드](finalize_report_QA.py)
'''
    kob=write('KO.md',ko.encode())
    review=dict(completed=True,PASS=True,kind='V223_INDEPENDENT_SAVED_SOURCE_ARITHMETIC_AND_FINAL_REPORT_SCOPE_QA',
                numeric_QA=numb,numeric_terminal='da077c exit0',report_bindings=bindings,
                final_report_QA_pending=False,table_metric_nodes=40,CSV_rows=16,
                independently_verified=num['input_fingerprint_grouping'],
                headline={'CF_DEV_CM':{'C68':[[14,4,0],[6,3,0]],'C69':[[13,5,0],[5,4,0]]},'CF_DEV_correct':[17,17],'CF_DEV_triplets':[2,1],'CF_DEV_recovered':1,'CF_DEV_damaged':1,'natural_DEV_correct':[9,9],'natural_predictions_changed':0},
                score_scope='Weak/CF PART both; strong actual supervised route C68 PART/C69 POINT.',
                replay_scope=num['source_replay'],
                report_scope_checks=dict(z_ge_zero=True,warmstart_and_encoder_transfer_limit=True,POINT_control6_unused_disclosed=True,weak_not_native_gold=True,GTquery_not_production_success=True,DEV_small_preexposed=True,no_adoption_no_privateforward_no_PnP=True,ROI_raster_not_GT3px=True,only_Root4_actualviews_not_all9=True,protected_metadata_exception_disclosed_no_globalzero_claim=True),
                relative_links=linkchecks,publication_pending_expected_links=sorted(pending),
                no_scientific_or_numeric_blocker=True,all_publication_links_currently_exist=False,
                evidence_not_replayed_by_B=['A actual checkpoint/log/CF CPU replay receipt06609a7f supplied by Root','Root four images','old V222 score files for exact replay','raw ROI/camera/GT or public teacher history payloads','external HTTP/browser render'],
                B_access_limits=dict(new_source_images_models_NPZ_PT_GT_protected_payloads=0,new_forward_training=0,broad_scan=0,science_writes=0,exact_comparison_directory_filename_inventory_only=True),
                own_folder_frozen_no_more_writes=True,KO=kob)
    rb=write('REVIEW.json',(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(completed=True,PASS=True,REVIEW=rb,KO=kob,report_bindings=bindings,pending_provenance_links=sorted(pending)),ensure_ascii=False))


if __name__=='__main__':main()
