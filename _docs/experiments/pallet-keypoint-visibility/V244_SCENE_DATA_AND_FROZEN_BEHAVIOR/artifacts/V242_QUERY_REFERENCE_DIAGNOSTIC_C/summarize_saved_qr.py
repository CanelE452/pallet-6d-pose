"""기존 independent data QA의 saved q/r 관측만 stdlib 집계. 새 pixel/model read 없음."""
import csv,hashlib,json,math,os,stat
from pathlib import Path

H=Path(__file__).absolute().parent
Q=H.parent/'V242_SCENE_DATA_QA_C/actual_review_v1'
PINS={'REVIEW.json':'33f996e71759f0d49a1587e1ff3b7c2927b6e7e61e0b55abecf53096df334bd4',
      'Q_VS_REFERENCE_OBSERVATIONS.json':'1c931969582cdfbeb70262e6760d55851754e16bb1eae0933f0dcd7742ff3ef8'}
KINDS=('continuous_reference','nearest_half_up_3x3')

def need(ok,message):
    if not ok:raise ValueError(message)

def safe(p):
    need(not any(x in str(p).casefold() for x in ('v112','__pycache__','.pyc')) and '..' not in p.parts,'lexical before FS')
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'non-directory/symlink parent')
    try:need(stat.S_ISREG(os.lstat(p).st_mode),'leaf not regular')
    except FileNotFoundError:pass
    return p

def read(p):
    with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def bind(p):
    b=read(p);return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}

def write_json(name,value):
    p=H/name
    with os.fdopen(os.open(safe(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return bind(p)

def run():
    inputs={};docs={}
    for name,sha in PINS.items():
        raw=read(Q/name);need(hashlib.sha256(raw).hexdigest()==sha,'input SHA changed')
        inputs[name]={'path':str(Q/name),'sha256':sha,'size_bytes':len(raw)};docs[name]=json.loads(raw)
    review=docs['REVIEW.json'];rows=docs['Q_VS_REFERENCE_OBSERVATIONS.json']
    need(review['completed'] is True and review['PASS'] is True and len(rows)==5406,'independent dataset QA terminal PASS/all5406')
    groups={(family,bin):{'family':family,'q_reference_bin':bin,'parents':0,'constructed':0,'r_proof_PASS_denominator':0,'q_both_footprints_and_SHAM_RGB_not_PASS':0,'footprints':{k:{'supported':0,'unsupported_UNKNOWN':0,'COVER_all_opaque_FAIL':0,'SHAM_all_clear_FAIL':0,'combined_FAIL':0,'both_FAIL':0,'SHAM_RGB_unchanged_FAIL':0,'full_footprint_proof_FAIL':0} for k in KINDS}} for family in (0,1) for bin in ('LE3','GT3')}
    ids=set()
    for i,r in enumerate(rows):
        need(r['pair_index']==i and r['parent_sample_index'] not in ids,'all5406 fixed unique pair order');ids.add(r['parent_sample_index'])
        dist=r['q_reference_distance_px'];need(isinstance(dist,(int,float)) and math.isfinite(dist) and dist>=0,'saved finite q-r distance')
        bin='LE3' if dist<=3 else 'GT3';need(r['q_reference_bin']==bin and r['family'] in (0,1),'saved distance bins <=3/>3')
        g=groups[r['family'],bin];g['parents']+=1;g['constructed']+=int(r['constructed'])
        need(r['labels_unchanged_by_q_diagnostic'] is True and (not r['r_proof_pass'] or r['constructed']),'labels unchanged/rproof requires construction')
        if not r['r_proof_pass']:continue
        g['r_proof_PASS_denominator']+=1;o=r['q_observation']
        need(o['teacher_reference_only'] is True and o['q_center_visibility_inferred'] is False and o['whole_intended_part_gold'] is False,'diagnostic authority not upgraded')
        g['q_both_footprints_and_SHAM_RGB_not_PASS']+=int(o['PASS'] is not True)
        flags=[]
        for kind in KINDS:
            x=o['checks'][kind];c=g['footprints'][kind]
            need(type(x['supported']) is bool,'supported bool')
            if not x['supported']:c['unsupported_UNKNOWN']+=1;flags.append(False);continue
            c['supported']+=1
            for key in ('COVER_all_opaque','SHAM_all_clear','SHAM_reference_RGB_unchanged'):need(type(x[key]) is bool,'saved footprint flags bool')
            a=not x['COVER_all_opaque'];b=not x['SHAM_all_clear'];rgb=not x['SHAM_reference_RGB_unchanged']
            c['COVER_all_opaque_FAIL']+=int(a);c['SHAM_all_clear_FAIL']+=int(b);c['combined_FAIL']+=int(a or b);c['both_FAIL']+=int(a and b);c['SHAM_RGB_unchanged_FAIL']+=int(rgb);c['full_footprint_proof_FAIL']+=int(a or b or rgb)
            flags.append(not(a or b or rgb))
        need(o['PASS']==all(flags),'q saved PASS equals two complete footprint checks')
    aggregate=[];table=[]
    for (family,bin),g in groups.items():
        cnt=review['counts'];prefix=f'family{family}_{bin}'
        need(g['parents']==cnt[prefix+'_parents'] and g['constructed']==cnt[prefix+'_constructed'] and g['r_proof_PASS_denominator']==cnt[prefix+'_proof_pass'],'independent review family/bin census')
        aggregate.append(g)
        for kind,c in g['footprints'].items():
            n=g['r_proof_PASS_denominator'];need(c['supported']+c['unsupported_UNKNOWN']==n and c['combined_FAIL']==c['COVER_all_opaque_FAIL']+c['SHAM_all_clear_FAIL']-c['both_FAIL'],'footprint denominator/inclusion-exclusion')
            table.append({**{k:g[k] for k in ('family','q_reference_bin','parents','constructed','r_proof_PASS_denominator')},'footprint':kind,**c,'combined_FAIL_pct_of_r_PASS':round(100*c['combined_FAIL']/n,6) if n else None})
    for f in (0,1):
        a=[g for g in aggregate if g['family']==f]
        need(sum(g['q_both_footprints_and_SHAM_RGB_not_PASS'] for g in a)==review['counts'][f'family{f}_rpass_q_combined_notpass'],'review combined diagnostic census')
    outcsv=H/'FOOTPRINT_FAILURES_BY_FAMILY_DISTANCE.csv'
    with os.fdopen(os.open(safe(outcsv),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    result={'completed':True,'PASS':True,'inputs':inputs,'dataset_complete_inherited_binding':review['dataset_complete'],'population':{'all_original_FIT_known_D_parents':5406,'r_proof_PASS':sum(g['r_proof_PASS_denominator'] for g in aggregate),'unconstructed_retained':sum(g['parents']-g['constructed'] for g in aggregate)},'denominator':'Each row uses only r_proof_PASS parents of that family and q-r bin. Unsupported q footprint is UNKNOWN, not FAIL. combined_FAIL is COVER_all_opaque FAIL OR SHAM_all_clear FAIL for that one supported footprint.','groups':aggregate,'new_arrays_RGB_weights_source_GT_model_or_protected_reads':0,'labels_parents_threshold_features_changed':False,'new_training_or_model_selection':False,'current_admission':False,'limitations':['Footprint mismatch is geometric opacity evidence for the saved synthetic composite, not literal-q physical visibility GT in real evaluation.','Public YCB q-r jitter is the existing teacher construction, not measured actual detector error.','Family0 SYN intended-native witness0 and family1 YCB associated-r auxiliary remain distinct teacher tasks.','No teacher relabeling, parent filtering, causal failure attribution or new model/architecture selection.']}
    json_binding=write_json('SUMMARY.json',result)
    md=['# 저장 q/r footprint 진단','',f"원 FIT known-D 부모 5,406개를 그대로 보존했습니다. r proof PASS {result['population']['r_proof_PASS']:,}개만 아래 실패율 분모로 사용하며, unconstructed 39개는 삭제하거나 재라벨하지 않았습니다.",'','| family | q−r | r PASS | footprint | COVER 불투명 실패 | SHAM clear 실패 | 둘 중 하나 실패 | 비율 |','|---|---|---:|---|---:|---:|---:|---:|']
    for t in table:md.append(f"| {t['family']} | {'≤3 px' if t['q_reference_bin']=='LE3' else '>3 px'} | {t['r_proof_PASS_denominator']} | {'continuous positive taps' if t['footprint']=='continuous_reference' else 'half-up 3×3'} | {t['COVER_all_opaque_FAIL']} | {t['SHAM_all_clear_FAIL']} | {t['combined_FAIL']} | {t['combined_FAIL_pct_of_r_PASS']:.2f}% |")
    md+=['','`둘 중 하나 실패`는 각 footprint에서 COVER all-opaque 실패 또는 SHAM all-clear 실패의 합집합입니다. 두 실패의 중복은 CSV의 both_FAIL에 보존합니다. SHAM RGB 불변 검사와 두 footprint 전체 PASS 여부는 JSON/CSV의 별도 열이며 위 opacity 합집합과 구분합니다. unsupported는 UNKNOWN으로 별도 집계합니다.','','family0 SYN의 intended-native witness0와 family1 YCB의 associated-r AUX는 서로 다른 교사입니다. r에서의 증명이 q 위치의 같은 opacity 증명을 자동 보장하지 않는다는 저장 관측이며, **q footprint 불일치는 literal pixel의 실사 가시성 GT 증거가 아닙니다.** 공개 YCB의 q−r GT jitter를 실제 detector 오차로 해석하지 않습니다.','','새 RGB·배열·가중치·원 source GT·모델·보호 payload를 열지 않았습니다. 기존 labels/parents/threshold/feature architecture는 바꾸지 않았고 학습·모델 선택·인과 원인 확정은 하지 않았습니다. current admission은 false입니다.','','[세부 수치](SUMMARY.json) · [CSV](FOOTPRINT_FAILURES_BY_FAMILY_DISTANCE.csv) · [고정 집계 코드](summarize_saved_qr.py)','']
    p=H/'KO.md'
    with os.fdopen(os.open(safe(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:f.write('\n'.join(md))
    receipt=write_json('COMPLETE.json',{'completed':True,'PASS':True,'code':bind(Path(__file__)),'inputs':inputs,'summary':json_binding,'CSV':bind(outcsv),'KO':bind(p),'actual_scope':'saved independent QA JSON only','new_payload_reads':0,'new_labels_or_training_or_model_selection':False,'frozen_after_complete':True})
    print(json.dumps({'COMPLETE':receipt,'table':table},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
