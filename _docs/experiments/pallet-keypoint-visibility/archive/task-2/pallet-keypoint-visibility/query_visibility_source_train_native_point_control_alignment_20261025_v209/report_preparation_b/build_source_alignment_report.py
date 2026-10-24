"""Saved V209 report + precommitted descriptive statistics on exactly24 clean PNGs."""
from pathlib import Path
from collections import Counter
import csv,hashlib,io,json,os,re,stat
import numpy as np
H=Path(__file__).parent;V=H.parent;P=V.parent
QA=V/'ROOT_ACTUAL_FIXED8_SOURCE_POINT_VISUAL_QA.json'
COMP=V/'source_train_native_point_control_alignment_v1/COMPLETE.json'
REVIEW=V/'independent_actual_alignment_review_b/REVIEW.json'
CENSUS=P/'query_visibility_native_point_source_head_readiness_20261025_v210/head_training_code_readiness_a/SOURCE_TRAIN_COVERAGE_CENSUS.json'
PINS={str(QA):'fbab80e9dc9675238ecf1d5f79ec51d72d01c4dd5da97644b2751b40f9cc74e7',str(COMP):'fe44f7650a570f4a364d231ecab7453aaf83a176c99239651970f2d406b1756b',str(REVIEW):'343069a1a0a2d1cd4db535af3e7865dcef45bbb036908123ef25fc5b48f2c5ce',str(CENSUS):'b022e525e8148e24e3cfd2c110ce8155820a963fae38cf0d7f6421568db0d533'}

def rb(p,png=False):
    p=Path(p);assert p.is_absolute()and P in p.parents and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert p.suffix in ({'.png'}if png else{'.json','.py','.csv','.md'})
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb')as f:return f.read()
def binding(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def get(b):
    z=rb(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return json.loads(z)
def pinned(p):return get(dict(path=str(p),sha256=PINS[str(p)]))
def put(p,obj):
    assert p.parent in(H,V)
    with p.open('x',encoding='utf-8')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvwrite(name,rr,columns):
    with(H/name).open('x',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rr)

def changes(before,after):
    assert before.shape==after.shape==(96,96,3)and before.dtype==after.dtype==np.uint8
    delta=np.abs(after.astype(np.int16)-before.astype(np.int16));mask=np.any(delta>0,axis=2);y,x=np.where(mask);n=int(mask.sum())
    return dict(changed_RGB_pixels=n,total_pixels=9216,changed_fraction=n/9216,
        RGB_change_bbox_xyxy_inclusive=None if n==0 else[int(x.min()),int(y.min()),int(x.max()),int(y.max())],
        sum_absolute_channel_change=int(delta.sum()),mean_absolute_channel_change_all=float(delta.mean()),
        mean_absolute_channel_change_changed_pixels=float(delta[mask].mean())if n else 0.,max_absolute_channel_change=int(delta.max()))

def main():
    qa=pinned(QA);complete=pinned(COMP);review=pinned(REVIEW);census=pinned(CENSUS)
    assert complete['PASS']and review['PASS']and qa['root_cards_actually_viewed']==8 and qa['full_context_triplets_actually_viewed']==2
    assert review['complete']==binding(COMP)and review['physical_rows']==1968 and review['variants']==complete['variants']
    render=get(qa['renderer_complete']);manifest=get(qa['image_manifest']);selection=get(qa['selected_before_RGB'])
    assert render['completed']and render['PASS']and render['PNGs']==136 and render['all8_original_crop_parities']==192
    assert selection['selection_count']==8 and selection['native_counts']=={'0':4,'1':4}
    image_map={r['path']:r for r in manifest['images']};allowed=[];selected_images=[]
    for case in selection['cases']:
        items={}
        for variant in ('BEFORE','COVER','SHAM'):
            found=[im for im in manifest['images']if im['attempt_id']==case['attempt_id']and Path(im['path']).name==variant+'_CLEAN96.png']
            assert len(found)==1 and found[0]['role']=='saved_unchanged_raw96'
            items[variant]=found[0]['binding'];allowed.append(found[0]['binding'])
        selected_images.append(dict(attempt_id=case['attempt_id'],native_slot=case['requested_native_slot'],images=items))
    assert len(allowed)==len({b['path']for b in allowed})==24
    # Code and complete image whitelist are sealed BEFORE any PNG filesystem access.
    precommit=dict(completed=True,prospective_before_any_clean_PNG_read=True,code=binding(Path(__file__)),image_manifest=qa['image_manifest'],fixed_selection=qa['selected_before_RGB'],
        exact24_cleanPNG=allowed,statistics='Against same-case BEFORE uint8 RAW96: RGB-change pixel = any channel differs; inclusive minimum bbox of changed output pixels; abs int16 channel difference sum/mean(all and changed pixels)/max; no threshold or case selection.',
        caveat='RGB-change bbox is NOT alpha footprint, part mask, opacity or nativegold. These fixed8 native0/1 cases are not representative of all82. Descriptive bytes do not measure recognition/learning information, human visibility or real-failure causality.',
        source_NPZ_GT_arrays_or_models_to_read=0)
    put(H/'RGB_STATISTICS_PRECOMMIT_PROTOCOL.json',precommit)
    before=np.zeros((96,96,3),np.uint8);after=before.copy();after[4,7]=[3,5,8]
    single=changes(before,after);same=changes(before,before)
    assert single['changed_RGB_pixels']==1 and single['sum_absolute_channel_change']==16 and single['RGB_change_bbox_xyxy_inclusive']==[7,4,7,4]
    assert same['changed_RGB_pixels']==0 and same['RGB_change_bbox_xyxy_inclusive']is None
    put(H/'INVENTED_STATISTICS_CHECKS.json',dict(completed=True,PASS=True,checks=['singlepixel_count_bbox_abs_sum','unchanged_no_bbox'],sourcePNG_read_before_checks=0))
    from PIL import Image
    statistics=[];opened=[]
    for case in selected_images:
        arrays={}
        for variant,b in case['images'].items():
            assert b in allowed
            z=rb(b['path'],True);assert hashlib.sha256(z).hexdigest()==b['sha256']
            with Image.open(io.BytesIO(z))as im:
                assert im.format=='PNG';arr=np.array(im)
            assert arr.shape==(96,96,3)and arr.dtype==np.uint8;arrays[variant]=arr;opened.append(b)
        for variant in ('COVER','SHAM'):statistics.append(dict(attempt_id=case['attempt_id'],native_slot=case['native_slot'],variant=variant,**changes(arrays['BEFORE'],arrays[variant])))
    csv_stats=[dict(r,RGB_change_bbox_xyxy_inclusive=json.dumps(r['RGB_change_bbox_xyxy_inclusive']))for r in statistics]
    csvwrite('FIXED8_RAW96_RGB_CHANGE_STATS.csv',csv_stats,list(csv_stats[0]))
    native_rows=[r for r in census['existing_DIRECT_TRAIN']['per_native']if r['domain']=='SYNTH_FINITE_END']
    native=[dict(input_native_slot=r['native_slot'],existing_weak_direct_COVER_count=r['E'])for r in sorted(native_rows,key=lambda x:x['native_slot'])]
    canonical=[dict(canonical_index=int(k),existing_weak_direct_COVER_count=n)for k,n in sorted(census['existing_DIRECT_TRAIN']['synthetic_COVER_canonical_mapping_counts'].items(),key=lambda x:int(x[0]))]
    assert sum(r['existing_weak_direct_COVER_count']for r in native)==sum(r['existing_weak_direct_COVER_count']for r in canonical)==82
    csvwrite('ALL82_INPUT_NATIVE_COVERAGE.csv',native,list(native[0]));csvwrite('ALL82_CANONICAL_COVERAGE.csv',canonical,list(canonical[0]))
    relation=[]
    for variant,data in review['variants'].items():
        for group,n in [('all_native',656),('requested_native',82)]:relation.append(dict(variant=variant,group=group,N=n,**{s:data[group].get(s,0)for s in ('CLEAR','COVERED','EDGE_UNKNOWN','UNKNOWN')}))
        relation.append(dict(variant=variant,group='other_native',N=574,**{s:review['other_native_relations'][variant].get(s,0)for s in ('CLEAR','COVERED','EDGE_UNKNOWN','UNKNOWN')}))
    csvwrite('ALL1968_ALPHA_RELATIONS.csv',relation,list(relation[0]));cross=review['source_point_vs_alpha_relation_physical_rows'];csvwrite('SOURCE_POINT_AND_ALPHA_SEPARATE_AXES.csv',cross,list(cross[0]))
    results=dict(completed=True,PASS=True,kind='SOURCE_SAVED_POINT_ALPHA_ALIGNMENT_DIAGNOSTIC_NOT_MODEL_EVALUATION',
        key_finding='All82 requested declared source points have CLEAR/COVERED/CLEAR raster relations. Different query center does not establish a wrong native-point label.',
        physical_rows=1968,TRAIN_attempts=82,original_parents=41,original_points=328,inputU=0,
        requested_patterns=review['requested_triplet_patterns'],other_native_CLEAR_relations=1722,source_point_vs_alpha=cross,variants=review['variants'],
        input_native_weak_COVER_counts=native,canonical_weak_COVER_counts=canonical,
        fixed_visual=dict(cases=8,unique_bases=8,native_counts=selection['native_counts'],root_actual_card_views=8,root_actual_full_context_views=2,root_actual_unique_PNG_views=10,
            renderer_crop_byte_parities=192,renderer_PNG_count=136,observations_scope='Root fixed8 source observations only; B decoded24 cleanPNG for byte statistics, no new toolviews',
            source_visual_hypothesis='Some fixed SOURCE interventions are small planar patches or similar in color. Not causal evidence for all82/protected real/model errors.',
            RGB_change_statistics=statistics,RGB_stat_decoded_PNGs=24,statistics_representative_or_learning_claim=False),
        provenance=dict(source_complete=binding(COMP),independent_actual_QA=binding(REVIEW),root_actual_visual_QA=binding(QA),renderer_complete=qa['renderer_complete'],image_manifest=qa['image_manifest'],fixed_selection=qa['selected_before_RGB'],V210_source_census=binding(CENSUS),statistics_precommit=binding(H/'RGB_STATISTICS_PRECOMMIT_PROTOCOL.json')),
        limitations=['Alpha relation is saved raster opacity support, not a new continuous ray/physical native D/E/S label.',
            'Fixed8 selection covers only input-native0/1; all82 input-native and canonical coverage are separate and not representative-case claims.',
            'RGB changed-pixel count/bbox/absolute difference are descriptive output bytes; not opacity, full part mask, visual signal/learning quantification or real causal explanation.',
            'No training/admission/model improvement or selection; no original source packets, GT or model reopened.'],
        original_q_L_native_initializer_sampler_unchanged=True,new_labels_or_training_admission=False,fit_or_forward=0,original_NPZ_GT_or_model_reopened=0)
    put(V/'ROOT_RESULTS_KO.json',results)
    lines=['# V209: SOURCE point와 기존 control의 대응','',
        '**82개 요청 point 모두 BEFORE CLEAR → COVER COVERED → SHAM CLEAR였다.** 원 query 중심과 선언 point가 다르다는 이유만으로 기존 point 타깃이 잘못됐다고 볼 수 없다. 이 결과는 SOURCE raster 대응 진단이며 새 native D/E 라벨·실사 개선·학습 admission이 아니다.','',
        '고정 TRAIN 82시도·41개 원 base·328개 원 point의 변형/native8을 보존한 1968행을 독립 확인했다. 원 q/L/native/initializer/sampler와 input U=0이 유지됐다. 요청 이외 native 관계 1722개는 모두 CLEAR였다.','',
        '| 변형 | 전체 행 | CLEAR | COVERED | 요청 point CLEAR/COVERED |','|---|---:|---:|---:|---:|']
    for variant,d in review['variants'].items():lines.append(f"| {variant} | {d['rows']} | {d['all_native'].get('CLEAR',0)} | {d['all_native'].get('COVERED',0)} | {d['requested_native'].get('CLEAR',0)}/{d['requested_native'].get('COVERED',0)} |")
    lines+=['','원 BEFORE point 상태와 현재 opacity는 별도 축이다. 원 point D×CLEAR 1640행, D×COVERED 82행, S×CLEAR 246행이며, 이를 새 D1640/E82/S246 학습라벨로 변환하지 않았다. CLEAR는 저장 raster의 양수 tap이 모두 비불투명이라는 뜻이며 깊이 tie/비전면도 포함할 수 있어 가시성 인증이 아니다.','',
        '## Coverage와 실제 본 범위','',
        '기존 SOURCE TRAIN finite-end COVER 약타깃 82건의 input-native0..7 건수는 **'+str([r['existing_weak_direct_COVER_count']for r in native])+'**다. canonical0..7 건수는 **'+str([r['existing_weak_direct_COVER_count']for r in canonical])+'**로 모두 등장한다. 두 순서를 혼동하거나 input-native 빈도를 곧바로 물리 부위 D/E 결손으로 해석하지 않는다.','',
        '시각화는 사전 hash·quota로 고정한 8 base, input-native0/1 각4개에 한정된다. root가 카드8개와 full-context triplet2개(고유 PNG10개)를 실제 보았다. renderer의 원 crop parity192개·PNG136개 생성은 실제 view 횟수와 다르다.','',
        '[고정8 원 clean/case gallery](fixed8_source_native_point_visual_v1/PORTABLE_GALLERY_KO.md) · [root 실제 view 영수증](ROOT_ACTUAL_FIXED8_SOURCE_POINT_VISUAL_QA.json)','',
        '일부 고정 SOURCE 예에서는 작거나 target과 유사색인 평면 개입이 관찰됐다. 이는 SOURCE appearance/scale 가설이며 전체82의 대표성, 학습 정보 증가, 실사 모델 오류의 원인을 입증하지 않는다. 원 source NPZ를 재개봉하지 않고 고정24 clean PNG만으로 COVER/SHAM 대 BEFORE 변화 픽셀수·최소 변화 bbox·절대 채널 변화를 기술 집계했다. RGB 변화 bbox는 opaque alpha footprint·정밀 부위 mask가 아니며 비슷한 색이면 opacity가 있어도 변화가 작을 수 있다.','',
        '| 고정 case | 변형 | 변화 pixel/9216 | 변화 bbox (inclusive) | 절대 채널차 합 |','|---|---|---:|---|---:|']
    for r in statistics:lines.append(f"| {r['attempt_id']} | {r['variant']} | {r['changed_RGB_pixels']} | {r['RGB_change_bbox_xyxy_inclusive']} | {r['sum_absolute_channel_change']} |")
    lines+=['','## 표·방법·검산','',
        '- [전체 1968행 alpha 관계 CSV](report_preparation_b/ALL1968_ALPHA_RELATIONS.csv)','- [원 point/alpha 별도축 CSV](report_preparation_b/SOURCE_POINT_AND_ALPHA_SEPARATE_AXES.csv)',
        '- [input-native coverage CSV](report_preparation_b/ALL82_INPUT_NATIVE_COVERAGE.csv)','- [canonical coverage CSV](report_preparation_b/ALL82_CANONICAL_COVERAGE.csv)',
        '- [고정8 RGB 변화 CSV](report_preparation_b/FIXED8_RAW96_RGB_CHANGE_STATS.csv)','- [PNG 통계 사전규칙](report_preparation_b/RGB_STATISTICS_PRECOMMIT_PROTOCOL.json)',
        '- [source82 실제 COMPLETE](source_train_native_point_control_alignment_v1/COMPLETE.json)','- [독립 실제 검산 PASS](independent_actual_alignment_review_b/REVIEW.json)',
        '- [고정8 렌더 COMPLETE](fixed8_source_native_point_visual_v1/COMPLETE.json)','- [이 보고서 builder](report_preparation_b/build_source_alignment_report.py)','',
        '저장 reference의 모든 엄격 양수 bilinear tap이 alpha True이면 COVERED, 모두 False면 CLEAR, 혼합이면 EDGE_UNKNOWN, 누락·프레임 밖이면 UNKNOWN으로 분리했다. RGB 차분·region state·crop-center owner로 native 상태를 대체하지 않았다. 새 labels/fit/admission/forward·q 교정·PnP는 0이며 목표는 미해결이다.','']
    md='\n'.join(lines)
    with(V/'PORTABLE_REPORT_KO.md').open('x',encoding='utf-8')as f:f.write(md)
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',md):
        assert not target.startswith(('/','file:','http:','https:'))
        path=V/target;assert 'v112'not in str(path).casefold()
        for component in [*path.parents,path]:assert not component.is_symlink()
        assert path.is_file(),target;links.append(target)
    assert len(statistics)==16 and len(opened)==24
    put(H/'COMPLETE.json',dict(completed=True,PASS=True,code=binding(Path(__file__)),results=binding(V/'ROOT_RESULTS_KO.json'),report=binding(V/'PORTABLE_REPORT_KO.md'),
        source_packets_or_GT_reopened=0,cleanPNG_read_bindings=opened,statistics_rows=16,local_relative_links_checked=len(links),
        root_actual_toolview_claims_only=True,new_labels_fit_or_model_improvement_claim=False))
    print(json.dumps(dict(completed=True,PASS=True,complete=binding(H/'COMPLETE.json'),report=binding(V/'PORTABLE_REPORT_KO.md'),results=binding(V/'ROOT_RESULTS_KO.json'),links=len(links),statistics_rows=16)))

if __name__=='__main__':main()
