"""Bounded V215 report from OWNED saved preview metadata and actual-view receipts.

No image decoder, numpy, source input, target, GT, checkpoint or score reader.
"""
from collections import Counter
import csv
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import stat

H = Path(__file__).absolute().parent
V = H.parent
COMPLETE = {'path': str(V/'fixed12_public_TRAIN_input_previews_v1/COMPLETE.json'), 'sha256': 'e19a3573217cf44d7c53ace4bc8b275b2b335345fee7d0e95103e59f5fb8ad1c'}
ROOT_VIEW = {'path': str(V/'ROOT_ACTUAL_FIXED6_SOURCE_FRAME_OBSERVATIONS.json'), 'sha256': 'd17cc398e368f6d6c2a007ebb8b815f972c7dbbff6633484bd082c19a6f32ee6'}
B_VIEW = {'path': str(V/'independent_visual_observations_b/ACTUAL_FIXED6_SOURCE_FRAME_OBSERVATIONS.json'), 'sha256': '28a35fb79787e88de957207c9d02e601d5ee674cce083215164ac6cbac72b3be'}
ILLUSTRATIONS = [0,6,10]  # Exact root report instruction, no score/error selection.
PROTO = H/'PROTOCOL_BEFORE_SAVED_METADATA_REPORT.json'


def need(v, m):
    if not v:
        raise AssertionError(m)


def safe(p):
    s = str(p)
    need('v112' not in s.casefold(), 'V112 lexical reject before FS')
    p = Path(s)
    need(p.is_absolute() and '..' not in p.parts and V in p.parents, 'Only owned V215 paths')
    return p


def opened(p, write=False):
    p = safe(p)
    fd = os.open('/', os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt = os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd)
            fd = nxt
        out = os.open(p.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
    finally:
        os.close(fd)
    need(stat.S_ISREG(os.fstat(out).st_mode),'Regular file, no symlink')
    return os.fdopen(out,'wb'if write else'rb')


def rb(p):
    with opened(p) as f:
        return f.read()


def binding(p):
    return dict(path=str(safe(p)),sha256=hashlib.sha256(rb(p)).hexdigest())


def checked(b):
    need(set(b)=={'path','sha256'} and binding(b['path'])==b,'Exact saved artifact binding')
    return b['path']


def read(b):
    need(Path(b['path']).suffix=='.json','Only saved owned JSON body')
    return json.loads(rb(checked(b)))


def text(p,s):
    with opened(p,True) as f:
        f.write(s.encode())


def write(p,d):
    text(p,json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def rel(p):
    return safe(p).relative_to(V).as_posix()


def link(label,p):
    return '['+label+']('+rel(p)+')'


def table(headers,rows):
    md=['|'+'|'.join(headers)+'|','|'+'|'.join('---'for _ in headers)+'|']
    md += ['|'+'|'.join(str(x).replace('|','/')for x in r)+'|'for r in rows]
    ht='<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'
    ht+=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</tbody></table>'
    return '\n'.join(md),ht


def equal_scalar(text_value,value):
    return text_value==''if value is None else float(text_value)==value


def gather():
    complete,rv,bv=read(COMPLETE),read(ROOT_VIEW),read(B_VIEW)
    need(complete['completed']and complete['PASS']and complete['frames']==12 and complete['native_rows']==96 and complete['original_inputU']==11,'Closed factual12/96/U11 preview')
    need(complete['all_original_fullRGB_RAW96_valid96_decoded_PNG_bytes_exact']and complete['models_labels_scores_GT_SAM_or_protected_reads']==0,'Bound preview exact PNG identity / no forbidden body reads')
    manifest=read(complete['manifest'])
    need(Path(complete['CSV']['path']).name=='ALL96_ORIGINAL_INPUT_IDENTITIES.csv','Only original preview identity CSV')
    csvrows=list(csv.DictReader(io.StringIO(rb(checked(complete['CSV'])).decode())))
    frames,slots=manifest['frames'],manifest['slots96']
    need(len(frames)==12 and len(slots)==len(csvrows)==96 and len({r['row_id']for r in slots})==96,'All original12/native96 identity census')
    need(manifest['original_qL_native_U_preserved']and all(s['original_raw_byte_identity']for s in slots),'All original qL/U and per-slot raw parity retained in saved manifest')
    for s,c in zip(slots,csvrows):
        need(s['role']=='TRAIN'and int(c['position'])==s['position']and int(c['native_slot'])==s['native_slot']and c['frame_key']==s['frame_key']and c['row_id']==s['row_id']and c['component_id']==s['component_id'],'CSV original row/native/frame/component/role exact')
        q=s['query_xy']
        need(equal_scalar(c['query_x'],None if q is None else q[0])and equal_scalar(c['query_y'],None if q is None else q[1])and equal_scalar(c['L'],s['L'])and c['original_query_available']==str(s['original_query_available']),'Original q/L/U saved CSV parity')
    need([f['position']for f in frames]==list(range(12))and all([s['native_slot']for s in slots if s['position']==i]==list(range(8))for i in range(12)),'Original frame order and native0..7 unchanged')
    need(sum(not s['original_query_available']for s in slots)==11 and len({f['component_id']for f in frames})==12 and Counter(f['prefix']for f in frames)=={p:3 for p in ('val/000000','val/000002','val/000005','val/000006')},'Original U11 /4prefixx3/12components')
    need(rv['completed']and rv['PASS']and bv['completed']and bv['PASS']and rv['preview_complete']==bv['preview_complete']==COMPLETE and rv['manifest']==bv['manifest']==complete['manifest'],'Both actual view receipts bind same exact preview')
    need(rv['fixed_positions']==[0,2,4,6,8,10]and bv['fixed_positions']==[1,3,5,7,9,11]and rv['actual_view_count']==bv['actual_generated_contact_toolviews']==6,'Prospective complementary6+6 actual views')
    observations={}
    for r in rv['actual_frames']:
        need(r['actual_image_seen']and r['header_qL_U_all8_tiles_footer_readable'],'Root actual card view/legibility')
        observations[r['position']]=dict(reader='root',frame_key=r['frame_key'],contact=r['image'],description_KO=r['descriptive_observation_KO'],native8=r['native8'])
    for r in bv['frames']:
        need(r['actual_tool_view_completed']and r['readable_header_and_all8_native_metadata']and r['position']not in observations,'B actual complementary card view/legibility')
        observations[r['position']]=dict(reader='B',frame_key=r['frame_key'],contact=r['contact'],description_KO=r['fullimage_crop_relation_KO'],native8=r['native8_local_appearance_observations'])
    need(sorted(observations)==list(range(12)),'Full12 view union, no replacement')
    for f in frames:
        o=observations[f['position']]
        need(o['frame_key']==f['frame_key']and o['contact']==f['all8contact']and len(o['native8'])==8,'Actual image SHA/frame/native join')
        original=[s for s in slots if s['position']==f['position']]
        for s,r in zip(original,o['native8']):
            need(s['native_slot']==r['native_slot']and s['original_query_available']==r['original_query_available']and r['fit_admission']is False,'Original native/U and no fit admission in both observations')
            if o['reader']=='root':
                need(r['native_D_E_S_label']is None,'Root no native label')
            else:
                need(r['native_DE_S_label_assigned']is False,'B no native label')
    return complete,manifest,csvrows,frames,slots,observations


class AttributeLinks(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in('href','src')and v)


def local_link_check(doc,source):
    if source.suffix=='.html':
        p=AttributeLinks();p.feed(doc);links=p.links
    else:
        links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc)
    for value in links:
        need(not value.startswith(('/', 'file:', 'http:', 'https:'))and '://'not in value,'Portable owned relative hyperlink')
        need('#'not in value and '?'not in value,'Simple literal local reference')
        target=source.parent/value
        safe(target)
        # Existence/type only. In particular, no generated PNG body is read here.
        with opened(target)as f:
            need(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'Referenced local regular file exists')
    return len(links)


def main():
    protocol=dict(completed=True,approved_saved_metadata_report=True,code=binding(__file__),inputs=dict(preview_COMPLETE=COMPLETE,root_actual6=ROOT_VIEW,B_actual6=B_VIEW),follow_only='Preview COMPLETE.manifest and COMPLETE.CSV, all owned V215 metadata; no other source bindings followed.',root_provided_accepted_selection_census=dict(TRAIN_frames=155,old_weak_frames=3,old_weak_components=3,frames_excluded_by_weak_components=55,eligible_frames=98,eligible_components=98,per_prefix_selected=3,total_selected=12,min_original_available=4),selection_census_authority='Accepted sealed selector preparation FINAL_READY f052d073 and explicit root report instruction. This report does not repeat original source metadata selection.',illustration_positions=ILLUSTRATIONS,illustration_rule='Root report instruction fixes0/6/10; illustrative only, no score/error selection or representativeness claim.',actual_source_binary_image_model_GT_score_reads=0,phase_freeze_or_Git=False)
    write(PROTO,protocol)
    complete,manifest,csvrows,frames,slots,obs=gather()
    rows=[]
    for f in frames:
        uu=sum(not s['original_query_available']for s in slots if s['position']==f['position'])
        rows.append(dict(position=f['position'],frame_key=f['frame_key'],prefix=f['prefix'],component_id=f['component_id'],original_available=8-uu,original_U=uu,actual_reader=obs[f['position']]['reader'],contact_relative=rel(f['all8contact']['path']),appearance_observation_KO=obs[f['position']]['description_KO'],formal_native_D_E_S_labels=0,semantic_native_annotation_status='UNKNOWN_UNANNOTATED',fit_admission=False))
    result=dict(completed=True,PASS=True,phase='V215',task_completed='FACTUAL_INPUT_COVERAGE_AND_COMPLEMENTARY_LOCAL_APPEARANCE_OBSERVATIONS',native_annotation_completed=False,model_goal_completed=False,fully_solved=False,inputs=dict(preview_complete=COMPLETE,manifest=complete['manifest'],identity_CSV=complete['CSV'],root_actual_views=ROOT_VIEW,B_actual_views=B_VIEW,selection_INERT=complete['selection'],selector_ready_INERT=complete['selector_ready']),selection_census=protocol['root_provided_accepted_selection_census'],selected=dict(frames=12,distinct_components=12,prefix_counts=dict(Counter(f['prefix']for f in frames)),native_rows=96,original_inputU=11,original_available=85,original_query_L_native_role_unchanged=True),actual_views=dict(root_positions=[0,2,4,6,8,10],B_positions=[1,3,5,7,9,11],union_positions=list(range(12)),frames=12,native_display_panels=96,complementary_readers_not_sameframe_independent_consensus=True,all_cards_legible_per_bound_receipts=True,HTML_browser_render_claim=False),native_semantic_annotation=dict(D=0,E=0,S=0,UNKNOWN_UNANNOTATED=96,new_labels_created=False,new_training_admission=False,physical_native_gold=False,local_appearance_is_not_native_label=True),per_native_original_input=[dict(native_slot=k,N=12,original_available=sum(s['original_query_available']for s in slots if s['native_slot']==k),original_U=sum(not s['original_query_available']for s in slots if s['native_slot']==k))for k in range(8)],frames=rows,illustration_positions=ILLUSTRATIONS,limits=['Deterministic SOURCE pilot; component identity is not a proof of physical independence or statistical generalization.','Wood/material presence and a pallet elsewhere in fullimage do not identify the intended native part in crop.','Background, another wooden object or an apparent foreign object is not native E/D by itself.','Complementary root/B views cover12 different frames; they are not two independent semantic annotations of the same96 slots.','Original U11 is input availability; UNKNOWN_UNANNOTATED96 is a different semantic annotation axis.','No native physical gold, corrected q/keypoint, new labels, model performance, adoption or goal-solved claim.'],next_annotation_axes=['selected_pallet_instance_identified_or_UNKNOWN','intended_native_part_correspondence_or_UNKNOWN','actual_local_visible_feature_witness_or_UNKNOWN','query_support_ON_NEAR_OFF_UNKNOWN_separate','external_SELF_location_unknown_evidence_separate'],source_images_arrays_GT_oldlabels_models_scores_opened_by_report=0,training_admission=False)
    write(V/'ROOT_RESULTS_KO.json',result)
    counts=[['선택','4prefix×3 =12frame /12component'],['원 입력','native96 /available85 /U11'],['실제 카드 관찰','root6 + B6 =12frame /96표시slot'],['새 native D/E/S','0 /0 /0;96slot 주석미확정'],['새 학습 admission /모델·평가 실행','false /0']]
    mdtable,httable=table(['항목','결과'],counts)
    lead='공개 SOURCE TRAIN에서 고정한12프레임의 원입력96slot을 보존하고, root와 B가 서로 다른6장씩 실제 확인했다. 새 native D/E/S 주석과 학습 admission은 만들지 않았다. 목재나 물체 표면이 crop에 보인다는 사실은 의도한 native 부위의 가시성 정답이 아니다.'
    paragraphs=[
        '기존 TRAIN155프레임에서 weak3와 같은3개 원본 component에 속한55프레임을 제외하고, 원 available query가4개 이상인98후보/98component에서 고정 hash로4prefix×3개를 선택했다. 부족할 때 조건을 완화하는 fallback은 없다. 이 graph component 구분은 물리적 독립성의 인증이 아니다.',
        '원 q/L·native0..7·role·RAW96/valid sampler와 U11은 유지했다. fullRGB·clean96·valid96의 원바이트 정합은 봉인된 preview COMPLETE가 기록한다. 이번 보고서는 생성된 JSON/CSV와 실제 관찰 receipt만 대조했으며 원 source 배열·RGB·GT·기존 라벨·모델·점수는 열지 않았다.',
        '카드는2304px로 생성됐고 실제 도구 표시에서 헤더·q/L/U·native8·clean96/valid96·푸터가 읽혔다. root/B의12장 합집합은 전체 입력 카드 확인이며, 같은 frame에 대한 두 관찰자의 독립 합의가 아니다. HTML을 브라우저로 렌더했다는 주장은 없다.',
        '원 U11과 새 의미주석 UNKNOWN96은 다른 축이다. available85도 intended 부위 식별85개를 뜻하지 않는다. 모든 native D/E/S는 아직 주석미확정이며 local appearance 관찰을 D/E/S로 변환하지 않았다.',
    ]
    md=['# V215 공개 SOURCE TRAIN 입력 coverage와 외관 관찰','',lead,'',mdtable,'']
    body='<h1>V215 공개 SOURCE TRAIN 입력 coverage와 외관 관찰</h1><p>'+html.escape(lead)+'</p>'+httable
    for s in paragraphs:
        md += [s,''];body+='<p>'+html.escape(s)+'</p>'
    md+=['## 사전 지정 설명 카드0/6/10','','아래3장은 root가 보고서용으로 고정한 설명 사례다. 빈도·대표성·모델 실패율을 나타내지 않으며, 그림을 근거로 q나 native 라벨을 교정하지 않았다.','']
    body+='<h2>사전 지정 설명 카드0/6/10</h2><p>설명 사례이며 빈도·대표성·모델 실패율이 아니다.</p>'
    for i in ILLUSTRATIONS:
        f=frames[i];s=obs[i]['description_KO'];url=rel(f['all8contact']['path'])
        md += ['### 위치'+str(i)+' · '+f['frame_key'],'',s,'',f'![고정 설명 카드{i}]({url})','']
        body+='<h3>위치'+str(i)+' · '+html.escape(f['frame_key'])+'</h3><p>'+html.escape(s)+'</p><a href="'+url+'"><img src="'+url+'" alt="고정 설명 카드'+str(i)+'"></a>'
    examples='B가 본 val/000002:0은 전면EPAL pallet와 뒤쪽 상자들이 같은 fullimage에 있으나 q8crop은 뒤쪽 나무 상자 상단 주변을 담았다. val/000005:24는 앞 열린 나무 상자·뒤 목재 구조·바닥이 crop별로 섞였다. val/000006:25는 작은 검은 손잡이와 바닥을 담았다. 이러한 외관 관찰은 selected-instance/native 오류 또는 true 외부가림의 정답 판정이 아니다.'
    md += [examples,''];body+='<p>'+html.escape(examples)+'</p>'
    md+=['## 전체12프레임과 원입력','',''+link('전체12 갤러리',complete['gallery']['path'])+' · '+link('원96slot ID/qL/U CSV',complete['CSV']['path'])+' · '+link('PNG 정합 manifest',complete['manifest']['path']),'']
    trows=[[r['position'],r['frame_key'],str(r['original_available'])+'/8',r['original_U'],r['actual_reader']]for r in rows]
    mt,ht=table(['위치','원 source frame','available','U','실제 관찰'],trows);md+=[mt,''];body+='<h2>전체12프레임</h2>'+ht
    for r in rows:
        md += [link('위치'+str(r['position'])+' '+r['frame_key']+' 카드',V/r['contact_relative']),'']
        body+='<p><a href="'+r['contact_relative']+'">위치'+str(r['position'])+' '+html.escape(r['frame_key'])+' 카드</a></p>'
    nexttext='다음 source 약관찰에서는 ①선택된 pallet instance 식별 ②의도한 native 부위 대응 ③국소 visible-feature witness ④query support/ON·NEAR·OFF ⑤외부·자기가림/위치 불확실성을 별도 축으로 기록한다. 판단할 수 없으면 UNKNOWN과 불일치를 보존한다. 배경·다른 목재·외부물체 표면을 native E/D로 바꾸거나, q/ROI를 이동해 통과시키지 않는다. 이는 후속 관찰의 준비이며 새 주석 규칙·학습 admission이 아니다.'
    md+=['## 남은 대응 문제','',nexttext,'','완료 범위는 factual 입력 coverage와 제한된 SOURCE 외관 관찰이다. 새 native annotation, 모델 개선·채택 또는 최종 목표 해결은 완료되지 않았다.','']
    body+='<h2>남은 대응 문제</h2><p>'+html.escape(nexttext)+'</p><p>새 native annotation·모델 개선/채택·목표 해결은 완료되지 않았다.</p>'
    links=[('수치/범위 JSON',V/'ROOT_RESULTS_KO.json'),('HTML 보고서',V/'ROOT_READOUT_KO.html'),('전체12 갤러리',complete['gallery']['path']),('Preview COMPLETE',COMPLETE['path']),('사전 선택 metadata',complete['selection']['path']),('Selector READY',complete['selector_ready']['path']),('Preview 코드',complete['code']['path']),('Preview protocol',complete['protocol']['path']),('root 실제6 receipt',ROOT_VIEW['path']),('B 실제6 receipt',B_VIEW['path']),('본 보고서 코드',Path(__file__)),('본 보고서 protocol',PROTO),('Saved scalar/text QA',H/'SAVED_REPORT_QA.json')]
    md+=['## 근거와 재현 파일','']
    for label,path in links:
        md += [link(label,path),''];body+='<p><a href="'+rel(path)+'">'+html.escape(label)+'</a></p>'
    source='\n'.join(md)
    text(V/'ROOT_REPORT_KO.md',source)
    text(V/'PORTABLE_REPORT_KO.md',source)
    text(V/'ROOT_READOUT_KO.html','<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V215 SOURCE 입력 coverage</title><style>body{max-width:1500px;margin:24px auto;padding:0 18px;font:17px/1.65 sans-serif;color:#182126}table{border-collapse:collapse;margin:16px 0}th,td{border:1px solid #bbb;padding:6px 12px;text-align:left}img{width:100%;height:auto}a{color:#125896}h1,h2,h3{line-height:1.3}</style>'+body+'</html>')
    artifacts={p.name:binding(p)for p in (V/'ROOT_RESULTS_KO.json',V/'ROOT_REPORT_KO.md',V/'PORTABLE_REPORT_KO.md',V/'ROOT_READOUT_KO.html')}
    # Create QA first so its declared report hyperlink can be checked without a cycle.
    qa=dict(completed=True,PASS=True,bounded_saved_metadata_only=True,code=binding(__file__),protocol=binding(PROTO),inputs=result['inputs'],artifacts=artifacts,checks=dict(original96CSV_META_qL_U_exact=True,original_frame_order_native8_and_TRAIN_role_preserved=True,selection12_components4x3_and_inputU11=True,actual_view_receipts_same_preview_SHA=True,complementary6plus6_exact_union12_not_consensus=True,all96_native_DE_S_unannotated_no_admission=True,predetermined_illustrations_exact0_6_10=True,source_array_RGB_GT_model_score_and_PNG_decode_count=0),portable_link_check='Completed below in the separate build COMPLETE; no semantic image re-evaluation.',HTML_browser_rendered=False)
    write(H/'SAVED_REPORT_QA.json',qa)
    counts={p.name:local_link_check(rb(p).decode(),p)for p in (V/'ROOT_REPORT_KO.md',V/'PORTABLE_REPORT_KO.md',V/'ROOT_READOUT_KO.html')}
    need(rb(V/'ROOT_REPORT_KO.md')==rb(V/'PORTABLE_REPORT_KO.md'),'ROOT and portable MD byte identity')
    write(H/'REPORT_BUILD_COMPLETE.json',dict(completed=True,PASS=True,code=binding(__file__),protocol=binding(PROTO),inputs=result['inputs'],artifacts=artifacts,QA=binding(H/'SAVED_REPORT_QA.json'),portable_local_link_counts=counts,all_local_links_regular_and_resolve=True,root_portable_MD_byteidentical=True,no_original_source_binary_GT_models_scores_or_new_image_decodes=True,no_phase_freeze_or_Git=True,own_preparation_frozen_no_further_edits=True))
    print(json.dumps(dict(completed=True,PASS=True,complete=binding(H/'REPORT_BUILD_COMPLETE.json'),QA=binding(H/'SAVED_REPORT_QA.json'),artifacts=artifacts,local_link_counts=counts),ensure_ascii=False))


if __name__=='__main__':
    main()
