"""Authorized V177 public saved-result readout. No model/NPZ/private reads."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import html
import json
import shutil
import textwrap
import traceback

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V=Path(__file__).resolve().parents[1]
RULES=V/'ROOT_VISUAL_READOUT_RULES_BEFORE_CONSENSUS_OR_SCORES.json'
RULES_SHA='b89fefa956bdd9a3b629e5a8017676ac70e676f7469e5f8bf4926a8e3ea8d226'
EVAL=V/'saved_public_held_owner_evaluation_v1/COMPLETE.json'
EVAL_SHA='ad7e8d329a9dcffd136a481e978285840fd356b4134fb12d56c02aff5ba0dc24'
RENDER=V/'blind_review_evidence_v1/ALL360_BLINDED_REVIEW_INPUTS.jsonl'
RENDER_SHA='d36712ced46a59d250d06421701178a412b12c8042b07f17e092a62fad576454'
B_AUDIT=V/'independent_actual_saved_evaluation_review_b/REVIEW.json'
B_SHA='816fd4b80f7b5fd993612cf8983ddb5b8367b6c451b45380493859097e12aea2'
MESH=V/'root_public_mesh_identity_audit/ROOT_ACTUAL_GEOMETRY_VISUAL_REVIEW.json'
O=V/'public_owner_readout_visual_v1'
OWNER=['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']
SHORT={OWNER[0]:'T',OWNER[1]:'F',OWNER[2]:'BG','UNKNOWN':'U'}
NEAREST=Image.Resampling.NEAREST
FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def checked(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def lines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def href(p,base=V):return str(Path(p).relative_to(base))
def percent(x):return f'{100*x:.1f}%'
def pred_name(x,tag):return 'U'if x<0 else ('T','NON_T')[x]if tag=='v61'else('T','F','BG')[x]


def select_cases(rules,joined,ledger):
    ordered=sorted(joined,key=lambda r:r['original_global_index']);used=set();selected=[]
    def match(i,r):
        p=ledger[r['original_global_index']];c=r['center_owner'];c48=p['C48'];v61=p['v61']
        return [c==OWNER[0]and c48==0 and v61==1,
                c==OWNER[0]and c48 in(1,2)and v61==0,
                c==OWNER[1]and c48==1,
                c==OWNER[1]and c48==0,
                c==OWNER[2]and c48==0,
                c in OWNER[1:]and r['crop_presence']=='YES',
                r['reviewer_A']['center_owner']!=r['reviewer_B']['center_owner'],
                not r['source_original_query_available']and r['source_crop_constructed']and r['crop_presence']=='YES'][i]
    for i,rule in enumerate(rules['rules']):
        matches=[r for r in ordered if match(i,r)];eligible=[r for r in matches if r['original_global_index']not in used]
        g=eligible[0]['original_global_index']if eligible else None
        if g is not None:used.add(g)
        selected.append(dict(bucket=i+1,rule=rule,matching_total=len(matches),eligible_after_distinct=len(eligible),
                             selected_original_global_index=g,status='SELECTED'if g is not None else'EMPTY_NO_MATCH'if not matches else'EMPTY_ALREADY_DISPLAYED'))
    return selected


def source_images(row,folder):
    entries={k:row['images'][k]for k in('raw96','center16','center2','valid96')};entries['context224']=row['context224']
    copied={};arrays={};facts=[]
    for key,b in entries.items():
        src=checked(b);dst=folder/(key+'.png');shutil.copy2(src,dst)
        assert sha(dst)==b['sha256']
        with Image.open(src)as im:a=np.asarray(im).copy();mode=im.mode
        with Image.open(dst)as im:assert im.mode==mode and np.array_equal(np.asarray(im),a)
        copied[key]=dst;arrays[key]=a
        facts.append(dict(key=key,source=b,copy=bind(dst),mode=mode,decoded_shape=list(a.shape),decoded_pixels_sha256=ash(a),
                          encoded_copy_byte_exact=True,decoded_copy_byte_exact=True))
    raw=arrays['raw96'];assert raw.shape==(96,96,3)and raw.dtype==np.uint8
    assert np.array_equal(arrays['center16'],raw[40:56,40:56])and np.array_equal(arrays['center2'],raw[47:49,47:49])
    assert ash(raw)==row['original_array_hashes_inherited_not_recomputed']['RGB96_HWC_sha256']
    valid=arrays['valid96'];assert valid.shape==(96,96)and set(np.unique(valid)).issubset({0,255})
    mask=valid>0;assert ash(mask)==row['original_array_hashes_inherited_not_recomputed']['valid96_sha256']
    assert np.array_equal(mask[47:49,47:49],row['center4_valid_bits'])
    assert arrays['context224'].shape==(224,224,3)
    return copied,facts


def card(item,row,observation,predictions,copies,folder):
    im=Image.new('RGB',(1800,900),'#f6f7fa');d=ImageDraw.Draw(im)
    title=ImageFont.truetype(str(FONT),27);body=ImageFont.truetype(str(FONT),20);small=ImageFont.truetype(str(FONT),18)
    d.text((36,24),f"Rule {item['bucket']} | original ID {row['original_global_index']} | frame {row['frame_index']} / input native slot {row['native_slot']}",font=title,fill='#132236')
    for j,line in enumerate(textwrap.wrap(item['rule'],110)):d.text((36,66+25*j),line,font=body,fill='#33445b')
    center=SHORT[observation['center_owner']];crop=observation['crop_presence']
    d.text((36,130),f'Assistant-weak consensus: center {center} | crop-presence {crop}  (different questions)',font=body,fill='#132236')
    d.text((36,161),'Frozen predictions: '+' | '.join(t+' '+pred_name(predictions[t],t)for t in('C47','C48','C49','v61')),font=body,fill='#132236')
    panels=[('context224',36,228,448,'Initial context224 x2 nearest'),('raw96',530,228,384,'Clean96 x4 nearest'),
            ('center16',960,228,256,'Clean center16 x16 nearest'),('center2',1260,228,160,'Exact2x2 x80 nearest'),
            ('valid96',1470,228,192,'Input valid96 x2 nearest')]
    for key,x,y,side,label in panels:
        d.text((x,198),label,font=small,fill='#33445b')
        with Image.open(copies[key])as src:view=src.convert('RGB').resize((side,side),NEAREST)
        im.paste(view,(x,y));d.rectangle((x-1,y-1,x+side,y+side),outline='#718198',width=1)
    q=row['query_xy'];l=row['L'];bits=row['center4_valid_bits'];valid_count=int(np.asarray(bits,bool).sum())
    notes=[f"q={q}; L={l}; original availability={row['original_query_available']}; center4 valid={valid_count}/4",
           f"Crop constructed={row['feature_constructed']}; raw96/center pixels and q/L are unchanged. Display-only nearest resizing.",
           'White valid-mask pixels are supported; black invalid/padded center pixels are not natural-background evidence.',
           'T/F/BG denotes literal center material only. Initial q8 context and mesh shape do not certify selected instance/native identity.']
    y=711
    for line in notes:
        for part in textwrap.wrap(line,150):d.text((36,y),part,font=small,fill='#33445b');y+=25
    path=folder/'CARD.png';im.save(path);return path


def scientific_plot(report):
    models=report['cohorts']['FULL360']['ALL']['models'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(15.5,10.5),gridspec_kw={'height_ratios':[.8,1.15]})
    ax=axes[0,0];counts=[24,96,84,156];colors=['#277da1','#e67e22','#6a994e','#b8bdc5'];left=0
    for label,n,color in zip(['T','F','BG','Unknown'],counts,colors):
        ax.barh([0],n,left=left,color=color,label=f'{label}: {n}')
        ax.text(left+n/2,0,str(n),ha='center',va='center',weight='bold',color='white'if label!='Unknown'else'#202b3c');left+=n
    ax.set_xlim(0,360);ax.set_yticks([]);ax.set_xlabel('Original queries (all 360 retained)');ax.set_title('A  Weak center-owner coverage: 204 known / 360')
    ax.set_ylim(-.6,1.0)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.26),ncol=4,frameon=False)
    ax.text(0,.95,'45 frames from one previously exposed public video\nInput U108; center-invalid112 overlaps it; truth U156 is separate.',transform=ax.transAxes,va='top',fontsize=10)
    ax=axes[0,1];tags=['v61','C47','C48','C49'];x=np.arange(4);w=.34
    tr=[100*models[t]['binary_T_nonT']['per_class']['T']['recall_including_prediction_U']for t in tags]
    nr=[100*models[t]['binary_T_nonT']['per_class']['NON_T']['recall_including_prediction_U']for t in tags]
    ax.bar(x-w/2,tr,w,color='#277da1',label='T recall (24 weak T)');ax.bar(x+w/2,nr,w,color='#6a994e',label='Non-T recall (180 weak F/BG)')
    for xx,ys in [(x-w/2,tr),(x+w/2,nr)]:
        for a,b in zip(xx,ys):ax.text(a,b+2,f'{b:.1f}',ha='center',fontsize=9)
    ax.set_xticks(x,tags);ax.set_ylim(0,117);ax.set_ylabel('Percent');ax.set_title('B  Fixed binary center-owner reference');ax.legend(frameon=False,loc='lower right',fontsize=9)
    ax.text(.02,.97,'T precision: '+', '.join(f"{t} {100*models[t]['binary_T_nonT']['per_class']['T']['precision_on_known_truth']:.1f}%"for t in tags),transform=ax.transAxes,va='top',fontsize=9)
    ax=axes[1,0];cm=np.asarray(models['C48']['owner3']['known_truth_confusion']);row=cm.sum(1,keepdims=True)
    ax.imshow(cm/row,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for i in range(3):
        for j in range(4):ax.text(j,i,str(cm[i,j]),ha='center',va='center',weight='bold',color='white'if cm[i,j]/row[i,0]>.65 else'#172a41')
    ax.set_xticks(range(4),['T','F','BG','U']);ax.set_yticks(range(3),['Weak T (24)','Weak F (96)','Weak BG (84)'])
    ax.set_xlabel('Frozen C48 point-owner prediction');ax.set_title('C  C48 three-way counts (known weak reference N204)')
    ax=axes[1,1];tags=['C47','C48','C49'];x=np.arange(3)
    p=[100*models[t]['owner3']['per_class'][OWNER[1]]['precision_on_known_truth']for t in tags]
    r=[100*models[t]['owner3']['per_class'][OWNER[1]]['recall_including_prediction_U']for t in tags]
    ax.bar(x-w/2,p,w,color='#e67e22',label='Point F precision');ax.bar(x+w/2,r,w,color='#f3b65d',label='Point F recall (96 weak F)')
    for xx,ys in[(x-w/2,p),(x+w/2,r)]:
        for a,b in zip(xx,ys):ax.text(a,b+2,f'{b:.1f}',ha='center',fontsize=10)
    ax.set_xticks(x,tags);ax.set_ylim(0,117);ax.set_ylabel('Percent');ax.set_title('D  Point-foreign reference; v61 has no three-way output');ax.legend(frameon=False,loc='lower right',fontsize=10)
    fig.suptitle('Natural public held queries: literal center-owner diagnostic',fontsize=18,weight='bold',y=.99)
    fig.text(.5,.015,'Assistant-weak observations; not native external/self visibility or physical keypoint gold. Fixed outputs; no model selection. Precision excludes unknown truth.',ha='center',fontsize=10)
    fig.tight_layout(rect=[.01,.05,.99,.955]);path=O/'PUBLIC_HELD_CENTER_OWNER_SCIENTIFIC.png';fig.savefig(path,dpi=170);plt.close(fig);return path


def main():
    assert sha(RULES)==RULES_SHA and sha(EVAL)==EVAL_SHA and sha(RENDER)==RENDER_SHA and sha(B_AUDIT)==B_SHA
    rules=read(RULES);complete=read(EVAL);report=read(checked(complete['outputs']['report']));joined=lines(checked(complete['outputs']['consensus']));render=lines(RENDER)
    with checked(complete['outputs']['ledger']).open(newline='')as f:ledger_rows=list(csv.DictReader(f))
    assert len(joined)==len(render)==len(ledger_rows)==360
    ledger={int(r['original_global_index']):{t:int(r[t+'_prediction'])for t in('C47','C48','C49','v61')}for r in ledger_rows}
    byid={r['original_global_index']:r for r in joined};images={r['original_global_index']:r for r in render}
    assert len(byid)==360 and set(byid)==set(images)==set(ledger)
    for g,r in byid.items():
        assert all(r[k]==images[g][k]for k in('query_id','frame_index','native_slot','split','query_xy','L'))
    assert Counter(r['center_owner']for r in joined)==Counter({OWNER[0]:24,OWNER[1]:96,OWNER[2]:84,'UNKNOWN':156})
    assert Counter(r['crop_presence']for r in joined)==Counter(YES=223,NO=45,UNKNOWN=92)
    assert read(B_AUDIT)['PASS']
    O.mkdir(exist_ok=False)
    try:
        selection=select_cases(rules,joined,ledger)
        write(O/'SELECTED8_BUCKETS_BEFORE_IMAGE_BODIES.json',dict(rules=bind(RULES),evaluation=bind(EVAL),items=selection,
              sort='original_global_index ascending, distinct across previous buckets',case_selection_before_any_image_body_read=True))
        cases=[];assetfacts=[]
        for item in selection:
            g=item['selected_original_global_index']
            if g is None:continue
            folder=O/f"RULE_{item['bucket']}_GLOBAL_{g:04d}";folder.mkdir()
            copies,facts=source_images(images[g],folder);assetfacts.extend(facts)
            path=card(item,images[g],byid[g],ledger[g],copies,folder)
            cases.append(dict(**item,query_id=byid[g]['query_id'],frame_index=byid[g]['frame_index'],native_slot=byid[g]['native_slot'],
                 center_owner=byid[g]['center_owner'],crop_presence=byid[g]['crop_presence'],reviewer_A=byid[g]['reviewer_A'],reviewer_B=byid[g]['reviewer_B'],
                 predictions=ledger[g],CARD=bind(path),original_copies={k:bind(v)for k,v in copies.items()},original_frame_sheet=images[g]['frame_sheet']))
        plot=scientific_plot(report)
        write(O/'ALL_CASE_IMAGE_BINDINGS_AND_PIXEL_PARITY.json',dict(completed=True,cases=cases,assets=assetfacts,
             all_copied_original_PNG_encoded_and_decoded_bytes_unchanged=True,exact_raw96_center16_center2_and_valid_hash_checks=True,
             nearest_display_only=True,source_RGB_NPZ_model_or_private_body_reads=0))
        models=report['cohorts']['FULL360']['ALL']['models'];binary={t:m['binary_T_nonT']for t,m in models.items()}
        native={str(k):dict(N=45,T=sum(r['native_slot']==k and r['center_owner']==OWNER[0]for r in joined),
                 F=sum(r['native_slot']==k and r['center_owner']==OWNER[1]for r in joined),BG=sum(r['native_slot']==k and r['center_owner']==OWNER[2]for r in joined),
                 U=sum(r['native_slot']==k and r['center_owner']=='UNKNOWN'for r in joined))for k in range(8)}
        results=dict(schema='V177_PUBLIC_HELD_CENTER_OWNER_READOUT_KO_v1',report_completed=True,fully_solved=False,adopted=False,selected_model=None,
            stage='공개 저장출력·약한 중심소속 평가 및 고정 사례 읽기',scope=dict(rows=360,frames=45,sequence='val/000001',input_native_slots=8,evaluation_only=True),
            coverage=report['coverage'],weak_center_counts=dict(T=24,F=96,BG=84,U=156),crop_counts=dict(YES=223,NO=45,U=92),
            binary_metrics_known204=binary,owner3_metrics_known204={t:m['owner3']for t,m in models.items()if'owner3'in m},
            reviewer_agreement=report['reviewer_agreement'],center_crop_cross=report['crop_associations']['consensus_center_vs_crop'],native_slot_denominators=native,
            core_findings_ko=[
              '공개 자연영상 한 개의 고정 질의에서 literal center-owner와 약한 관찰의 대응을 확인했습니다. 원 native 외부가림 성능 향상이나 완전 전이 성공을 뜻하지 않습니다.',
              'C48/C49의 약한 T 재현율은22/24(91.7%)이며 v61은7/24(29.2%)입니다. C48/C49 T precision은22/48(45.8%)이고, non-T 재현율154/180(85.6%)은 v61의164/180(91.1%)보다 낮습니다.',
              'C48의 점 F precision/recall은53/57(93.0%)/53/96(55.2%), C49는57/61(93.4%)/57/96(59.4%)입니다. v61에는 F/BG를 가르는3way 출력이 없습니다.',
              '전체360행 중 약한 중심 known204(T24/F96/BG84), UNKNOWN156을 보존했습니다. 원 입력U108과 중심invalid112는 겹치며 합산하지 않습니다. crop presence known268(YES223/NO45)은 별도 관찰입니다.',
              '새 head3개는 원 입력U108, v61은 자체가용성U109입니다. all4 paired251과 known204를 분리했고 known204에서는 모델U가0입니다.',
              '두 assistant의 새 점수 blind 관찰이지만 이45프레임 영상은 이전 native 평가에서 반복 노출됐습니다. 사람·픽셀·물리 native gold가 아니며 selected-instance 모호성은 UNKNOWN입니다.',
              '공식 obj1 mesh가 팔레트 형상이라는 확인은 해당 프레임에서 초기 q8이 지칭하는 물리 instance나 native 대응을 인증하지 않습니다.',
              '새 모델/문턱/데이터 선택·라벨 교정·학습 편입은 없습니다. 기존 전체 native qualification과 최종 과제는 미해결입니다.'],
            evidence=dict(evaluation=bind(EVAL),report=complete['outputs']['report'],ledger=complete['outputs']['ledger'],consensus=complete['outputs']['consensus'],
                          reviewer_seals=complete['reviewer_seals'],numeric_audit=bind(B_AUDIT),mesh_identity_limit=bind(MESH),rules=bind(RULES),
                          selection=bind(O/'SELECTED8_BUCKETS_BEFORE_IMAGE_BODIES.json'),pixel_parity=bind(O/'ALL_CASE_IMAGE_BINDINGS_AND_PIXEL_PARITY.json'),plot=bind(plot)),
            visual_case_counts=dict(buckets=8,selected=len(cases),empty=sum(x['selected_original_global_index']is None for x in selection)),
            visual_QA='PNG actual view pending; offline HTML source/link validation only, browser rendering not claimed.',
            no_new_fit_forward_or_label_changes=True,private_current_body_reads_this_readout=0,training_admission=False)
        write(V/'ROOT_RESULTS_KO.json',results)
        rows=''.join('<tr><td>'+t+'</td><td>'+percent(binary[t]['per_class']['T']['precision_on_known_truth'])+'</td><td>'+percent(binary[t]['per_class']['T']['recall_including_prediction_U'])+'</td><td>'+percent(binary[t]['per_class']['NON_T']['recall_including_prediction_U'])+'</td><td>'+str(binary[t]['prediction_U_all'])+'/360</td></tr>'for t in('v61','C47','C48','C49'))
        case_html=[]
        for item in selection:
            g=item['selected_original_global_index']
            if g is None:
                case_html.append(f"<article><h3>규칙 {item['bucket']}: 없음</h3><p>{html.escape(item['rule'])} — {item['status']}. 대체 사례를 고르지 않았습니다.</p></article>");continue
            c=next(x for x in cases if x['selected_original_global_index']==g)
            notes=''.join('<p><b>'+t+'</b>: 중심 '+html.escape(c['reviewer_'+t]['center_owner'])+' / crop '+html.escape(c['reviewer_'+t]['crop_presence'])+'<br>'+html.escape(c['reviewer_'+t]['observation'])+'</p>'for t in('A','B'))
            links=' · '.join('<a href="'+href(b['path'])+'">'+k+'</a>'for k,b in c['original_copies'].items())
            case_html.append('<article><h3>규칙 '+str(item['bucket'])+' · 원 ID '+str(g)+'</h3><img class="card" src="'+href(c['CARD']['path'])+'" alt="고정 규칙의 원본 crop 및 중심 확대"><p>'+links+' · <a href="'+href(c['original_frame_sheet']['path'])+'">원 reviewer frame sheet</a></p><details><summary>A/B 원 관찰</summary>'+notes+'</details></article>')
        caveats=''.join('<li>'+html.escape(x)+'</li>'for x in results['core_findings_ko'][3:])
        doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V177 공개 중심소속 평가</title><style>body{font:17px/1.65 system-ui,sans-serif;color:#1b2b40;background:#f7f8fa;margin:0}main{max-width:1180px;margin:auto;padding:36px}h1{font-size:30px;line-height:1.3}h2{margin-top:36px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border-bottom:1px solid #d8dfe8;text-align:left}img{max-width:100%;height:auto}.card{image-rendering:pixelated}article{padding:20px;background:white;border:1px solid #d8dfe8;margin:22px 0}aside{background:#e7eef5;padding:18px;border-left:4px solid #277da1}small{color:#52647a}a{color:#185d94}details{margin:14px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><main><h1>공개 자연영상: 중심 소속을 읽는 진전과 남은 한계</h1>'''
        doc+='<p>'+html.escape(results['core_findings_ko'][0])+'</p><aside><b>범위:</b> 공개 val/000001 영상45프레임 · 원 질의360행. 약한 중심 known204(T24/F96/BG84), UNKNOWN156. 평가 전용이며 모델 채택은 없습니다.</aside><h2>고정 출력 비교</h2><p>'+html.escape(results['core_findings_ko'][1])+'</p><table><tr><th>모델</th><th>T precision</th><th>T recall</th><th>non-T recall</th><th>출력 U</th></tr>'+rows+'</table><small>P/R 분모는 중심 known204입니다. truth UNKNOWN156은 오답으로 세지 않고 별도 보존합니다. known204에서 모델U는0입니다.</small><p>'+html.escape(results['core_findings_ko'][2])+'</p><img src="'+href(plot)+'" alt="Literal center owner scientific diagnostic plot"><h2>분모와 해석</h2><ul>'+caveats+'</ul><p>all4 입력 paired251과 중심 known204는 서로 다른 분모입니다. 중심 소속과 crop 어느 곳에든 팔레트가 보이는지는 별개입니다. F/BG를 진짜 native 외부가림으로 바꾸어 해석하지 않습니다.</p><h2>동결 규칙의 사례</h2><p>8개 규칙마다 원global ID 오름차순의 첫 미표시 사례만 선택했습니다. 빈 규칙은 그대로 남겼습니다. 사례는 결과별 설명용이며 대표 표본이 아닙니다. 깨끗한 원96·center16[40:56]·center2[47:49]는 byte 보존했고 배율은 nearest display입니다. 카드의 글자·배치는 모델 입력이 아닙니다.</p>'+''.join(case_html)
        doc+='<h2>원 기록</h2><p><a href="'+href(EVAL)+'">평가 완료</a> · <a href="'+href(complete['outputs']['report']['path'])+'">모든 cohort/native8 교차표</a> · <a href="'+href(complete['outputs']['ledger']['path'])+'">전360행 ledger</a> · <a href="'+href(B_AUDIT)+'">독립 숫자 감사</a> · <a href="'+href(MESH)+'">obj1 형상 확인의 범위</a> · <a href="'+href(V/'blind_review_evidence_v1/index.html')+'">원45 frame gallery</a></p><details><summary>기계 판독 결과·해시</summary><a href="ROOT_RESULTS_KO.json">ROOT_RESULTS_KO.json</a><pre>'+html.escape(json.dumps(results['coverage'],ensure_ascii=False,indent=2))+'</pre></details><small>이 파일은 V177 공개 평가 기록입니다. HTML 브라우저 렌더 검증은 수행하지 않았습니다. PNG actual-view QA는 별도 영수증으로 구분합니다. 최종 해결/채택 false, selected model null.</small></main></html>'
        (V/'ROOT_READOUT_KO.html').write_text(doc)
        write(O/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),rules=bind(RULES),evaluation=bind(EVAL),
             selected_cases=len(cases),empty_buckets=sum(x['selected_original_global_index']is None for x in selection),
             source_PNG_assets_verified=len(assetfacts),CARDs=[x['CARD']for x in cases],scientific_plot=bind(plot),
             selection=bind(O/'SELECTED8_BUCKETS_BEFORE_IMAGE_BODIES.json'),pixel_parity=bind(O/'ALL_CASE_IMAGE_BINDINGS_AND_PIXEL_PARITY.json'),
             visual_actual_view_QA_pending=True,HTML_browser_rendered=False,new_model_forward_fit_or_label_change=False,private_current_body_reads=0))
        print(json.dumps(dict(completed=True,selected_cases=len(cases),empty_buckets=sum(x['selected_original_global_index']is None for x in selection),
                             plot=str(plot),cards=[x['CARD']['path']for x in cases],complete=bind(O/'COMPLETE.json'))))
    except BaseException:
        write(O/'FAILED.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()
