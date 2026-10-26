"""Truthful result summary and actual full-frame success/failure overview."""
from pathlib import Path
import csv
import hashlib
import io
import json
import sys
import types
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

HERE=Path(__file__).absolute().parent
PREV=HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'
MODELS=('V267_CS','HIGHRES_REFINE','BOUNDARY_REFINE')
NAMES={'V267_CS':'이전 전체 CLIP','HIGHRES_REFINE':'고해상도 보정 A','BOUNDARY_REFINE':'경계 손실 추가 B',
       'V264_HEATMAP_LOCAL':'이전 V264','v61':'이전 v61','CAL_LEGACY':'과거 실사학습 모델'}


def safe(p):
    p=Path(p)
    assert p.is_absolute()and not any(x in str(p).lower()for x in('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink()
    return p


def read(b):
    raw=safe(Path(b['path'])).read_bytes();assert len(raw)==b['size_bytes']and hashlib.sha256(raw).hexdigest()==b['sha256'];return raw


def dump(p,d):
    p=safe(p)
    if p.exists():
        assert json.loads(p.read_bytes())==d
        return
    with p.open('x')as f:json.dump(d,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')


def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',size)


def panel(image,w=290,h=220):
    base=Image.new('RGB',(w,h),'#1e2939');x=image.copy();x.thumbnail((w,h),Image.Resampling.BICUBIC);base.paste(x,((w-x.width)//2,(h-x.height)//2));return base


def marked(image,coords,chosen):
    x=image.copy();d=ImageDraw.Draw(x);r=max(2,round(x.width/240))
    for i,q in enumerate(coords):
        if q is None:continue
        a,b=q
        try:valid=bool(np.isfinite([a,b]).all())
        except TypeError:valid=False
        if not valid:continue
        color='#ffe735'if i==chosen else'#34d6ff'
        d.ellipse((a-r,b-r,a+r,b+r),outline=color,width=2)
    return x


def main():
    torch.set_num_threads(1)
    with safe(HERE/'evaluation_v1/CLASSIFICATION.csv').open()as f:metrics=list(csv.DictReader(f))
    with safe(HERE/'evaluation_v1/ALL1784_COMPARISON.csv').open()as f:rows=list(csv.DictReader(f))
    def metric(model,group='ALL',cohort='PRIMARY'):return next(r for r in metrics if r['model']==model and r['group']==group and r['cohort']==cohort)
    checks=json.loads(safe(HERE/'ROOT_INDEPENDENT_METRICS_CHECK.json').read_bytes());assert checks['PASS']
    syn=json.loads(safe(HERE/'syn_mask_diagnostic_v1/ORIGINAL_PIXEL_RESULTS.json').read_bytes())
    # Fixed previous cases plus deterministic first-index new changes, before RGB.
    recovered=checks['primary_transitions']['V267_CS']['HIGHRES_REFINE']['visible_recovered']
    regressed=checks['primary_transitions']['V267_CS']['HIGHRES_REFINE']['external_new_miss']
    selected=[370,min(recovered),min(regressed),200]
    dump(HERE/'ROOT_VISUAL_SELECTION_BEFORE_RGB.json',dict(completed=True,selected_global_indices=selected,
        rule='FixedpreviousconeG370;firstsortedAvisible_recovered;firstsortedAexternal_newmiss;fixedpreviouscornerG200.',
        root_RGB_reads_before_selection=0,representative_frequency_estimate=False,new_fits_or_policy_selection=0))
    receipt=json.loads(safe(HERE/'CURRENT_PREDICTIONS_FROZEN.json').read_bytes());cfg=json.loads(read(receipt['configuration']));policy=json.loads(read(cfg['protocol']))
    inp=json.loads(read(policy['current_input_complete']));frames=[json.loads(x)for x in read(inp['frames']).splitlines()];queries=[json.loads(x)for x in read(inp['queries_manifest']).splitlines()]
    helper=types.ModuleType('v268_overview_original_rgb_io');helper.__file__=policy['current_input_helper']['path'];sys.modules[helper.__name__]=helper
    exec(compile(read(policy['current_input_helper']),helper.__file__,'exec'),helper.__dict__)
    with np.load(io.BytesIO(read(policy['V267_current_frame_masks'])),allow_pickle=False)as z:old=np.array(z['FULL_CLIP_SURGERY_ARCH_logit'],copy=True)
    index=json.loads(read(receipt['frame_masks']));maps={'V267_CS':old}
    for arm,b in index['arms'].items():
        with np.load(io.BytesIO(read(b)),allow_pickle=False)as z:maps[arm]=np.array(z['logits'],copy=True)
    canvas=Image.new('RGB',(1560,1860),'#f6f8fb');d=ImageDraw.Draw(canvas)
    d.text((24,18),'V268 · 전체 이미지 경계 보정: 두 목표의 동시 개선 실패',font=font(31),fill='#16263e')
    d.text((24,66),'합성768학습/235검증 · 실사223평가만 · 입력 KP crop 없음 · 새 학습2회×1000',font=font(23),fill='#475b78')
    for x,t in [(24,'방법'),(430,'보임 정답 /1147'),(770,'외부 가림 정답 /90'),(1190,'보임 손상 / 가림 손상')]:d.text((x,109),t,font=font(22),fill='#16263e')
    for j,model in enumerate(MODELS):
        r=metric(model);y=149+41*j
        a={'V267_CS':'—','HIGHRES_REFINE':'43 / 6','BOUNDARY_REFINE':'93 / 0'}[model]
        for x,t in [(24,NAMES[model]),(430,f"{r['D_D']} ({float(r['D_recall'])*100:.2f}%)"),(770,f"{r['E_E']} ({float(r['E_recall'])*100:.2f}%)"),(1190,a)]:d.text((x,y),t,font=font(23),fill='#16263e')
    d.text((24,285),'A: 보임67복구/43손상 · 가림6새누락   B: 보임10복구/93손상 · 기존 가림84유지',font=font(23),fill='#ad420c')
    titles=['G370 · 콘 가림 유지','G'+str(selected[1])+' · A의 보임 복구','G'+str(selected[2])+' · A의 콘 가림 새 누락','G200 · 보이는 모서리 누락은 남음']
    captions=[]
    for k,(g,title)in enumerate(zip(selected,titles)):
        y=340+k*365;q=queries[g];fi=q['frame_index'];rgb=helper.read_rgb(frames[fi]);h,w=rgb.shape[:2]
        qq=[queries[fi*8+j]['original_q0']for j in range(8)];slot=g%8
        pvalues={m:float(rows[g][m+'_p_target'])for m in MODELS}
        predvalues={m:rows[g][m+'_pred']for m in MODELS}
        d.text((24,y),title,font=font(25),fill='#16263e')
        labels=['전체 RGB · 모델 입력 범위','이전 CLIP P(target) 0–1','A P(target) 0–1','B P(target) 0–1','초기 q 확대 · 표시 전용']
        visual=[marked(Image.fromarray(rgb),qq,slot)]
        for model in MODELS:
            z=torch.from_numpy(maps[model][fi]).reshape(1,1,*maps[model].shape[-2:])
            z=F.interpolate(z,(h,w),mode='bilinear',align_corners=False)[0,0].numpy()
            p=1/(1+np.exp(-np.clip(z,-80,80)));color=np.empty((h,w,3),np.uint8);color[:,:,0]=(14+10*p).astype(np.uint8);color[:,:,1]=(25+230*p).astype(np.uint8);color[:,:,2]=(19+10*p).astype(np.uint8)
            visual.append(marked(Image.fromarray(color),qq,slot))
        a,b=q['original_q0'];crop=Image.fromarray(rgb).crop((round(a)-48,round(b)-48,round(a)+48,round(b)+48)).resize((220,220),Image.Resampling.NEAREST)
        cd=ImageDraw.Draw(crop);cd.line((105,110,115,110),fill='#ffe735',width=2);cd.line((110,105,110,115),fill='#ffe735',width=2);visual.append(crop)
        for j,(lab,im)in enumerate(zip(labels,visual)):
            x=24+j*307;d.text((x,y+41),lab,font=font(18),fill='#475b78');canvas.paste(panel(im),(x,y+70))
        text='  /  '.join(f'{NAMES[m]} {predvalues[m]} · P={pvalues[m]:.3f}'for m in MODELS)
        d.text((24,y+302),text,font=font(20),fill='#16263e');captions.append(dict(global_index=g,title=title,p_target=pvalues,predictions=predvalues))
    d.text((24,1810),'E는 비대상 마스크 대응: 배경/초기좌표오차/외부가림은 서로 다름 · 새 모델 채택 없음',font=font(22),fill='#ad420c')
    canvas.save(safe(HERE/'RESULT_OVERVIEW_KO.png'))
    dump(HERE/'FINAL_RESULT_SUMMARY.json',dict(completed=True,new_model_or_gate_adopted=False,joint_improvement_achieved=False,
        primary=[metric(m)for m in MODELS],manual_D_initial_LE3=[metric(m,cohort='MANUAL_CLICK_D_LE3_VALID_INPUT')for m in MODELS],
        transitions=checks['primary_transitions']['V267_CS'],actual_overview_cases=captions,
        original_SYN_pixel_metrics=syn['results'],new_fits=2,total_optimizer_updates=2000,current_fit_or_threshold_or_checkpoint_selection=0))
    table=''
    for group,title in [('ALL','전체'),('WOOD_PRIMARY105','목재105장'),('SQUARE118','정사각형118장')]:
        table+=f'<h2>{title}</h2><table><tr><th>방식</th><th>D→D/E/U</th><th>E→D/E/U</th><th>D P%</th><th>D R%</th><th>E P%</th><th>E R%</th><th>보류%</th></tr>'
        for model in(*MODELS,'V264_HEATMAP_LOCAL','v61','CAL_LEGACY'):
            r=metric(model,group);table+='<tr><td>'+NAMES[model]+'</td><td>'+'/'.join(r['D_'+x]for x in'DEU')+'</td><td>'+'/'.join(r['E_'+x]for x in'DEU')+'</td>'
            table+=''.join(f'<td>{float(r[x])*100:.2f}</td>'for x in('D_precision','D_recall','E_precision','E_recall','U_rate'))+'</tr>'
        table+='</table>'
    synth=''
    for key,title in [('V267_CS32','이전 CS'),('HIGHRES_REFINE','고해상도 A'),('BOUNDARY_REFINE','경계 B')]:
        r=syn['results'][key];synth+=f"<tr><td>{title}</td><td>{100*r['micro_IoU']:.2f}</td><td>{100*r['boundary1px']['F1']:.2f}</td><td>{100*r['boundary3px']['F1']:.2f}</td></tr>"
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V268 전체 영상 경계 보정 비교</title><style>body{font-family:sans-serif;max-width:1560px;margin:30px auto;padding:0 24px;line-height:1.7;color:#17253c;background:#f5f7fb}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d4deeb;padding:9px}.note{padding:18px;background:#fff0ce}a{color:#165db8}</style>
<h1>전체 이미지 경계 보정 비교 완료 — 두 목표의 동시 개선은 얻지 못함</h1>
<p class="note">A는 보이는 점67개를 복구했지만43개를 새로 놓쳐 정답595→619, 외부 가림은84→78로 줄었습니다. B는 가림84개를 유지했지만 보임 정답이512개로 줄었습니다. <b>새 모델이나 게이트를 채택하지 않았습니다.</b> 계획한 두 학습 이후 추가 fit·문턱 조정은 없습니다.</p>
<p>전체 이미지 방식 유지, 원래 합성768학습/235검증, 실사223평가만입니다. KP crop으로 입력하지 않았습니다. 전체 RGB512→평균256과 동결된 전체512 CLIP32특징/기존32logit를 읽어 새256지도를 만들었습니다. 아래 crop은 표시 전용입니다.</p>
<p><a href="evaluation_v1/INDEX_KO.html">실사111사례 전체 영상·3방식 확률/마스크·확대</a> · <a href="syn_mask_diagnostic_v1/INDEX_KO.html">원본 합성 M4/예측 대조</a> · <a href="RUN_GUIDE_KO.md">재현 안내</a> · <a href="PROTOCOL_BEFORE_FIT.json">학습 전 동결 조건</a></p>
<img src="RESULT_OVERVIEW_KO.png">
<p class="note">D=보임, E=외부 가림 주석에 비대상 마스크를 대응한 <b>MASK_PRESENCE_PROXY</b>, U=입력 불가입니다. 실제 외부 가림 원인을 별도로 학습한 분류기가 아닙니다. 자기 가림은 E와 합치지 않았습니다. PRIMARY1237=D1147/E90; 새 모델U0, 전체1784의 기존45U 보존. 과거 실사학습 모델의 CAL 노출 이력도 유지합니다.</p>
'''+table+'''
<p><a href="evaluation_v1/CLASSIFICATION.csv">모든 평가군72행 CM/P/R/U</a> · <a href="evaluation_v1/SELF_RESPONSES.csv">자기 가림18행</a> · <a href="evaluation_v1/ALL1784_COMPARISON.csv">1784점 결과</a> · <a href="evaluation_v1/EXTERNAL90_TRANSITIONS.csv">가림90점 전이</a></p>
<h2>무엇이 됐고 무엇이 안 됐는가</h2>
<p>G370/G908/G1515의 콘 제외는 유지합니다. A는 G15/G904 등 일부 보이는 모서리를 회복하지만, G800/G803/G837/G1274/G1692/G1772의 가림을 새로 D로 읽습니다. 실제 영상에서 밤의 콘 몸체와 낮의 받침 경계까지 대상 영역이 넓어지는 사례가 있습니다. G1274에서는 배경의 파란 덮개에도 대상 반응이 늘어납니다.</p>
<p>G1692/G1772의 초기 q에는 다른 보이는 팔레트 부분이 섞이며, 가려진 해당 native 참조점과는13.91/32.46px 떨어집니다. 이 둘을 명백한 콘 몸체 오분할이라고 부르면 안 됩니다. 특정 코너 가시성 주석과 현재 픽셀의 대상 존재가 다른 구체 사례입니다. 단일 원인으로 모든 새 누락을 설명하지 않았습니다.</p>
<p>G200은 초기 오차1.77px인 보이는 점인데 세 방법 모두 E입니다. B의 지도는 슬랫·빈틈을 더 자세히 구분해 보여도 native q의 보임 정답이 되는 것은 아닙니다. 실사 픽셀 GT가 없으므로 시각적으로 자세한 지도가 더 정확하다고 확정하지 않습니다. G996의 사용자 판단/기존 E 주석 이견은 그대로 유지했습니다.</p>
<p>초기 오차≤3px인 수동 보임376점의 정답은 이전194/A209/B166입니다. A는24복구/9손상, B는4복구/32손상입니다. 위치 오차만으로 실패 전체를 설명할 수 없고, 정확한 픽셀 분할과 의미상 코너 보임의 차이가 남습니다.</p>
<h2>합성 경계 개선이 실사 두 목표의 동시 개선으로 이어지지 않음</h2>
<p>아래는 같은 합성 DEV235장의 <b>원래 해상도 M4</b> 기준입니다.32와256 target-grid IoU를 직접 비교하지 않았습니다. 경계는 원영상1px/3px 허용 P/R을 모두 계산했으며 문턱이나 허용 반경을 선택하지 않았습니다.</p>
<table><tr><th>방식</th><th>원본 픽셀 micro IoU%</th><th>경계F1@1px%</th><th>경계F1@3px%</th></tr>'''+synth+'''</table>
<p>B의 합성 IoU 증가는 FP1,179,713→985,444 감소와 함께 FN619,015→645,409 증가도 포함합니다. 분할 IoU와 native 보임 recall은 같은 목표가 아닙니다. 작은 어두운 팔레트 미검출, 다른 팔레트 선택, 목재 무늬를 구멍처럼 깎는 문제도 남습니다. 실사 분할 IoU는 GT가 없어 보고하지 않습니다.</p>
<h2>학습 방법·한계·검산</h2>
<p>동결된 V267 CLIP Surgery 영상 구조 특징과 기존 head는 학습하지 않았습니다. 새26177개 파라미터 full-frame refiner 두 개만 같은 초기값·배치·seed17로 각1000업데이트했습니다. M4 원PNG>127→512bilinear antialias→평균256의 soft target을 새로 만듭니다. A=BCE+Dice, B=동일 손실+0.5양쪽 경계 균형 BCE입니다. 경계 band는 손실 가중치에만 쓰며 정답이나 출력 마스크를 팽창시키지 않습니다.</p>
<p>새 지도는 기존32logit를256으로 보간한 값+6tanh(새 residual/6)입니다. 같은256격자의 기존|logit|>6에서 부호 보존을 검증했습니다. 기존32에서 직접 읽는 q와256으로 다시 보간한 q는 완전히 같지 않아, 학습 없는 재보간만의 판정 변화2/1739점도 별도 기록했습니다. 이 수치로 정책을 선택하지 않았습니다.</p>
<p>두 final1000을 합성 DEV 평가 전에 동결하고 실사 지도·1784점 예측을 저장한 뒤에만 주석을 붙였습니다. 실사223은 반복 개발 평가이며 독립 블라인드 검증이 아닙니다. 신모델은 실사 통계·정규화·문턱·체크포인트 적합에 사용하지 않았고 새 PnP/좌표 보정은0입니다.</p>
<p><a href="INDEPENDENT_SCOPE_TRAINING_REVIEW.json">원본1003 SHA·10개 전처리·실제 학습 독립 감사</a> · <a href="ROOT_INDEPENDENT_METRICS_CHECK.json">72행 독립 검산</a> · <a href="fullframe_replay_v1/COMPLETE.json">CPU5전체영상 head/1784점 재현</a> · <a href="current_inference_v1/NOOP_256_RESAMPLING_DIAGNOSTIC_BEFORE_GT.json">재보간 수치 진단</a></p>
<h2>Full CLIP Surgery가 좋아 보이는데 왜 보류했는가</h2>
<p><b>전체 팔레트 영역을 찾는 앞단으로 사용할 수 있습니다.</b> 보류한 것은 지도의 한 점만으로 특정 native의 보임/외부 가림을 확정하는 용도입니다. 초록 지도는 CLIP Surgery 영상 구조 특징+합성 M4 head의 출력이며, 논문의 text feature surgery/full zero-shot pipeline을 실행한 결과가 아닙니다.</p>
<p>V267만 따로 G200/G852의 보이는 모서리 누락, G665의 작은 야간 대상 누락, G1354의 외부 가림 오판, G300의 큰 query/숨은 native 대응 차이를 시각화했습니다. 초기 오차≤3px 수동D376점에서 실제 q는194D, 수동 참조좌표를 넣은 평가용 oracle는200D입니다. 참조점에서도176E이므로 좌표 오차 하나로 모두 설명되지는 않습니다. GT좌표를 쓰는 진단이지 새로운 배포 성능이나 좌표 보정은 아닙니다.</p>
<p><a href="v267_baseline_failures_v1/INDEX_KO.html">V267만 보는 실패5개와 수동좌표 진단</a> · <a href="PRIMARY_SOURCE_DCLIP_REVIEW_KO.md">DeCLIP/DCLIP 공식 논문·구현 확인</a> · <a href="syn_pixel_replay_v1/README_KO.md">원본 데이터 없는 합성235 픽셀 지표 CPU재현</a></p>
<p>확인된 한계는 전체 이미지 인식이 전혀 안 된다는 뜻이 아닙니다. 콘과 팔레트 몸체의 구분은 일부 성공합니다. 후속에서는 전체 문맥을 유지하면서 <b>native 주변 구조가 보이는가</b>라는 감독과 판정을 연결해야 합니다. 마스크 밖을 바로 외부 가림으로 읽는 목표 차이와 실제 전이 실패를 함께 다뤄야 합니다. 이 보고 이후 추가 학습은 진행하지 않았습니다.</p></html>'''
    safe(HERE/'ROOT_READOUT_KO.html').write_text(html)
    print(json.dumps(dict(completed=True,report=str(HERE/'ROOT_READOUT_KO.html'),selected_cases=selected)))


if __name__=='__main__':main()
