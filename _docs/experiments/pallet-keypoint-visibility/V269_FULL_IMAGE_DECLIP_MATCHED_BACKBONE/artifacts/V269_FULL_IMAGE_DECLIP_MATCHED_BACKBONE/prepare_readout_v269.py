"""Build the Korean readout from frozen metrics and actual compact RGBs."""
from pathlib import Path
import csv, hashlib, html, io, json, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

W=Path(__file__).absolute().parent
ARMS=('EVA_STANDARD','EVA_CSA','EVA_DECLIP')
MODELS=('OPENAI_FULL_CLIP','V267_CS',*ARMS)
NAMES=dict(OPENAI_FULL_CLIP='OpenAI 일반 CLIP',V267_CS='기존 Full CLIP Surgery',
           EVA_STANDARD='EVA 일반',EVA_CSA='EVA + CSA',EVA_DECLIP='EVA + DeCLIP')

def safe(p):
    p=Path(p);assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112','cache','.pyc','..'))
    for a in reversed((p,*p.parents)):assert not a.is_symlink()
    return p

def read(p):return safe(p).read_bytes()
def bound(b):
    r=read(b['path']);assert len(r)==b['size_bytes'] and hashlib.sha256(r).hexdigest()==b['sha256'];return r
def dump(p,j):
    with safe(p).open('x') as f:json.dump(j,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',n)
def fit(im,w=290,h=218):
    b=Image.new('RGB',(w,h),'#1c293b');im=im.copy();im.thumbnail((w,h),Image.Resampling.BICUBIC);b.paste(im,((w-im.width)//2,(h-im.height)//2));return b
def marked(im,xy):
    im=im.copy();d=ImageDraw.Draw(im);x,y=xy;r=max(3,round(im.width/200));d.ellipse((x-r,y-r,x+r,y+r),outline='#ffea25',width=2);return im

def main():
    torch.set_num_threads(1)
    metrics=list(csv.DictReader(io.StringIO(read(W/'evaluation_v1/CLASSIFICATION.csv').decode())))
    points=list(csv.DictReader(io.StringIO(read(W/'evaluation_v1/ALL1784_COMPARISON.csv').decode())))
    def row(m,g='ALL',c='PRIMARY'):return next(r for r in metrics if r['model']==m and r['group']==g and r['cohort']==c)
    independent=json.loads(read(W/'ROOT_INDEPENDENT_METRICS_CHECK.json'));assert independent['PASS']
    syn=json.loads(read(W/'syn_mask_diagnostic_v1/ORIGINAL_PIXEL_RESULTS.json'))
    selection=json.loads(read(W/'ROOT_VISUAL_SELECTION_BEFORE_RGB.json'))
    vis=json.loads(read(W/'evaluation_v1/compact_visual_inputs/INPUTS.json'))
    frames={r['frame_index']:r for r in vis['frame_metadata']}
    rgbfiles={r['frame_index']:r['file'] for r in vis['full_RGBs']}
    with np.load(io.BytesIO(read(W/'CURRENT_FRAME_MASKS.npz')),allow_pickle=False) as z:
        maps={a:np.array(z[a+'_logit'],copy=True) for a in ARMS}
    # Historical maps are included in the compact visual fixture.
    with np.load(io.BytesIO(bound(vis['masks'])),allow_pickle=False) as z:
        old_by_frame={int(fi):np.array(field,copy=True) for fi,field in zip(z['frame_index'],z['V267_CS_logit'])}
    maps['V267_CS']=old_by_frame
    new=row('EVA_DECLIP');old=row('V267_CS');matched=row('EVA_CSA')
    joint=float(new['D_recall'])>float(old['D_recall']) and float(new['E_recall'])>=float(old['E_recall'])
    title='전체 집계 보임·가림 개선 / 정사각형 보임 악화' if joint else '보임·외부 가림 동시 개선은 미달'
    selected=selection['cases'];H=400+len(selected)*350+70
    canvas=Image.new('RGB',(1860,H),'#f6f8fb');d=ImageDraw.Draw(canvas)
    d.text((24,16),'V269 · 전체 영상 DeCLIP 실제 비교: '+title,font=font(31),fill='#182a42')
    d.text((24,66),'전체 RGB512 · 합성768학습/235검증 · 새 head 3개×1000회 · 실사223평가만',font=font(23),fill='#485b74')
    for j,m in enumerate(MODELS):
        r=row(m);t=f"{NAMES[m]}   보임 {r['D_D']}/1147 ({100*float(r['D_recall']):.2f}%)   외부 가림 {r['E_E']}/90 ({100*float(r['E_recall']):.2f}%)"
        d.text((24,113+j*42),t,font=font(23),fill='#182a42')
    d.text((24,337),'노란 원=원래 초기 query · 전 지도 raw logit 보간 후 sigmoid, 고정색 0–1 · crop은 표시 전용',font=font(23),fill='#a54813')
    details=[]
    for k,case in enumerate(selected):
        g=case['global_index'];fi=g//8;meta=frames[fi];xy=meta['original_query_xy8'][g%8]
        with Image.open(io.BytesIO(bound(rgbfiles[fi]))) as im:rgb=np.array(im.convert('RGB'),copy=True)
        h,w=rgb.shape[:2];y=400+k*350
        d.text((24,y),f"G{g} · "+case['title'],font=font(25),fill='#182a42')
        panels=[marked(Image.fromarray(rgb),xy)]
        for a in ('V267_CS',*ARMS):
            z=F.interpolate(torch.from_numpy(maps[a][fi]).reshape(1,1,32,32),(h,w),mode='bilinear',align_corners=False)[0,0].numpy()
            p=1/(1+np.exp(-np.clip(z,-80,80)));color=np.stack((14+10*p,25+230*p,19+10*p),-1).astype('uint8')
            panels.append(marked(Image.fromarray(color),xy))
        xq,yq=xy;crop=Image.fromarray(rgb).crop((round(xq)-48,round(yq)-48,round(xq)+48,round(yq)+48)).resize((218,218),Image.Resampling.NEAREST)
        cd=ImageDraw.Draw(crop);cd.line((103,109,115,109),fill='#ffea25',width=2);cd.line((109,103,109,115),fill='#ffea25',width=2);panels.append(crop)
        labs=['전체 RGB','기존 Surgery P(T)','EVA 일반 P(T)','EVA + CSA P(T)','EVA + DeCLIP P(T)','초기 q 확대 / 표시만']
        for j,(lab,im) in enumerate(zip(labs,panels)):
            x=24+307*j;d.text((x,y+39),lab,font=font(18),fill='#485b74');canvas.paste(fit(im),(x,y+67))
        outputs={a:dict(pred=points[g][a+'_pred'],p_target=float(points[g][a+'_p_target'])) for a in ('V267_CS',*ARMS)}
        d.text((24,y+296),' / '.join(f"{NAMES[a]} {v['pred']} P={v['p_target']:.3f}"for a,v in outputs.items()),font=font(20),fill='#182a42')
        details.append(dict(**case,outputs=outputs))
    d.text((24,H-47),'M4=보이는 대상 픽셀 · non-target은 외부 가림 원인과 다름 · COCO 적응 가중치 이력 공개 · 새 PnP/좌표 변경0',font=font(22),fill='#a54813')
    canvas.save(safe(W/'RESULT_OVERVIEW_KO.png'))
    if '--figure-only' in sys.argv:
        print(json.dumps(dict(completed=True,PASS=True,figure_only=True,new_fits_or_forwards=0)))
        return
    summary=dict(completed=True,PASS=True,joint_improvement_vs_CS=joint,
        primary=[row(m) for m in MODELS],manual_D_LE3=[row(m,c='MANUAL_CLICK_D_LE3_VALID_INPUT') for m in MODELS],
        new_fits=3,optimizer_updates=3000,real_new_fit_or_selection=0,new_PnP=0,
        DeCLIP_vs_CS_D_count_delta=int(new['D_D'])-int(old['D_D']),DeCLIP_vs_CS_E_count_delta=int(new['E_E'])-int(old['E_E']),
        matched_DeCLIP_vs_CSA_D_count_delta=int(new['D_D'])-int(matched['D_D']),matched_DeCLIP_vs_CSA_E_count_delta=int(new['E_E'])-int(matched['E_E']),
        actual_cases=details,original_SYN_pixel_metrics=syn['results'],gate_or_deployment_adopted=False)
    summary['material_CS_to_DeCLIP']={g:dict(D_correct=[int(row('V267_CS',g)['D_D']),int(row('EVA_DECLIP',g)['D_D'])],
        D_total=int(row('EVA_DECLIP',g)['truth_D']),E_correct=[int(row('V267_CS',g)['E_E']),int(row('EVA_DECLIP',g)['E_E'])],
        E_total=int(row('EVA_DECLIP',g)['truth_E']))for g in ('WOOD_PRIMARY105','SQUARE118')}
    summary['joint_improvement_all_materials']=False
    dump(W/'FINAL_RESULT_SUMMARY.json',summary)
    table=''
    for group,title2 in [('ALL','전체'),('WOOD_PRIMARY105','목재105장'),('SQUARE118','정사각형118장')]:
        table+=f'<h2>{title2}</h2><table><tr><th>방법</th><th>D→D/E/U</th><th>E→D/E/U</th><th>D P/R%</th><th>E P/R%</th><th>보류%</th></tr>'
        for m in (*MODELS,'V264_HEATMAP_LOCAL','CAL_LEGACY'):
            r=row(m,group);table+=f"<tr><td>{NAMES.get(m,m)}</td><td>{r['D_D']}/{r['D_E']}/{r['D_U']}</td><td>{r['E_D']}/{r['E_E']}/{r['E_U']}</td><td>{100*float(r['D_precision']):.2f}/{100*float(r['D_recall']):.2f}</td><td>{100*float(r['E_precision']):.2f}/{100*float(r['E_recall']):.2f}</td><td>{100*float(r['U_rate']):.2f}</td></tr>"
        table+='</table>'
    near='<table><tr><th>방법</th><th>초기오차≤3px 수동 보임 정답 /376</th></tr>'+''.join(f"<tr><td>{NAMES[m]}</td><td>{row(m,c='MANUAL_CLICK_D_LE3_VALID_INPUT')['D_D']}</td></tr>"for m in MODELS)+'</table>'
    synthetic='<table><tr><th>방법</th><th>원본 M4 IoU%</th><th>경계F1@1px%</th><th>경계F1@3px%</th></tr>'
    for m in MODELS:
        r=syn['results'][m];synthetic+=f"<tr><td>{NAMES[m]}</td><td>{100*r['micro_IoU']:.2f}</td><td>{100*r['boundary1px']['F1']:.2f}</td><td>{100*r['boundary3px']['F1']:.2f}</td></tr>"
    synthetic+='</table>'
    findings=read(W/'ROOT_VISUAL_FINDINGS_KO.md').decode()
    paras=''.join('<p>'+html.escape(p)+'</p>'for p in findings.split('\n\n') if p.strip())
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V269 전체 영상 DeCLIP 실제 비교</title><style>body{{font:17px/1.7 sans-serif;max-width:1860px;margin:30px auto;padding:0 24px;background:#f6f8fb;color:#182a42}}table{{border-collapse:collapse;background:white}}td,th{{border:1px solid #cad6e6;padding:9px}}img{{max-width:100%}}.note{{background:#fff0cf;padding:18px}}</style>
<h1>전체 영상 DeCLIP 실제 적용 완료 — {title}</h1>
<p class="note">기존 Surgery 대비 DeCLIP의 보임 정답 변화 {summary['DeCLIP_vs_CS_D_count_delta']:+d}/1147, 외부 가림 정답 변화 {summary['DeCLIP_vs_CS_E_count_delta']:+d}/90입니다. 같은 EVA+CSA 경로에서 백본 가중치 패키지를 바꾸고 동일 조건으로 head를 학습하면 보임 {summary['matched_DeCLIP_vs_CSA_D_count_delta']:+d}, 외부 가림 {summary['matched_DeCLIP_vs_CSA_E_count_delta']:+d}입니다. 실사 결과에 맞춘 추가 학습·문턱·체크포인트 선택은0이며 새 게이트를 배포하지 않았습니다.</p>
<p><a href="evaluation_v1/INDEX_KO.html">실제 영상 전체 사례</a> · <a href="evaluation_v1/CLASSIFICATION.csv">84행 CM/P/R/보류</a> · <a href="evaluation_v1/SELF_RESPONSES.csv">자기 가림21행</a> · <a href="evaluation_v1/EXTERNAL90_TRANSITIONS.csv">외부 가림90점</a> · <a href="syn_mask_diagnostic_v1/INDEX_KO.html">합성 RGB/M4 대조</a> · <a href="RUN_GUIDE_KO.md">학습·재현 안내</a></p>
<img src="RESULT_OVERVIEW_KO.png">{table}
<h2>백본과 DeCLIP 효과를 구분한 방법</h2><p>OpenAI 일반 CLIP→EVA 일반은 백본 구조와 일반 사전학습 패키지가 함께 바뀝니다. EVA 일반→EVA+CSA는 원본 가중치 그대로 마지막 attention/readout(잔차·FFN 처리 포함)이 바뀝니다. EVA+CSA→DeCLIP은 구조·CSA 읽기·특징폭·분할 head·학습 배치를 맞추고 공개 dense 적응 가중치 패키지만 바뀝니다. 기존 Surgery는 OpenAI ViT-B/16의 공식 Surgery 영상 구조를 사용한 역사적 참조이며, 공식 text feature surgery/zero-shot 전체 파이프라인 재현은 아닙니다.</p>
<p>새 세 방식은 전체 RGB512→공식 projected512 특징32×32→L2 정규화→동일26721개 파라미터 head입니다. 같은 M4 target, seed17 초기값, 학습768/DEV235, 배치8·AdamW0.001/0.0001·고정1000회를 사용했습니다. 백본은 동결했고 KP crop/sliding 입력은 없습니다. 세 final을 DEV 전에 동결하고, 실사223장 전체 지도를 만든 후 초기 q를 읽었습니다.</p>
<p>공개 DeCLIP은 COCO 이미지와 DINOv2 교사를 통한 적응 이력이 있습니다. 이번 팔레트용 head 학습은 합성만 사용했습니다. <a href="https://github.com/xiaomoguhz/DeCLIP">공식 코드</a> · <a href="https://arxiv.org/html/2505.04410v1">공식 논문</a> · <a href="OFFICIAL_ASSETS_COMPLETE.json">실제 source/weights commit·SHA</a></p>
<h2>실제 영상에서 확인한 점</h2>{paras}{near}
<h2>같은 원본 합성235장 픽셀 평가</h2>{synthetic}
<p>모든 raw32 logit를 원래 H/W로 bilinear 복원한 뒤 M4 uint8&gt;127과 비교했습니다. 경계는 원본1/3px 고정 허용입니다. 합성 IoU와 실사 native 보임 recall은 다른 지표입니다. 실사 픽셀 GT가 없어 실사 IoU는 보고하지 않습니다.</p>
<h2>해석과 재현 범위</h2><p>M4는 보이는 대상 픽셀이고 여집합에는 배경·틈·다른 물체가 섞입니다. 초기 q의 대상 여부를 특정 native 코너의 보임/외부 가림에 대응한 proxy입니다. 자기 가림은 E에 합치지 않았습니다. PRIMARY1237=D1147/E90, 원래 입력불가45점(U)을 보존했습니다. CAL은 과거 실사 학습 이력이 있는 참조입니다. G996은 사용자와 기존 E 주석의 해석 이견을 보존했습니다.</p>
<p>실사223장은 학습·정규화 통계·문턱 선택에 넣지 않았지만 이미 반복 열람된 개발 평가입니다. 독립 블라인드 성능이나 목표90% 달성을 주장하지 않습니다. 새 PnP·위치 보정·추가 fit은 없습니다.</p>
<p><a href="PROTOCOL_BEFORE_FIT.json">학습 전 조건</a> · <a href="CURRENT_PREDICTIONS_FROZEN.json">예측 동결</a> · <a href="ROOT_INDEPENDENT_METRICS_CHECK.json">독립 검산</a> · <a href="fullframe_replay_v2/COMPLETE.json">CPU head·query 재현</a> · <a href="RUN_GUIDE_KO.md">재현 경로</a></p></html>'''
    safe(W/'ROOT_READOUT_KO.html').write_text(body)
    print(json.dumps(dict(completed=True,PASS=True,report=str(W/'ROOT_READOUT_KO.html'),joint_improvement=joint,cases=[r['global_index']for r in selected])))

if __name__=='__main__':main()
