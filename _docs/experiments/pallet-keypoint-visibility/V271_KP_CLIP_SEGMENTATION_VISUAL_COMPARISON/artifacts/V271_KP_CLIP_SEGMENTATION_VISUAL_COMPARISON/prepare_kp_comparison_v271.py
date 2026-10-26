"""Saved-output native-ID classification tables and whole-frame display panels only."""
import csv,json,hashlib,math,html
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).parent;R=W.parent
SOURCE=R/'V270_INITIAL_KP_JOINT_VISIBILITY_QUICKCHECK/evaluation_v4/ALL1784_CLASSIFICATION_COORDINATE_JOIN.csv'
REFERENCE=SOURCE.parent/'RESULTS.json'
V61=R.parent.parent/'target_presence_classifier_20261022_v61'
FRAMES=R/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK/current_fullframe_inputs_v2/FRAMES223_INPUT_ONLY.jsonl'
METHODS=['CAL_LEGACY','V61_PRESENCE_PROXY','HEATMAP_LOCAL','FULL_CLIP','FULL_CLIP_SURGERY_ARCH','EVA_STANDARD','EVA_CSA','EVA_DECLIP','VIS_ONLY','JOINT']
DISPLAY=['CAL_LEGACY','HEATMAP_LOCAL','FULL_CLIP_SURGERY_ARCH','EVA_DECLIP','VIS_ONLY','JOINT']
FIXED=[0,25,46,100,105,106,113,124,189,83]
COLORS={'D':'#d6f2df','E':'#f8d4db','U':'#e6e9ee','S':'#eadfff','UNKNOWN':'#e6e9ee'}
def safe(p):
    p=Path(p);assert p.is_absolute() and not any(t in str(p).lower() for t in ('v112','cache','.pyc','..'))
    for a in [p,*p.parents]:assert not a.is_symlink()
    return p
def bind(p):
    p=safe(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),size_bytes=p.stat().st_size)
def dump(p,x):safe(p).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def writecsv(p,rows):
    with safe(p).open('w',newline='') as f:
        d=csv.DictWriter(f,fieldnames=list(rows[0]));d.writeheader();d.writerows(rows)
def ratio(a,b):return a/b if b else None
def cm(rows,method):
    c={t+'_'+p:sum(r['truth']==t and r[method]==p for r in rows) for t in ('D','E') for p in ('D','E','U')}
    dd,de,du,ed,ee,eu=[c[k] for k in ('D_D','D_E','D_U','E_D','E_E','E_U')];n=len(rows);dr=ratio(dd,dd+de+du);er=ratio(ee,ed+ee+eu)
    return dict(n=n,actual_D=dd+de+du,actual_E=ed+ee+eu,confusion_matrix=c,D_precision=ratio(dd,dd+ed),D_recall=dr,D_F1=ratio(2*dd,2*dd+de+ed+du),E_precision=ratio(ee,ee+de),E_recall=er,E_F1=ratio(2*ee,2*ee+de+ed+eu),U_rate=ratio(du+eu,n),coverage=ratio(n-du-eu,n),accuracy_all=ratio(dd+ee,n),accepted_accuracy=ratio(dd+ee,n-du-eu),balanced_recall=(dr+er)/2 if dr is not None and er is not None else None)
def font(size):return ImageFont.truetype(str(safe(Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))),size)
def show(d,xy,text,size=24,fill='#19304a',**kw):d.text(xy,str(text),font=font(size),fill=fill,**kw)
def gtlabel(row):
    return dict(DIRECT_VISIBLE='D 보임',EXTERNAL_OCCLUDED='E 외부 가림',SELF_OCCLUDED='S 자기 가림',OUT_OF_FRAME='OOB 밖',UNANNOTATED='미주석',UNKNOWN_MAPPING='대응 불명')[row['human_state']]
def panel(out,fi,rows,frame):
    ib=frame['image_binding'];assert bind(Path(ib['path']))==ib
    rgb=Image.open(safe(Path(ib['path']))).convert('RGB');assert hashlib.sha256(rgb.tobytes()).hexdigest()==frame['decoded_RGB_sha256']
    image=Image.new('RGB',(2240,1050),'#f1f5fa');d=ImageDraw.Draw(image)
    show(d,(35,20),f"F{fi:03d} | {frame['frame_id']}",36)
    show(d,(35,76),'같은 native 번호의 판정 비교 · 새 학습 / 새 추론 / 좌표 재매칭 없음',24)
    show(d,(35,121),'이미지의 번호 위치 = 저장된 기존 query 좌표',25)
    scale=min(1050/rgb.width,650/rgb.height);ww,hh=round(rgb.width*scale),round(rgb.height*scale);px,py=35,180+(650-hh)//2
    image.paste(rgb.resize((ww,hh),Image.Resampling.LANCZOS),(px,py));missing=[];placed=[]
    for row in rows:
        k=row['native_slot'];x,y=row['original_q_x'],row['original_q_y']
        if not (math.isfinite(x) and math.isfinite(y) and 0<=x<rgb.width and 0<=y<rgb.height):missing.append(k);continue
        cx,cy=px+x*scale,py+y*scale;fill={'D':'#45d77c','E':'#ff5878','S':'#bc8df0'}.get(row['truth'],'#dbe1ea')
        d.ellipse((cx-7,cy-7,cx+7,cy+7),fill=fill,outline='black',width=2)
        offsets=[(-20,-37),(8,-37),(8,9),(-28,9),(32,-15),(-53,-15),(5,36),(-20,-62)]
        for ox,oy in offsets:
            lx=max(px+4,min(px+ww-34,cx+ox));ly=max(py+4,min(py+hh-40,cy+oy));box=(lx,ly,lx+32,ly+38)
            if not any(not(box[2]<b[0] or box[0]>b[2] or box[3]<b[1] or box[1]>b[3]) for b in placed):break
        placed.append(box);d.line((cx,cy,lx+16,ly+19),fill='black',width=2);d.rounded_rectangle(box,radius=6,fill='#ffffff',outline='black',width=2);show(d,(lx+16,ly+19),k,23,'#101923',anchor='mm')
    show(d,(35,866),'점 색 = 사람 주석: 초록 D / 분홍 E / 보라 S / 회색 미확정',22)
    show(d,(35,904),'JOINT·VIS는 새 모델의 같은 native 출력입니다.',23)
    show(d,(35,940),'그 모델들이 그림의 기존 q 위치를 입력으로 썼다는 뜻은 아닙니다.',22)
    if missing:show(d,(35,980),'그림 밖 기존 q 번호: '+', '.join(map(str,missing)),20)
    tx=1130;widths=[70,170,135,135,135,135,135,135];titles=['KP','사람 GT','기존 CAL','local HM','CLIP CS','DeCLIP','VIS','JOINT'];xs=[tx]
    for v in widths:xs.append(xs[-1]+v)
    for j,title in enumerate(titles):
        d.rectangle((xs[j],170,xs[j+1]-2,226),fill='#20354e');show(d,((xs[j]+xs[j+1])/2,198),title,22,'white',anchor='mm')
    for ir,row in enumerate(rows):
        yy=234+ir*70;vals=[str(row['native_slot']),gtlabel(row)+(' *' if row['truth'] in ('D','E') and not row['primary'] else '')]+[row[m] for m in DISPLAY]
        for j,text in enumerate(vals):
            pred=vals[j] if j>=2 else row['truth'];fill=COLORS.get(pred,'#e6e9ee') if j else '#ffffff';border='#ffffff';bw=2
            if j>=2 and row['primary']:
                border='#16864d' if text==row['truth'] else '#bd2448';bw=4
            d.rectangle((xs[j],yy,xs[j+1]-2,yy+64),fill=fill,outline=border,width=bw);show(d,((xs[j]+xs[j+1])/2,yy+32),text,25 if j>=2 else 22,anchor='mm')
    show(d,(tx,825),'D=보임 / E=외부 가림 / U=방법의 보류',23)
    show(d,(tx,866),'주평가 D/E 행의 테두리: 초록=정답, 빨강=오답·보류',21)
    show(d,(tx,907),'S·OOB·미주석·대응 불명은 D/E 정답으로 합치지 않습니다.',21)
    show(d,(tx,948),'* = D/E 주석은 있으나 공통 주평가1237점에서는 제외',20)
    show(d,(tx,988),'CLIP 계열 판정은 팔레트 mask 존재의 대리값입니다.',20)
    image.save(safe(out/f'FRAME_{fi:03d}_KP_COMPARISON.png'))
def main():
    out=safe(W/'kp_comparison');assert not out.exists();out.mkdir()
    source=list(csv.DictReader(safe(SOURCE).open()));v61=list(csv.DictReader(safe(V61/'REAL_QUERY_PREDICTIONS.csv').open()));assert len(source)==len(v61)==1784
    points=[];maxdiff=0
    for i,(r,v) in enumerate(zip(source,v61)):
        assert int(r['global_index'])==i and r['frame_id']==v['id'] and r['native_slot']==v['native_corner'] and r['group']==v['group']
        x,y=float(r['old_query_x']),float(r['old_query_y']);maxdiff=max(maxdiff,abs(x-float(v['query_x'])),abs(y-float(v['query_y'])))
        p=dict(global_index=i,frame_index=int(r['frame_index']),native_slot=int(r['native_slot']),frame_id=r['frame_id'],row_id=r['row_id'],group=r['group'],truth=r['truth'],human_state=r['human_state'],primary=r['primary']=='True',original_q_x=x,original_q_y=y)
        for m in METHODS:
            p[m]=('U' if v['valid']!='True' else 'D' if v['predicted_present']=='True' else 'E') if m=='V61_PRESENCE_PROXY' else r[m+'_D_E_U']
            assert p[m] in ('D','E','U')
        p.update(V61_presence_probability=float(v['presence_probability']),VIS_ONLY_p_E=float(r['VIS_ONLY_p_E']),JOINT_p_E=float(r['JOINT_p_E']),new_VIS_ONLY_x=float(r['VIS_ONLY_x']),new_VIS_ONLY_y=float(r['VIS_ONLY_y']),new_JOINT_x=float(r['JOINT_x']),new_JOINT_y=float(r['JOINT_y']))
        points.append(p)
    assert maxdiff==0 and sum(p['primary'] for p in points)==1237
    metrics=[];ref=json.loads(safe(REFERENCE).read_text());groups=['ALL','WOOD_PRIMARY105','SQUARE118']
    for cohort in ('PRIMARY_PAIRED1237','ALL_REGISTERED_D_E'):
        base=[p for p in points if p['truth'] in ('D','E') and (p['primary'] if cohort=='PRIMARY_PAIRED1237' else True)]
        for group in groups:
            gr=[p for p in base if group=='ALL' or p['group']==group]
            for slot in [None,*range(8)]:
                sr=[p for p in gr if slot is None or p['native_slot']==slot]
                for m in METHODS:
                    result=cm(sr,m)
                    if slot is None and m!='V61_PRESENCE_PROXY':assert result==ref['metrics'][cohort][group][m],(cohort,group,m)
                    metrics.append(dict(cohort=cohort,group=group,native_slot='ALL' if slot is None else slot,method=m,**{k:v for k,v in result.items() if k!='confusion_matrix'},**result['confusion_matrix']))
    nonde=[]
    for state in sorted(set(p['human_state'] for p in points)):
        for m in METHODS:
            rows=[p for p in points if p['human_state']==state];count=Counter(p[m] for p in rows);nonde.append(dict(human_state=state,method=m,N=len(rows),pred_D=count['D'],pred_E=count['E'],pred_U=count['U'],interpretation='D/E confusion metrics only for DIRECT_VISIBLE/EXTERNAL_OCCLUDED; all other states retained separately'))
    sources=dict(scored_join=bind(SOURCE),reference_results=bind(REFERENCE),V61_predictions=bind(V61/'REAL_QUERY_PREDICTIONS.csv'),V61_protocol=bind(V61/'PROTOCOL.json'),V61_freeze=bind(V61/'MODEL_FREEZE.json'),frame_manifest=bind(FRAMES),code=bind(Path(__file__)))
    scope=dict(primary='Human native D/E common PRIMARY1237; SELF/OOB/unannotated/mapping-unknown separate',query_display='All panel image markers use same frozen historical q only. VIS_ONLY/JOINT classify new raw-model native slots and do not consume those q coordinates. New coordinates are separate numeric fields and never substituted into shared display.',CLIP='Mask-pallet presence readout is an external-occlusion proxy, not physical owner/causal gold.',V61='Saved fixed threshold .5 center-pixel target-presence; D=present,E=absent,U=invalid. Semantic D/E transfer proxy, not real target-pixel accuracy.',legacy='Existing real-CAL-trained reference; distinct training provenance from SYN-only experiments.',repeated_real_set='Existing repeatedly examined223 development images, not new independent real validation.',new_fits=0,new_model_forwards=0,new_PnP=0)
    writecsv(out/'POINT_MODEL_COMPARISON.csv',points);dump(out/'POINT_MODEL_COMPARISON.json',dict(PASS=True,sources=sources,scope=scope,methods=METHODS,rows=points));writecsv(out/'METRICS_OVERALL_MATERIAL_NATIVE.csv',metrics);dump(out/'METRICS_OVERALL_MATERIAL_NATIVE.json',dict(PASS=True,rows=metrics));writecsv(out/'ALL_HUMAN_STATE_DISTRIBUTIONS.csv',nonde)
    frames=[json.loads(x) for x in safe(FRAMES).read_text().splitlines()]
    dump(out/'PANEL_SELECTION_BEFORE_RGB.json',dict(PASS=True,frame_indices=FIXED,selection='Exact root-fixed ten frames; chosen before this script opens RGB. No model/threshold/case search.',sources=sources))
    for fi in FIXED:
        rows=[p for p in points if p['frame_index']==fi];assert [p['native_slot'] for p in rows]==list(range(8));panel(out,fi,rows,frames[fi])
    pct=lambda x:'—' if x is None else f'{x*100:.2f}%'
    summary=[r for r in metrics if r['cohort']=='PRIMARY_PAIRED1237' and r['group']=='ALL' and r['native_slot']=='ALL']
    table='<table><tr><th>方法</th><th>D recall</th><th>E recall</th><th>E precision</th><th>保留U</th><th>D→D/E/U</th><th>E→D/E/U</th></tr>'+''.join(f"<tr><td>{r['method']}</td><td>{pct(r['D_recall'])}</td><td>{pct(r['E_recall'])}</td><td>{pct(r['E_precision'])}</td><td>{pct(r['U_rate'])}</td><td>{r['D_D']}/{r['D_E']}/{r['D_U']}</td><td>{r['E_D']}/{r['E_E']}/{r['E_U']}</td></tr>" for r in summary)+'</table>'
    style='<style>body{font:18px sans-serif;color:#16304c;background:#f2f5fa;max-width:1900px;margin:25px auto;padding:16px}p{line-height:1.7}article,section{background:white;padding:22px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}table{border-collapse:collapse}th,td{padding:10px;border:1px solid #bac6d4}a{color:#155db0}</style>'
    intro='<h1>같은 native 번호에서 KP 판정 비교</h1><p>저장된223장×8점, 총1784행. 주평가1237점(D1147/E90)의 혼동행렬과 precision/recall을 같은 대상에서 계산했습니다. S/밖/미주석/대응 불명은 별도 보존했습니다. v61은 존재/부재의 semantic 전이 proxy입니다. 기존 CAL은 실사 CAL 학습 이력이 있는 기준이며 새 SYN 실험과 학습 조건이 다릅니다.</p><p><b>그림의 점 위치는 모두 기존 q입니다.</b> JOINT·VIS_ONLY는 같은 native 번호의 새 full-image 모델 판정이며, 그 q를 입력으로 쓰거나 그 위치의 mask를 읽은 값이 아닙니다. CLIP은 target-mask 존재의 대리 판정입니다. 따라서 동일한 픽셀을 모든 모델이 판단했다는 비교가 아닙니다.</p><p>JOINT의 slot2/3/6/7 전부 E 패턴과 CLIP의 이미지별 mask/query 실패를 구분해 보세요. 재평가·시각화로 checkpoint나 임계값을 선택하지 않았습니다.</p>'
    links='<p><a href="POINT_MODEL_COMPARISON.html">전체1784점 검색표</a> · <a href="POINT_MODEL_COMPARISON.csv">행별 CSV</a> · <a href="METRICS_OVERALL_MATERIAL_NATIVE.csv">전체/재질/native별 지표 CSV</a> · <a href="ALL_HUMAN_STATE_DISTRIBUTIONS.csv">모든 주석 상태 분포</a></p>'
    cards=''.join(f'<article><h2>F{fi:03d} — {html.escape(frames[fi]["frame_id"])}</h2><a href="FRAME_{fi:03d}_KP_COMPARISON.png"><img src="FRAME_{fi:03d}_KP_COMPARISON.png" alt="full frame and native classification table"></a></article>' for fi in FIXED)
    safe(out/'INDEX_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>KP native 판정 비교</title>'+style+intro+links+'<section>'+table+'</section>'+cards+'</html>')
    fields=['frame_index','frame_id','native_slot','human_state','primary',*METHODS]
    datajson=json.dumps([{k:p[k] for k in fields} for p in points],ensure_ascii=False).replace('</','<\\/')
    script='<script>const rows='+datajson+';const keys='+json.dumps(fields)+';function draw(){const q=document.getElementById("q").value.toLowerCase();const sel=rows.filter(r=>Object.values(r).join(" ").toLowerCase().includes(q));document.getElementById("n").textContent=sel.length+" / 1784";document.getElementById("t").innerHTML="<tr>"+keys.map(k=>"<th>"+k+"</th>").join("")+"</tr>"+sel.map(r=>"<tr>"+keys.map(k=>"<td>"+r[k]+"</td>").join("")+"</tr>").join("")}draw();</script>'
    safe(out/'POINT_MODEL_COMPARISON.html').write_text('<!doctype html><meta charset="utf-8"><title>全1784 KP比較</title>'+style+intro+'<p><a href="INDEX_KO.html">画像比較へ</a></p><input id="q" oninput="draw()" placeholder="frame/material/state/native 검색" style="font-size:22px;width:650px"><span id="n"></span><div style="overflow:auto"><table id="t"></table></div>'+script)
    outputs={p.name:bind(p) for p in sorted(out.iterdir()) if p.is_file()}
    dump(out/'COMPLETE.json',dict(PASS=True,completed=True,sources=sources,scope=scope,rows=1784,frames=223,methods=METHODS,primary_rows=1237,V61_same_all1784_IDs=True,V61_max_original_q_coordinate_difference_px=maxdiff,reference_metric_blocks_exact=54,panel_frames=FIXED,outputs=outputs,new_model_forwards=0,new_fits=0))
    print(json.dumps(dict(PASS=True,output=str(out),points=1784,metrics=len(metrics),panels=10,V61_max_original_q_difference=maxdiff,primary_summary=summary),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
