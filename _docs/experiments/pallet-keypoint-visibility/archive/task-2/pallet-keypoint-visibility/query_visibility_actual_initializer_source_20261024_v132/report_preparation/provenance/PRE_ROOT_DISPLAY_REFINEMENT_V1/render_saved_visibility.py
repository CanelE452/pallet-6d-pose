"""Saved V132 source scores/crops only; no model calls or source relabeling.

Root executes after both prediction seals and the saved evaluator are complete.
Every displayed crop is read from the C16 input-only saved RGB96 chunk.
"""
from pathlib import Path
import argparse, hashlib, html, json, math, traceback
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
POLICY = V / 'ROOT_INITIALIZER_POLICY_BEFORE_INPUTS.json'
PROTOCOL = H / 'REPORT_PROTOCOL_BEFORE_RGB.json'
OUT = V / 'saved_visibility_visual_report_v1'
FONT = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
MODELS = ('C16', 'v61')
VARIANTS = ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def bind(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def check(binding):
    assert sha(binding['path']) == binding['sha256'], binding['path']
    return Path(binding['path'])


def same(a, b):
    return a['path'] == b['path'] and a['sha256'] == b['sha256']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def ash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pixel_to_display(x, scale, origin=0):
    return origin + (float(x) + .5) * scale - .5


def reference_in_crop(q, ref, side):
    if q is None or ref is None or side is None:
        return None
    q, ref = np.asarray(q, np.float64), np.asarray(ref, np.float64)
    if q.shape != (2,) or ref.shape != (2,) or not np.isfinite(q).all() or not np.isfinite(ref).all():
        return None
    return ((ref - q) / float(side) + .5) * 96 - .5


def ratio(a, b):
    return float(a/b) if b else None


def calculate(rows, mask, model):
    selected = [r for r, take in zip(rows, mask) if take]
    cm = np.zeros((2, 3), np.int64)
    for r in selected:
        if r['raw_state'] in ('D', 'E'):
            y = 0 if r['raw_state'] == 'D' else 1
            pred = r[model]['prediction']
            cm[y, 2 if pred == -1 else pred] += 1
    return dict(CM=cm.tolist(), planned_rows=len(selected), known_DE=int(cm.sum()),
                teacher_states={s:sum(r['raw_state'] == s for r in selected) for s in 'DESU'},
                D_precision=ratio(cm[0, 0], cm[:, 0].sum()), D_recall=ratio(cm[0, 0], cm[0].sum()),
                E_precision=ratio(cm[1, 1], cm[:, 1].sum()), E_recall=ratio(cm[1, 1], cm[1].sum()),
                prediction_U_all=sum(r[model]['prediction'] == -1 for r in selected),
                prediction_U_all_rate=ratio(sum(r[model]['prediction'] == -1 for r in selected), len(selected)),
                prediction_U_known=int(cm[:, 2].sum()), prediction_U_known_rate=ratio(cm[:, 2].sum(), cm.sum()))


def percent(x):
    return '—' if x is None else '%.1f%%' % (100*x)


def pred_name(x):
    return {-1:'U(입력 보류)', 0:'D(보임)', 1:'E(외부가림)'}[int(x)]


def number(x, digits=3):
    if x is None:
        return '없음'
    x = float(x)
    return ('%.*f' % (digits, x)) if np.isfinite(x) else '미추론'


def coordinates(x):
    return '없음' if x is None else '(%.2f, %.2f)' % (float(x[0]),float(x[1]))


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONT), size)


def text(draw, xy, value, size=22, fill='#172235'):
    draw.text(xy, str(value), font=font(size), fill=fill)


def cross(draw, xy, color, radius=7, width=3):
    x, y = map(float, xy)
    draw.line((x-radius, y, x+radius, y), fill=color, width=width)
    draw.line((x, y-radius, x, y+radius), fill=color, width=width)


def authorize(entrypath):
    e = read(entrypath)
    assert e['approved_saved_visual_report'] is True
    assert all(e[k] is False for k in ('model_forward', 'fit', 'current223', 'V112', 'PnP', 'label_edit', 'automatic_retry'))
    assert same(e['code'], bind(__file__)) and same(e['protocol'], bind(PROTOCOL))
    p = read(PROTOCOL)
    assert same(p['code'], e['code']) and same(p['policy'], bind(POLICY))
    assert not OUT.exists(), 'Preserve existing output/failure; no automatic retry.'
    # Both model seals are complete and unchanged before any GT/RGB payload.
    for key, mode in (('C16_prediction_seal', 'c16'), ('v61_prediction_seal', 'v61')):
        s = read(check(e[key]))
        assert s['completed'] and s['PASS'] and s['mode'] == mode and s['rows'] == 6144
        assert s['all768_scenes6144_slots_preserved'] and s['GT_read'] is False
        assert s['fit_current_V112_PnP_q_correction'] == 0
    evaluation_seal = read(check(e['evaluation_seal']))
    assert evaluation_seal['completed'] and evaluation_seal['PASS']
    for key in ('C16_prediction_seal','v61_prediction_seal'):
        assert same(evaluation_seal[key],e[key]),key
    for key, target in (('results','evaluation_results'),('scores','scores'),('selection','selection')):
        assert same(evaluation_seal[key],e[target]),key
    # Saved evaluation already joined the fixed GT convention. No new teacher.
    for key in ('evaluation_results', 'selection', 'scores', 'C16_input_crop_facts'):
        check(e[key])
    c16 = read(e['C16_prediction_seal']['path'])
    assert same(c16['input_crop_facts'], e['C16_input_crop_facts'])
    return e, p


def load_rows(e):
    packet = read(e['C16_input_crop_facts']['path'])
    assert packet['completed'] and packet['GT_read'] is False and len(packet['rows']) == 6144
    inputs = packet['rows']
    for i, r in enumerate(inputs):
        assert r['occurrence_index'] == i and r['scene_index'] == i//8 and r['native_slot'] == i%8
    # Evaluator schema is fixed in the preparation protocol. Input identities
    # and scores are retained; visual categories are never recalculated on RGB.
    scores = [json.loads(line) for line in Path(e['scores']['path']).read_text().splitlines() if line]
    assert len(scores) == len(inputs)
    for i, (r, original) in enumerate(zip(scores, inputs)):
        assert r['global_index'] == i and r['scene_index'] == original['scene_index']
        assert r['base_index'] == original['base_index'] and r['native_slot'] == original['native_slot']
        assert r['split'] == original['split'] and r['variant'] == original['variant']
        assert r['raw_state'] in 'DESU' and isinstance(r['convention_boundary'], bool)
        for model in MODELS:
            assert r[model]['prediction'] in (-1, 0, 1)
            assert r[model]['available'] == (r[model]['prediction'] != -1)
    # Recheck saved score identity/polarity, including noquery rawNaN/U. No model.
    for model, sealkey, rawkey, predkey in (
        ('C16','C16_prediction_seal','raw_head_logit_before_U','prediction'),
        ('v61','v61_prediction_seal','raw_visible_logit','prediction_D0_E1_Uminus1')):
        seal=read(e[sealkey]['path'])
        with np.load(check(seal['predictions']),allow_pickle=False)as f:
            assert np.array_equal(f['global_index'],np.arange(6144))
            assert np.array_equal(f['occurrence_id'],np.array([r['occurrence_id']for r in inputs]))
            z=f[rawkey];pred=f[predkey];available=f['query_available']
            for i,r in enumerate(scores):
                assert r[model]['prediction']==int(pred[i])and r[model]['available']==bool(available[i])
                stored=r[model]['raw_logit']
                assert stored is None if np.isnan(z[i]) else stored==float(z[i])
    selection = read(e['selection']['path'])
    assert selection['completed'] and selection['PASS'] and selection['max_cases'] == 8 and len(selection['cases']) <= 8
    for case in selection['cases']:
        i = case['global_index']
        assert 0 <= i < 6144 and case['base_index'] == scores[i]['base_index'] and case['native_slot'] == scores[i]['native_slot']
    return inputs, scores, selection


class SavedPixels:
    def __init__(self):
        self.chunks = {}
        self.images = {}

    def crop(self, row):
        b = row['rgb96_chunk']
        key = (b['path'], b['sha256'])
        if key not in self.chunks:
            check(b)
            with np.load(b['path'], allow_pickle=False) as f:
                self.chunks[key] = (np.array(f['rgb96'], copy=True), np.array(f['valid96'], copy=True),
                                    np.array(f['global_index'], copy=True), np.array(f['crop_constructed'], copy=True))
        rgb, valid, indices, constructed = self.chunks[key]
        j = row['rgb96_row']
        assert indices[j] == row['occurrence_index'] and bool(constructed[j]) == row['crop_constructed']
        assert ash(rgb[j]) == row['rgb96_sha256'] and ash(valid[j]) == row['valid96_sha256']
        return rgb[j].copy()

    def scene(self, row):
        from PIL import Image
        b = row['raw_RGB_PNG']
        key = (b['path'], b['sha256'])
        if key not in self.images:
            check(b)
            assert Path(b['path']).suffix == '.png'
            image = np.array(Image.open(b['path']).convert('RGB'), np.uint8)
            assert list(image.shape) == row['raw_hw'] + [3] and ash(image) == row['decoded_RGB_sha256']
            self.images[key] = image
        return self.images[key].copy()


def case_panel(case, ordinal, inputs, scores, pixels, output):
    from PIL import Image, ImageDraw
    base, slot = case['base_index'], case['native_slot']
    selected = case['global_index']
    image = Image.new('RGB', (1600, 1220), '#f7f9fc')
    d = ImageDraw.Draw(image)
    text(d, (28, 18), 'V132 | 합성 source · 실제 초기 추론 q | 사례 %02d' % ordinal, 29)
    text(d, (28, 63), '분류: %s  |  base %03d · native slot %d · 선택행 %d' % (case['category'], base, slot, selected), 23)
    text(d, (28, 104), '각 variant에서 q를 다시 추론했습니다. 서로 다른 q이며 same-q 대조가 아닙니다.', 22, '#a04400')
    text(d, (28, 143), '노랑 + : 초기 q   /   청록 + : 적용 convention의 reference   /   raw crop에는 표시 없음', 22)
    for x, title in ((28,'원본 장면 + 별도 표시'), (314,'저장 raw96 ×2'), (530,'별도 marked96 ×2'), (752,'라벨·좌표·저장 모델 출력')):
        text(d, (x, 186), title, 20)
    evidence = []
    for vi, variant in enumerate(VARIANTS):
        index = (base*4+vi)*8+slot
        inp, row = inputs[index], scores[index]
        assert inp['base_index'] == base and inp['native_slot'] == slot and inp['variant'] == variant
        y = 224+vi*238
        if index == selected:
            d.rounded_rectangle((18, y-6, 1578, y+228), radius=9, outline='#c74c26', width=3)
        # Display context only; raw96 in the next column is unchanged.
        scene = Image.fromarray(pixels.scene(inp)).resize((220,220), Image.Resampling.BILINEAR)
        scene_draw = ImageDraw.Draw(scene)
        q, ref = inp['query_xy'], row['reference_xy']
        if q is not None:
            cross(scene_draw, [pixel_to_display(q[0],220/512),pixel_to_display(q[1],220/512)], '#ffe43b', 5, 2)
        if ref is not None:
            cross(scene_draw, [pixel_to_display(ref[0],220/512),pixel_to_display(ref[1],220/512)], '#00e5e0', 5, 2)
        image.paste(scene, (28,y))
        raw = Image.fromarray(pixels.crop(inp)).resize((192,192), Image.Resampling.NEAREST)
        image.paste(raw, (314,y+19))
        marked = raw.copy()
        md = ImageDraw.Draw(marked)
        if inp['crop_constructed']:
            cross(md, (95.5,95.5), '#ffe43b', 8, 3)
            local_ref = reference_in_crop(q, ref, inp['L'])
            if local_ref is not None:
                xy = [pixel_to_display(v,2) for v in local_ref]
                if all(-10 <= v <= 201 for v in xy):
                    cross(md, xy, '#00e5e0', 8, 3)
        else:
            md.rectangle((0,0,191,191), outline='#9f2932', width=3)
        image.paste(marked,(530,y+19))
        text(d,(752,y+2),'%s  ·  %s  ·  scene %d' % (variant,row['split'],inp['scene_index']),23)
        text(d,(752,y+36),'teacher %s | error %spx | L %s | boundary %s' %
             (row['raw_state'],number(row['error_px'],2),number(inp['L'],2),str(row['convention_boundary'])),21)
        text(d,(752,y+68),'적용 convention: '+str(row['applied_convention']),19)
        for mi, model in enumerate(MODELS):
            z = row[model]
            color = '#53617a'
            if row['raw_state'] in ('D','E') and z['prediction'] != -1:
                color = '#145d43' if z['prediction'] == {'D':0,'E':1}[row['raw_state']] else '#8b3c30'
            text(d,(752,y+102+mi*34),'%s: %s | raw %s | available %s' %
                 (model,pred_name(z['prediction']),number(z['raw_logit']),str(z['available'])),21,color)
        text(d,(752,y+176),'q %s  /  reference %s' % (coordinates(q),coordinates(ref)),17)
        text(d,(752,y+202),'crop 생성 %s · C16 입력 %s' % (inp['crop_constructed'],inp['query_available']),17)
        evidence.append(dict(global_index=index, scene_index=inp['scene_index'], variant=variant,
                             raw_RGB_PNG=inp['raw_RGB_PNG'], crop_chunk=inp['rgb96_chunk'], crop_row=inp['rgb96_row'],
                             rgb96_sha256=inp['rgb96_sha256'], query_xy=q, reference_xy=ref,
                             raw_state=row['raw_state'], error_px=row['error_px'], convention_boundary=row['convention_boundary']))
    text(d,(28,1185),'S=자기가림, U=teacher 불확실. 둘 다 외부 가림 negative로 합치지 않았습니다. Native semantic 대응은 미인증입니다.',18,'#53617a')
    path=output/('CASE_%02d_BASE%03d_SLOT%d.png'%(ordinal,base,slot))
    image.save(path)
    return dict(image=bind(path), selected_global_index=selected, category=case['category'], evidence=evidence)


def metrics_figure(rows, output):
    from PIL import Image, ImageDraw
    image=Image.new('RGB',(1650,1260),'#f7f9fc')
    d=ImageDraw.Draw(image)
    text(d,(28,18),'V132 | 합성 source · 실제 초기 q · 저장 C16/v61 비교',30)
    text(d,(28,65),'원래 6,144슬롯 보존. 행=teacher D/E, 열=예측 D/E/U. U도 recall 분모에 포함.',23)
    text(d,(28,103),'Self S와 teacher U는 D/E 혼동행렬에서 제외하고 별도 집계합니다. Native convention 대응은 미인증.',21,'#994800')
    folds=np.array([r['split']for r in rows])
    error=np.array([r['error_px']if r['error_px']is not None else np.nan for r in rows],np.float64)
    boundary=np.array([r['convention_boundary']for r in rows],bool)
    scopes=[('TRAIN 전체',folds=='TRAIN'),('VALIDATION 전체',folds=='VALIDATION'),
            ('VALIDATION · nonboundary · error ≤3px',(folds=='VALIDATION')&~boundary&np.isfinite(error)&(error<=3))]
    result={}
    for ri,(title,mask)in enumerate(scopes):
        top=160+ri*350
        text(d,(28,top),title,25)
        result[title]={m:calculate(rows,mask,m)for m in MODELS}
        state=result[title]['C16']['teacher_states']
        text(d,(28,top+38),'slots %d | teacher D%d E%d S%d U%d'%(int(mask.sum()),state['D'],state['E'],state['S'],state['U']),20)
        for mi,model in enumerate(MODELS):
            x=28+mi*812
            item=result[title][model]
            text(d,(x,top+78),model,25)
            cm=item['CM']
            for ci,label in enumerate(('pred D','pred E','pred U')):
                text(d,(x+95+ci*108,top+117),label,18)
            for yi,label in enumerate(('D','E')):
                text(d,(x+13,top+157+yi*48),label,23)
                for ci,value in enumerate(cm[yi]):
                    color='#dcefe6'if ci==yi else'#f2dfdc'if value else'#e9eef5'
                    d.rounded_rectangle((x+85+ci*108,top+150+yi*48,x+185+ci*108,top+192+yi*48),radius=5,fill=color)
                    text(d,(x+101+ci*108,top+158+yi*48),value,22)
            text(d,(x+438,top+125),'D P/R  %s / %s'%(percent(item['D_precision']),percent(item['D_recall'])),21)
            text(d,(x+438,top+161),'E P/R  %s / %s'%(percent(item['E_precision']),percent(item['E_recall'])),21)
            text(d,(x+438,top+197),'U 전체  %d (%s)'%(item['prediction_U_all'],percent(item['prediction_U_all_rate'])),20)
            text(d,(x+438,top+231),'U D/E  %d (%s)'%(item['prediction_U_known'],percent(item['prediction_U_known_rate'])),20)
    text(d,(28,1210),'C16: 외부가림 logit ≥0 → E.  v61: visible logit ≥0 → D. 서로 다른 학습 목표의 출력입니다.',21,'#53617a')
    path=output/'ACTUAL_INITIALIZER_METRICS.png'
    image.save(path)
    return dict(image=bind(path),scopes=result)


def run(entrypath):
    e,p=authorize(entrypath)
    inputs,scores,selection=load_rows(e)
    OUT.mkdir()
    OUT.joinpath('figures').mkdir()
    try:
        # This fixed evaluator selection was sealed before any new RGB read.
        write(OUT/'SELECTION_COPY_BEFORE_RGB.json',dict(selection=e['selection'],cases=selection['cases'],
              missing_categories=selection.get('missing_categories',[]),source='SYNTHETIC_ONLY',labels_edited=False))
        pixels=SavedPixels()
        panels=[case_panel(c,i+1,inputs,scores,pixels,OUT/'figures')for i,c in enumerate(selection['cases'])]
        metrics=metrics_figure(scores,OUT/'figures')
        report=dict(completed=True,PASS=True,entry=bind(entrypath),code=e['code'],protocol=e['protocol'],
                    evaluation_seal=e['evaluation_seal'],selection=e['selection'],source='SYNTHETIC_ONLY',
                    metrics=metrics,cases=panels,missing_categories=selection.get('missing_categories',[]),
                    actual_model_calls=0,new_private_access=0,label_edits=0,
                    raw96_comes_from_saved_C16_input_chunks=True,marks_are_separate_display_copy=True,
                    actual_queries_can_change_across_variants=True,semantic_native_convention_certified=False,
                    fully_solved=False)
        write(OUT/'REPORT_RESULT.json',report)
        cards=''.join('<article><h2>'+html.escape(c['category'])+'</h2><img src="'+html.escape(c['image']['path'])+'"></article>'for c in panels)
        missing=html.escape(str(report['missing_categories']))
        page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V132 실제 초기 q 합성 진단</title><style>body{font-family:sans-serif;margin:28px;background:#f5f7fb;color:#172235}main{max-width:1700px;margin:auto}img{width:100%;height:auto;background:white}article{margin:30px 0}p{line-height:1.7}.note{background:#fff0d7;padding:16px;border-radius:8px}a{color:#2459a6}</style><main><h1>V132: 실제 초기 q에서 보임·가림 판단</h1><p class="note">이 보고서는 합성 source입니다. 현재 실사 평가셋 성능이나 문제 해결을 뜻하지 않습니다. 초기 q는 실제 동결 producer 출력이며, 네 variant에서 서로 달라질 수 있습니다. Native semantic convention 대응은 아직 인증되지 않았습니다.</p><p>원래 768장 × 8슬롯을 유지했습니다. raw96은 저장된 모델 입력 그대로이고, 노랑 q·청록 reference 표시는 별도 사본에만 추가했습니다. teacher S/U를 외부 가림 negative로 합치지 않았습니다.</p><img src="'''+html.escape(metrics['image']['path'])+'"><p>선택 범주 중 사례 없음: '+missing+'</p>'+cards+'<p><a href="'+html.escape(e['evaluation_results']['path'])+'">전체 saved 평가 결과</a> · <a href="'+html.escape(e['selection']['path'])+'">RGB 열기 전 고정한 사례 선택</a> · <a href="'+html.escape(str(OUT/'REPORT_RESULT.json'))+'">표시 근거와 이미지 해시</a></p></main></html>'
        path=OUT/'READOUT_DRAFT_KO.html'
        with path.open('x')as f:f.write(page)
        seal=dict(completed=True,PASS=True,entry=bind(entrypath),result=bind(OUT/'REPORT_RESULT.json'),
                  HTML=bind(path),figures=[metrics['image']]+[c['image']for c in panels],
                  needs_root_actual_image_review=True,browser_rendered=False,model_calls=0)
        write(OUT/'REPORT_SEAL.json',seal)
        print(json.dumps(dict(completed=True,seal=bind(OUT/'REPORT_SEAL.json'),case_panels=len(panels))),flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(traceback=traceback.format_exc(),automatic_retry=False))
        raise


def prepare():
    assert FONT.is_file()
    # Generated geometry only: no actual score, GT, crop, RGB or model read.
    assert np.array_equal(reference_in_crop([10.,20.],[10.,20.],16.),[47.5,47.5])
    assert np.array_equal(reference_in_crop([10.,20.],[13.,20.],16.),[65.5,47.5])
    assert pixel_to_display(47.5,2)==95.5 and reference_in_crop(None,None,None)is None
    fixture=[dict(raw_state=s,C16=dict(prediction=p))for s,p in [('D',0),('E',0),('D',-1),('S',1)]]
    m=calculate(fixture,np.ones(4,bool),'C16')
    assert m['CM']==[[1,0,1],[1,0,0]]and m['D_recall']==.5and m['E_recall']==0and m['prediction_U_known_rate']==1/3
    assert m['prediction_U_all_rate']==.25and m['teacher_states']['S']==1
    generated=dict(completed=True,PASS=True,crop_pixelcentre_and3px_transform=True,
                   noquery_no_reference_mark=True,original_denominator_U_and_self_exclusion=True,
                   actual_source_GT_scores_RGB_or_model_reads=0)
    gp=H/'GENERATED_DISPLAY_GEOMETRY.json'
    write(gp,generated)
    p=dict(completed=True,code=bind(__file__),policy=bind(POLICY),generated=bind(gp),output=str(OUT),
           maximum_cases=8,source='SYNTHETIC_ONLY',selection='Evaluator pre-RGB fixedmax8; missing categories shown, never RGB-based replacements',
           panels='Each fixedbase/slot shows all4variants at their own actual q; nativePNG context, untouched saved96crop, separate markedcopy,teacher/error/convention/scores',
           marks='qyellow/referencecyan, exact crop96 pixelcentre and nativePNG transforms; no coordinate correction',
           metrics='All6144 original slots; TRAIN/VAL and VALnonboundary error<=3. D/E CM2x3 withU retained; S/U excluded from external loss/metric',
           required_scores_schema='JSONL6144 global_index,scene_index,base_index,native_slot,split,variant,reference_xy,error_px,raw_state,convention_boundary,applied_convention,C16/v61:{prediction,raw_logit,available}',
           required_selection_schema='{completed,PASS,max_cases:8,cases:[{category,global_index,base_index,native_slot}],missing_categories:[]}',
           no_model_or_private_or_PnP_or_label_changes=True,root_owns_actual_execution=True)
    write(PROTOCOL,p)
    template=dict(approved_saved_visual_report=False,model_forward=False,fit=False,current223=False,V112=False,
                  PnP=False,label_edit=False,automatic_retry=False,code=p['code'],protocol=bind(PROTOCOL),
                  C16_prediction_seal=None,v61_prediction_seal=None,evaluation_seal=None,evaluation_results=None,
                  selection=None,scores=None,C16_input_crop_facts=None)
    tp=H/'ROOT_REPORT_ENTRY_TEMPLATE.json'
    write(tp,template)
    print(json.dumps(dict(completed=True,code=p['code'],protocol=bind(PROTOCOL),template=bind(tp),
                         command='/home/minjae/anaconda3/bin/python '+str(Path(__file__).resolve())+' --entry ROOT_REPORT_ENTRY.json',
                         actual_report_RGB_score_GT_model_reads=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--entry',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.entry:run(args.entry)
    else:raise SystemExit('Use --prepare or root --entry')
