"""Saved-output-only, GT-first review gallery for the exact decoder-probe DEV319.

Contact sheets provide review coverage; generating them does not adjudicate labels.
No label edits, model forward, metric replacement, desktop opening or notification.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import html
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROBE = ROOT / 'data/pallet/results/pallet_dht_decoder_probe_v1'
DEFAULT_RUN = ROOT / 'data/pallet/results/pallet_dht_gt_audit_v1'
CASE = 'eval_pallet07:1778652166837872128'
TITLE = 'GT Review · 수동 GT와 P90 오류 확인'
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
PRIOR_FLAGS = ['eval_pallet07:1778652176547299328','eval_pallet09:1778653634641026304',
               'eval_pallet07:1778652174598774528','eval_pallet07:1778652140531310080']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
ASSETS = Path.home()/'.claude/agents/viz-expert/assets'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class Inputs:
    def __init__(self): self.hashes={}
    def bind(self,path,expected=None):
        path=Path(path).resolve(); digest=sha(path)
        if expected is not None: assert digest==expected, f'Input changed: {path}'
        self.hashes[str(path)]=digest;return path
    def read(self,path): return json.loads(self.bind(path).read_text())
    def verify(self):
        for path,digest in self.hashes.items(): assert sha(path)==digest,f'Input changed during rendering: {path}'


def palette(inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use(inputs.bind(ASSETS/'analysis.mplstyle'))
    spec=importlib.util.spec_from_file_location('gt_review_palette',inputs.bind(ASSETS/'palette.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return dict(gt=module.color_for('gt'),halo=module.CV['text'],baseline=module.color_for('baseline'),
                line=module.color_for('after'),occluded=module.CV['occluded'],excluded=module.color_for('pred'))


def collect(probe, inputs):
    manifest=inputs.read(probe/'MANIFEST.json')
    original=[r for r in manifest['records'] if r['population']=='real_dev']
    predictions=inputs.read(probe/'PREDICTIONS.json');metrics=inputs.read(probe/'FRAME_METRICS.json')
    preds={r['id']:r for r in predictions['records'] if r['population']=='real_dev'}
    met={r['id']:r for r in metrics['records'] if r['population']=='real_dev'}
    protocol=inputs.read(probe/'TRAIN_PROTOCOL.json')
    canonical_path=inputs.bind(protocol['data']['real_manifest'],protocol['data']['real_manifest_sha256'])
    canonical={r['frame_id']:r for r in inputs.read(canonical_path)['items']}
    assert len(original)==319 and len({r['id'] for r in original})==319
    assert set(r['id'] for r in original)==set(preds)==set(met)==set(canonical)
    diag=inputs.read(probe/'DIAGNOSTIC.json')
    metadata_ids=list(dict.fromkeys(r['frame_id'] for r in diag['records']))
    assert len(metadata_ids)==14 and CASE in metadata_ids
    prior_doc=ROOT/'_docs/audits/accuracy_root_cause_v1/GT_REVIEW_RESULT.md'
    if prior_doc.is_file(): inputs.bind(prior_doc)
    frames=[]; pooled=[]
    for source in original:
        ident=source['id'];p=preds[ident];m=met[ident];c=canonical[ident]
        path=inputs.bind(source['image'],source['image_sha256'])
        assert p['image_sha256']==source['image_sha256']
        with Image.open(path) as im: assert im.size==(source['width'],source['height'])
        ann_path=inputs.bind(ROOT/c['gt_v2_path']);ann=inputs.read(ann_path);obj=ann['objects'][0]
        kp=obj['keypoint_annotations']; assert len(kp)==9
        gt=[k['xy'] for k in kp];vis=[int(k['visibility']) for k in kp]
        assert np.array_equal(np.asarray(gt),np.asarray(m['gt_points']))
        assert [v>0 for v in vis]==m['gt_supervised']
        errors=m['arms']['baseline']['errors_px'];pooled.extend(e for e in errors if e is not None)
        migration=ann.get('real_gt_v2_migration',{})
        legacy_path=migration.get('source_label')
        if legacy_path and (ROOT/legacy_path).is_file(): inputs.bind(ROOT/legacy_path,migration.get('source_sha256'))
        mask=obj.get('extrapolated_mask')
        if mask is not None: assert len(mask)==9
        frames.append(dict(id=ident,session=source['session_id'],width=source['width'],height=source['height'],
            raw_image=str(path),raw_image_sha256=sha(path),image='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),
            annotation=str(ann_path),annotation_sha256=sha(ann_path),gt=gt,visibility=vis,
            sources=[k.get('source','unknown') for k in kp],reasons=[k.get('reason','unknown') for k in kp],
            in_frame=[k.get('in_frame') for k in kp],extrapolated_mask=mask,
            migration_status=obj.get('migration_status'),manual_review_reasons=obj.get('manual_review_reasons',[]),
            legacy_source=legacy_path,legacy_source_sha256=migration.get('source_sha256'),
            prior_pose_review_flag=ident in PRIOR_FLAGS,metadata14=ident in metadata_ids,
            baseline=p['baseline']['points'],line_fusion=p['arms']['line_fusion']['points'],
            baseline_valid=p['baseline']['point_valid'],line_valid=p['arms']['line_fusion']['point_valid'],
            baseline_errors_px=errors,line_errors_px=m['arms']['line_fusion']['errors_px'],
            baseline_match_iou50=m['baseline_match_iou50'],view=m.get('view',{})))
    p90=float(np.quantile(pooled,.9))
    for f in frames:
        f['p90_boundary_points']=[i for i,e in enumerate(f['baseline_errors_px']) if e is not None and abs(e-p90)<=2]
        f['tail_points']=[i for i,e in enumerate(f['baseline_errors_px']) if e is not None and e>=p90]
        f['frame_boundary_points']=[i for i,(x,y) in enumerate(f['gt']) if min(x,y,f['width']-1-x,f['height']-1-y)<=10]
    return frames,dict(n_frames=319,n_sessions=len({f['session'] for f in frames}),
        n_gt_points=319*9,n_supervised_points=sum(v>0 for f in frames for v in f['visibility']),
        n_observed_points=len(pooled),baseline_p90_px=p90,
        visibility_counts=dict(Counter(str(v) for f in frames for v in f['visibility'])),
        point_source_counts=dict(Counter(s for f in frames for s in f['sources'])),
        metadata14_ids=metadata_ids,prior_pose_review_flag_ids=PRIOR_FLAGS,
        prior_review_scope='Historical A/B pose-overlay review, not proof that every current 2D point is wrong.' )


def annotate(image, frame, colors, scale=1.):
    out=image.copy();draw=ImageDraw.Draw(out);font=ImageFont.truetype(FONT,max(11,round(13*scale)))
    # No connecting cuboid lines by default: the point labels are the review target.
    for i,((x,y),v) in enumerate(zip(frame['gt'],frame['visibility'])):
        x,y=x*scale,y*scale;r=max(3,4*scale)
        if x<0 or y<0 or x>=out.width or y>=out.height: continue
        if v==2:
            draw.ellipse((x-r,y-r,x+r,y+r),outline=colors['halo'],width=5)
            draw.ellipse((x-r,y-r,x+r,y+r),outline=colors['gt'],width=2)
        elif v==1:
            polygon=[(x,y-r-1),(x+r+1,y),(x,y+r+1),(x-r-1,y),(x,y-r-1)]
            draw.line(polygon,fill=colors['halo'],width=6);draw.line(polygon,fill=colors['gt'],width=2)
        else:
            for a,b in [((x-r,y-r),(x+r,y+r)),((x-r,y+r),(x+r,y-r))]:
                draw.line([a,b],fill=colors['halo'],width=6);draw.line([a,b],fill=colors['gt'],width=2)
        label=str(i);tx=min(max(x+6,1),out.width-20);ty=min(max(y-17,1),out.height-18)
        draw.text((tx,ty),label,font=font,fill=colors['gt'],stroke_width=2,stroke_fill=colors['halo'])
    return out


def sheets(run_dir,frames,summary,colors):
    folder=run_dir/'contact_sheets';folder.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype(FONT,13);small=ImageFont.truetype(FONT,11);outputs=[]
    def sheet(group,name,scope):
        # Fit previews with one isotropic scale; original resolutions may differ.
        rows=(len(group)+3)//4;im=Image.new('RGB',(1320,62+rows*285),colors['halo']);d=ImageDraw.Draw(im)
        d.text((10,8),'GT ONLY | circle v2 / diamond v1 / cross v0 | IDs unchanged',font=font,fill=colors['gt'])
        d.text((10,29),scope,font=small,fill=colors['gt'])
        for j,f in enumerate(group):
            x=10+(j%4)*330;y=62+(j//4)*285
            factor=min(320/f['width'],240/f['height'])
            raw=Image.open(f['raw_image']).convert('RGB').resize((round(f['width']*factor),round(f['height']*factor)),Image.Resampling.LANCZOS)
            im.paste(annotate(raw,f,colors,factor),(x,y));d.text((x,y+243),f['id'],font=small,fill=colors['gt'])
            suffix=' PRIOR POSE REVIEW' if f['prior_pose_review_flag'] else ''
            d.text((x,y+260),'v2/v1/v0 '+('/'.join(str(f['visibility'].count(v)) for v in (2,1,0)))+suffix,font=small,fill=colors['gt'])
        path=folder/name;im.save(path);outputs.append(dict(path=str(path),sha256=sha(path),frame_ids=[f['id'] for f in group],scope=scope))
    metadata=[next(f for f in frames if f['id']==ident) for ident in summary['metadata14_ids']]
    sheet(metadata,'metadata14.png','Fixed metadata-selected 14 images; no prediction/error ranking or model overlays')
    details=folder/'metadata14';details.mkdir(exist_ok=True)
    for j,f in enumerate(metadata):
        raw=Image.open(f['raw_image']).convert('RGB');im=Image.new('RGB',(2*raw.width,raw.height+70),colors['halo']);d=ImageDraw.Draw(im)
        d.text((8,5),f['id']+' | Raw / GT ONLY | v2 circle, v1 diamond, v0 cross',font=font,fill=colors['gt'])
        im.paste(raw,(0,32));im.paste(annotate(raw,f,colors),(raw.width,32))
        d.text((8,raw.height+38),'Sources: '+', '.join(f'{i}:{s}' for i,s in enumerate(f['sources'])),font=small,fill=colors['gt'])
        path=details/f'{j+1:02d}.png';im.save(path);outputs.append(dict(path=str(path),sha256=sha(path),frame_ids=[f['id']],scope='Fixed metadata-selected GT-only raw/full-resolution overlay'))
    write(folder/'METADATA14_READY.json',dict(complete=True,records=[x for x in outputs if 'metadata14' in x['path']]))
    for start in range(0,len(frames),16):sheet(frames[start:start+16],f'all_{start//16+1:02d}.png',f'Original manifest order {start+1}-{min(start+16,319)} / 319 | availability for triage, not completed manual validation')
    sheet([f for f in frames if f['prior_pose_review_flag']],'prior_pose_review.png','Historical A/B pose-overlay flags; NOT current point-label adjudications')
    sheet([f for f in frames if f['p90_boundary_points']][:16],'p90_boundary.png','Posthoc baseline-error selection: a point lies within P90 +/- 2 px; GT ONLY display')
    sheet(sorted(frames,key=lambda f:max([e for e in f['baseline_errors_px'] if e is not None] or [-1]),reverse=True)[:16],
          'tail_16.png','Posthoc baseline max-point-error ranking; GT ONLY display; no automatic label decision')
    return outputs


def numeric_section(audit):
    if audit is None:return '<p>독립 수치 감사가 아직 보고서 입력에 없습니다.</p>'
    assert audit['complete'] and audit['audit_integrity_PASS']
    entries=[('All original supervised IDs',audit['metric_reconstruction']['summary'])]
    entries += [('Visibility v'+k,v) for k,v in audit['by_visibility'].items() if k in ('1','2')]
    entries += [('Source: '+k,v) for k,v in audit['by_annotation_source'].items()]
    entries += [('Resolution '+k,v) for k,v in audit['by_image_resolution'].items()]
    entries += [('Historical frame/GT-overlay OK frames',audit['prior_human_review']['prior_ok'])]
    rows=[]
    for label,r in entries:
        vals=[label,r['n_frames'],r['n_gt_points'],r['n_observed_points']]
        for arm in ('baseline','point_only','line_fusion'):
            a=r['arms'][arm];vals += [f"{a['median_px']:.2f}" if a['median_px'] is not None else 'missing',f"{a['p90_px']:.2f}" if a['p90_px'] is not None else 'missing']
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>')
    headers=['Subset (descriptive)','Frames','GT points','Observed','Baseline median','Baseline P90','Point median','Point P90','Line median','Line P90']
    table='<div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    boundary=audit['metric_reconstruction']['p90_boundary'];lo=boundary['lower'];hi=boundary['upper']
    calculation=(f"관측된 같은 ID 점 {audit['metric_reconstruction']['summary']['n_observed_points']:,}개의 오류를 정렬해 "
        f"{boundary['lower_rank_one_based']:,}번째 {lo['error_baseline_px']:.5f}px와 "
        f"{boundary['upper_rank_one_based']:,}번째 {hi['error_baseline_px']:.5f}px를 보간한 값이 P90입니다. "
        "프레임별 P90의 평균이나 GT 자체의 오차가 아닙니다.")
    changed=audit['manual_vs_scored_xy']
    jumps=' '.join('<button class="jump" data-id="'+html.escape(row['id'])+'" data-point="'+str(row['point_id'])+'">'+label+' · '+html.escape(row['id'])+' · point '+str(row['point_id'])+'</button>' for label,row in [('P90 lower',lo),('P90 upper',hi)])
    return '<p>'+calculation+'</p><div class="controls">'+jumps+'</div>'+table+(
        '<p class="muted">All errors in raw pixels. 표는 기존 점 집합을 설명용으로 나눈 것입니다. 직접 클릭·가시점 subset 결과를 새 공식 점수로 바꾸지 않습니다. '
        '서로 다른 원해상도가 섞여 있으므로 pixel 오류의 상대적 크기를 해석할 때 해상도도 확인합니다. '
        '이전 프레임·외삽 리뷰 ok는 frame ID 연결이며 현재 좌표의 픽셀 정밀도 인증이 아닙니다. '
        f"manual_kps와 현재 평가 xy가 다른 점은 {changed['n_nonidentical_frames']}장·{changed['n_nonidentical_points']}개입니다. "
        '두 좌표와 출처를 보존했고 현재 평가 GT는 수정하지 않았습니다.</p>'
        '<p><a href="NUMERIC_AUDIT.json">Independent numeric audit</a> · <a href="NUMERIC_AUDIT.csv">Point-level audit CSV</a></p>')


def render(run_dir,probe_dir,skip_sheets=False):
    run_dir=Path(run_dir).resolve();probe_dir=Path(probe_dir).resolve()
    assert (run_dir/'PURPOSE.md').is_file(),'Purpose-declared audit root required'
    inputs=Inputs();inputs.bind(__file__);colors=palette(inputs);frames,summary=collect(probe_dir,inputs)
    sheet_index_path=run_dir/'CONTACT_SHEETS.json'
    if skip_sheets:
        index=json.loads(sheet_index_path.read_text());outputs=index['sheets']
        for item in outputs:assert sha(item['path'])==item['sha256']
    else:
        outputs=sheets(run_dir,frames,summary,colors)
        write(sheet_index_path,dict(complete=True,n_unique_frames=319,manual_validation_complete=False,
            scope='GT-only review assets, not a verdict on annotation accuracy.',sheets=outputs))
    conclusion_path=run_dir/'AUDIT_CONCLUSION.json'
    conclusion=inputs.read(conclusion_path) if conclusion_path.exists() else None
    if conclusion is not None:assert conclusion['analysis_complete'] and conclusion['gt_fully_certified'] is False and conclusion['gt_corrections_applied']==0
    queue_path=run_dir/'GT_REVIEW_QUEUE.json'
    queue=inputs.read(queue_path) if queue_path.exists() else None
    if queue is not None:assert queue['canonical_GT_edited'] is False
    numeric_path=run_dir/'NUMERIC_AUDIT.json'
    numeric=inputs.read(numeric_path) if numeric_path.exists() else None
    if numeric is not None:
        assert abs(numeric['metric_reconstruction']['summary']['arms']['baseline']['p90_px']-summary['baseline_p90_px'])<1e-12
    public=dict(schema='pallet_dht_gt_review_v1',complete=True,summary=summary,frames=frames,
                colors=colors,edges=EDGES,required_case=CASE,manual_validation_complete=False,conclusion=conclusion,review_queue=queue)
    write(run_dir/'GT_REVIEW_DATA.json',public)
    links=''.join('<a href="'+html.escape(str(Path(x['path']).relative_to(run_dir)))+'">'+html.escape(Path(x['path']).name)+'</a> ' for x in outputs if len(Path(x['path']).relative_to(run_dir).parts)==2)
    if (run_dir/'root_review_details/INDEX.json').is_file():
        details=inputs.read(run_dir/'root_review_details/INDEX.json')
        links+='<p>추가 P90 경계·tail 상세 검토 (사후 선택, GT-only): '+''.join('<a href="root_review_details/'+Path(x['path']).name+'">'+html.escape(x['id'])+'</a> ' for x in details['records'])+'</p>'
    if (run_dir/'root_review_details/ENDPOINT_CROP.json').is_file():
        crop=inputs.read(run_dir/'root_review_details/ENDPOINT_CROP.json');inputs.bind(crop['image'],crop['sha256'])
        links+='<p><a href="root_review_details/'+Path(crop['image']).name+'">Independent endpoint ambiguity crop · plastic_night_01:037376</a></p>'
    for name,label in [('ROOT_GT_VISUAL_REVIEW.json','Root GT-only review'),('specialist_gt_review.json','Specialist GT-only review'),('specialist_gt_review.md','Specialist review notes'),('MASK_POLICY_DIAGNOSTIC.json','Separate unchanged-mask diagnostic')]:
        if (run_dir/name).is_file():
            inputs.bind(run_dir/name);links+='<a href="'+name+'">'+label+'</a> '
    headline=html.escape(conclusion['headline_ko']) if conclusion else 'GT와 저장된 예측 오차를 원본좌표에서 분리해 검토합니다.'
    text=TEMPLATE.replace('@@TITLE@@',TITLE).replace('@@DATA@@',json.dumps(public,ensure_ascii=False,allow_nan=False).replace('</','<\\/')).replace('@@SHEETS@@',links).replace('@@NUMERIC@@',numeric_section(numeric)).replace('@@HEADLINE@@',headline)
    assert '@@' not in text and '<script src=' not in text
    page=run_dir/'gt_review.html';page.write_text(text)
    inputs.verify()
    receipt=dict(complete=True,PASS=True,html=str(page),html_sha256=sha(page),title=TITLE,n_frames=319,
        input_sha256=inputs.hashes,contact_sheet_index_sha256=sha(sheet_index_path),
        gt_review_data_sha256=sha(run_dir/'GT_REVIEW_DATA.json'),manual_validation_complete=False,
        scope='Actual saved-data GT-only review rendering; human GT adjudication remains separate.',
        no_inference=True,no_label_edit=True,no_desktop_open=True)
    write(run_dir/'REPORT_RENDER.json',receipt);print(json.dumps({k:receipt[k] for k in ('complete','html','html_sha256')}))
    return receipt


def browser_qa(run_dir):
    """Check the actual HTML and controls; this is not annotation adjudication."""
    from scripts.research.pallet_dht_coupling_v2.visual_qa import Browser, DataScript
    run_dir=Path(run_dir).resolve();receipt=json.loads((run_dir/'REPORT_RENDER.json').read_text())
    page=Path(receipt['html']);assert sha(page)==receipt['html_sha256']
    for p,h in receipt['input_sha256'].items():assert sha(p)==h
    parser=DataScript();parser.feed(page.read_text());assert not parser.external
    data=json.loads(''.join(parser.chunks));assert len(data['frames'])==319
    for f in data['frames']:
        assert hashlib.sha256(base64.b64decode(f['image'].split(',',1)[1])).hexdigest()==f['raw_image_sha256']
        source=json.loads(Path(f['annotation']).read_text())['objects'][0]['keypoint_annotations']
        assert f['gt']==[k['xy'] for k in source] and f['visibility']==[k['visibility'] for k in source]
        assert f['sources']==[k.get('source','unknown') for k in source]
    directory=run_dir/'visual_qa';directory.mkdir(exist_ok=True);b=Browser(directory);shots={}
    try:
        b.call('Page.navigate',{'url':page.as_uri()});b.ready()
        assert b.js('document.title')==TITLE and b.js('REPORT_FRAME_COUNT')==319
        assert b.js('REPORT_CURRENT_FRAME')==CASE
        assert b.js("$('showgt').checked && !$('showbase').checked && !$('showline').checked")
        def capture(name,element=None):
            if element:
                clip=b.js(f"(()=>{{const r=document.getElementById('{element}').getBoundingClientRect();return {{x:r.left+scrollX,y:r.top+scrollY,width:r.width,height:r.height,scale:1}}}})()")
            else:clip=b.js('({x:0,y:scrollY,width:innerWidth,height:innerHeight,scale:1})')
            path=directory/name;b.screenshot(path,clip);shots[str(path)]=sha(path)
        capture('home.png')
        decoded=b.js("(async()=>{let n=0;for(let i=0;i<DATA.frames.length;i+=8)await Promise.all(DATA.frames.slice(i,i+8).map(async f=>{const im=new Image();im.src=f.image;await im.decode();if(im.width!==f.width||im.height!==f.height)throw Error('image shape');n++}));return n})()")
        assert decoded==319
        subsets={}
        for kind in ('metadata','p90','tail','boundary','prior','queue','all'):
            b.select('subset',kind);subsets[kind]=b.js('available.length');assert subsets[kind]>0
        b.select('point',4);capture('required_case_gt_only.png','review')
        # One wide-resolution frame checks crop coordinates and native-size mode.
        wide=next(f for f in data['frames'] if f['width']==1280)
        index=next(i for i,f in enumerate(data['frames']) if f['id']==wide['id'])
        b.select('frame',index);b.select('point',8);b.select('span',80)
        assert b.js("[$('full').width,$('full').height]")==[1280,720]
        b.js("$('native').click();true");assert b.js("Math.round($('full').getBoundingClientRect().width)")==1280
        b.js("$('native').click();$('raw').click();true");b.ready()
        assert b.js("!$('showgt').checked && !$('showbase').checked && !$('showline').checked")
        b.js("$('raw').click();$('showbase').checked=true;$('showbase').dispatchEvent(new Event('change'));true");b.ready()
        b.js("$('showline').checked=true;$('showline').dispatchEvent(new Event('change'));true");b.ready()
        assert b.js("!$('comparison').classList.contains('hidden')")
        b.js("$('showbase').checked=false;$('showline').checked=false;true")
        b.js("document.querySelector('.jump').click();true");b.ready()
        b.select('span',160);capture('p90_boundary_gt_only.png','review')
        b.select('subset','queue')
        queue_index=b.js("available.findIndex(f=>f.id==='wood_night_01:032043')")
        assert queue_index>=0;b.select('frame',queue_index);b.select('point',5);b.select('span',80)
        assert b.js("available[Number($('frame').value)].visibility[5]")==2
        capture('out_of_frame_v2_gt_only.png','review')
        b.js("document.getElementById('numeric').querySelector('details').open=true;true")
        capture('numeric_audit_table.png','numeric')
        assert not b.errors and not b.console_errors and not b.external_requests
        result=dict(complete=True,PASS=True,html_sha256=sha(page),n_exact_gt_frames=319,
            n_exact_raw_images=319,raw_browser_decoded=decoded,default_gt_only=True,default_case=CASE,
            visibility_sources_coordinates_exact=True,subsets_checked=subsets,wide_resolution_native_check=True,
            raw_and_prediction_toggles_checked=True,centroid_and_crop_checked=True,screenshots=shots,
            js_errors=b.errors,console_errors=b.console_errors,external_requests=b.external_requests,
            manual_gt_adjudication=False,scope='Saved-data equality and actual browser checks only.',
            no_inference=True,no_desktop_open=True,no_notification=True)
    finally:b.close()
    write(run_dir/'VISUAL_QA.json',result)
    print(json.dumps({k:result[k] for k in ('complete','PASS','html_sha256','n_exact_gt_frames')}))
    return result


TEMPLATE='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>@@TITLE@@</title><style>
:root{font:15px Arial,sans-serif;color:#e6edf5;background:#0d1a28;color-scheme:dark}*{box-sizing:border-box}main{max-width:1450px;margin:auto;padding:22px}h1{font-size:27px}h2{font-size:20px;font-weight:normal}p{line-height:1.7}.notice,section{background:#172c3d;border:1px solid #42576a;padding:18px;margin:18px 0;border-radius:8px}.notice{border-left:4px solid #9acde4}.muted{color:#bbcad6;font-size:13px}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}label{font-size:13px}select,button{background:#253e51;color:inherit;border:1px solid #708698;padding:9px;border-radius:4px}select{display:block;max-width:420px;margin-top:5px}a{color:#9bd8ef}.panels{display:grid;grid-template-columns:1.2fr 1fr;gap:16px;align-items:start}.panel{min-width:0}canvas{display:block;width:100%;height:auto;background:#ffffff}.native #full{width:auto;max-width:none}.native .panel{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:left;white-space:nowrap}thead{border-top:1px solid #748695;border-bottom:1px solid #748695}tbody{border-bottom:1px solid #748695}th{font-weight:normal}.scroll{overflow:auto}.links{line-height:2}.hidden{display:none}summary{cursor:pointer}pre{white-space:pre-wrap;font-size:12px}.badge{display:inline-block;padding:4px 8px;margin:3px;background:#354d60;border-radius:4px}@media(max-width:800px){.panels{grid-template-columns:1fr}main{padding:9px}}
</style></head><body><main><h1>@@TITLE@@</h1><p class="notice">@@HEADLINE@@<br>같은 실사 319장과 원래 GT를 보존한 검토 화면입니다. 기본은 GT만 표시하며 예측은 숨깁니다. P90은 예측과 GT 사이의 오류 분포 요약이지, GT 자체가 그만큼 부정확하다는 뜻이 아닙니다. 이 화면 생성은 319장의 수동 GT 정확도 검증 완료를 의미하지 않습니다.</p><div id="summary"></div><section id="review"><h2>원본 영상과 점별 출처 확인</h2><p class="muted">Circle = v2 · Diamond = v1 · Cross = v0. v&gt;0은 기존 평가 감독 대상입니다. 가시성 표기와 좌표 출처는 별개입니다. manual_kps 필드가 있다는 이유만으로 직접 클릭한 점으로 해석하지 않습니다. source unknown은 그대로 미상입니다.</p><div class="controls"><label>Subset<select id="subset"><option value="all">All 319 · original order</option><option value="metadata">Fixed metadata 14 · GT-only review</option><option value="p90">P90 ± 2 px · posthoc error selection</option><option value="tail">At least one point ≥ P90 · posthoc</option><option value="boundary">GT within 10 px of image boundary / outside</option><option value="prior">Historical pose-review flags</option><option value="queue">Current review queue · no corrections</option></select></label><label>Frame<select id="frame"></select></label><label>Point ID<select id="point"></select></label><label>Crop span<select id="span"><option value="160">160 raw px · 480 canvas px</option><option value="80">80 raw px · 480 canvas px</option><option value="320">320 raw px · 480 canvas px</option></select></label><label><input id="showgt" type="checkbox" checked>GT overlay</label><label><input id="showv0" type="checkbox" checked>v0 (excluded)</label><label><input id="edges" type="checkbox">Cuboid guide lines</label><label><input id="showbase" type="checkbox">Baseline prediction</label><label><input id="showline" type="checkbox">Line-fusion prediction</label><button id="raw">Clean raw / GT</button><button id="native">Native 1:1 pixels</button><button id="previous">Previous</button><button id="next">Next</button></div><p id="info" class="notice"></p><div class="panels"><div class="panel"><p>Original image · unchanged coordinates</p><canvas id="full"></canvas></div><div class="panel"><p id="croptitle">Selected point · raw-coordinate crop</p><canvas id="crop"></canvas></div></div><p id="coord" class="muted"></p><p id="comparison" class="notice hidden"></p><div class="scroll"><table><thead><tr><th>ID</th><th>x (px)</th><th>y (px)</th><th>v / reason</th><th>Coordinate source</th><th>Legacy extrapolated mask</th><th>In frame</th></tr></thead><tbody id="points"></tbody></table></div><p id="provenance" class="muted"></p><details><summary>현재 프레임의 이전 검토 메모와 이관 상태</summary><p id="flags"></p></details></section><section id="numeric"><h2>P90 계산과 GT 출처별 분해</h2><details><summary>원래 지표를 보존한 독립 숫자 감사 · expand</summary>@@NUMERIC@@</details></section><section><h2>전체 GT contact sheets</h2><p>전체 319장은 원래 manifest 순서의 20장 시트로 제공합니다. 축소 시트는 검토 대상을 빠르게 찾는 용도이며, 실제 판단은 원본과 확대 crop으로 확인해야 합니다. 고정14장은 영상 메타데이터로 선택했으며 모델 오류 기반 선택이 아닙니다. tail과 P90 경계 시트는 오류를 보고 고른 사후 검토 표본입니다.</p><div class="links">@@SHEETS@@</div><p><a href="CONTACT_SHEETS.json">Sheet membership / SHA</a> · <a href="GT_REVIEW_DATA.json">GT and display data</a> · <a href="REPORT_RENDER.json">Source bindings</a></p></section><section><h2>판단 범위</h2><p>v1에는 가려진 점과 자동 중심점 등이 포함될 수 있습니다. 화면에서 직접 보이는 물리적 경계와 일치하는지, 해당 번호가 의미하는 모서리인지, 좌표가 직접 클릭인지 투영·외삽인지 구분해 검토합니다. 큰 예측 오류나 이전 pose-overlay의 불합격 메모만으로 현재 GT를 잘못됐다고 판정하지 않습니다. 현재 라벨, 평가 분모, 점 번호와 P90 계산은 수정하지 않습니다.</p><div id="conclusion"></div></section></main><script id="report-data" type="application/json">@@DATA@@</script><script>
const DATA=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id),images=new Map();let available=[],token=0;const c=DATA.colors;
function opt(el,value,text){const x=document.createElement('option');x.value=value;x.textContent=text;el.append(x)}
function point(ctx,x,y,v,label,scale=1){const r=4;ctx.save();ctx.lineJoin='round';ctx.font='13px Arial';function shape(){ctx.beginPath();if(v===2)ctx.arc(x,y,r,0,Math.PI*2);else if(v===1){ctx.moveTo(x,y-r-1);ctx.lineTo(x+r+1,y);ctx.lineTo(x,y+r+1);ctx.lineTo(x-r-1,y);ctx.closePath()}else{ctx.moveTo(x-r,y-r);ctx.lineTo(x+r,y+r);ctx.moveTo(x-r,y+r);ctx.lineTo(x+r,y-r)}}shape();ctx.strokeStyle=c.halo;ctx.lineWidth=6;ctx.stroke();shape();ctx.strokeStyle=c.gt;ctx.lineWidth=2;ctx.stroke();const tx=Math.min(Math.max(x+6,2),ctx.canvas.width-22),ty=Math.min(Math.max(y-7,14),ctx.canvas.height-3);ctx.strokeStyle=c.halo;ctx.lineWidth=4;ctx.strokeText(label,tx,ty);ctx.fillStyle=c.gt;ctx.fillText(label,tx,ty);ctx.restore()}
function seg(ctx,a,b,color,width=1){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}
function overlay(ctx,f,transform){if($('showgt').checked){if($('edges').checked)DATA.edges.forEach(([a,b])=>{if((f.visibility[a]>0&&f.visibility[b]>0)||$('showv0').checked){seg(ctx,transform(f.gt[a]),transform(f.gt[b]),c.halo,4);seg(ctx,transform(f.gt[a]),transform(f.gt[b]),c.gt,1)}});f.gt.forEach((p,i)=>{if(f.visibility[i]===0&&!$('showv0').checked)return;const q=transform(p);if(q[0]>=0&&q[1]>=0&&q[0]<ctx.canvas.width&&q[1]<ctx.canvas.height)point(ctx,...q,f.visibility[i],String(i))})}
for(const [toggle,key,valid,color,prefix] of [['showbase','baseline','baseline_valid',c.baseline,'B'],['showline','line_fusion','line_valid',c.line,'L']])if($(toggle).checked)f[key].forEach((p,i)=>{if(!p||!f[valid][i])return;const q=transform(p);ctx.strokeStyle=c.halo;ctx.lineWidth=5;ctx.beginPath();ctx.arc(...q,6,0,7);ctx.stroke();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();ctx.font='12px Arial';ctx.strokeStyle=c.halo;ctx.lineWidth=3;ctx.strokeText(prefix+i,q[0]+8,q[1]+13);ctx.fillStyle=color;ctx.fillText(prefix+i,q[0]+8,q[1]+13)})}
function reset(){const kind=$('subset').value;available=DATA.frames.filter(f=>kind==='all'||kind==='metadata'&&f.metadata14||kind==='p90'&&f.p90_boundary_points.length||kind==='tail'&&f.tail_points.length||kind==='boundary'&&f.frame_boundary_points.length||kind==='prior'&&f.prior_pose_review_flag||kind==='queue'&&(DATA.review_queue?.items||[]).some(q=>q.frame_id===f.id));$('frame').replaceChildren();available.forEach((f,i)=>opt($('frame'),i,f.id));if(kind==='all')$('frame').value=String(available.findIndex(f=>f.id===DATA.required_case));render()}
async function render(){window.REPORT_READY=false;const seq=++token,f=available[Number($('frame').value)||0];if(!f)return;let im=images.get(f.id);if(!im){im=new Image();im.src=f.image;await im.decode();images.set(f.id,im)}if(seq!==token)return;const id=Number($('point').value),p=f.gt[id],span=Number($('span').value),origin=[p[0]-span/2,p[1]-span/2],scale=480/span;
const full=$('full');full.width=f.width;full.height=f.height;const x=full.getContext('2d');x.drawImage(im,0,0);overlay(x,f,p=>p);
const crop=$('crop');crop.width=480;crop.height=480;const y=crop.getContext('2d');y.fillStyle=c.halo;y.fillRect(0,0,480,480);y.save();y.translate(-origin[0]*scale,-origin[1]*scale);y.scale(scale,scale);y.imageSmoothingEnabled=false;y.drawImage(im,0,0);y.restore();overlay(y,f,q=>[(q[0]-origin[0])*scale,(q[1]-origin[1])*scale]);
$('croptitle').textContent='Point '+id+' · '+span+'×'+span+' raw px → 480×480 canvas pixels';$('info').textContent=f.id+' | '+f.session+' | '+f.width+'×'+f.height+' | v2/v1/v0: '+[2,1,0].map(v=>f.visibility.filter(x=>x===v).length).join('/')+(f.prior_pose_review_flag?' | Historical pose-review flag':'');$('coord').textContent='Crop origin ('+origin.map(v=>v.toFixed(2)).join(', ')+') raw px. White area beyond the image is blank canvas, not reconstructed image content. Point ID8 is the centroid; all IDs are unchanged.';
$('points').replaceChildren();f.gt.forEach((p,i)=>{const tr=document.createElement('tr');[i,...p.map(v=>v.toFixed(3)),f.visibility[i]+' / '+f.reasons[i],f.sources[i],f.extrapolated_mask===null?'not recorded':String(f.extrapolated_mask[i]),String(f.in_frame[i])].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});tr.onclick=()=>{$('point').value=i;render()};$('points').append(tr)});
$('provenance').textContent='GT: '+f.annotation+' | SHA '+f.annotation_sha256+' | Original source: '+(f.legacy_source||'not recorded');$('flags').textContent='Migration status: '+f.migration_status+'; reasons: '+f.manual_review_reasons.join(', ')+'. '+(f.prior_pose_review_flag?'Historical A/B pose overlays were both rejected. This is not a current per-point 2D GT adjudication.':'No listed historical A/B pose-review flag for this frame.');
const comparison=$('showbase').checked||$('showline').checked;$('comparison').classList.toggle('hidden',!comparison);const fmt=e=>e==null?'not observed':e.toFixed(2)+' px';$('comparison').textContent='Saved prediction comparison · point '+id+': baseline '+fmt(f.baseline_errors_px[id])+' / line fusion '+fmt(f.line_errors_px[id])+'. Pooled P90 '+DATA.summary.baseline_p90_px.toFixed(2)+' px; frame selection is never used to replace GT.';window.REPORT_CURRENT_FRAME=f.id;window.REPORT_READY=true;}
for(let i=0;i<9;i++)opt($('point'),i,i===8?'8 · centroid':String(i));$('point').value='4';const s=DATA.summary;$('summary').textContent='319 images / '+s.n_sessions+' sessions · '+s.n_supervised_points+' supervised GT (v>0) · '+s.n_observed_points+' observed matched points · baseline P90 '+s.baseline_p90_px.toFixed(2)+' px. Source counts: '+Object.entries(s.point_source_counts).map(([k,v])=>k+' '+v).join(' / ')+'.';
if(DATA.conclusion){const list=document.createElement('ul');DATA.conclusion.findings_ko.forEach(text=>{const li=document.createElement('li');li.textContent=text;li.style.marginBottom='10px';list.append(li)});$('conclusion').append(list);const detail=document.createElement('details'),summary=document.createElement('summary');summary.textContent='검토 한계와 다음 확인';detail.append(summary);DATA.conclusion.limitations.forEach(text=>{const p=document.createElement('p');p.textContent=text;detail.append(p)});const next=document.createElement('p');next.textContent=DATA.conclusion.review_next_ko;detail.append(next);$('conclusion').append(detail);const heading=document.createElement('h2');heading.textContent='재검토 후보 · GT 수정 0건';$('conclusion').append(heading);(DATA.review_queue?.items||[]).forEach(q=>{const div=document.createElement('p'),button=document.createElement('button');button.className='jump';button.dataset.id=q.frame_id;button.dataset.point=q.point_ids[0];button.textContent=q.frame_id+' · point '+q.point_ids.join(',');div.append(button,document.createTextNode(' '+q.reason_ko+' ['+q.status+']'));$('conclusion').append(div)});const links=document.createElement('p');links.innerHTML='<a href="AUDIT_CONCLUSION.json">Audit conclusion</a> · <a href="GT_REVIEW_QUEUE.json">Review queue (no corrected coordinates)</a> · <a href="ROOT_GT_VISUAL_REVIEW.json">Root visual review</a>';$('conclusion').append(links)}else $('conclusion').textContent='독립 숫자 감사와 수동 검토 결론은 별도 작성 중입니다. 이 화면 자체로 라벨의 정오를 자동 판정하지 않습니다.';
$('subset').onchange=reset;['frame','point','span','showgt','showv0','edges','showbase','showline'].forEach(k=>$(k).onchange=render);$('raw').onclick=()=>{$('showgt').checked=!$('showgt').checked;$('showbase').checked=false;$('showline').checked=false;render()};$('native').onclick=()=>$('review').classList.toggle('native');for(const [key,delta] of [['previous',-1],['next',1]])$(key).onclick=()=>{$('frame').value=(Number($('frame').value)+delta+available.length)%available.length;render()};document.querySelectorAll('.jump').forEach(button=>button.onclick=()=>{$('subset').value='all';reset();$('frame').value=available.findIndex(f=>f.id===button.dataset.id);$('point').value=button.dataset.point;$('showbase').checked=false;$('showline').checked=false;$('showgt').checked=true;render();$('review').scrollIntoView()});window.REPORT_FRAME_COUNT=DATA.frames.length;reset();
</script></body></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--probe-dir',type=Path,default=DEFAULT_PROBE)
    parser.add_argument('--skip-sheets',action='store_true',help='Reuse verified existing GT-only contact sheets during HTML-only refresh.')
    parser.add_argument('--qa',action='store_true',help='After rendering, check actual HTML in isolated headless Chrome; no desktop opening.')
    args=parser.parse_args();render(args.run_dir,args.probe_dir,args.skip_sheets)
    if args.qa:browser_qa(args.run_dir)
