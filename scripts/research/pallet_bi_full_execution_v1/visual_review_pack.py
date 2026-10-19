"""Authenticate derived galleries; assemble contacts from derived JPEGs only."""
from pathlib import Path
import argparse,gzip,html,json,hashlib
from . import stage_gallery as G

NAMES=('S1_STAGE_GALLERY_V4','S3_S4_STAGE_GALLERY_V3','STAGE_GALLERY_S3_BI_VS_S3_COMMON_V2',
 'STAGE_GALLERY_S4_CONTOUR_VS_S4_PL_V2','STAGE_GALLERY_S4_IMAGE_VS_S4_PL_V2','STAGE_GALLERY_S4_IMAGE_VS_S4_CONTOUR_V2')
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt')as f:return [json.loads(l)for l in f if l.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def public_guard(value):
    if isinstance(value,dict):
        if set(value)&{'R_physical','R_fixed','t_gt','actual_pose','centroid','reference_pose','reference'}:raise ValueError('Pose/GT field in derived case JSON')
        for v in value.values():public_guard(v)
    elif isinstance(value,list):
        for v in value:public_guard(v)
def run(campaign,output):
    r,o=map(Path,(campaign,output));o.mkdir(parents=True,exist_ok=False)
    score={x['id']:x for x in rows(r/'S3_S4_EVALUATION/SCORED_ROWS.jsonl.gz')}
    banks={'S1':rows(r/'POST_BANK_S1_VALIDATION_V2/BANK_COVERAGE_ROWS.jsonl.gz'),'S3':rows(r/'S3_S4_BANK_ANALYSIS/BANK_COVERAGE_ROWS.jsonl.gz')}
    records=[]
    for name in NAMES:
        d=r/name;m=read(d/'GALLERY_MANIFEST.json')
        if not m['complete']or m['display_source_closure']!=G.display_sources():raise ValueError('Gallery source/complete differs')
        for p,digest in m['artifact_sha256'].items():
            if sha(d/p)!=digest:raise ValueError('Derived gallery artifact differs')
        cases={(c['arm'],c['id']):c for c in m['cases']};required=set();present=set();is_s1=name.startswith('S1_')
        baseline='D1_H1_MLP'
        if '_VS_'in name:baseline=name.split('_VS_',1)[1].rsplit('_V',1)[0]
        arms={a for a,_ in cases}
        for b in banks['S1'if is_s1 else'S3']:
            if b['arm']not in arms or b['partition']not in G.PRIMARY:continue
            for seed in G.SEEDS:
                selected=b['selected'][str(seed)]
                before=selected['baseline_metric']['success']if baseline=='D1_H1_MLP'else score[b['id']]['metrics'][f'{baseline}_s{seed}']['success']
                after=selected['metric']['success']
                if before!=after:required.add((b['arm'],b['id'],seed,'RECOVERY'if after else'DAMAGE'))
        for (arm,fid),c in cases.items():
            data=read(d/c['json']);public_guard(data)
            if data['partition']not in G.PRIMARY or data.get('CAL_RGB_opened',0):raise ValueError('CAL display prohibited')
            if data['reference_pose_or_GT_published']or data['raw_RGB_copied']or data['new_cost_evaluations']or data['new_pose_calls']or data['new_fits']:raise ValueError('Derived scope differs')
            for category in c['categories']:
                if category['category']in ('RECOVERY','DAMAGE'):present.add((arm,fid,category['seed'],category['category']))
        if required!=present:raise ValueError('All3seed transition coverage differs')
        if is_s1:
            expected={'plastic_day_01:011497','wood_day_01:005174','wood_day_01:005249'}
            if not expected.issubset({fid for _,fid in cases}):raise ValueError('PF new-good fixed examples missing')
        records.append(dict(name=name,manifest_sha256=sha(d/'GALLERY_MANIFEST.json'),cases=len(cases),all_transition_arm_id_seed_events=len(required),all_transition_ids=len({v[1]for v in required}),exact_complete_transition_coverage=True,public_case_JSON_pose_GT_guard_PASS=True,all_primary_only=True,all_artifact_hashes_PASS=True))
    # These entries were actually inspected using the image-viewing tool. This
    # is limited display/privacy QA,not annotation or physical-owner truth.
    viewed=[('S1_STAGE_GALLERY_V4','case_0012.jpg'),('S1_STAGE_GALLERY_V4','case_0013.jpg'),
        ('S3_S4_STAGE_GALLERY_V3','case_0002.jpg'),('S3_S4_STAGE_GALLERY_V3','case_0102.jpg'),('S3_S4_STAGE_GALLERY_V3','case_0114.jpg'),
        ('S3_S4_STAGE_GALLERY_V3','case_0002_context.jpg'),('S3_S4_STAGE_GALLERY_V3','case_0114_families.jpg'),
        ('STAGE_GALLERY_S3_BI_VS_S3_COMMON_V2','case_0004.jpg'),('STAGE_GALLERY_S4_IMAGE_VS_S4_CONTOUR_V2','case_0001.jpg'),
        ('S1_ORIENTATION_SUPPLEMENT','ORIENTATION_NATIVE_AXES.jpg'),('S1_S3_S4_COMPARISON_FIGURES','REPRESENTATIVE_FULL.png')]
    qa=dict(scope='Actual limited displayed-pixel spot checks; not all panels and not physical ownership certification',
        reviewed=[dict(path=str(r/d/n),sha256=sha(r/d/n))for d,n in viewed],
        findings=['Object ROI readable; outside automatic outline sample count disclosed; full context available',
        'Baseline/new/private-reference/best-good colors and three-seed labels separated; some long textual aliases are clipped in image panels and full metadata remains in linked JSON',
        'Full context case0002 shows three privacy blocks; cropped object panel excludes those background locations',
        'PL panels show genuine saved refinement and metrics; missing G13 family pose is explicit NA,not invented',
        'Native orientation IDs/axes demonstrate C2-only near-square ambiguity; projected S model corners are not image observations',
        'Scatter uses T horizontal/R vertical; full outliers retained; dense near-origin overlap remains visible through optional explicit zoom'],
        all_pixels_reviewed=False,independent_physical_owner_labels=0,physical_owner_certification=False)
    parent=r/'ROOT_VISUAL_SPOT_REVIEW_S1_V4.json'
    if parent.exists():qa['parent_independent_spot_review']=dict(path=str(parent),sha256=sha(parent))
    (o/'LIMITED_PIXEL_REVIEW.json').write_text(json.dumps(qa,indent=2)+'\n')
    from PIL import Image,ImageDraw
    contact_items=[('S1_STAGE_GALLERY_V4','case_0012.jpg'),('S1_STAGE_GALLERY_V4','case_0013.jpg'),
        ('S3_S4_STAGE_GALLERY_V3','case_0102.jpg'),('S3_S4_STAGE_GALLERY_V3','case_0114.jpg'),
        ('STAGE_GALLERY_S3_BI_VS_S3_COMMON_V2','case_0004.jpg'),('STAGE_GALLERY_S4_IMAGE_VS_S4_CONTOUR_V2','case_0001.jpg')]
    canvas=Image.new('RGB',(1600,900),'white');draw=ImageDraw.Draw(canvas)
    for i,(directory,name)in enumerate(contact_items):
        image=Image.open(r/directory/name).convert('RGB');crop=image.crop((0,300,min(790,image.width),min(560,image.height)));crop.thumbnail((790,260))
        x=(i%2)*800;y=(i//2)*300;draw.text((x+5,y+4),directory+' / '+name,fill='black');canvas.paste(crop,(x,y+28))
    canvas.save(o/'REVIEW_CONTACT_MAIN.jpg',quality=88)
    figure=read(r/'S1_S3_S4_COMPARISON_FIGURES/FIGURE_MANIFEST.json')
    if not figure['complete']or figure['datasets']!=30:raise ValueError('All ten figure pairs by three seeds required')
    for p,digest in figure['artifacts'].items():
        if sha(r/'S1_S3_S4_COMPARISON_FIGURES'/p)!=digest:raise ValueError('Figure artifact differs')
    orientation=read(r/'S1_ORIENTATION_SUPPLEMENT/DISPLAY_SEAL.json')
    for p,digest in orientation['artifact_sha256'].items():
        if sha(r/'S1_ORIENTATION_SUPPLEMENT'/p)!=digest:raise ValueError('Orientation artifact differs')
    links=['<li><a href="../'+n+'/index.html">'+html.escape(n)+'</a></li>'for n in NAMES]
    links += ['<li><a href="../S1_S3_S4_COMPARISON_FIGURES/index.html">Interactive full192 error scatter:ten pairs × three head seeds</a></li>',
        '<li><a href="../S1_ORIENTATION_SUPPLEMENT/ORIENTATION_NATIVE_AXES.jpg">Saved native-axis orientation supplement</a></li>']
    (o/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Derived visual handoff</title><h1>Completed S1/S3/S4 derived visuals</h1><p>PRIMARY-only approved redacted derivatives. Limited pixel review,not physical-owner certification. Every all3seed transition is authenticated. No final F output yet.</p><ul>'+''.join(links)+'</ul><p>Contact crops show the first saved head seed;full galleries retain all seeds and metadata.</p><img width="1600" src="REVIEW_CONTACT_MAIN.jpg">')
    seal=dict(complete=True,component='display coverage/authentication/contact packaging',galleries=records,
        figure_manifest_sha256=sha(r/'S1_S3_S4_COMPARISON_FIGURES/FIGURE_MANIFEST.json'),orientation_seal_sha256=sha(r/'S1_ORIENTATION_SUPPLEMENT/DISPLAY_SEAL.json'),
        display_source_closure=G.display_sources(),packaging_source_sha256=sha(__file__),all_public_pose_GT_guards_PASS=True,
        new_scoring_or_metric_evaluations=0,new_costs=0,new_pose_calls=0,new_fits=0,new_raw_RGB_reads=0,new_reference_decodes=0,
        actual_views=qa,pixel_review='LIMITED_SPOT_REVIEW_ONLY',whole_campaign_complete=False,
        artifacts={p.name:sha(p)for p in o.iterdir()if p.is_file()})
    (o/'VISUAL_REVIEW_SEAL.json').write_text(json.dumps(seal,indent=2)+'\n');return {k:seal[k]for k in ('complete','pixel_review','galleries')}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.run,a.output)))
