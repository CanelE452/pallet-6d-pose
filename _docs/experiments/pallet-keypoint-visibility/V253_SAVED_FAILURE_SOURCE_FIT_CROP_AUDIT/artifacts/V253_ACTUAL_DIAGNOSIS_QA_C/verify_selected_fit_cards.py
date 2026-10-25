"""V253 고정 SOURCE FIT 66행의 원 cache→miniNPZ/clean PNG byte 검산."""
import argparse, hashlib, io, json, os, stat
from pathlib import Path
import numpy as np
from PIL import Image

EXPECTED = '804517de9e6bdc8381b439148119f63c087b789514dfe3eb8d2910c3facf4734'
checked = []


def safe(p):
    p=Path(p)
    assert p.is_absolute() and '..'not in p.parts and not any(x in str(p).casefold()for x in ('v112','cache','.pyc','current223','publicheld','source_holdout'))
    for a in reversed((p,*p.parents)):
        assert not stat.S_ISLNK(os.lstat(a).st_mode)
    assert stat.S_ISREG(os.lstat(p).st_mode)
    return p


def read(binding, retain=True):
    p=safe(binding['path']);h=hashlib.sha256();size=0;parts=[]
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b);size+=len(b)
            if retain:parts.append(b)
    assert h.hexdigest()==binding['sha256'] and size==binding['size_bytes'],str(p)
    checked.append(dict(binding))
    return b''.join(parts) if retain else None


def run(path):
    p=safe(path);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
    c=json.loads(raw);assert c['completed']is True and c['PASS']is True
    complete_binding={'path':str(p),'sha256':EXPECTED,'size_bytes':len(raw)}
    selection=json.loads(read(c['selection_manifest']))
    diag=json.loads(read(c['inputs']['diagnosis_complete']))
    assert diag['outputs']['FIT_TILE_SELECTION.json']==c['selection_manifest']
    # 별도 독립 산술 단계에서 검산한 봉인 selection을 따르며 여기서 재선택하지 않는다.
    selection_rows={r['sample_index']:r for g in selection['groups']for r in g['selected']}
    assert len(selection['groups'])==36 and sum(g['empty']for g in selection['groups'])==2
    indices=sorted(selection_rows)
    assert indices==selection['selected_sample_indices']==c['actual_numeric_pixel_indices'] and len(indices)==66
    assert all(r['role']=='FIT'for r in selection_rows.values())
    crop=json.loads(read(c['inputs']['crop_complete']))
    assert crop['completed']is True and crop['PASS']is True
    for k,name in [('raw_rgb96','raw_rgb96.npy'),('raw_valid96','raw_valid96.npy')]:
        assert [b for b in crop['outputs']if Path(b['path']).name==name]==[c['inputs'][k]]
        read(c['inputs'][k],retain=False)
    rgb=np.load(safe(c['inputs']['raw_rgb96']['path']),mmap_mode='r',allow_pickle=False)
    valid=np.load(safe(c['inputs']['raw_valid96']['path']),mmap_mode='r',allow_pickle=False)
    assert rgb.dtype==np.uint8 and rgb.shape==(7617,3,96,96)
    assert valid.dtype==bool and valid.shape==(7617,1,96,96)
    mini_bytes=read(c['selected_raw_miniNPZ'])
    with np.load(io.BytesIO(mini_bytes),allow_pickle=False)as z:
        mini={k:z[k]for k in z.files}
    assert set(mini)=={'rgb96','valid96','sample_index','row_id','family','target','original_native_address','RAW96_q_xy'}
    assert mini['sample_index'].tolist()==indices
    assert mini['rgb96'].dtype==np.uint8 and mini['rgb96'].shape==(66,3,96,96)
    assert mini['valid96'].dtype==bool and mini['valid96'].shape==(66,1,96,96)
    assert mini['RAW96_q_xy'].dtype==np.float64 and np.array_equal(mini['RAW96_q_xy'],np.full((66,2),47.5))
    cases=json.loads(read(c['case_data']))
    assert cases['numeric_source_row_access_only_indices']==indices and cases['selected_FIT_only']is True
    assert [r['sample_index']for r in cases['cases']]==indices
    man=json.loads(read(c['manifest']));root=p.parent;allowed={}
    for b in man['files']:
        rel=Path(b['relative_path']);assert not rel.is_absolute()and '..'not in rel.parts
        leaf={'path':str(root/rel),'sha256':b['sha256'],'size_bytes':b['size_bytes']}
        read(leaf,retain=False);allowed[str(root/rel)]=leaf
    evidence=[]
    for j,(i,case)in enumerate(zip(indices,cases['cases'])):
        row=selection_rows[i]
        assert case['saved_diagnostic_row']==row and case['role']=='FIT'
        assert case['row_id']==row['row_id'] and mini['row_id'][j]==row['row_id']
        for key in ('family','target'):
            assert int(mini[key][j])==row[key]
        assert int(mini['original_native_address'][j])==(-1 if row['original_native_address']is None else row['original_native_address'])
        x=np.array(rgb[i],copy=True);v=np.array(valid[i],copy=True)
        assert x.tobytes()==mini['rgb96'][j].tobytes() and v.tobytes()==mini['valid96'][j].tobytes()
        xsha=hashlib.sha256(x.tobytes()).hexdigest();vsha=hashlib.sha256(v.tobytes()).hexdigest()
        assert xsha==case['RAW96_CHW_uint8_sha256'] and vsha==case['RAW96_valid_bool_sha256']
        for key,wanted,isrgb in [('raw_RGB96',x,True),('raw_VALID96',v,False)]:
            b=case[key];assert allowed[b['path']]==b
            with Image.open(io.BytesIO(read(b)))as im:
                if isrgb:
                    got=np.array(im.convert('RGB')).transpose(2,0,1).copy()
                else:
                    a=np.array(im);assert np.isin(a,[0,255]).all();got=(a>0)[None]
            assert got.tobytes()==wanted.tobytes() and got.shape==wanted.shape
        marker=bool(row['available'])and bool(v.any())
        expected_center=[24+(47.5+.5)*4-.5,159+(47.5+.5)*4-.5]if marker else None
        assert case['RAW96_q_address']==[47.5,47.5] and case['display_q_marker_drawn']is marker and case['display_pixel_center']==expected_center
        assert case['clean96_model_pixels_changed']is False and case['new_gold_or_model_input']is False
        assert allowed[case['card']['path']]==case['card']
        with Image.open(io.BytesIO(read(case['card'])))as card:
            assert card.size==(1120,680)
            if marker:
                a=np.array(card.convert('RGB'))
                region=a[339:363,203:228]
                assert np.any(np.all(region==[0,255,255],axis=2))
        evidence.append({'sample_index':i,'role':'FIT','RAW96_CHW_sha256':xsha,'valid_bool_sha256':vsha,'q_display_center':expected_center})
    assert c['counts']=={'fixed_cells':36,'empty_cells':2,'unique_selected_FIT_rows':66,'source_numeric_rows_decoded':66,'SOURCE_DEV_numeric_pixel_rows':0}
    return {'completed':True,'PASS':True,'render_complete':complete_binding,'diagnosis_complete':c['inputs']['diagnosis_complete'],'selection_manifest':c['selection_manifest'],'verified_bindings':checked,'checks':{'selected_FIT_rows':66,'selected_DEV_numeric_pixel_rows':0,'fixed_cells':36,'empty_cells':2,'miniNPZ_original_selected_row_byte_exact':True,'clean_PNG_RGB_and_valid_byte_exact':True,'all_case_saved_scores_and_reference9_exact_selection_rows':True,'q_display_math47_5_exact':True,'card_dimensions_and_q_marker_checked':True,'whole_source_containers_opaque_SHA_and_headers_include_DEV_bytes':True,'original_unselected_numeric_pixel_rows_read':0},'selected_row_proofs':evidence,'current_or_held_payloads':0,'new_training_or_model_forward_or_new_gold':0,'limits':['원 Source cache의 선택 FIT 행만 수치 픽셀로 읽었다. 전체 원 컨테이너 opaque hash 및 header에는 DEV bytes가 포함된다.','clean PNG/miniNPZ byte와 q 표시 수학 검산이며 intended physical part/instance/native gold 또는 전체 카드의 semantic/layout 판독을 주장하지 않는다.','saved selection의 first2 산술 독립 검산은 별도 actual_review_v1이며 이 픽셀 단계는 봉인된 동일 selection을 따른다.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);assert out.is_absolute()and '..'not in out.parts and not any(t in str(out).casefold()for t in('v112','cache','.pyc'))
    for q in reversed((out,*out.parents)):
        try:s=os.lstat(q)
        except FileNotFoundError:continue
        assert not stat.S_ISLNK(s.st_mode)
    out.mkdir(exist_ok=False)
    try:
        r=run(a.complete);code=Path(__file__).read_bytes();r['checker']={'path':str(Path(__file__)),'sha256':hashlib.sha256(code).hexdigest(),'size_bytes':len(code)}
        b=(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();f=out/'REVIEW.json';f.write_bytes(b)
        print(json.dumps({'completed':True,'PASS':True,'review':{'path':str(f),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)},'checks':r['checks']},ensure_ascii=False))
    except BaseException as e:
        (out/'FAILURE.json').write_text(json.dumps({'completed':False,'PASS':False,'error_type':type(e).__name__,'error':str(e),'automatic_retry':False},ensure_ascii=False,indent=2)+'\n');raise
