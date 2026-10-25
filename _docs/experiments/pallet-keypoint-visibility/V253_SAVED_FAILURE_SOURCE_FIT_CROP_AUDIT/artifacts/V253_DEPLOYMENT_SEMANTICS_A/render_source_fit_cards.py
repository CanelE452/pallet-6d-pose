"""V253 selected SOURCEFIT clean96 cards only; no model or new selection."""
import argparse,csv,hashlib,html,json,os,stat,sys,textwrap
from pathlib import Path
sys.dont_write_bytecode=True
ROOT_PROTOCOL_SHA='e9f2a2217930ce38fde59a8c1ebd922cd97f7ad8fe2db02cfd76dafa8511b513'
ROOT_PROTOCOL_SIZE=8448
MATH_AUDIT_SHA='fe09539ee3d01fe1b77fabe865b383cd63b5ea79f480a1f789d4219888ad6762'
MATH_AUDIT_SIZE=2662
GROUPS=('D_BASEcorrect_TREATfalseE','D_TREATcorrect','E_TREATcorrect','E_TREATmiss')
INPUT_KEYS={'diagnosis_complete','selection_manifest','raw_rgb96','raw_valid96','crop_complete'}


def lexical(value):
    s=str(value)
    if any(t in s.casefold() for t in ('v112','cache','.pyc','current223','publicheld','public_held','oldval','oldhold','/hold/','source_holdout','source_val','sourceval','v177','v178','v182','private_evaluation')) or '\x00' in s or '..' in Path(s).parts:
        raise ValueError('Forbidden path before filesystem access')
    return s

def guard(value,exists=False,regular=False):
    s=lexical(value)
    if not Path(s).is_absolute():raise ValueError('Explicit absolute path required')
    p=Path(os.path.abspath(s))
    for parent in reversed((p,*p.parents)):
        try:info=os.lstat(parent)
        except FileNotFoundError:
            if exists:raise
            continue
        if stat.S_ISLNK(info.st_mode):raise ValueError('Symlink artifact/ancestor')
    if regular and not stat.S_ISREG(os.lstat(p).st_mode):raise ValueError('Regular file required')
    return p

def raw(p):
    with os.fdopen(os.open(guard(p,True,True),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def binding(p):
    p=guard(p,True,True);h=hashlib.sha256();n=0
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b);n+=len(b)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}

def verify(p,record):
    actual=binding(p)
    if any(actual[k]!=record[k] for k in ('sha256','size_bytes')):raise ValueError('SHA/size mismatch: '+str(p))
    return actual

def jread(p):return json.loads(raw(p))

def jwrite(p,value):
    p=guard(p);data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'wb')as f:f.write(data)
    return binding(p)

def content_same(actual,expected):
    if any(actual[k]!=expected[k]for k in ('sha256','size_bytes')):raise ValueError('Saved artifact binding identity mismatch')

def file_new(path):return guard(path).open('xb')

def validate_selection(selection):
    if selection['source_role']!='FIT'or len(selection['groups'])!=36:raise ValueError('Exactly36 fixed SOURCEFIT cells')
    expected={(0,k,g)for k in range(8)for g in GROUPS}|{(1,None,g)for g in GROUPS};seen=set();rows=[]
    for cell in selection['groups']:
        f=cell['family'];k=cell['original_native_address'];g=cell['failure_group'];key=(f,k,g)
        if key not in expected or key in seen:raise ValueError('Fixed SYN addresses/YCB no-native cells')
        seen.add(key);chosen=cell['selected'];n=cell['eligible_count']
        if type(n)is not int or n<0 or len(chosen)!=min(2,n)or cell['empty']is not(n==0):raise ValueError('First2/empty-cell contract')
        ids=[r['sample_index']for r in chosen]
        if ids!=sorted(set(ids)):raise ValueError('Ascending unique sample indices per cell')
        for row in chosen:
            i=row['sample_index']
            if type(i)is not int or not 0<=i<7617 or row['family']!=f or row['original_native_address']!=k or row['role']!='FIT'or row['target']not in (0,1)or type(row['available'])is not bool:raise ValueError('Original selected SourceFIT identity')
            if any(row[m+'_prediction']not in ('D','E','U')for m in ('BASE','CONTROL','TREAT')):raise ValueError('Saved prediction namespace')
            if g.startswith('D_')and row['target']!=0 or g.startswith('E_')and row['target']!=1:raise ValueError('Fixed failure group target')
            if g=='D_BASEcorrect_TREATfalseE'and(row['BASE_prediction']!='D'or row['TREAT_prediction']!='E'):raise ValueError('Fixed BASE damage group')
            if g in ('D_TREATcorrect','E_TREATcorrect')and row['TREAT_prediction']!=('D'if row['target']==0 else'E'):raise ValueError('Fixed correct group')
            if g=='E_TREATmiss'and row['TREAT_prediction']=='E':raise ValueError('Fixed E miss group')
            rows.append((dict(row),g))
    ids=[r['sample_index']for r,g in rows]
    if seen!=expected or len(ids)>72 or len(set(ids))!=len(ids)or sorted(selection['selected_sample_indices'])!=sorted(ids):raise ValueError('Closed unique max72 selection')
    return sorted(rows,key=lambda p:p[0]['sample_index'])


def render(entry_path):
    e=jread(entry_path)
    if e.get('approved_SOURCE_FIT_selected_crop_render')is not True or e.get('automatic_retry')is not False:raise ValueError('Root later true entry required before pixels')
    if verify(__file__,e['code'])!=e['code']:raise ValueError('Self code binding')
    verify(e['root_protocol']['path'],e['root_protocol'])
    if e['root_protocol']['sha256']!=ROOT_PROTOCOL_SHA or e['root_protocol']['size_bytes']!=ROOT_PROTOCOL_SIZE:raise ValueError('Frozen V253 protocol')
    protocol=jread(e['root_protocol']['path'])
    if set(e['inputs'])!=INPUT_KEYS:raise ValueError('Closed renderer inputs')
    for key in ('raw_rgb96','raw_valid96','crop_complete'):
        if e['inputs'][key]!=protocol['inputs'][key]:raise ValueError('Exact source crop binding')
    paths={k:guard(v['path'],True,True)for k,v in e['inputs'].items()}
    proofs={k:verify(paths[k],v)for k,v in e['inputs'].items()}
    diagnosis=jread(paths['diagnosis_complete']);crop=jread(paths['crop_complete']);selection_raw=raw(paths['selection_manifest']);selection=json.loads(selection_raw)
    if diagnosis.get('completed')is not True or diagnosis.get('PASS')is not True or crop.get('completed')is not True or crop.get('PASS')is not True:raise ValueError('Completed saved selection/crop provenance')
    content_same(proofs['selection_manifest'],diagnosis['outputs']['FIT_TILE_SELECTION.json'])
    for key,filename in [('raw_rgb96','raw_rgb96.npy'),('raw_valid96','raw_valid96.npy')]:
        choices=[b for b in crop['outputs']if Path(b['path']).name==filename]
        if len(choices)!=1:raise ValueError('Exact original crop output')
        content_same(proofs[key],choices[0])
    math_path=guard(__file__,True,True).parent/'CROP_MATH_AUDIT.json';math_binding=verify(math_path,{'sha256':MATH_AUDIT_SHA,'size_bytes':MATH_AUDIT_SIZE})
    fixed=validate_selection(selection);out=guard(e['output_directory'])
    if out.exists():raise ValueError('Fresh output only/no retry')
    out.mkdir(parents=True);made=[]
    def keep(p):
        b=binding(p);made.append({'relative_path':str(p.relative_to(out)),'sha256':b['sha256'],'size_bytes':b['size_bytes']});return b
    def jsonout(name,value):p=out/name;jwrite(p,value);return keep(p)
    def png(name,im):
        p=out/name
        with file_new(p)as f:im.save(f,format='PNG')
        return keep(p)
    try:
        import numpy as np
        from PIL import Image,ImageDraw,ImageFont
        with file_new(out/'FIT_TILE_SELECTION.json')as f:f.write(selection_raw)
        keep(out/'FIT_TILE_SELECTION.json')
        rgb=np.load(paths['raw_rgb96'],mmap_mode='r',allow_pickle=False);valid=np.load(paths['raw_valid96'],mmap_mode='r',allow_pickle=False)
        if rgb.dtype!=np.uint8 or rgb.shape!=(7617,3,96,96)or valid.dtype!=bool or valid.shape!=(7617,1,96,96):raise ValueError('Original CHW RGB/valid npy schema')
        for name in ('raw96','cards'):(out/name).mkdir()
        font=ImageFont.load_default(size=18);small=ImageFont.load_default(size=16);titlefont=ImageFont.load_default(size=23)
        records=[];rgbmini=[];validmini=[];thumbs=[]
        for row,group_name in fixed:
            i=row['sample_index'];im=np.array(rgb[i],copy=True);v=np.array(valid[i],copy=True)
            generic=np.asarray(row['generic_ref9'],np.float64);target=np.asarray(row['target_ref9'],np.float64)
            if generic.shape!=(9,)or target.shape!=(9,)or not np.isfinite(generic).all()or not np.isfinite(target).all():raise ValueError('Saved selected-row finite ref9')
            scores=[float(row[k+'_E_logit'])for k in ('BASE','CONTROL','TREAT')]
            if not np.isfinite(scores).all():raise ValueError('Saved finite logits')
            chwsha=hashlib.sha256(im.tobytes()).hexdigest();validsha=hashlib.sha256(v.tobytes()).hexdigest();base=Image.fromarray(im.transpose(1,2,0));stem=f'S{i:06d}'
            rawpng=png('raw96/'+stem+'_RGB96.png',base);validpng=png('raw96/'+stem+'_VALID96.png',Image.fromarray(v[0].astype(np.uint8)*255))
            with Image.open(rawpng['path'])as check:restored=np.array(check.convert('RGB')).transpose(2,0,1).copy()
            with Image.open(validpng['path'])as check:restoredv=(np.array(check)>0)[None]
            if restored.dtype!=np.uint8 or restored.shape!=(3,96,96)or restored.tobytes()!=im.tobytes()or restoredv.tobytes()!=v.tobytes():raise ValueError('Lossless PNG RGB CHW/valid replay')
            card=Image.new('RGB',(1120,680),'#f8fafc');draw=ImageDraw.Draw(card)
            family='SYN intended-native-address witness0 MAIN'if row['family']==0 else'YCB associated-r AUX; native/SELF UNKNOWN'
            draw.text((24,18),stem+' | SOURCE FIT | inherited '+('D'if row['target']==0 else'E'),font=titlefont,fill='#111827')
            draw.text((24,55),family+' | address='+str(row['original_native_address']),font=font,fill='#334155')
            for j,line in enumerate(textwrap.wrap(str(row['row_id']),width=105)[:2]):draw.text((24,86+j*21),line,font=small,fill='#475569')
            draw.text((24,132),'Original96, nearest x4; cyan q=(47.5,47.5)',font=small,fill='#334155');card.paste(base.resize((384,384),Image.Resampling.NEAREST),(24,159))
            marker=bool(row['available'])and bool(v.any());center=(24+48*4-.5,159+48*4-.5)
            if marker:
                cx,cy=center;draw.line((cx-11,cy,cx+11,cy),fill='#00ffff',width=2);draw.line((cx,cy-11,cx,cy+11),fill='#00ffff',width=2)
            draw.text((440,147),group_name,font=font,fill='#111827')
            for j,name in enumerate(('BASE','CONTROL','TREAT')):draw.text((440,183+29*j),f'{name}: {row[name+"_prediction"]}   raw E logit {scores[j]:+.6f}',font=font,fill='#111827')
            draw.text((440,281),'Reference: selected / next / margin / support',font=font,fill='#334155')
            for label,arr,y in [('TARGET',target,310),('GENERIC',generic,412)]:
                draw.text((440,y),label+' (all 3 fixed views)',font=small,fill='#1d4ed8')
                for j,name in enumerate(('fullCLS','fixedQ','centreCLS')):draw.text((440,y+23+23*j),f'{name:9s} {arr[j]:+.4f} / {arr[3+j]:+.4f} / {arr[j]-arr[3+j]:+.4f} / {arr[6+j]:.0f}',font=small,fill='#111827')
            draw.text((24,552),f'Input available={row["available"]}; valid={int(v.sum())}/9216; central4={bool(v[0,47:49,47:49].all())}',font=small,fill='#334155')
            card.paste(Image.fromarray(v[0].astype(np.uint8)*255).convert('RGB').resize((96,96),Image.Resampling.NEAREST),(440,542));draw.text((550,555),'Input validity only; not material/occlusion truth.',font=small,fill='#475569')
            draw.text((24,596),'q cross is display geometry, not a literal-q sampled pixel or teacher reference.',font=small,fill='#475569')
            draw.text((24,624),'No GT/r/pose/recrop/new labels. Intended part/instance can remain UNKNOWN.',font=small,fill='#475569')
            cardpng=png('cards/'+stem+'.png',card);thumbs.append(card.resize((448,272),Image.Resampling.NEAREST))
            records.append({'sample_index':i,'row_id':row['row_id'],'family':row['family'],'role':'FIT','original_native_address':row['original_native_address'],'failure_group':group_name,'saved_diagnostic_row':row,'raw_RGB96':rawpng,'raw_VALID96':validpng,'card':cardpng,'RAW96_CHW_uint8_sha256':chwsha,'RAW96_valid_bool_sha256':validsha,'RAW96_q_address':[47.5,47.5],'display_q_marker_drawn':marker,'display_pixel_center':list(center)if marker else None,'clean96_model_pixels_changed':False,'new_gold_or_model_input':False})
            rgbmini.append(im);validmini.append(v)
        n=len(records);mini=out/'SELECTED_FIT_RAW96.npz'
        with file_new(mini)as dest:np.savez(dest,rgb96=np.stack(rgbmini)if n else np.empty((0,3,96,96),np.uint8),valid96=np.stack(validmini)if n else np.empty((0,1,96,96),bool),sample_index=np.asarray([r['sample_index']for r in records],np.int64),row_id=np.asarray([r['row_id']for r in records],dtype=str),family=np.asarray([r['family']for r in records],np.int8),target=np.asarray([r['saved_diagnostic_row']['target']for r in records],np.int8),original_native_address=np.asarray([-1 if r['original_native_address']is None else r['original_native_address']for r in records],np.int8),RAW96_q_xy=np.full((n,2),47.5,np.float64))
        mini_binding=keep(mini);jsonout('CASE_DATA.json',{'completed':True,'PASS':True,'cases':records,'selected_FIT_only':True,'raw_source_containers_include_DEV_bytes':True,'opaque_whole_source_file_hashes_checked':True,'numeric_source_row_access_only_indices':[r['sample_index']for r in records],'miniNPZ':mini_binding,'q_math_audit':math_binding})
        overview=Image.new('RGB',(1344,max(1,(n+2)//3)*272),'#e2e8f0')
        for j,im in enumerate(thumbs):overview.paste(im,((j%3)*448,(j//3)*272))
        if not n:ImageDraw.Draw(overview).text((24,24),'All fixed cells EMPTY; no substitute examples.',font=titlefont,fill='#111827')
        overview_binding=png('OVERVIEW.png',overview)
        cells=[]
        for cell in selection['groups']:
            links=' '.join('<a href="cards/S%06d.png">S%06d</a>'%(r['sample_index'],r['sample_index'])for r in cell['selected'])or'EMPTY'
            cells.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%s</td></tr>'%(cell['family'],cell['original_native_address'],html.escape(cell['failure_group']),cell['eligible_count'],links))
        sections=''.join('<article><h2>S%06d — %s</h2><img src="cards/S%06d.png" alt="SOURCE FIT card"><p><a href="raw96/S%06d_RGB96.png">clean96</a> · <a href="raw96/S%06d_VALID96.png">validity</a></p></article>'%(r['sample_index'],html.escape(r['failure_group']),r['sample_index'],r['sample_index'],r['sample_index'])for r in records)
        page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V253 SOURCE FIT 고정 crop 진단</title><style>body{font:16px sans-serif;max-width:1160px;margin:30px auto;color:#172033}img{max-width:100%;height:auto}td,th{border:1px solid #cbd5e1;padding:8px}table{border-collapse:collapse;width:100%}article{margin:32px 0}p{line-height:1.65}</style><h1>V253 SOURCE FIT 고정 crop 진단</h1><p>봉인된 36 cell에서 sample_index가 작은 2개까지 선택했습니다. 빈 cell은 대체하지 않았습니다. SYN은 원 입력 주소의 합성 witness0, YCB는 associated-r AUX이며 물리적 native gold/교사 instance 정답을 새로 만들지 않습니다. 실제 part/instance는 이 crop만으로 UNKNOWN일 수 있습니다. 이는 사후 SOURCE 진단이며 현재 성능이나 원인 일반화의 증거가 아닙니다.</p><p>원 파일에는 SOURCE DEV bytes도 있으므로 전체 파일 opaque SHA/header 검산 사실을 공개합니다. 숫자 pixel 접근은 아래 선택된 FIT index뿐입니다. clean96은 정확한 원 CHW cache row의 무손실 PNG이며, cyan q 표식은 별도 표시 카드에만 그렸습니다. 원 q/L이나 모델 입력을 보정하지 않았습니다. validity는 입력 지지이며 물체/가림 정답이 아닙니다.</p><p><a href="SELECTED_FIT_RAW96.npz">선택 원 RGB/valid miniNPZ</a> · <a href="FIT_TILE_SELECTION.json">봉인 선택표</a> · <a href="CASE_DATA.json">pixel hash/원 scalar 진단</a> · <a href="OVERVIEW.png">overview</a></p><table><tr><th>family</th><th>original address</th><th>fixed cell</th><th>eligible</th><th>chosen</th></tr>'+''.join(cells)+'</table>'+sections+'</html>\n'
        with file_new(out/'PORTABLE_GALLERY_KO.html')as f:f.write(page.encode())
        keep(out/'PORTABLE_GALLERY_KO.html')
        manifest=jsonout('MANIFEST.json',{'completed':True,'PASS':True,'files':made.copy(),'input_bindings':proofs,'q_math_audit':math_binding,'source_pixels_modified':False,'new_model_labels_or_reselection':False})
        result={'completed':True,'PASS':True,'entry':binding(entry_path),'code':binding(__file__),'root_protocol':e['root_protocol'],'inputs':proofs,'selection_manifest':proofs['selection_manifest'],'manifest':manifest,'overview':overview_binding,'portable_gallery':binding(out/'PORTABLE_GALLERY_KO.html'),'case_data':binding(out/'CASE_DATA.json'),'selected_raw_miniNPZ':mini_binding,'counts':{'fixed_cells':36,'empty_cells':sum(c['empty']for c in selection['groups']),'unique_selected_FIT_rows':n,'source_numeric_rows_decoded':n,'SOURCE_DEV_numeric_pixel_rows':0},'container_DEV_bytes_opaque_hash_traversed':True,'model_CP_forward_or_new_GT':0,'raw_asset_or_RGB_archive_reread':False,'original96_pixels_unchanged':True,'new_semantic_annotation':False,'q_math_audit':math_binding,'actual_numeric_pixel_indices':[r['sample_index']for r in records]}
        jwrite(out/'COMPLETE.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        jwrite(out/'FAILURE.json',{'completed':False,'PASS':False,'error_type':type(exc).__name__,'error':str(exc),'automatic_retry':False});raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--entry',required=True);a=p.parse_args();render(a.entry)
