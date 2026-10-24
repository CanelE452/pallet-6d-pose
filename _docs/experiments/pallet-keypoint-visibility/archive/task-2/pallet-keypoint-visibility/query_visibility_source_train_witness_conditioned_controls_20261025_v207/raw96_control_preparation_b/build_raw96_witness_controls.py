"""ROOT-ENTRY-only post-sampler RAW96 controls; not physical/native E labels."""
from pathlib import Path
import argparse,csv,hashlib,html,io,json,os,stat,sys,traceback
import numpy as np
import raw96_control_geometry as G

H=Path(__file__).parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_RAW96_WEAK_WITNESS_CONTROL_PILOT_SCOPE.json'
SCOPE_SHA='a61ee52c41e4ad8e8e21d514ec8801cdaf6d32863438cc8c084cc4cbf37cf342'
OUT=V/'raw96_weak_witness_control_pilot_v1'

def read_safe(path,extensions):
    p=Path(path)
    assert p.is_absolute() and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert P in p.parents and p.suffix in extensions
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:
            nxt=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        final=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(final).st_mode)
    with os.fdopen(final,'rb')as f:return f.read()

def binding(path):
    b=read_safe(path,{'.py','.json','.jsonl','.csv','.md'})
    return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest())

def bound_json(binding_):
    b=read_safe(binding_['path'],{'.json'})
    assert hashlib.sha256(b).hexdigest()==binding_['sha256']
    return json.loads(b)

class PayloadReader:
    def __init__(self,whitelist,authorized):
        assert authorized is True
        self.allowed={b['path']:b for b in whitelist};assert len(self.allowed)==26
        self.opened=[]
    def read(self,b):
        assert self.allowed.get(b['path'])==b,'Exact encoded binding whitelist'
        content=read_safe(b['path'],{'.png','.npz'})
        assert hashlib.sha256(content).hexdigest()==b['sha256']
        if 'size_bytes'in b:assert len(content)==b['size_bytes']
        self.opened.append(dict(path=b['path'],sha256=b['sha256']))
        return content

def decode_png(data,kind):
    from PIL import Image
    with Image.open(io.BytesIO(data))as image:
        assert image.format=='PNG'
        array=np.array(image)
    if kind=='CLEAN96':assert array.shape==(96,96,3) and array.dtype==np.uint8
    else:
        if array.shape==(96,96,3):
            assert np.array_equal(array[:,:,0],array[:,:,1]) and np.array_equal(array[:,:,0],array[:,:,2]);array=array[:,:,0]
        assert array.shape==(96,96) and array.dtype==np.uint8 and set(np.unique(array))<={0,255}
        array=array>0
    return array

def decode_cutout(data,donor):
    with np.load(io.BytesIO(data),allow_pickle=False)as packet:
        assert set(packet.files)=={'rgb','alpha'}
        rgb=packet['rgb'];alpha=packet['alpha']
    assert rgb.dtype==np.uint8 and alpha.dtype==np.bool_ and rgb.shape[:2]==alpha.shape and rgb.shape[2]==3
    for name,array in [('rgb',rgb),('alpha',alpha)]:
        proof=donor['recorded_array_proof'][name]
        assert list(array.shape)==proof['shape'] and str(array.dtype)==proof['dtype']
        assert hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()==proof['array_sha256']
    return rgb,alpha

def put(path,obj):
    assert OUT in path.parents and 'v112'not in str(path).casefold()
    with path.open('x',encoding='utf-8')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def output_binding(path):
    assert OUT in path.parents and 'v112'not in str(path).casefold()
    b=read_safe(path,{'.png','.json','.jsonl','.csv','.md','.html'})
    return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def save_png(path,array):
    from PIL import Image
    assert not path.exists();Image.fromarray(array).save(path)
    return dict(**output_binding(path),decoded_sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())

def contact(path,before,cover,sham,case,label):
    from PIL import Image,ImageDraw
    panels=[before,cover,sham];canvas=Image.new('RGB',(900,354),'white');draw=ImageDraw.Draw(canvas)
    for i,(array,title)in enumerate(zip(panels,('BEFORE fixed RAW96','COVER crop augmentation','SHAM same donor transform'))):
        if array is not None:
            im=Image.fromarray(array).resize((288,288),Image.Resampling.NEAREST);canvas.paste(im,(i*300,42))
            for key,col in [('ROI_A','red'),('ROI_C','orange')]:
                if case[key]is not None:
                    x0,y0,x1,y1=case[key];draw.rectangle((i*300+x0*3,42+y0*3,i*300+(x1+1)*3-1,42+(y1+1)*3-1),outline=col,width=2)
            draw.line((i*300+142.5,177,i*300+142.5,192),fill='cyan',width=2)
            draw.line((i*300+135,184.5,i*300+150,184.5),fill='cyan',width=2)
        draw.text((i*300+4,26),title,fill='black')
    draw.text((4,4),case['row_id'],fill='black');draw.text((4,334),label,fill='black')
    assert not path.exists();canvas.save(path)
    return output_binding(path)

def main(entry_path):
    entry=json.loads(read_safe(entry_path,{'.json'}))
    scope=binding(SCOPE);assert scope['sha256']==SCOPE_SHA
    expected=dict(scope=scope,code=binding(Path(__file__)),geometry=binding(H/'raw96_control_geometry.py'),
        protocol=binding(H/'PROTOCOL_BEFORE_RAW96_OR_DONOR_PAYLOADS.json'),selection=binding(H/'FIXED8_X_TRAIN10_DONOR80_SELECTION.json'),
        payload_whitelist=binding(H/'EXACT26_PAYLOAD_WHITELIST.json'),output_dir=str(OUT))
    G.validate_gate(entry,expected)
    # Metadata checks completed before this reader can touch any payload.
    selection=bound_json(expected['selection']);whitelist=bound_json(expected['payload_whitelist'])
    assert selection['attempt_count']==80 and len(selection['cases'])==8 and len(selection['donors'])==10
    assert len(whitelist['files'])==26 and whitelist['source_binary_access_before_root_entry']==0
    reader=PayloadReader(whitelist['files'],True)
    assert OUT.parent==V and 'v112'not in str(OUT).casefold()
    OUT.mkdir(exist_ok=False)
    try:
        (OUT/'cases').mkdir();(OUT/'attempts').mkdir();(OUT/'fixed8').mkdir()
        donors={d['donor_id']:d for d in selection['donors']};transformed={}
        for donor_id,donor in donors.items():
            rgb,alpha=decode_cutout(reader.read(donor['compact_RGBA']),donor)
            for longside in G.LONGSIDES:
                for rotation in G.ROTATIONS:transformed[(donor_id,longside,rotation)]=G.transform(rgb,alpha,longside,rotation)
        attempts=[];previews=[];pngs=[]
        for case in selection['cases']:
            position=case['position'];case_dir=OUT/'cases'/f'CASE_{position:02d}';case_dir.mkdir()
            clean_bytes=reader.read(case['CLEAN96']);valid_bytes=reader.read(case['VALID96'])
            before=decode_png(clean_bytes,'CLEAN96');valid=decode_png(valid_bytes,'VALID96')
            assert np.all(before[~valid]==0),'Original invalid RGB must remain zero'
            for name,data in [('BEFORE_CLEAN96.png',clean_bytes),('ORIGINAL_VALID96.png',valid_bytes)]:
                (case_dir/name).write_bytes(data);pngs.append(output_binding(case_dir/name))
            feature,status=G.obligation(case);case_attempts=[];render_arrays={}
            for assignment in [a for a in selection['assignments']if a['case_position']==position]:
                donor_id=assignment['donor_id'];record=dict(assignment,original_meta={k:case[k]for k in ('row_id','frame_key','native_slot','original_query_xy','original_L','original_query_available')},
                    ROI_A=case['ROI_A'],ROI_C=case['ROI_C'],authority='POST_SAMPLER_RAW96_2D_WEAK_FEATURE_CONTROL',training_admission=False,new_native_E_label=False,
                    status=status,transforms_tested=[],outputs=None)
                chosen=None
                if feature is not None:
                    for longside,rotation in G.ordered_transforms(case['row_id'],donor_id):
                        texture,alpha=transformed[(donor_id,longside,rotation)]
                        covers,shams=G.feasible_positions(alpha,valid,feature)
                        record['transforms_tested'].append(dict(longside=longside,rotation=rotation,alpha_pixels=int(alpha.sum()),COVER_positions=len(covers),SHAM_positions=len(shams)))
                        if covers and shams:
                            chosen=(longside,rotation,G.choose_position(covers,case['row_id'],donor_id,longside,rotation,'COVER'),G.choose_position(shams,case['row_id'],donor_id,longside,rotation,'SHAM'));break
                    record['status']='GEOMETRY_EXHAUSTED'if chosen is None else'GEOMETRY_FEASIBLE'
                if chosen is not None:
                    longside,rotation,cp,sp=chosen;texture,alpha=transformed[(donor_id,longside,rotation)]
                    cover,ca=G.paste(before,texture,alpha,cp);sham,sa=G.paste(before,texture,alpha,sp)
                    proof=G.verify_actual(before,cover,sham,ca,sa,valid,case)
                    record.update(transform=dict(longside=longside,rotation=rotation),COVER_origin=list(cp),SHAM_origin=list(sp),proof=proof,
                        status='CONSTRUCTED_RAW96_CONTROL_NO_LABEL_ADMISSION'if proof['actual_change_PASS']else'ACTUAL_CHANGE_GUARD_FAILED_NO_RETRY')
                    if proof['actual_change_PASS']:
                        ad=OUT/'attempts'/assignment['attempt_id'];ad.mkdir()
                        record['outputs']={name:save_png(ad/(name+'.png'),array)for name,array in [('COVER_CLEAN96',cover),('SHAM_CLEAN96',sham),('COVER_ALPHA',ca.astype(np.uint8)*255),('SHAM_ALPHA',sa.astype(np.uint8)*255)]}
                        pngs.extend(record['outputs'].values());render_arrays[assignment['attempt_id']]=(cover,sham)
                attempts.append(record);case_attempts.append(record)
            feasible=[a for a in case_attempts if a['status']=='CONSTRUCTED_RAW96_CONTROL_NO_LABEL_ADMISSION']
            selected=min(feasible,key=lambda a:a['assignment_hash'])if feasible else None
            cover,sham=render_arrays[selected['attempt_id']]if selected else(None,None)
            label=('No native E/fit admission. '+selected['donor_id'])if selected else'No constructed pair; factual weak evidence retained.'
            card=contact(OUT/'fixed8'/f'CASE_{position:02d}_CONTACT.png',before,cover,sham,case,label);pngs.append(card)
            previews.append(dict(position=position,row_id=case['row_id'],chosen_attempt_id=selected['attempt_id']if selected else None,card=card,
                BEFORE=output_binding(case_dir/'BEFORE_CLEAN96.png'),VALID=output_binding(case_dir/'ORIGINAL_VALID96.png'),
                pair_outputs=selected['outputs']if selected else None,all10_attempts_retained=True))
        assert len(attempts)==80 and len(previews)==8 and len(reader.opened)==26
        with (OUT/'ALL80_ATTEMPTS.jsonl').open('x')as f:
            for row in attempts:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        with (OUT/'ALL80_STATUS.csv').open('x',newline='')as f:
            cols=['attempt_id','case_position','row_id','donor_id','object_id','status','assignment_hash'];writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
            for row in attempts:writer.writerow({k:row[k]for k in cols})
        from collections import Counter
        put(OUT/'FIXED8_PREVIEW_MANIFEST.json',dict(completed=True,PASS=True,cards=previews,pngs=pngs,actual_tool_views=0))
        lines=['# 고정 SOURCE TRAIN 8개 RAW96 control pilot','',
            '원 q/L/native와 RAW96 샘플은 유지했다. 아래 개입은 샘플링 후 2D crop augmentation이며 물리 렌더링·native E gold·학습 admission이 아니다. 모든 80개 시도와 불가 상태를 보존했다.',
            '', '[전체 80개 상태 CSV](ALL80_STATUS.csv) · [상세 JSONL](ALL80_ATTEMPTS.jsonl)','']
        for p in previews:
            rel=Path(p['card']['path']).relative_to(OUT).as_posix();lines.extend([f"## {p['position']:02d} {p['row_id']}",'',f'![고정 contact]({rel})',''])
            if p['pair_outputs']:
                for k in ('COVER_CLEAN96','SHAM_CLEAN96'):
                    rel2=Path(p['pair_outputs'][k]['path']).relative_to(OUT).as_posix();lines.append(f'[{k}]({rel2})')
            lines.append('')
        (OUT/'PORTABLE_GALLERY_KO.md').write_text('\n'.join(lines),encoding='utf-8')
        table=''.join('<tr><td>'+html.escape(str(r['attempt_id']))+'</td><td>'+html.escape(r['row_id'])+'</td><td>'+html.escape(r['donor_id'])+'</td><td>'+html.escape(r['status'])+'</td></tr>'for r in attempts)
        cards=''.join('<h2>'+html.escape(p['row_id'])+'</h2><img width="900" src="'+Path(p['card']['path']).relative_to(OUT).as_posix()+'">'for p in previews)
        (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V207 RAW96 control pilot</title><h1>8 cases / 80 retained attempts</h1><p>Post-sampler 2D weak-feature controls; no native E or fit admission.</p><a href="PORTABLE_GALLERY_KO.md">Portable gallery</a>'+cards+'<table>'+table+'</table>',encoding='utf-8')
        outputs={name:output_binding(OUT/name)for name in ('ALL80_ATTEMPTS.jsonl','ALL80_STATUS.csv','FIXED8_PREVIEW_MANIFEST.json','PORTABLE_GALLERY_KO.md','index.html')}
        put(OUT/'COMPLETE.json',dict(completed=True,PASS=True,authority='POST_SAMPLER_RAW96_2D_WEAK_FEATURE_CONTROL_ONLY',
            scope=scope,entry=binding(Path(entry_path)),code=expected['code'],geometry=expected['geometry'],protocol=expected['protocol'],selection=expected['selection'],
            exact_payload_whitelist=expected['payload_whitelist'],payloads_opened=reader.opened,attempts=80,fixed_cases=8,
            statuses=dict(Counter(r['status']for r in attempts)),outputs=outputs,original_qL_native_U_unchanged=True,
            no_geometry_retries_based_on_RGB_changes=True,fit_admission=False,new_native_E_label=False,physical_rendering_claim=False,actual_tool_views=0))
        print(json.dumps(dict(completed=True,PASS=True,complete=output_binding(OUT/'COMPLETE.json'),statuses=dict(Counter(r['status']for r in attempts)))))
    except Exception as error:
        put(OUT/'FAILURE.json',dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);main(parser.parse_args().entry)
