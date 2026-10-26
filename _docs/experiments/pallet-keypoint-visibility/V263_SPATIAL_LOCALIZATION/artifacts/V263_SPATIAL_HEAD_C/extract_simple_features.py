"""Root-authorized original7040 SYN RGB96 -> fresh spatial intermediate tokens."""
import hashlib,io,json,os,sys,time,types
from pathlib import Path
import numpy as np
import torch

H=Path(__file__).absolute().parent

def module(path,name):
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m


def run(entry_path):
    # Own presealed helper supplies lexical/ancestor checks and exact byte pins.
    p=module(H/'probe_runtime.py','v263_bound_io')
    e=json.loads(p.safe(entry_path).read_bytes())
    assert e['approved_actual_original7040_spatial_features'] is True
    assert e['simple_factor']==1.0 and e['official_layer_index']==8
    assert p.binding(Path(__file__).absolute())==e['code']
    out=Path(e['output_directory']);p.safe(out.parent);assert not out.exists();out.mkdir()
    started=time.time()
    try:
        packet=json.loads(p.raw(e['label_packet_complete']))
        assert packet['completed'] and packet['PASS'] and packet['include_V255'] is False
        groups=[json.loads(x) for x in p.raw(packet['outputs']['groups']).splitlines()]
        rows=[json.loads(x) for x in p.raw(packet['outputs']['rows']).splitlines()]
        assert len(groups)==880 and len(rows)==7040
        crop_binding=groups[0]['RGB96_SOURCE_binding']
        assert all(g['RGB96_SOURCE_binding']==crop_binding and g['RGB96_member']=='rgb96' for g in groups)
        assert crop_binding==e['original_crop_input']
        with np.load(io.BytesIO(p.raw(crop_binding)),allow_pickle=False) as z:
            rgb=np.array(z['rgb96'],copy=True)
            valid=np.array(z['valid96'],copy=True)
        assert rgb.shape==(7040,96,96,3) and rgb.dtype==np.uint8
        if valid.shape==(7040,96,96):valid=valid[:,None]
        assert valid.shape==(7040,1,96,96) and valid.dtype==bool
        assert all(r['source_index']==i and 0<=r['native_slot']<8 for i,r in enumerate(rows))
        available=np.array([r['original_available'] for r in rows],bool)
        train=module(H/'train_spatial_v2.py','v263_train_tensor_api')
        support=train.token_support_from_valid(torch.from_numpy(valid)).numpy()&available[:,None,None]
        active=np.flatnonzero(support.reshape(7040,-1).any(1))
        np.save(out/'TOKEN_SUPPORT.npy',support)
        features=np.lib.format.open_memmap(out/'TOKENS_FP16.npy',mode='w+',dtype=np.float16,shape=(7040,256,384))
        features[:]=0
        closure=json.loads(p.raw(e['official_dependency_closure']))
        p.module(closure['code']['crop_module'],'crop_modules')
        loader=p.module(closure['code']['model_loader'],'v263_simple_official_loader')
        model=loader.load_backbone(closure['model_loader_config'],'cuda:0',treatment=False).eval()
        before=loader.tensor_state_sha(model.state_dict(),base=True)
        assert before==closure['expected_encoder_state_sha256'] and not any(v.requires_grad for v in model.parameters())
        # Literal pinned preprocess function; only its full view feeds block8.
        import ast
        from torch.nn import functional as F
        src=closure['code']['preprocess'];tree=ast.parse(p.raw(src))
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_views']
        ns={'torch':torch,'F':F,'MEAN':(.485,.456,.406),'STD':(.229,.224,.225)}
        exec(compile(ast.Module(body=funcs,type_ignores=[]),src['path'],'exec'),ns)
        repeat=True
        for start in range(0,len(active),32):
            ix=active[start:start+32]
            x=torch.from_numpy(rgb[ix].transpose(0,3,1,2).copy());v=torch.from_numpy(valid[ix])
            views,_=ns['preprocess_views'](x,v)
            tokens=train.extract_spatial(model,views[:len(ix)].cuda())
            if start==0:
                again=train.extract_spatial(model,views[:len(ix)].cuda())
                repeat=bool(torch.equal(tokens,again));assert repeat
            features[ix]=tokens.cpu().numpy()
            if start%320==0:print(json.dumps({'rows_done':min(start+32,len(active)),'supported_rows':len(active)}),flush=True)
        features.flush();del features
        after=loader.tensor_state_sha(model.state_dict(),base=True);assert before==after
        census={'all_rows':7040,'original_unavailable':int((~available).sum()),'model_supported':len(active),
                'FIT_rows':sum(r['fit'] for r in rows),'DEV_rows':sum(not r['fit'] for r in rows)}
        summary={'completed':True,'PASS':True,'entry':p.binding(Path(entry_path)),'code':e['code'],
          'label_packet_complete':e['label_packet_complete'],'original_crop_input':crop_binding,
          'outputs':{n:p.binding(out/n) for n in ['TOKENS_FP16.npy','TOKEN_SUPPORT.npy']},
          'census':census,'crop':'SIMPLE_2D original q/L; original stored rgb96 bytes',
          'representation':{'layer':8,'norm':True,'shape':[7040,256,384],'storage_dtype':'float16','forward_dtype':'float32'},
          'encoder_state_before':before,'encoder_state_after':after,'encoder_frozen':True,
          'first_batch_repeat_FP16_byte_exact':repeat,'GT_pose_reference_or_labels_in_encoder_inputs':False,
          'actual_current_or_held_reads':0,'head_optimizer_updates':0,'elapsed_seconds':time.time()-started}
        (out/'COMPLETE.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'completed':True,'PASS':True,'COMPLETE':p.binding(out/'COMPLETE.json'),'census':census}),flush=True)
    except BaseException as ex:
        (out/'FAILURE.json').write_text(json.dumps({'completed':False,'PASS':False,'error_type':type(ex).__name__,'error':str(ex),'automatic_retry':False},indent=2)+'\n')
        raise

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--entry',required=True);run(Path(a.parse_args().entry))
