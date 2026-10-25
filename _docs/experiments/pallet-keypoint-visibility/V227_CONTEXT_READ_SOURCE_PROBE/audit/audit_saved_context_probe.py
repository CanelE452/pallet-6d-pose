"""허가된 V227 SOURCE 결과의 재현/가산분해/사전고정 4행 표시. 새 학습 없음."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch
from torch.nn import functional as F

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
HERE = R / 'V227_ACTUAL_MODEL_AUDIT_A'
FIT_COMPLETE = R / 'V227_ROOT_CONTEXT_PROBE/source_pair_fit_v1/COMPLETE.json'
FIT_SHA = '2c658c83fb1ac66134cafdddf991bf09efe4dedd8cd442725aef35ea17fdc536'
FEATURE_COMPLETE = R / 'V227_ROOT_CONTEXT_PROBE/context_features_v1/COMPLETE.json'
FEATURE_COMPLETE_SHA = '1e7fe53291e77588230949a14f62b5b78f407edf248347607d21a22c71f7ca11'
HEAD = R / 'V227_ATTENTION_IMPLEMENTATION_A/head.py'
SELECTION = HERE / 'FIXED_DEV_FIRST2D_FIRST2E_BEFORE_SCORES.json'
REPLAY_ATOL = 1e-5


def safe(path):
    p = Path(os.path.abspath(os.fspath(path)))
    assert 'v112' not in str(p).casefold(), 'V112 lexical reject before FS'
    for q in reversed((p, *p.parents)):
        assert not q.is_symlink(), str(q)
    return p


def read(path):
    return safe(path).read_bytes()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def bind(path):
    b = read(path)
    return {'path': str(safe(path)), 'sha256': sha_bytes(b), 'size_bytes': len(b)}


def bound(binding):
    b = read(binding['path'])
    assert sha_bytes(b) == binding['sha256'], binding['path']
    if 'size_bytes' in binding:
        assert len(b) == binding['size_bytes']
    return b


def js(path, obj):
    with safe(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load_roots():
    a = read(FIT_COMPLETE)
    b = read(FEATURE_COMPLETE)
    assert sha_bytes(a) == FIT_SHA and sha_bytes(b) == FEATURE_COMPLETE_SHA
    fit, feat = json.loads(a), json.loads(b)
    assert fit['completed'] and fit['PASS'] and feat['completed'] and feat['PASS']
    assert fit['feature_packet'] == feat['features']
    return fit, feat


def select():
    assert not SELECTION.exists(), '선정 기록 덮어쓰기 금지'
    fit, feat = load_roots()
    # 점수 CSV와 PT는 이 단계에서 열지 않는다.
    with np.load(io.BytesIO(bound(fit['feature_packet'])), allow_pickle=False) as z:
        y, is_fit = z['target'], z['fit_bool']
        ids, frames, natives = z['row_id'], z['frame_index'], z['native']
        assert y.shape == (368,) and int(is_fit.sum()) == 65
        indices = [int(i) for label in (0, 1) for i in np.flatnonzero((~is_fit) & (y == label))[:2]]
        assert len(indices) == 4 and len(set(indices)) == 4
        rows = [{'packet_row_index': i, 'row_id': str(ids[i]), 'frame_index': int(frames[i]), 'native': int(natives[i]), 'truth_D0_E1': int(y[i]), 'role': 'DEV'} for i in indices]
    inputs = [json.loads(line) for line in bound(feat['input_manifest']).splitlines() if line]
    facts = json.loads(bound(feat['input_facts']))['frames']
    fm = {r['original_frame_index']: r for r in inputs}
    ff = {r['original_frame_index']: r for r in facts}
    for r in rows:
        frame = fm[r['frame_index']]
        q = next(q for q in frame['queries'] if q['row_id'] == r['row_id'])
        assert q['original_query_available'] and q['existing_role_target'] == r['truth_D0_E1']
        assert q['native_slot'] == r['native']
        r.update({'query_xy': q['query_xy'], 'L': q['L'], 'source_id': frame['source_id'], 'canonical_RGB_NPY': frame['canonical_RGB_NPY'], 'raw_hw': frame['raw_hw'], 'context_box_xyxy': ff[r['frame_index']]['context_box_xyxy']})
    js(SELECTION, {'completed': True, 'selection_rule': 'DEV packet order first2 D and first2 E before score/PT reads', 'selected_rows': rows, 'unique_frames': sorted({r['frame_index'] for r in rows}), 'fit_complete': bind(FIT_COMPLETE), 'feature_complete': bind(FEATURE_COMPLETE), 'input_manifest': feat['input_manifest'], 'input_facts': feat['input_facts'], 'scores_or_PT_read_before_selection': 0, 'no_new_case_or_model_selection': True})
    print(json.dumps({'completed': True, 'selection': bind(SELECTION), 'selected_rows': [{k:r[k] for k in ('packet_row_index','row_id','frame_index','native','truth_D0_E1')} for r in rows]}, ensure_ascii=False))


def state_hash(state):
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        a = t.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(json.dumps(list(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def cm(y, logits):
    p = (np.asarray(logits) >= 0).astype(int)
    m = [[int(np.sum((y == a) & (p == b))) for b in (0,1)] + [0] for a in (0,1)]
    out = {'rows':len(y), 'CM_D_E_by_D_E_U':m, 'U':0}
    for k, name in enumerate(('D','E')):
        den = m[0][k] + m[1][k]
        out[name] = {'precision': m[k][k]/den if den else None, 'recall': m[k][k]/sum(m[k]), 'truth_count':sum(m[k]), 'prediction_count':den}
    return out


def pure_function(path, expected, names, namespace):
    b = read(path); assert sha_bytes(b) == expected
    tree = ast.parse(b)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def render(selection, feat, weights_by_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    P = R.parent
    v90 = P / 'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
    v107 = P / 'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
    ns = {'np': np}
    pure_function(v90, '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b', ['crop'], ns)
    pure_function(v107, 'aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3', ['valid_box','target_context_grid','sample_target_context'], ns)
    manifest = [json.loads(line) for line in bound(feat['input_manifest']).splitlines() if line]
    fact_rows = json.loads(bound(feat['input_facts']))['frames']
    fm = {r['original_frame_index']:r for r in manifest}
    facts = {r['original_frame_index']:r for r in fact_rows}
    images = {}
    allowed = {r['canonical_RGB_NPY']['path']:r['canonical_RGB_NPY'] for r in selection['selected_rows']}
    for path, b in allowed.items():
        a = np.load(io.BytesIO(bound(b)), allow_pickle=False)
        assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3
        images[path] = a
    outputs, details = [], []
    for order, r in enumerate(selection['selected_rows']):
        frame, fact = fm[r['frame_index']], facts[r['frame_index']]
        rgb = images[r['canonical_RGB_NPY']['path']]
        assert list(rgb.shape[:2]) == r['raw_hw']
        assert sha_bytes(np.ascontiguousarray(rgb).tobytes()) == fact['decoded_RGB_sha256']
        local, valid = ns['crop'](rgb, r['query_xy'], r['L'], 96)
        k = fact['original_query_rows'].index(r['row_id'])
        assert sha_bytes(np.ascontiguousarray(local).tobytes()) == fact['original96_RGB_SHA'][k]
        assert sha_bytes(np.ascontiguousarray(valid).tobytes()) == fact['original96_valid_SHA'][k]
        context, context_valid = ns['sample_target_context'](rgb, r['context_box_xyxy'])
        w = weights_by_id[r['row_id']].reshape(16,16)
        entropy = float(-np.sum(w.astype(np.float64) * np.log(np.maximum(w, 1e-300))))
        box = np.asarray(r['context_box_xyxy'], dtype=float)
        q = np.asarray(r['query_xy'], dtype=float)
        q224 = (q-box[:2])/(box[2:]-box[:2])*224-.5
        fig, axs = plt.subplots(1,4, figsize=(18,4.8), gridspec_kw={'width_ratios':[2.4,1,1,1.15]})
        axs[0].imshow(rgb)
        for native, q8 in enumerate(frame['query_xy']):
            if q8 is None: continue
            color = 'cyan' if frame['original_query_available'][native] else 'gray'
            axs[0].plot(*q8, '+', color=color, ms=5)
            axs[0].text(q8[0],q8[1],str(native),color=color,fontsize=8,clip_on=True)
        axs[0].add_patch(patches.Rectangle(box[:2], box[2]-box[0], box[3]-box[1], fill=False, edgecolor='yellow',linewidth=1))
        axs[0].plot(*q,'+',color='cyan',ms=13,mew=2)
        axs[0].set_title('Original source RGB / initializer q8 + context box')
        axs[0].set_xlim(0,rgb.shape[1]); axs[0].set_ylim(rgb.shape[0],0)
        axs[1].imshow(np.clip(context/255,0,1));axs[1].plot(*q224,'+',color='cyan',ms=13,mew=2);axs[1].set_title('Same context224 / original active q')
        axs[2].imshow(local.transpose(1,2,0));axs[2].plot(47.5,47.5,'+',color='cyan',ms=13,mew=2);axs[2].set_title('Same original local96 / q=47.5')
        axs[3].imshow(np.clip(context/255,0,1),extent=(-.5,15.5,15.5,-.5))
        h = axs[3].imshow(w,cmap='magma',vmin=0,vmax=.02,alpha=.65,extent=(-.5,15.5,15.5,-.5))
        q16=(q224+.5)/14-.5
        axs[3].plot(*q16,'+',color='cyan',ms=10,mew=2)
        axs[3].set_title(f'TREAT attention / entropy {entropy:.4f}')
        cb = fig.colorbar(h, ax=axs[3],fraction=.05,pad=.04)
        cb.set_ticks([0,1/256,.01,.02]);cb.set_ticklabels(['0','uniform 1/256','.01','.02'])
        for ax in axs: ax.set_xticks([]); ax.set_yticks([])
        label = 'D' if r['truth_D0_E1']==0 else 'E'
        fig.suptitle(f"Fixed DEV {label} | {r['row_id']} | native {r['native']} | original q/L unchanged",fontsize=12)
        fig.text(.5,.015,'Attention weights are a model read, not truth, an occlusion mask, a corrected query, or causal proof. Reused SOURCE weak labels; not blind DEV.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.04,1,.93))
        path = HERE / f"FIXED_DEV_{order:02d}_N{r['native']}_PANEL.png"
        assert not safe(path).exists()
        fig.savefig(path,dpi=140);plt.close(fig)
        outputs.append(bind(path))
        details.append({'row_id':r['row_id'],'entropy_nats':entropy,'uniform_entropy_nats':float(np.log(256)),'max_attention_weight':float(w.max()),'heatmap_fixed_vmin':0,'heatmap_fixed_vmax':.02,'local96_RGB_SHA_exact':True,'local96_valid_SHA_exact':True,'panel':outputs[-1]})
    return outputs, details, [dict(b) for b in allowed.values()], {'crop':bind(v90),'context_sampler':bind(v107)}


def run():
    assert not (HERE/'REVIEW.json').exists(), '한 번 완료 결과 덮어쓰기 금지'
    selection = json.loads(read(SELECTION))
    assert selection['scores_or_PT_read_before_selection'] == 0
    fit, feat = load_roots()
    barrier = json.loads(bound(fit['all2_final_barrier']))
    assert barrier['completed'] and barrier['all2_frozen'] and barrier['checkpoints'] == fit['checkpoints']
    assert barrier['DEV_descriptor_or_logit_calls_before_barrier'] == 0
    assert read(HEAD) and bind(HEAD)['sha256'] == fit['head']['sha256']
    spec = importlib.util.spec_from_file_location('v227_audit_bound_head', HEAD)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with np.load(io.BytesIO(bound(fit['feature_packet'])),allow_pickle=False) as z:
        data = {key:z[key] for key in z.files}
    y, is_fit = data['target'], data['fit_bool']
    assert data['localCLS'].shape == (368,384) and int(is_fit.sum()) == 65
    assert np.array_equal(data['frame_index'], data['context_frame_index'][data['context_index']])
    for r in selection['selected_rows']:
        i = r['packet_row_index']; assert not is_fit[i] and str(data['row_id'][i]) == r['row_id'] and int(y[i]) == r['truth_D0_E1']
    saved = list(csv.DictReader(io.StringIO(bound(fit['row_logits']).decode())))
    assert len(saved) == 736
    scoremap = {(r['arm'],r['row_id']):r for r in saved}; assert len(scoremap) == 736
    results, additive, all_replay, attention_rows = {}, [], [], {}
    initial = module.ContextReadProbe(False)
    initial_sha = state_hash(initial.state_dict())
    saved_model_bindings = {}
    for arm, use_attention in [('CONTROL_FIXED_Q',False),('TREAT_ATTENTION',True)]:
        raw = bound(fit['checkpoints'][arm])
        checkpoint = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        assert checkpoint['arm'] == arm and checkpoint['updates'] == 300 and checkpoint['seed'] == 17 and checkpoint['use_attention'] == use_attention
        assert checkpoint['parameters'] == 4561 and checkpoint['initial_state_sha256'] == initial_sha
        model = module.ContextReadProbe(use_attention);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval().requires_grad_(False)
        assert sum(p.numel() for p in model.parameters()) == 4561
        assert len(model.state_dict()) == 6 and all(torch.isfinite(t).all() for t in model.state_dict().values())
        opt = checkpoint['optimizer_state_dict']['state']
        assert len(opt) == (6 if use_attention else 2)
        assert all(float(v['step']) == 300 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in opt.values())
        if not use_attention:
            assert all(torch.equal(checkpoint['state_dict'][k],initial.state_dict()[k]) for k in ('query.weight','query.bias','key.weight','key.bias'))
        logs = json.loads(bound(fit['logs'][arm]))
        assert len(logs['rows']) == 300 and [r['update'] for r in logs['rows']] == list(range(1,301))
        assert logs['DEV_used_for_fit_or_normalization'] is False
        assert all(np.isfinite([r['balanced_BCE'],r['L2_sum'],r['total_loss']]).all() and abs(r['total_loss']-r['balanced_BCE']-.001*r['L2_sum']) < 2e-6 for r in logs['rows'])
        result = {'parameters':4561,'finite_state':True,'state_tensors':6,'final_state_sha256':state_hash(model.state_dict()),'optimizer_state_count':len(opt),'updates300':True,'initial_state_exact':True,'replay':{},'DEV_additive':{}}
        max_replay = 0.0; max_additive = 0.0
        for role, mask in [('FIT',is_fit),('DEV',~is_fit)]:
            ix = np.flatnonzero(mask);ci = data['context_index'][ix]
            tensors = tuple(torch.from_numpy(np.ascontiguousarray(data[k][ci if k in ('contextCLS','tokens') else ix])) for k in ('localCLS','fixedQ','contextCLS','tokens','condition32','normalized_active_q'))
            with torch.no_grad():
                logit = model(*tensors).numpy()
                local, fixed, cls, tokens, condition, activeq = tensors
                if use_attention:
                    descriptor, weights = model.attention_read(tokens,condition,activeq)
                else:
                    descriptor = F.normalize(fixed,p=2,dim=-1,eps=1e-12)
                    weights = None
                x = torch.cat([F.normalize(local,p=2,dim=-1,eps=1e-12),descriptor,F.normalize(cls,p=2,dim=-1,eps=1e-12),condition],1)
                w = model.classifier.weight[0]
                blocks = torch.stack([(x[:,a:b]*w[a:b]).sum(1) for a,b in [(0,384),(384,768),(768,1152),(1152,1184)]],1).numpy()
                bias = float(model.classifier.bias[0]); summed = blocks.sum(1) + bias
            old = []
            for j,i in enumerate(ix):
                r = scoremap[(arm,str(data['row_id'][i]))]
                assert r['role'] == role and int(r['truth_D0_E1']) == int(y[i]) and int(r['native']) == int(data['native'][i]) and int(r['frame_index']) == int(data['frame_index'][i])
                old.append(float(r['logit']))
                assert int(r['pred_D0_E1']) == int(logit[j]>=0)
                all_replay.append({'arm':arm,'row_id':str(data['row_id'][i]),'role':role,'truth_D0_E1':int(y[i]),'saved_logit':float(r['logit']),'replay_logit':float(logit[j]),'abs_difference':abs(float(r['logit'])-float(logit[j]))})
                if role == 'DEV':
                    additive.append({'arm':arm,'row_id':str(data['row_id'][i]),'frame_index':int(data['frame_index'][i]),'native':int(data['native'][i]),'truth_D0_E1':int(y[i]),'local384':float(blocks[j,0]),'read384':float(blocks[j,1]),'contextCLS384':float(blocks[j,2]),'condition32':float(blocks[j,3]),'bias':bias,'sum':float(summed[j]),'factual_logit':float(logit[j]),'condition_plus_bias_logit':float(blocks[j,3]+bias)})
                    if weights is not None:attention_rows[str(data['row_id'][i])] = weights[j].numpy()
            err = float(np.max(np.abs(np.asarray(old)-logit)))
            add_err = float(np.max(np.abs(summed-logit)))
            assert err <= REPLAY_ATOL and add_err <= REPLAY_ATOL, (arm,role,err,add_err)
            max_replay=max(max_replay,err);max_additive=max(max_additive,add_err)
            result['replay'][role] = cm(y[ix],logit)
            if role == 'DEV':
                result['DEV_additive'] = {'condition_plus_bias_only_no_refit':cm(y[ix],blocks[:,3]+bias),'block_mean_signed':{name:float(blocks[:,j].mean()) for j,name in enumerate(['local','read','contextCLS','condition'])},'block_mean_abs':{name:float(np.abs(blocks[:,j]).mean()) for j,name in enumerate(['local','read','contextCLS','condition'])},'bias':bias,'attribution_not_causal_intervention':True,'partial_logit_is_posthoc_OOD_diagnostic':True}
        result.update({'maxabs_saved_replay':max_replay,'replay_signs_exact':True,'maxabs_additive_sum_minus_logit':max_additive})
        results[arm]=result;saved_model_bindings[arm]=fit['checkpoints'][arm]
    for name, rows in [('ALL736_REPLAY_LOGITS.csv',all_replay),('ALL606_DEV_ADDITIVE_LOGIT_CONTRIBUTIONS.csv',additive)]:
        with safe(HERE/name).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    panels, panel_details, rgb_bindings, math_bindings = render(selection,feat,attention_rows)
    entropy = [-np.sum(w.astype(np.float64)*np.log(np.maximum(w,1e-300))) for w in attention_rows.values()]
    result = {'completed':True,'PASS':True,'scope':'V227 SOURCE151/368 only; trained checkpoint replay and posthoc additive diagnosis; no optimizer or new fit','fit_complete':bind(FIT_COMPLETE),'feature_complete':bind(FEATURE_COMPLETE),'code':bind(__file__),'head':fit['head'],'selection_before_scores':bind(SELECTION),'replay_atol_predeclared':REPLAY_ATOL,'results':results,'DEV_attention_entropy_nats':{'rows':len(entropy),'mean':float(np.mean(entropy)),'min':float(np.min(entropy)),'max':float(np.max(entropy)),'uniform':float(np.log(256))},'rows':{'replay':len(all_replay),'DEV_additive':len(additive)},'checkpoints':saved_model_bindings,'canonical_RGB_opened_only_selected':rgb_bindings,'sampler_code':math_bindings,'figures':panels,'figure_details':panel_details,'limits':['SOURCE DEV303 reused/exposed, not blind or protected practical evaluation.','Weak associated native-role labels do not certify physical coordinates.','Block dot products are exact additive attribution, not a causal proof.','Condition+bias removes image terms after fitting: posthoc OOD partial-logit diagnostic, no refit.','Attention weights are neither truth nor occlusion masks; original query unchanged.','Fixed first2D/first2E examples are not representative; no new model/head/epoch/threshold choice.'],'new_optimizer_or_updates':0,'current_or_protected_access':0,'GT_reference_pose_access':0,'V112_access':0,'source_branch_failed_no_retraining_or_current_inference':True}
    js(HERE/'REVIEW.json',result)
    html = '<!doctype html><meta charset="utf-8"><title>V227 fixed SOURCE DEV attention diagnostics</title><h1>고정 DEV 첫 D2/E2: attention은 정답·가림 mask·교정 query가 아닙니다</h1><p>원 q/L/native 유지. SOURCE weak label, 재사용 DEV. 색축 0..0.02, 균일값 1/256. 추가 학습 없음.</p>' + ''.join(f'<figure><img style="max-width:100%" src="{Path(b["path"]).name}"><figcaption>{d["row_id"]}; entropy={d["entropy_nats"]:.6f}</figcaption></figure>' for b,d in zip(panels,panel_details))
    with safe(HERE/'FIXED4_GALLERY_KO.html').open('x',encoding='utf-8') as f:f.write(html)
    js(HERE/'COMPLETE.json',{'completed':True,'PASS':True,'review':bind(HERE/'REVIEW.json'),'replay_CSV':bind(HERE/'ALL736_REPLAY_LOGITS.csv'),'additive_CSV':bind(HERE/'ALL606_DEV_ADDITIVE_LOGIT_CONTRIBUTIONS.csv'),'gallery':bind(HERE/'FIXED4_GALLERY_KO.html'),'selection':bind(SELECTION),'figures':panels,'source_branch_failed':True,'no_additional_fit_or_current_inference':True})
    print(json.dumps({'completed':True,'PASS':True,'COMPLETE':bind(HERE/'COMPLETE.json'),'REVIEW':bind(HERE/'REVIEW.json'),'results':results},ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--select',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    assert a.select != a.run
    try:select() if a.select else run()
    except BaseException:
        failure=HERE/('SELECTION_FAILED.json' if a.select else 'ACTUAL_AUDIT_FAILED.json')
        if not failure.exists():js(failure,{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'no_automatic_retry':True})
        raise
