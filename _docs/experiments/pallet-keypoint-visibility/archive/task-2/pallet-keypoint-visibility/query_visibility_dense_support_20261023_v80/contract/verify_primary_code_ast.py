"""Read-only AST proof of the unchanged primary path; no model import/run."""
from pathlib import Path
import ast,hashlib,json
R=Path(__file__).resolve().parent
P=R.parent.parent
OLD=P/'query_visibility_reference_local_support_20261023_v76/mixed_fit/query_center_support_head.py'
NEW=R.parent/'model/dense_support_head.py'
def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def functions(path):
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeContextSupportHead')
    out={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    out.update({n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)})
    return out
def normalized(node,new=False):
    node=ast.parse(ast.unparse(node)).body[0]
    body=[]
    for s in node.body:
        if isinstance(s,ast.Expr) and isinstance(s.value,ast.Constant) and isinstance(s.value.value,str):continue
        if new and node.name=='__init__' and 'aux_decoder' in ast.unparse(s):continue
        if new and node.name=='initialize_from_proxy' and isinstance(s,ast.Expr) and 'model.aux_decoder' in ast.unparse(s):continue
        if new and node.name=='forward':
            if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='aux' for t in s.targets):continue
            if isinstance(s,ast.Return) and isinstance(s.value,ast.Call):
                assert isinstance(s.value.func,ast.Name) and s.value.func.id=='dict'
                s.value.keywords=[x for x in s.value.keywords if x.arg!='aux_mask_logits']
        body.append(s)
    node.body=body
    return ast.dump(node,include_attributes=False)
def main():
    old,new=functions(OLD),functions(NEW)
    matches={name:normalized(old[name])==normalized(new[name],True) for name in ('__init__','_binary','forward','initialize_from_proxy')}
    assert all(matches.values()),matches
    forward=new['forward']
    assert [a.arg for a in forward.args.args]==['self','p3_roi','local_rgb96','local_valid96','target_rgb128','target_valid128','context42']
    assert forward.args.kwarg is None and forward.args.vararg is None
    text=NEW.read_text();assert 'nn.Conv2d(24,2,1)' in text
    d=dict(PASS=True,test=b(Path(__file__).resolve()),old_model=b(OLD),new_model=b(NEW),
        normalized_primary_AST_equal=matches,
        normalization='Ignore docstrings; remove only declared aux init, aux=f(fused), aux output key, and aux re-zero in initializer. All remaining primary operations and function signatures match.',
        inference_six_tensor_whitelist_unchanged=True,new_decoder_declared_24x2_1x1=True,
        model_import_or_forward=0,actualsource_RGB_GT_mask_feature_score_weights_reads=0,fit_current_PnP=0)
    with (R/'GENERATED_PRIMARY_AST_EQUIVALENCE.json').open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,report=b(R/'GENERATED_PRIMARY_AST_EQUIVALENCE.json'))))
if __name__=='__main__':main()
