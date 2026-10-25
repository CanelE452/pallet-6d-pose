"""Root-entry-only frozen encoder transport. Import performs no payload I/O."""
import argparse, ast, hashlib, json, os, platform, stat, sys, traceback, types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
R = HERE.parent
POLICY = {"path":str(R/"V260_ROOT_PUBLIC_FEATURE_PILOT/PROSPECTIVE_TRANSPORT_PILOT_POLICY.json"),
          "sha256":"f84da771510ffe6f42e3f86565e94f623a42140961bcabeaee18e4355a567e60","size_bytes":1154}
CLOSURE = {"path":str(HERE/"DEPENDENCY_CLOSURE.json"),
           "sha256":"900ea2373160158c25ef324f5171e3a49ee0bbac52776d32f93d1eae99ab5b17","size_bytes":8660}
OFFSETS = [[0,0],[3,0],[-3,0],[0,3],[0,-3],[6,0],[-6,0],[0,6],[0,-6]]
QUERY_KEYS = set("reference_grid_index r_xy r_derived_label r_derived_state r_depth_eligible r_known r_raw_observed_minus_render_m r_raw_residual_finite r_exclusion_reasons native_slot SELF_state record_id q_offset_index q_offset_xy q_xy q_inside_image q_sam_target_present q_pixel_DE_gold r_label_copied_as_query_label q_is_public_emulation q_is_actual_initializer q_local_L_pixels U_is_model_abstention".split())

def lexical(value):
    s = str(value)
    if any(t in s.casefold() for t in ("v112","cache",".pyc","current223","publicheld","oldval","source_holdout","private_evaluation")):
        raise ValueError("Forbidden path before filesystem access")
    p = Path(s)
    if not p.is_absolute() or ".." in p.parts:
        raise ValueError("Absolute path without traversal required")
    return p

def guard(value, file=False, fresh=False):
    p = lexical(value)
    for a in reversed([p,*p.parents]):
        try: m = os.lstat(a).st_mode
        except FileNotFoundError: continue
        if stat.S_ISLNK(m): raise ValueError("Symlink artifact/ancestor rejected")
    if fresh and p.exists(): raise ValueError("Fresh output required")
    if file and not stat.S_ISREG(os.lstat(p).st_mode): raise ValueError("Regular artifact required")
    return p

def binding_key(b):
    if set(b) != {"path","sha256","size_bytes"}: raise ValueError("Exact binding schema")
    lexical(b["path"])
    if len(b["sha256"]) != 64 or type(b["size_bytes"]) is not int or b["size_bytes"] < 0:
        raise ValueError("Binding SHA/size")
    return b["path"],b["sha256"],b["size_bytes"]

def checked(b, whitelist):
    if binding_key(b) not in {binding_key(x) for x in whitelist}: raise ValueError("Exact whitelist required before file access")
    p = guard(b["path"],file=True); h = hashlib.sha256(); size = 0
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block); size += len(block)
    if h.hexdigest() != b["sha256"] or size != b["size_bytes"]: raise ValueError("Bound SHA/size mismatch")
    return p

def bind(p):
    p=guard(p,file=True); b=p.read_bytes()
    return {"path":str(p),"sha256":hashlib.sha256(b).hexdigest(),"size_bytes":len(b)}

def json_write(p, value):
    with guard(p,fresh=True).open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False); f.write("\n")

def module(b, whitelist, name):
    p=checked(b,whitelist); m=types.ModuleType(name); m.__file__=str(p); sys.modules[name]=m
    exec(compile(p.read_bytes(),str(p),"exec"),m.__dict__); return m

def pure(b, whitelist, names, constants=None):
    p=checked(b,whitelist); tree=ast.parse(p.read_bytes()); nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names): raise ValueError("Pinned pure definitions absent")
    ns={"np":np,"torch":torch,"F":F,"hashlib":hashlib,"json":json}
    ns.update(constants or {}); exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),"exec"),ns)
    return types.SimpleNamespace(**{k:ns[k] for k in names})

def authorize(entry):
    if entry.get("execution_authorized") is not True or entry.get("encoder_only") is not True:
        raise ValueError("Separate Root true entry required; no output or payload access")
    if entry["policy"] != POLICY or entry["dependency_closure"] != CLOSURE: raise ValueError("Pinned policy/closure")
    scope_b=entry["root_scope"]; scope=json.loads(checked(scope_b,[scope_b]).read_bytes())
    contract={k:v for k,v in entry.items() if k != "root_scope"}
    if scope.get("approved") is not True or scope.get("entry_contract") != contract: raise ValueError("Exact Root scope contract required")
    metadata=entry["metadata_whitelist"]; codes=entry["code_whitelist"]
    if {binding_key(x) for x in metadata} != {binding_key(POLICY),binding_key(CLOSURE)} or len(metadata) != 2:
        raise ValueError("Only prospective policy and closed implementation metadata")
    policy=json.loads(checked(POLICY,metadata).read_bytes()); closure=json.loads(checked(CLOSURE,metadata).read_bytes())
    if entry["model_loader_config"] != closure["model_loader_config"]: raise ValueError("Official loader config changed")
    if entry["runtime"] != closure["runtime"]: raise ValueError("Fixed runtime contract")
    if checked(entry["driver_code"],codes) != guard(__file__,file=True): raise ValueError("Exact executing driver binding")
    required_codes=list(closure["code"].values())+closure["model_loader_config"]["code_whitelist"]+[entry["driver_code"],entry["core_code"]]
    if {binding_key(x) for x in codes} != {binding_key(x) for x in required_codes} or len(codes) != len({binding_key(x) for x in required_codes}):
        raise ValueError("Closed exact implementation CODE whitelist")
    for b in closure["code"].values(): checked(b,codes)
    for b in closure["model_loader_config"]["code_whitelist"]: checked(b,codes)
    checked(entry["core_code"],codes)
    frames=entry["frames"]
    if [f["frame_id"] for f in frames] != ["0000","0011"]: raise ValueError("Fixed two public frames")
    expected=[]
    for f in frames:
        if set(f) != {"frame_id","rgb","query_records","selected_reference_grid_indices"}: raise ValueError("Frame input schema")
        if f["selected_reference_grid_indices"] != policy["selected_reference_grid_indices"][f["frame_id"]]: raise ValueError("Frozen label-independent selection")
        expected += [f["rgb"],f["query_records"]]
    expected += [closure["model_loader_config"]["official_state"]]
    if len(entry["payload_whitelist"]) != 5 or len({binding_key(x) for x in expected}) != 5 or {binding_key(x) for x in entry["payload_whitelist"]} != {binding_key(x) for x in expected}:
        raise ValueError("Only two RGB/two record manifests/one official state admitted")
    guard(entry["output_dir"],fresh=True)
    return policy,closure

def load_records(frame, payloads, core):
    p=checked(frame["query_records"],payloads); raw=p.read_bytes().splitlines(keepends=True)
    if len(raw) != 43200: raise ValueError("All 4800x9 original records retained per frame")
    selected=set(frame["selected_reference_grid_indices"])
    if sorted(selected) != core.selected_grid_indices(frame["frame_id"]): raise ValueError("Hash policy independent recalculation")
    rows=[]
    for i,line in enumerate(raw):
        row=json.loads(line)
        if set(row) != QUERY_KEYS: raise ValueError("Declared original query schema changed")
        g,off=divmod(i,9); r=[4+8*(g%80),4+8*(g//80)]; q=[r[0]+OFFSETS[off][0],r[1]+OFFSETS[off][1]]
        if row["reference_grid_index"] != g or row["q_offset_index"] != off or row["r_xy"] != r or row["q_offset_xy"] != OFFSETS[off] or row["q_xy"] != q or row["q_local_L_pixels"] != 96:
            raise ValueError("Original grid/q/L/offset identity required; no snap or recrop")
        if row["record_id"] != f"r_y{r[1]}_x{r[0]}_q{off}" or row["q_inside_image"] != (0 <= q[0] < 640 and 0 <= q[1] < 480): raise ValueError("Original record/inside identity")
        if row["native_slot"] is not None or row["SELF_state"] != "UNKNOWN" or row["q_pixel_DE_gold"] is not None or row["r_label_copied_as_query_label"] is not False or row["q_is_public_emulation"] is not True or row["q_is_actual_initializer"] is not False or row["U_is_model_abstention"] is not False:
            raise ValueError("AUX emulation namespace remains separate from native/SELF/query gold")
        rows.append({"original_record":row,"source_query_row_index":i,"source_line_sha256":hashlib.sha256(line).hexdigest(),"selected_for_feature":g in selected})
    return rows

def run(entry, entry_binding=None):
    policy,closure=authorize(entry)
    runtime=entry["runtime"]
    if platform.python_version()!=runtime["python"] or np.__version__!=runtime["numpy"] or torch.__version__!=runtime["torch"]: raise ValueError("Strict declared runtime versions")
    payloads=entry["payload_whitelist"];codes=entry["code_whitelist"]; cb=closure["code"]
    core=module(entry["core_code"],codes,"v260_single_query_core")
    crop_math=pure(cb["spatial_crop"],codes,["crop"])
    preprocess=pure(cb["preprocess"],codes,["preprocess_views"],{"MEAN":(.485,.456,.406),"STD":(.229,.224,.225)})
    canonical=pure(cb["canonical"],codes,["state_sha","node_pixel_batch"])
    crop_module=module(cb["crop_module"],codes,"crop_modules")
    loader=module(cb["model_loader"],codes,"v260_official_loader")
    loader.configure_runtime()
    records=[load_records(f,payloads,core) for f in entry["frames"]]
    if sum(x["selected_for_feature"] for rows in records for x in rows) != 144: raise ValueError("Exactly 144 selected feature rows")
    # Scope and all declarations validated before model construction/official weight access.
    checked(closure["model_loader_config"]["official_state"],payloads)
    model=loader.load_backbone(entry["model_loader_config"],runtime["device"],treatment=False)
    model.eval().requires_grad_(False)
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()): raise ValueError("Frozen/eval encoder")
    before=canonical.state_sha(model.state_dict())
    if before != closure["expected_encoder_state_sha256"]: raise ValueError("Official encoder tensor state")
    out=guard(entry["output_dir"],fresh=True);out.mkdir()
    features=np.zeros((144,1186),np.float32); metadata=[]; decoded=[]; first_repeat=None; k=0
    try:
        from PIL import Image, __version__ as pillow_version
        for frame,rows in zip(entry["frames"],records):
            image_path=checked(frame["rgb"],payloads)
            with Image.open(image_path) as im: rgb=np.asarray(im.convert("RGB"),dtype=np.uint8).copy()
            if rgb.shape != (480,640,3) or rgb.dtype != np.uint8: raise ValueError("Declared public RGB HWC 480x640")
            decoded.append({"frame_id":frame["frame_id"],"encoded":frame["rgb"],"decoded_RGB_sha256_HWC":hashlib.sha256(rgb.tobytes()).hexdigest(),"shape":list(rgb.shape),"decoder":"PIL.Image.open.convert('RGB')","Pillow_version":pillow_version})
            pending=[]
            for item in rows:
                item.update({"frame_id":frame["frame_id"],"namespace":"PUBLIC_SURFACE_AUX_QUERY_EMULATION","feature_row_index":-1,"input_support_status":-1,"encoder_forwarded":False})
                if not item["selected_for_feature"]: continue
                row=item["original_record"];raw,valid,supported=core.query_crop(rgb,row["q_xy"],row["q_local_L_pixels"],crop_math.crop)
                item["feature_row_index"]=k;item["input_support_status"]=int(supported)
                meta={"feature_row_index":k,"frame_id":frame["frame_id"],"source_query_row_index":item["source_query_row_index"],"record_id":row["record_id"],"q_xy":row["q_xy"],"L_pixels":row["q_local_L_pixels"],"input_supported":supported,"RAW96_RGB_sha256_CHW":hashlib.sha256(raw.tobytes()).hexdigest(),"RAW96_valid_sha256_bool":hashlib.sha256(valid.tobytes()).hexdigest(),"fractions":[float(valid.mean()),float(valid[:,32:64,32:64].mean())]}
                metadata.append(meta)
                if supported: pending.append((k,raw,valid,item))
                k+=1
            for start in range(0,len(pending),runtime["batch_size"]):
                batch=pending[start:start+runtime["batch_size"]];raw=np.stack([x[1] for x in batch]);valid=np.stack([x[2] for x in batch])
                vv=torch.from_numpy(valid).float()
                expected_fraction=torch.stack((vv.mean((1,2,3)),vv[:,:,32:64,32:64].mean((1,2,3))),1).numpy()
                with torch.inference_mode():
                    feat,fractions,central4=core.encode_queries(model,raw,valid,canonical.node_pixel_batch,preprocess.preprocess_views,crop_module,runtime["device"])
                    if first_repeat is None:
                        again,af,ac=core.encode_queries(model,raw,valid,canonical.node_pixel_batch,preprocess.preprocess_views,crop_module,runtime["device"])
                        first_repeat={"rows":len(batch),"features_byteexact":feat.tobytes()==again.tobytes(),"fractions_byteexact":fractions.tobytes()==af.tobytes(),"central4_byteexact":central4.tobytes()==ac.tobytes()}
                        if not all(first_repeat[x] for x in ("features_byteexact","fractions_byteexact","central4_byteexact")): raise ValueError("First supported batch repeat mismatch")
                if not np.array_equal(fractions,expected_fraction) or not np.array_equal(feat[:,1152:1154],fractions): raise ValueError("Canonical/geometric fractions parity")
                for j,(index,_,_,item) in enumerate(batch):
                    if not central4[j]: raise ValueError("Input-only central4 support")
                    features[index]=feat[j];item["encoder_forwarded"]=True
        after=canonical.state_sha(model.state_dict())
        if before != after or not np.isfinite(features).all() or np.any(features[:,1154:]): raise ValueError("Frozen state/finite/zero AUX condition")
        if k != 144: raise ValueError("Selected feature row count")
        np.save(out/"PUBLIC_AUX_FEATURES1186.npy",features,allow_pickle=False)
        with (out/"ALL86400_QUERY_TRANSPORT_LEDGER.jsonl").open("x",encoding="utf-8") as f:
            for rows in records:
                for item in rows: f.write(json.dumps(item,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")
        with (out/"SELECTED144_FEATURE_ROWS.jsonl").open("x",encoding="utf-8") as f:
            for item in metadata: f.write(json.dumps(item,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")
        qa={"completed":True,"PASS":True,"all_records":86400,"selected_feature_rows":144,"unselected_feature_row_index_minus1":86256,"encoder_forwarded_rows":sum(x["encoder_forwarded"] for rows in records for x in rows),"selected_unsupported_rows":sum(not x["input_supported"] for x in metadata),"unselected_input_support_not_assessed":86256,"layout":core.LAYOUT,"condition32_all_zero":True,"native_slot":None,"SELF":"UNKNOWN","first_supported_batch_repeat":first_repeat,"encoder_state_sha256_before":before,"encoder_state_sha256_after":after,"decoded_RGB":decoded,"teacher_metadata_loaded_as_sidecar_only":True,"teacher_values_used_in_crop_encoder_or_selection":False,"r_derived_label_is_query_pixel_gold":False,"classifier_forwards":0,"optimizer_updates":0,"PnP_or_query_optimization":0,"training_admission":False,"full_capture_disjointness_from_protected_public45_certified":False,"current_or_held_payloads":0}
        if first_repeat is None: raise ValueError("No supported feature batch for transport proof")
        json_write(out/"TRANSPORT_QA.json",qa)
        outputs={n:bind(out/n) for n in ("PUBLIC_AUX_FEATURES1186.npy","ALL86400_QUERY_TRANSPORT_LEDGER.jsonl","SELECTED144_FEATURE_ROWS.jsonl","TRANSPORT_QA.json")}
        complete={"completed":True,"PASS":True,"diagnostic_only":True,"entry":entry_binding,"policy":POLICY,"dependency_closure":CLOSURE,"inputs":entry["payload_whitelist"],"code":entry["code_whitelist"],"outputs":outputs,"feature_shape":[144,1186],"feature_dtype":"float32","pixel1154_obtainable_as_slice":True,"no_duplicate_pixel_array":True,"runtime":runtime,"encoder_state_sha256_before":before,"encoder_state_sha256_after":after,"classifier_forwards":0,"optimizer_updates":0,"teacher_metadata_loaded_as_sidecar_only":True,"training_admission":False}
        json_write(out/"COMPLETE.json",complete)
        return complete
    except BaseException as exc:
        json_write(out/"FAILED_PARTIAL.json",{"completed":False,"error_type":type(exc).__name__,"error":str(exc),"automatic_retry":False,"classifier_forwards":0,"optimizer_updates":0})
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--entry",required=True);a=ap.parse_args()
    p=guard(a.entry,file=True);entry=json.loads(p.read_bytes())
    # Keep scope equality on the exact written entry; provenance binding stays outside it.
    result=run(entry,bind(p));print(json.dumps({"completed":True,"PASS":result["PASS"],"output_dir":entry["output_dir"]}))

if __name__ == "__main__": main()
