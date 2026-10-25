"""Root-only public r=q frozen feature batch. No classifier/training/teacher channels."""
import argparse, ast, gzip, hashlib, json, os, platform, stat, sys, types
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
R = HERE.parent
OLD = R / "V260_SINGLE_QUERY_FEATURE_TRANSPORT_B"
POLICY = {"path":str(R/"V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON/PROSPECTIVE_FINAL_ONE_COMPARISON_POLICY.json"),"sha256":"5bfeaf8d690b39c6e9c287d1b25340a1ee7b538d90d2d9447be9fa61e94da54c","size_bytes":4732}
SPLIT = {"path":str(R/"V260_ROOT_PUBLIC_BATCH/PROSPECTIVE_BATCH_SCOPE_AND_SPLIT.json"),"sha256":"55ef69da9044616761452a3db17b346777ea061d6a3242e068769006fa60ca68","size_bytes":2247}
CLOSURE = {"path":str(OLD/"DEPENDENCY_CLOSURE.json"),"sha256":"900ea2373160158c25ef324f5171e3a49ee0bbac52776d32f93d1eae99ab5b17","size_bytes":8660}
CORE = {"path":str(OLD/"single_query_core.py"),"sha256":"57a3056c72b5feef2299900faf7b1af274df765f4c5f96efc4cf14fe82902352","size_bytes":1911}
OLD_DRIVER = {"path":str(OLD/"build_single_query_features.py"),"sha256":"c7fd59a252cde3f535b4a1b6286fd4867888486e3beddc8319d0e7091be8e1c2","size_bytes":16839}
SALT = "V261_PUBLIC_POINT_STRATIFIED64_V1|"
REF_KEYS = set("reference_grid_index r_xy r_derived_label r_derived_state r_depth_eligible r_known r_raw_observed_minus_render_m r_raw_residual_finite r_exclusion_reasons native_slot SELF_state".split())

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



def unique(bindings):
    return list({binding_key(b): b for b in bindings}.values())


def authorize(entry):
    if entry.get("execution_authorized") is not True or entry.get("encoder_only") is not True:
        raise ValueError("Separate Root true entry required; no payload/model/output access")
    if entry["policy"] != POLICY or entry["split_policy"] != SPLIT or entry["dependency_closure"] != CLOSURE or entry["core_code"] != CORE:
        raise ValueError("Pinned final policy/split/closure/core required")
    sb = entry["root_scope"]
    scope = json.loads(checked(sb, [sb]).read_bytes())
    if scope.get("approved") is not True or scope.get("entry_contract") != {k:v for k,v in entry.items() if k != "root_scope"}:
        raise ValueError("Root scope must bind exact entry minus root_scope")
    meta, codes, payloads = entry["metadata_whitelist"], entry["code_whitelist"], entry["payload_whitelist"]
    if {binding_key(b) for b in meta} != {binding_key(b) for b in (POLICY,SPLIT,CLOSURE)} or len(meta) != 3:
        raise ValueError("Only fixed policy/split/dependency metadata admitted")
    policy = json.loads(checked(POLICY,meta).read_bytes())
    split = json.loads(checked(SPLIT,meta).read_bytes())
    closure = json.loads(checked(CLOSURE,meta).read_bytes())
    if policy["public_selection"]["rank_salt"] != SALT or policy["public_selection"]["per_frame_per_class_max"] != 64:
        raise ValueError("Fixed per-class64 SHA selection")
    if entry["model_loader_config"] != closure["model_loader_config"] or entry["runtime"] != closure["runtime"]:
        raise ValueError("Unchanged official model/runtime closure")
    required = unique(list(closure["code"].values())+closure["model_loader_config"]["code_whitelist"]+[entry["driver_code"],CORE,OLD_DRIVER])
    if {binding_key(x) for x in codes} != {binding_key(x) for x in required} or len(codes) != len(required):
        raise ValueError("Exact closed CODE whitelist")
    if checked(entry["driver_code"],codes) != guard(__file__,file=True):
        raise ValueError("Executing driver pin")
    for b in required: checked(b,codes)
    frames = entry["frames"]
    if len(frames) != policy["public_complete_frames"] or len(frames) != split["all_complete_bundles"]:
        raise ValueError("All predeclared complete public frames required")
    if len({f["frame_id"] for f in frames}) != len(frames) or len({f["publisher_frame_path"] for f in frames}) != len(frames):
        raise ValueError("Unique public frame identities")
    for f in frames:
        if set(f) != {"frame_id","publisher_frame_path","coarse_group_id","source_role","rgb","reference_records"}:
            raise ValueError("Closed frame schema")
        if split["adopted_group_roles"].get(f["coarse_group_id"]) != f["source_role"]:
            raise ValueError("Only fixed coarse-group TRAIN/DEV role")
        if f["publisher_frame_path"].split("/",1)[0] != f["coarse_group_id"]:
            raise ValueError("Publisher frame/coarse-group identity")
    if dict(Counter(f["source_role"] for f in frames)) != split["bundle_role_counts"]:
        raise ValueError("Fixed split frame counts")
    expected = unique([b for f in frames for b in (f["rgb"],f["reference_records"])]+[closure["model_loader_config"]["official_state"]])
    if len(payloads) != len(expected) or {binding_key(b) for b in payloads} != {binding_key(b) for b in expected}:
        raise ValueError("Only all frame RGB/referencegzip plus official state payloads")
    guard(entry["output_dir"],fresh=True)
    return policy,split,closure


def validate_reference(row, index):
    if set(row) != REF_KEYS or row["reference_grid_index"] != index or row["r_xy"] != [4+8*(index%80),4+8*(index//80)]:
        raise ValueError("All original4800 r-grid rows/order/coordinates required")
    label = row["r_derived_label"]
    if type(label) is not int or label not in (-1,0,1) or row["r_derived_state"] != {-1:"U",0:"D",1:"E"}[label]:
        raise ValueError("D0/E1/Uminus1 teacher schema")
    if row["r_known"] is not (label >= 0) or row["native_slot"] is not None or row["SELF_state"] != "UNKNOWN":
        raise ValueError("Public surface AUX namespace only")
    if not isinstance(row["r_exclusion_reasons"],list) or (label == -1 and not row["r_exclusion_reasons"]):
        raise ValueError("Teacher U exclusions retained")


def select_indices(rows, publisher_frame_path):
    selected, ranks = set(), {}
    for label in (0,1):
        candidates = [i for i,r in enumerate(rows) if r["r_derived_label"] == label]
        order = sorted(candidates,key=lambda i:(hashlib.sha256((SALT+publisher_frame_path+"|"+str(label)+"|"+str(i)).encode("utf-8")).digest(),i))
        ranks.update({i:k for k,i in enumerate(order)})
        selected.update(order[:64])
    return sorted(selected),ranks


def json_line(row):
    return (json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode("utf-8")


def run(entry,entry_binding):
    policy,split,closure = authorize(entry)
    runtime = entry["runtime"]
    if platform.python_version()!=runtime["python"] or np.__version__!=runtime["numpy"] or torch.__version__!=runtime["torch"]:
        raise ValueError("Strict MLPY Python/NumPy/Torch runtime")
    out = guard(entry["output_dir"],fresh=True); out.mkdir()
    payloads,codes,cb = entry["payload_whitelist"],entry["code_whitelist"],closure["code"]
    plans,selected,frame_meta = [],[],[]
    try:
        json_write(out/"STARTED.json",{"entry":entry_binding,"listed_frames":[f["frame_id"] for f in entry["frames"]],"automatic_retry":False})
        ledger_path = out/"ALL_REFERENCE_SELECTION.jsonl.gz"
        plain_hash=hashlib.sha256();plain_size=all_rows=0
        with guard(ledger_path,fresh=True).open("xb") as rawout:
            with gzip.GzipFile(filename="",mode="wb",fileobj=rawout,mtime=0,compresslevel=6) as gzout:
                for frame_index,frame in enumerate(entry["frames"]):
                    p=checked(frame["reference_records"],payloads)
                    with gzip.open(p,"rb") as f: lines=list(f)
                    if len(lines)!=4800: raise ValueError("All4800 teacher reference rows per frame")
                    rows=[json.loads(line) for line in lines]
                    for i,row in enumerate(rows): validate_reference(row,i)
                    chosen,ranks=select_indices(rows,frame["publisher_frame_path"])
                    chosen_set=set(chosen);base=len(selected);indices={i:base+j for j,i in enumerate(chosen)}
                    counts=Counter(r["r_derived_state"] for r in rows)
                    selected_counts=Counter(rows[i]["r_derived_state"] for i in chosen)
                    fm={"frame_index":frame_index,"frame_id":frame["frame_id"],"publisher_frame_path":frame["publisher_frame_path"],"coarse_group_id":frame["coarse_group_id"],"source_role":frame["source_role"],"fit":frame["source_role"]=="TRAIN","label_counts":{k:counts[k] for k in ("D","E","U")},"selected_counts":{k:selected_counts[k] for k in ("D","E")},"selected_reference_grid_indices":chosen,"source_bindings":{"RGB":frame["rgb"],"REFERENCE_ALL":frame["reference_records"]}}
                    frame_meta.append(fm)
                    pending=[]
                    for i,(row,line) in enumerate(zip(rows,lines)):
                        take=i in chosen_set
                        item={"frame_id":frame["frame_id"],"publisher_frame_path":frame["publisher_frame_path"],"coarse_group_id":frame["coarse_group_id"],"source_role":frame["source_role"],"original_reference":row,"source_line_sha256":hashlib.sha256(line).hexdigest(),"selected_for_feature":take,"sample_index":indices[i] if take else -1,"class_SHA_rank":ranks.get(i),"selection_reason":"selected_known" if take else "teacher_U_excluded_not_model_abstention" if row["r_derived_label"]==-1 else "known_class_rank_beyond_cap64"}
                        encoded=json_line(item);gzout.write(encoded);plain_hash.update(encoded);plain_size+=len(encoded);all_rows+=1
                        if take:
                            record={"sample_index":indices[i],"row_id":"PUBLIC_SURFACE_AUX|"+frame["publisher_frame_path"]+"|r"+str(i),"frame_id":frame["frame_id"],"publisher_frame_path":frame["publisher_frame_path"],"coarse_group_id":frame["coarse_group_id"],"source_role":frame["source_role"],"reference_grid_index":i,"q_xy":row["r_xy"],"r_xy":row["r_xy"],"q_offset_index":0,"L_pixels":96,"target":row["r_derived_label"],"family":1,"fit":frame["source_role"]=="TRAIN","available":False,"native_slot":None,"SELF_state":"UNKNOWN","teacher_namespace":"PUBLIC_REAL_SURFACE_DEPTH_SAM_AUX","r_equals_explicit_query":True,"actual_native_initializer":False,"r_is_displaced_query_label":False,"teacher_reference":row,"source_bindings":fm["source_bindings"],"encoder_forwarded":False}
                            selected.append(record);pending.append(record)
                    plans.append((frame,pending))
        if all_rows!=len(entry["frames"])*4800: raise ValueError("All reference ledgers retained")
        json_write(out/"SELECTION_FROZEN_BEFORE_ENCODER.json",{"policy":POLICY,"split_policy":SPLIT,"frames":frame_meta,"selected_rows":len(selected),"all_reference_rows":all_rows,"selection_uses_model_scores":False})
        n=len(selected)
        if not n: raise ValueError("No known rows under frozen selection; no model execution")
        # Original canonical RGB-only core. Do not call the old pilot run/selection.
        core=module(CORE,codes,"v261_single_query_core")
        crop_math=pure(cb["spatial_crop"],codes,["crop"])
        preprocess=pure(cb["preprocess"],codes,["preprocess_views"],{"MEAN":(.485,.456,.406),"STD":(.229,.224,.225)})
        canonical=pure(cb["canonical"],codes,["state_sha","node_pixel_batch"])
        crop_module=module(cb["crop_module"],codes,"crop_modules")
        loader=module(cb["model_loader"],codes,"v261_official_loader")
        loader.configure_runtime()
        checked(closure["model_loader_config"]["official_state"],payloads)
        model=loader.load_backbone(entry["model_loader_config"],runtime["device"],treatment=False)
        model.eval().requires_grad_(False)
        if any(x.training for x in model.modules()) or any(p.requires_grad for p in model.parameters()): raise ValueError("Frozen/eval official encoder")
        before=canonical.state_sha(model.state_dict())
        if before!=closure["expected_encoder_state_sha256"]: raise ValueError("Official encoder tensor state")
        features=np.lib.format.open_memmap(out/"PUBLIC_AUX_FEATURES1186.npy",mode="w+",dtype=np.float32,shape=(n,1186));features[:]=0
        repeat=None;decoded=[]
        from PIL import Image,__version__ as pillow_version
        for frame,items in plans:
            rgb_path=checked(frame["rgb"],payloads)
            if not items:
                decoded.append({"frame_id":frame["frame_id"],"RGB":frame["rgb"],"pixel_decode":False,"reason":"no_selected_known_rows_frame_retained"});continue
            with Image.open(rgb_path) as im: rgb=np.asarray(im.convert("RGB"),dtype=np.uint8).copy()
            if rgb.shape!=(480,640,3): raise ValueError("Original public RGB480x640")
            decoded.append({"frame_id":frame["frame_id"],"RGB":frame["rgb"],"pixel_decode":True,"decoded_RGB_sha256_HWC":hashlib.sha256(rgb.tobytes()).hexdigest(),"Pillow":pillow_version})
            batches=[]
            for record in items:
                raw,valid,supported=core.query_crop(rgb,record["q_xy"],96,crop_math.crop)
                record.update(available=bool(supported),RAW96_RGB_sha256_CHW=hashlib.sha256(raw.tobytes()).hexdigest(),RAW96_valid_sha256_bool=hashlib.sha256(valid.tobytes()).hexdigest(),input_valid_fractions=[float(valid.mean()),float(valid[:,32:64,32:64].mean())])
                if supported: batches.append((record,raw,valid))
            for start in range(0,len(batches),32):
                batch=batches[start:start+32];raw=np.stack([x[1] for x in batch]);valid=np.stack([x[2] for x in batch]);vv=torch.from_numpy(valid).float()
                expected_fraction=torch.stack((vv.mean((1,2,3)),vv[:,:,32:64,32:64].mean((1,2,3))),1).numpy()
                with torch.inference_mode():
                    feat,fractions,central4=core.encode_queries(model,raw,valid,canonical.node_pixel_batch,preprocess.preprocess_views,crop_module,runtime["device"])
                    if repeat is None:
                        other,of,oc=core.encode_queries(model,raw,valid,canonical.node_pixel_batch,preprocess.preprocess_views,crop_module,runtime["device"])
                        repeat={"rows":len(batch),"features_byteexact":feat.tobytes()==other.tobytes(),"fractions_byteexact":fractions.tobytes()==of.tobytes(),"central4_byteexact":central4.tobytes()==oc.tobytes()}
                        if not all(repeat[k] for k in ("features_byteexact","fractions_byteexact","central4_byteexact")):raise ValueError("First batch exact repeat")
                if not np.array_equal(fractions,expected_fraction) or not np.array_equal(feat[:,1152:1154],fractions):raise ValueError("Canonical fractions")
                for j,(record,_,_) in enumerate(batch):
                    if not central4[j]:raise ValueError("Geometric central4 support")
                    features[record["sample_index"]]=feat[j];record["encoder_forwarded"]=True
            print(json.dumps({"frame_id":frame["frame_id"],"selected_rows":len(items),"supported_rows":len(batches)}),flush=True)
        after=canonical.state_sha(model.state_dict());features.flush()
        if before!=after or not np.isfinite(features).all() or np.any(features[:,1154:]):raise ValueError("Frozen state/finite/zero32")
        if repeat is None:raise ValueError("No supported batch for required exact-repeat proof")
        with guard(out/"samples.npz",fresh=True).open("xb") as f:
            np.savez_compressed(f,target=np.array([r["target"] for r in selected],np.int8),family=np.ones(n,np.int8),fit=np.array([r["fit"] for r in selected],bool),available=np.array([r["available"] for r in selected],bool),row_id=np.array([r["row_id"] for r in selected],str),frame_id=np.array([r["frame_id"] for r in selected],str),coarse_group_id=np.array([r["coarse_group_id"] for r in selected],str))
        with guard(out/"SELECTED_FEATURE_ROWS.jsonl",fresh=True).open("wb") as f:
            for record in selected:f.write(json_line(record))
        json_write(out/"FRAME_MANIFEST.json",{"frames":frame_meta,"all_frames_retained":True,"all_reference_rows":all_rows,"selected_rows":n,"source_role_is_fixed_coarse_group_only":True,"independent_sessions_not_certified":True})
        qa={"completed":True,"PASS":True,"selected_rows":n,"all_reference_rows":all_rows,"selected_counts":dict(Counter(r["target"] for r in selected)),"selected_unsupported":sum(not r["available"] for r in selected),"layout":core.LAYOUT,"condition32_all_zero":True,"family":1,"native_slot":None,"SELF":"UNKNOWN","first_supported_batch_repeat":repeat,"encoder_state_sha256_before":before,"encoder_state_sha256_after":after,"decoded_RGB":decoded,"teacher_label_depth_pose_or_mask_values_used_in_crop_or_encoder":False,"teacher_labels_used_for_frozen_sampling":True,"q_equals_supervision_r":True,"teacher_U_is_model_abstention":False,"classifier_forwards":0,"optimizer_updates":0,"actual_native_initializer_or_physical_native_gold":False}
        json_write(out/"FEATURE_QA.json",qa)
        outputs={name:bind(out/name) for name in ("PUBLIC_AUX_FEATURES1186.npy","samples.npz","SELECTED_FEATURE_ROWS.jsonl","FRAME_MANIFEST.json","ALL_REFERENCE_SELECTION.jsonl.gz","SELECTION_FROZEN_BEFORE_ENCODER.json","FEATURE_QA.json","STARTED.json")}
        outputs["ALL_REFERENCE_SELECTION.jsonl.gz"].update(encoding="gzip_jsonl",gzip_mtime=0,gzip_filename="",uncompressed_sha256=plain_hash.hexdigest(),uncompressed_size_bytes=plain_size,records=all_rows)
        complete={"completed":True,"PASS":True,"entry":entry_binding,"policy":POLICY,"split_policy":SPLIT,"dependency_closure":CLOSURE,"outputs":outputs,"features":outputs["PUBLIC_AUX_FEATURES1186.npy"],"samples":outputs["samples.npz"],"records":outputs["SELECTED_FEATURE_ROWS.jsonl"],"frame_manifest":outputs["FRAME_MANIFEST.json"],"feature_shape":[n,1186],"feature_dtype":"float32","runtime":runtime,"encoder_state_sha256_before":before,"encoder_state_sha256_after":after,"official_base":closure["model_loader_config"]["official_state"],"asset_identity":closure["model_loader_config"]["asset_identity"],"payload_whitelist":payloads,"code_whitelist":codes,"classifier_forwards":0,"optimizer_updates":0,"native_or_querypixel_gold":False,"final_one_comparison_policy_only_no_automatic_fit":True}
        json_write(out/"COMPLETE.json",complete)
        return complete
    except BaseException as exc:
        json_write(out/"FAILED_PARTIAL.json",{"completed":False,"PASS":False,"error_type":type(exc).__name__,"error":str(exc),"automatic_retry":False,"listed_frames":[f["frame_id"] for f in entry["frames"]],"selected_plan_frames":len(plans),"classifier_forwards":0,"optimizer_updates":0})
        raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--entry",required=True);args=ap.parse_args()
    p=guard(args.entry,file=True);entry=json.loads(p.read_bytes())
    result=run(entry,bind(p));print(json.dumps({"completed":True,"PASS":result["PASS"],"output_dir":entry["output_dir"],"feature_shape":result["feature_shape"]}),flush=True)


if __name__=="__main__":main()
