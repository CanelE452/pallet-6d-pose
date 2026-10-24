"""Manual decoder repair: unchanged canonical RGB bytes, lossless transport only."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
POLICY=V/'ROOT_SOURCE_INITIALIZER_ASSOCIATION_POLICY_BEFORE_FORWARD.json';POLICY_SHA='99e3efb5693da7ecb0866bdb0b931d6c21d1f56b31351ceaf7e2db50c6866d41'
INPUTS=V/'initializer_preparation/INPUTS16_RGB_ONLY.jsonl';PROTOCOL=H/'PROTOCOL_BEFORE_CANONICAL_RGB_TRANSPORT.json'
FAILED=V/'actual_public_initializer_v1/FAILED_EXECUTION.json';STARTED=V/'actual_public_initializer_v1/STARTED.json'
CANONICAL=V/'decoder_failure_diagnosis/CANONICAL_PY313_PIL_ONLY_RESULT.json';PINNED=V/'decoder_failure_diagnosis/PINNED_PY310_DECODER_ONLY_RESULT.json'
OUT=V/'canonical_RGB_transport_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def prepare():
    assert sha(POLICY)==POLICY_SHA and read(CANONICAL)['PASS']and read(PINNED)['first_failure_index']==0
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),inputs=bind(INPUTS),failed_v1=bind(FAILED),started_v1=bind(STARTED),
        canonical_decoder_diagnosis=bind(CANONICAL),pinned_decoder_diagnosis=bind(PINNED),output=str(OUT),frames=16,
        canonical_runtime=dict(executable='/home/minjae/anaconda3/bin/python',Python='3.13',Pillow='12.0.0',jpeglib='9.0'),
        operation='Fixed16 originalRGBs canonical PILRGB decode -> unchanged expected valueSHA -> lossless uint8 HWC NPY. No expectedSHA change, model, selection or resampling.',
        existing_v1_outputs_preserved=True,automatic_retry=False,pixels_read_during_prepare=0)
    write(PROTOCOL,p);path=H/'ROOT_CANONICAL_TRANSPORT_ENTRY_TEMPLATE.json';write(path,dict(approved_canonical_lossless_transport=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),inputs=p['inputs'],
        model_forward=False,fit=False,current223=False,V112=False,PnP=False,expected_SHA_change=False,source_replacement=False))
    print(json.dumps(dict(prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),template=bind(path))))
def run(entry):
    import PIL
    from PIL import Image
    e=read(entry);assert e['approved_canonical_lossless_transport']is True
    assert all(e[k]is False for k in('model_forward','fit','current223','V112','PnP','expected_SHA_change','source_replacement'))
    assert sha(POLICY)==POLICY_SHA and e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)
    p=read(PROTOCOL)
    for k in('code','policy','inputs'):assert e[k]==p[k];check(p[k])
    for k in('failed_v1','started_v1','canonical_decoder_diagnosis','pinned_decoder_diagnosis'):check(p[k])
    assert Path(sys.executable).resolve()==Path(p['canonical_runtime']['executable']).resolve()and sys.version_info[:2]==(3,13)and PIL.__version__=='12.0.0'and Image.core.jpeglib_version=='9.0'
    assert not OUT.exists();OUT.mkdir();rows=[json.loads(l)for l in INPUTS.read_text().splitlines()];assert len(rows)==16
    arrays=[]
    for row in rows:
        assert sha(row['image'])==row['image_sha256'];rgb=np.asarray(Image.open(row['image']).convert('RGB'));value=hashlib.sha256(rgb.tobytes()).hexdigest()
        assert rgb.dtype==np.uint8 and list(rgb.shape)==row['raw_hw']+[3]and value==row['decoded_RGB_sha256']
        path=OUT/f'RGB_{row["frame_index"]:02d}.npy';np.save(path,rgb,allow_pickle=False)
        assert np.array_equal(np.load(path,allow_pickle=False),rgb)
        arrays.append(dict(frame_index=row['frame_index'],id=row['id'],source=row['source'],source_id=row['source_id'],split=row['split'],raw_hw=row['raw_hw'],original_image=dict(path=row['image'],sha256=row['image_sha256']),
            canonical_RGB_NPY=bind(path),decoded_RGB_sha256=value,expected_decoded_RGB_sha256_unchanged=row['decoded_RGB_sha256']))
    manifest=OUT/'CANONICAL16_RGB_ARRAY_MANIFEST_INPUT_ONLY.json';write(manifest,dict(completed=True,rows=arrays,source_inputs=e['inputs'],all_expected_SHAs_unchanged=True))
    assert sum(path.stat().st_size for path in OUT.iterdir())<256*1024**2
    seal=OUT/'TRANSPORT_COMPLETE.json';write(seal,dict(completed=True,PASS=True,code=bind(__file__),policy=e['policy'],protocol=bind(PROTOCOL),entry=bind(entry),inputs=e['inputs'],failed_v1=p['failed_v1'],
        manifest=bind(manifest),frames=16,all_original_RGB_expected_SHAs_and_HW_exact=True,no_resampling_or_new_selection=True,lossless=True,source_folds_unchanged=True,
        runtime=dict(python=sys.executable,Python=sys.version,NumPy=np.__version__,Pillow=PIL.__version__,jpeglib=Image.core.jpeglib_version),model_forward_fit_private_V112_PnP=0))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(seal),manifest=bind(manifest))))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use inert --prepare or explicit root transport --entry')
