import subprocess
import sys
from . import common as C

def main():
    C.protocol();a=C.read(C.DOC/'TRAIN_INPUT_AUDIT.json');assert a['PASS']
    files=[C.HERE/name for name in ('common.py','data.py','train.py','inference.py','test_contracts.py','pretrain.py')]
    C.save(C.DOC/'CODE_LOCK.json',dict(files=[C.bind(p) for p in files]))
    result=subprocess.run([sys.executable,'-m','pytest','-q',str(C.HERE/'test_contracts.py')],capture_output=True,text=True)
    C.save(C.DOC/'PRETRAIN_TESTS.json',dict(PASS=result.returncode==0,stdout=result.stdout,stderr=result.stderr,
        skipped='post-training/post-inference only, rerun at completion',code=C.bind(C.DOC/'CODE_LOCK.json')))
    print(result.stdout,result.stderr,flush=True);assert result.returncode==0

if __name__=='__main__':main()
