"""Actual-shaped text-only identity fixture; no source/stat/header/payload reads."""
from pathlib import Path
import importlib.util,io,json,hashlib
H=Path(__file__).parent
sp=importlib.util.spec_from_file_location('v2',H/'diagnose_source_train_modalities.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
path=str(m.P/'DECLARED_TRAIN_FIXTURE_ONLY.png');body=b'authored-fixture-not-an-image';sha=hashlib.sha256(body).hexdigest()
whitelist=dict(path=path,sha256=sha,kind='WEAK_RGB_PNG');row=dict(path=path,sha256=sha,size_bytes=len(body));m.PAYLOAD[path]=whitelist
m.need(m.payload_identity(whitelist)==m.payload_identity(row),'Extra annotations not identity')
# Actual production payload_file is exercised, with filesystem opened replaced by in-memory authored bytes.
original=m.opened;calls=[]
def in_memory(p):calls.append(str(p));return io.BytesIO(body)
m.opened=in_memory
with m.payload_file(row)as f:m.need(f.read()==body,'Production acceptance with size vs kind annotations')
accepted_calls=len(calls)
wrong=dict(row,sha256='0'*64)
try:m.payload_file(wrong);rejected=False
except AssertionError:rejected=True
m.need(rejected and len(calls)==accepted_calls,'Wrong SHA rejected before filesystem')
try:m.payload_file(dict(row,path=str(m.P/'NOT_DECLARED.png')));rejected=False
except AssertionError:rejected=True
m.need(rejected and len(calls)==accepted_calls,'Missing whitelist path rejected before filesystem')
m.opened=original
result=dict(completed=True,PASS=True,code=m.bind(H/'diagnose_source_train_modalities.py'),checks=['actual-shaped row{path,sha,size_bytes} equals whitelist{path,sha,kind}','production payload_file accepts same path/SHA despite annotations','same path wrongSHA rejected before filesystem','undeclared path rejected before filesystem'],actual_source_files_opened_or_stat_hash_header=0,only_in_memory_authored_bytes=True)
with(H/'PAYLOAD_ANNOTATION_IDENTITY_INVENTED_CHECKS.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
