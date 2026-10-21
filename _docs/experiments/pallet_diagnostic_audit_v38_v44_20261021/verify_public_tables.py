from pathlib import Path
import csv,json,hashlib
R=Path(__file__).resolve().parent
m=json.loads((R/'MANIFEST.json').read_text())
for f in m['files']:
 p=R/f['path']; assert p.stat().st_size==f['bytes']; assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
pairs=list(csv.DictReader((R/'results/v44_paired_rows.csv').open()))
for pop,n,expected in [('CAL46',46,1),('PRIMARY192',192,19)]:
 rows=[r for r in pairs if r['population']==pop]; assert len(rows)==5*n; assert len({r['id']for r in rows})==n
 assert len({r['id']for r in rows if r['comparator_success_OCPR_fail']=='True'})==expected
print('PASS: published byte hashes and saved scalar unions; no model or pose evaluation')
