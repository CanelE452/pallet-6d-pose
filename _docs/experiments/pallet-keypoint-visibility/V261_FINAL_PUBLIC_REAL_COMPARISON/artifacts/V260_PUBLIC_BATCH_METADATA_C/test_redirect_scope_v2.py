"""Invented URLs only; no network, file payload, or real signed query values."""
import ast
import hashlib
import json
from pathlib import Path
import types
import urllib.request

H = Path(__file__).absolute().parent
code = (H/'public_batch_v2.py').read_bytes()
m = types.ModuleType('invented_redirect_scope_only')
m.__file__ = str(H/'public_batch_v2.py')
exec(compile(code, m.__file__, 'exec'), m.__dict__)
results = []
initial = 'https://rdr.kuleuven.be/api/access/datafile/357303'
base = 'https://rdmo.icts.kuleuven.be/dataverse/10.48804/DJLDFS/'


def test(name, call):
    call()
    results.append({'name': name, 'PASS': True})


def reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected explicit rejection')


def accepted_origins():
    assert m.storage_url_in_scope(base+'invented-object')
    assert m.storage_url_in_scope(base.replace('.be/', '.be:443/')+'invented-object')
    assert m.storage_url_in_scope(base+'invented-object?X-Amz-Signature=invented-value')
test('HTTPS_exact_dataset_storage_default_or443', accepted_origins)


def rejected_origins():
    bad = [base.replace('https:', 'http:'), base.replace('.be/', '.be:444/'),
           base.replace('rdmo.icts.kuleuven.be', 'other.example'),
           base.replace('https://', 'https://user@'), base.replace('https://', 'https://u:p@'),
           base.replace('DJLDFS/', 'OTHER/'), base.replace('DJLDFS/', 'DJLDFS-other/'),
           base+'../OTHER/object', base+'%2e%2e/OTHER/object', base+'object#fragment',
           base+'a\\b']
    assert all(not m.storage_url_in_scope(x) for x in bad)
test('reject_other_origin_DOI_port_userinfo_path_traversal_fragment', rejected_origins)


def one_redirect_and_redaction():
    secret = 'invented-sensitive-token-not-to-persist'
    query = 'X-Amz-Signature='+secret+'&X-Amz-Expires=123'
    url = base+'invented-object?'+query
    h = m.DatasetScopedRedirect(initial)
    request = h.redirect_request(urllib.request.Request(initial), None, 303, 'See Other', {}, url)
    assert request.full_url == url and len(h.events) == 1
    serialized = json.dumps(h.events)
    assert secret not in serialized and query not in serialized and url not in serialized
    recorded = h.events[0]['storage']
    assert recorded['query_sha256'] == hashlib.sha256(query.encode()).hexdigest()
    assert recorded['query_keys'] == ['X-Amz-Expires', 'X-Amz-Signature']
    reject(lambda: h.redirect_request(urllib.request.Request(url), None, 303, 'See Other', {}, base+'another'))
test('official303_once_query_values_never_in_record_chain_rejected', one_redirect_and_redaction)


def source_and_target_guard():
    bad_initials = [initial.replace('rdr.kuleuven.be', 'evil.example'), initial.replace('https:', 'http:'),
                    initial+'?q=anything', initial+'#fragment', initial.replace('https://', 'https://user@')]
    for url in bad_initials:
        reject(lambda u=url: m.DatasetScopedRedirect(u))
    for target in ('https://evil.example/object', base.replace('DJLDFS/', 'OTHER/')+'object'):
        h = m.DatasetScopedRedirect(initial)
        reject(lambda t=target: h.redirect_request(urllib.request.Request(initial), None, 303, 'See Other', {}, t))
    h = m.DatasetScopedRedirect(initial)
    reject(lambda: h.redirect_request(urllib.request.Request(initial.replace('357303', '999999')), None, 303, 'See Other', {}, base+'object'))
test('official_initial_exact_request_and_exact_dataset_target', source_and_target_guard)


def final_response_predicate():
    allow = lambda url: url == initial or m.storage_url_in_scope(url)
    assert allow(initial) and allow(base+'object?X-Amz-Signature=invented')
    assert not allow(initial.replace('357303', '999999')) and not allow('https://evil.example/object')
test('final_HTTP200_origin_accepts_initial_or_same_dataset_only', final_response_predicate)


def unchanged_collector():
    old = ast.parse((H/'public_batch.py').read_bytes())
    new = ast.parse(code)
    a = {n.name: ast.dump(n, include_attributes=False) for n in old.body if isinstance(n, (ast.FunctionDef,ast.ClassDef))}
    b = {n.name: ast.dump(n, include_attributes=False) for n in new.body if isinstance(n, (ast.FunctionDef,ast.ClassDef))}
    assert [k for k in a if k in b and a[k] != b[k]] == ['download']
    assert a['collect'] == b['collect']
test('collector_and_existing_nondownload_function_AST_unchanged', unchanged_collector)

receipt = {'completed': True, 'PASS': True, 'groups': len(results), 'tests': results,
           'code_sha256': hashlib.sha256(code).hexdigest(), 'code_size_bytes': len(code),
           'only_invented_URLs_used': True, 'network_or_payload_access': 0,
           'real_signed_query_values_seen_or_stored': False}
with (H/'INVENTED_REDIRECT_SCOPE_CHECKS_V2.json').open('x') as f:
    json.dump(receipt, f, ensure_ascii=False, sort_keys=True, indent=2);f.write('\n')
print(json.dumps(receipt, sort_keys=True))
