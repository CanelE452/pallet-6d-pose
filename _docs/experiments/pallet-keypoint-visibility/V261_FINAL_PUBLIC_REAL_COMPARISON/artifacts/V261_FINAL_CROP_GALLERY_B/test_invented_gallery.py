"""Invented rows/RGB only: fixed cells, same-q crop, no GT recentering."""
import hashlib,json,sys,types
from pathlib import Path
import numpy as np
H=Path(__file__).absolute().parent

def run():
    m=types.ModuleType('v261_gallery_invented');m.__file__=str(H/'build_final_gallery.py');sys.modules[m.__name__]=m
    exec(compile(Path(m.__file__).read_bytes(),m.__file__,'exec'),m.__dict__)
    ledger=[]
    for i in range(1784):
        human=m.D if i<1147 else m.E if i<1237 else'UNKNOWN';truth='D'if human==m.D else'E'
        ledger.append({'global_index':str(i),'frame_index':str(i//8),'native_corner':str(i%8),'row_id':f'INVENTED_{i:04d}','human_state':human,'paired_same_query_RGB_D_vs_E':str(i<1237),'group':m.GROUPS[0]if i<840 else m.GROUPS[1],m.TREAT+'_D_E_U':truth if i%2 else('E'if truth=='D'else'D'),m.CONTROL+'_D_E_U':'D','v61_D_E_U':'D','annotation_coordinate_source':'manual_click'if i<376 else'UNKNOWN','initial_reference_error_px':'2'if i<376 else'5','original_query_valid':'True'})
    a=m.select_cases(ledger);b=m.select_cases(list(ledger))
    assert a==b and len(a['slots'])==12 and a['primary_N']==1237 and a['manual_click_LE3_ledger_derived_count']==376
    assert a['empty_cells_preserved']==4
    for cell in a['slots']:
        if not cell['empty']:assert cell['selected_hash']==hashlib.sha256((m.SALT+cell['selected_row_id']).encode()).hexdigest()
    ns={'np':np};import ast
    raw=m.checked(m.CROP,[m.CROP]);node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name=='crop');exec(compile(ast.Module(body=[node],type_ignores=[]),m.CROP['path'],'exec'),ns)
    rgb=np.arange(128*128*3,dtype=np.uint32).reshape(128,128,3).astype(np.uint8)
    row={'query_x':'63.5','query_y':'63.5','roi_side':'96','human_reference_x':'70','human_reference_y':'63.5','original_query_valid':'True',m.TREAT+'_U_reason':''}
    original,marked,fact=m.render_case(rgb,row,ns['crop'])
    assert np.array_equal(np.asarray(original),rgb[16:112,16:112])and fact['q_RAW96_xy']==[47.5,47.5]and fact['reference_RAW96_xy_annotation_only']==[54.,47.5]
    moved=dict(row,human_reference_x='110');other,_,_=m.render_case(rgb,moved,ns['crop']);assert np.array_equal(np.asarray(original),np.asarray(other))
    assert marked.size==(384,384)and fact['no_recrop_or_snap']
    try:m.run('/V112_FORBIDDEN_INVENTED')
    except ValueError:pass
    else:raise AssertionError('Forbidden path accessed')
    return{'completed':True,'PASS':True,'checks':['Fixed12cells/hashmin/all4empties retained, manual-click LE3 extraD only','RAW96 equals original sameq/L crop, unchanged by moving reference marker; q47.5','Forbidden lexical path before filesystem'],'actual_current_GT_RGB_predictions_or_model_reads':0,'model_forwards':0,'optimizer_updates':0}

if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,sort_keys=True,indent=2))
