# V258 parser-only V2

`rgbd_geometry_v2.py` preserves the V1 geometry API and every non-parser function AST. Only `parse_obj_text` now accepts standard `v x y z`, homogeneous `v x y z 1`, and Meshlab `v x y z r g b`. Every numeric value must be finite; RGB must lie within0..1 and is ignored for geometry. The first3 XYZ values are preserved exactly. RGBA is explicitly unsupported. No coordinate/scale/triangulation/render/pose/depth/threshold change occurs.

Root's actual V1 parser failure and all original code/toys remain unchanged. The exact authorized non-frame OBJ was SHA/size checked and its27598 vertex lines all had6 numeric fields, with finite RGB within0..0.905882. No frame data was read by C.

`test_geometry_v2.py` retains the original10 analytic tests and adds xyz/xyz1/xyzRGB exactXYZ/depth equality, invalid color and RGBA rejection.11 groups PASS, terminal e7a190 exit0, NumPy2.3.5. The original deliberate synthetic NaN→uint16 rejection test still emits its disclosed RuntimeWarning. No real frame/model was involved in the toy.

API remains `parse_obj(path,binding=None) -> vertices,faces,parser_report`, `render_depth(v,f,T,K,hw) -> depth,support,render_report`, `describe_observed_depth(observed_m,rendered_m,mask_bool) -> stats,raw_residual`, `diagnostic_pair(...) -> arrays,report`. See original README_KO.md for the fixed+Z/integer-center/reciprocal-depth/near-crossing policy. Use V2 code binding in a fresh Root actual entry/output; this file grants no additional frame access or label admission.
