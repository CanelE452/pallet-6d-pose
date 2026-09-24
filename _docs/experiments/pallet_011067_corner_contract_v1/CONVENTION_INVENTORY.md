# Convention inventory

|출처|실제 정의|한계|
|---|---|---|
|annotation tool|near/front 0..3, local -Z; X right/Y down; LR image x, TB image y, FR camera depth|LR/TB/FR can admit two front-role candidates; diagnostic area call may clip to image|
|historical converter|physical top/bottom; parallel side-pair max projected-area difference; larger side front; image-x LR|Not proven to be current renderer writer; no independent origin3D in real011067|
|paper physical/CF frame|canonical physical XYZ vs camera-facing frame; proper yaw role transforms; W/D swap at90/270|Rectangular physical yaw180 equivalence is not arbitrary C4 scoring invariance|
|semantics audit|DLT/PnP alone may absorb semantic C4; role and physical symmetry differ|Older two-sample renderer discussion superseded in coverage by later60k audit; not contradiction|
|renderer evidence|60000/60000 stored front = max side-normal dot object-center-to-camera ray|Source evidence, not real011067 authority; writer implementation unavailable|
