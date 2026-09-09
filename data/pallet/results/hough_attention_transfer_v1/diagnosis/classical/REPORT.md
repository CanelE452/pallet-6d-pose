# Fixed classical edge-candidate diagnostic

This is GT-oracle candidate coverage, not semantic detection accuracy.

| Method | Available GT roles | Oracle matched | Coverage | Median candidates/frame |
|---|---:|---:|---:|---:|
| HoughLinesP | 622 | 213 | 34.2% | 975.0 |
| LSD | 622 | 302 | 48.6% | 992.5 |

Canny 50/150; HoughLinesP rho=1px, theta=1deg, threshold=30, minimum segment length=15px, maximum gap=5px. No tuning.
Oracle gate: angle <=5deg; mean full-GT-endpoint distance <=1% image diagonal; candidate projection overlaps >=25% of the image-clipped GT segment.

- GT-oracle candidate availability is not deployed semantic detection precision or model accuracy.
- There is no learned role association, and one candidate can cover multiple GT roles.
- Background lines can match a GT line accidentally within the tolerance.
- Cuboid supporting lines can be occluded or geometrically virtual, with no physical visible edge.
- Failure of these fixed classical settings does not prove that the image lacks usable information.
- LSD uses its native default scale/refinement and no Hough-specific length filter; compare as separate candidate generators.
- Original full GT endpoints determine line distance; overlap uses only the GT segment clipped into the original image.

Predetermined panels (one SHA256-smallest ID per session):

- [eval_outside__1778651618049498624](panels/eval_outside__011d6288a979.png)
- [eval_noapril__1775201430551447040](panels/eval_noapril__07b09b708519.png)
- [eval_cad__1778653017736058368](panels/eval_cad__02be94939f55.png)
