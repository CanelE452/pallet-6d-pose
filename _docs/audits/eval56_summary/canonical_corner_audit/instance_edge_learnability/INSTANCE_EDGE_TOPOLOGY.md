# Twelve-edge topology

Derived from `annotate_pnp.make_pallet_keypoints_3d` by the rule that a physical edge is a corner pair differing along exactly one local axis.  No index is written by hand.

```
id      corners         class
-----------------------------
0        (0, 1)     top_width
1        (0, 3)      vertical
2        (0, 4)     top_depth
3        (1, 2)      vertical
4        (1, 5)     top_depth
5        (2, 3)    base_width
6        (2, 6)    base_depth
7        (3, 7)    base_depth
8        (4, 5)     top_width
9        (4, 7)      vertical
10       (5, 6)      vertical
11       (6, 7)    base_width

class counts  {"top_width": 2, "top_depth": 2, "base_width": 2, "base_depth": 2, "vertical": 4}
polarity pairs [[0, 5], [2, 7], [4, 6], [8, 11]]
sha256        9c0aafa1292eba3e844f429bc59e852b7950f155d1eca725d9cd79fa37d8103d
```
