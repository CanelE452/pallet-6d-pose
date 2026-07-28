import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.fit_fwd_model import d_of_T, fit_fwd, t_of_d


def test_d_of_T_piecewise_shape():
    # t0=0, t1=4, a=0.075 → vmax=0.3, d_acc=0.6
    assert d_of_T(0.0, t0=0.0, t1=4.0, a=0.075) == 0.0          # 정지
    assert abs(d_of_T(2.0, t0=0.0, t1=4.0, a=0.075) - 0.15) < 1e-9   # 가속: 0.5*0.075*4
    assert abs(d_of_T(4.0, t0=0.0, t1=4.0, a=0.075) - 0.6) < 1e-9    # d_acc
    assert abs(d_of_T(6.0, t0=0.0, t1=4.0, a=0.075) - (0.6 + 0.3*2)) < 1e-9  # 정속


def test_fit_recovers_params():
    true = dict(t0=-0.02, t1=4.2, a=0.072)
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0])   # 가속~정속 분포
    d = np.array([d_of_T(t, **true) for t in T])
    rng = np.random.default_rng(0)
    d_noisy = d + rng.normal(0, 0.002, size=d.shape)           # mm급 노이즈
    res = fit_fwd(T, d_noisy)
    assert abs(res["a"] - true["a"]) < 0.01
    assert abs(res["t1"] - true["t1"]) < 0.3
    assert res["r2"] > 0.99


def test_t_of_d_is_inverse():
    # t(d)는 d(T)의 역함수여야 한다 (정속 구간 점)
    p = dict(t0=-0.02, t1=4.2, a=0.072)
    d = 1.5
    t = t_of_d(d, **p)
    assert abs(d_of_T(t, **p) - d) < 1e-6
