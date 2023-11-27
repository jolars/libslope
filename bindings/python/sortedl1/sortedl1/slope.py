import _sortedl1 as sl1
import numpy as np


def slope(x, y, lam, alph):
    x = np.array(x)
    y = np.array(y)

    lam = np.array(lam)
    alpha = np.array(alph)

    res = sl1.fit_slope(x, y, lam, alpha)

    return res
