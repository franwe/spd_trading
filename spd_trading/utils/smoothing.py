import numpy as np
import scipy.interpolate as interpolate  # B-Spline


def bspline(x, y, sections, degree=3):
    idx = np.linspace(0, len(x) - 1, sections + 1, endpoint=True).round(0).astype("int")
    x = x[idx]
    y = y[idx]

    t, c, k = interpolate.splrep(x, y, s=0, k=degree)
    spline = interpolate.BSpline(t, c, k, extrapolate=True)
    pars = {"t": t, "c": c, "deg": k}
    points = {"x": x, "y": y}
    return pars, spline, points


def interpolate_to_same_interval(x1, y1, x2, y2, interval=[0.5, 1.5]):
    """Interpolates y1 and y2 to the same interval, especially same interval values!

    Args:
        x1 (np.array): X values of first function.
        y1 (np.array): Y values of first function.
        x2 (np.array): X values of second function.
        y2 (np.array): Y values of second function.
        interval (list, optional): Interval in which the functions are interpolated. Defaults to [0.5, 1.5].

    Returns:
        tuple of np.arrays: Interpolated functions y1 and y2 and their common X values.
    """
    _, spline1, _ = bspline(x1, y1, sections=15, degree=2)
    _, spline2, _ = bspline(x2, y2, sections=15, degree=2)
    X = np.linspace(interval[0], interval[1], 100)

    Y1 = spline1(X)
    Y2 = spline2(X)
    return Y1, Y2, X


def remove_tails(y_base, y_other, X, cut_tail_percent):
    """Cuts the tails of y_base if they are having a value, less than `cut_tail_percent`.

    Also adjusts y_other and X to the new length of y_base.

    Args:
        y_base (np.array): The reference function. The tails, that are smaller than `cut_tail_percent` will be
            cut off
        y_other (np.array): Another function that will be cut at the same places as the reference function.
        X (np.array): The X values of the functions. Need to be cut also.
        cut_tail_percent (float): Percent threshold of when to cut off tails.

    Returns:
        tuple of np.arrays: The shortened arrays in the same order as input.
    """
    mask = y_base > y_base.max() * cut_tail_percent
    return y_base[mask], y_other[mask], X[mask]
