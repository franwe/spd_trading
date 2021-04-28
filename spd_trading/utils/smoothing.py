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


def interpolate_to_same_interval(curve_1, curve_2, interval=[0.5, 1.5]):
    """Interpolates curve_1 and curve_2 to the same interval, especially same interval values!

    Args:
        curve_1 (dict):  dict of np.arrays "x" and "y"
        curve_2 (dict):  dict of np.arrays "x" and "y"
        interval (list, optional): Interval in which the functions are interpolated. Defaults to [0.5, 1.5].

    Returns:
        tuple of np.arrays: Interpolated curves curve_1 and curve_2 to new interval.
    """
    _, spline1, _ = bspline(curve_1["x"], curve_1["y"], sections=15, degree=2)
    _, spline2, _ = bspline(curve_2["x"], curve_2["y"], sections=15, degree=2)
    X = np.linspace(interval[0], interval[1], 100)

    new_curve_1 = {"x": X, "y": spline1(X)}
    new_curve_2 = {"x": X, "y": spline2(X)}
    return new_curve_1, new_curve_2


def remove_tails(base, other, cut_tail_percent):
    """Cuts the tails of y_base if they are having a value, less than `cut_tail_percent`.

    Important: base and other need to have same domain values!
    Also adjusts y_other and X to the new length of y_base.

    Args:
        base (dict): The reference function. The tails, that are smaller than `cut_tail_percent` will be
            cut off
        other (dict): Another function that will be cut at the same places as the reference function.
        cut_tail_percent (float): Percent threshold of when to cut off tails.

    Returns:
        tuple of np.arrays: The shortened arrays in the same order as input.
    """

    mask = base["y"] > base["y"].max() * cut_tail_percent
    new_base = {"x": base["x"][mask], "y": base["y"][mask]}
    new_other = {"x": other["x"][mask], "y": other["y"][mask]}
    return new_base, new_other
