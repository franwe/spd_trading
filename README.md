# spd_trading

This package estimates the Risk Neutral Density (RND) and Historical Density (HD) of an underlying and suggests a trading 
strategy based on the Pricing Kernel:

<a href="https://www.codecogs.com/eqnedit.php?latex=K=\frac{RND}{HD}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?K=\frac{RND}{HD}" title="K=\frac{RND}{HD}" /></a>


The **RND** is estimated by *Rookley's Method*, which uses the option table of one trading day. ``risk_neutral_density.Calculator.get_rnd``

The **HD** is estimated by a *GARCH(1,1) Model*, which uses a timeseries of the underlying. ``historical_density.Calculator.get_hd``

The package is part of a Master Thesis, which will be published after grading [1]_. The thesis explains the theoretical background 
in more detail and gives more references. Furthermore an actual trading strategy was implemented and backtested on real data 
(BTC options March-September 2019). 

The concious desicion of *not* implementing the actual strategy in the package is due to the high responsibility that would come 
with publishing such a risky tool. However, the construction of strategies based on the kernels are explained and analyized in the 
thesis as well. 

## Installation

Via pip
```
    pip install spd_trading
```

Or via download from git:

```
    pip install git+https://github.com/franwe/localpoly#egg=spd-trading
```

Note that in order to avoid potential conflicts with other packages it is strongly recommended to use a virtual environment (venv) or a conda environment.

![See image in README.md in GitHub repo.](https://github.com/franwe/spd_trading/blob/main/docs/_static/example_kernel.png?raw=true)