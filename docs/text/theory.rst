Theoretical Background
================================================================

In the setting of a perfect financial market, 
the arbitrage free price of an option is derived from its expected payoff at maturity :math:`T` (under the risk neutral 
measure :math:`\mathbb{Q}`), discounted to the time of interest :math:`t` via the riskfree interest rate :math:`r` that one would get if 
invested in the underlying directly. [1]_ 

.. math::
    \begin{equation*}
        \psi_t = \mathbb{E}_\mathbb{Q}\left[ \psi_T e^{-r\tau}\rvert \mathcal{F}_t \right] 
            = e^{-r\tau} \int_{0}^{\infty} \psi_T(S_T) q_{t}(S_T) \,dS_T
    \end{equation*}

Where we call :math:`q_{t}(S_T)` the Risk Neutral Density (RND) or State Price Density of the underlying :math:`S_T` at time :math:`t`,
which is derived from the option prices of the underlying in the market under the risk neutral measure :math:`\mathbb{Q}`. 

Under the physical measure :math:`\mathbb{P}`, the "true" measure, the price of the option can not easily be expressed in the 
same way, since the discounted price process is not a martingale under this measure. 

The Radon-Nykodym Lemma which evolves from the more general Girsanov Theorem states,
how a change from :math:`\mathbb{P}` to :math:`\mathbb{Q}` is possible:

Let :math:`\mathbb{P}` and 
:math:`\mathbb{Q}` be equivalent measures and :math:`X_t` and :math:`\mathcal{F}_t`-adapetd process, then

.. math::
    \begin{equation*}
        \mathbb{E}_\mathbb{Q}\left[ X_t \right] = \mathbb{E}_\mathbb{P}\left[ \frac{d \mathbb{Q}}{d \mathbb{P}} X_t \right] 
    \end{equation*}

where we define the Radon-Nikodym derivative of :math:`\mathbb{Q}` with respect to :math:`\mathbb{P}` to be 
:math:`K_t := \mathbb{E}_\mathbb{P}\left[ \frac{d \mathbb{Q}}{d \mathbb{P}} \rvert \mathcal{F}_t  \right]`. 

The price of the option can now be expressed via the physical density :math:`\mathbb{P}` as:

.. math::
    \begin{equation*}
        \psi_t = \mathbb{E}_\mathbb{P}\left[ \psi_T K_{t}\rvert \mathcal{F}_t \right] 
            = \int_{0}^{\infty} \psi_T(S_T) K_{t}(S_T) p_{t}(S_T) \,dS_T
    \end{equation*}

Where we call :math:`p_{t}(S_T)` the Historical Density (HD) or Empirical Density of the underlying :math:`S_T` at time :math:`t`, which
is dericed from the timeseries of historical values of the underlying itself. 

Risk Neutral Density 
----------------------------------------------------------------
The Risk Neutral Density (RND) is derived from the options that are offered on the market. On one trading day, options 
with different maturity :math:`T` are offered. A set of options with the same maturity consists of options with different 
strike prices :math:`K` (and therefore different price of option, different implied volatility). 

These information are sufficient to use a semi-parametric approach to estimate the Risk Neutral Density. 
It uses the results of Breeden and Litzenberger [2]_, where Arrow-Debreu prices are 
replicated using butterfly spreads on Europeak options.
\mytodo{add how the replication strategy works?}

As an end result they find:

.. math::
    \begin{equation}
        q_t^*(S_T) = e^{r_{t,\tau}\tau} \frac{\partial^2C_t(\cdot)}{\partial K^2}\Big|_{K=S_T}
    \end{equation}\label{equation:breeden_litzenberger}
    
**Rookley's Method**

Rookley found an analytical solution to the equation above by representing the call-price by the Black-Scholes call
option pricing formula [3]_. 

.. math::
    \begin{align*}
        C_{BS}(S_t, K, \tau, r_{t,\tau}, \delta_{t,\tau}, \sigma) 
            &= S_t e^{-\delta_{t,\tau}\tau} \Phi(d_1)
                - K e^{-r_{t,\tau}\tau}\Phi(d_2) \\
        d_1 &= \frac{log(S_t/K) + (r_{t,\tau} - \delta_{t,\tau} + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau} }\\
        d_2 &= d_1 - \sigma\sqrt{\tau}  
    \end{align*}

The analytical solution is quite lengthy, therefore the equations are not listed here. 
One crucial point is, that in order to use Rookely's Method, we need the volatility smile and its first and second derivatives 
:math:`(\sigma(M), \frac{\partial \sigma(M)}{\partial M}, \frac{\partial^2 \sigma(M)}{\partial M^2})`. 
In practice, we will estimate them with Local Polynomial Regression (install package: ``localpoly`` (documentation_))

| The algorithm to estimate the RND:
| step 0 : fit iv-smile to iv-over-M option values
| step 1 : calculate spd for every option-point “Rookley’s method” 
| step 2 : Rookley results (points in K-domain) - fit density curve
| step 3 : transform density POINTS from K- to M-domain
| step 4 : density points in M-domain - fit density curve
| (All fits are obtained by Local Polynomial Regression)

The steps of the algorithm are shown in the following graphic. 

.. image:: ../_static/example_rnd.png

Historical Density
----------------------------------------------------------------

**GARCH(1,1) Model**

The Historical Density (HD) is derived from the timeseries of historical returns of the underlying. 
A simple but sufficient model is the GARCH(1,1) model, since it accounts for the heteroscedasticity of the timeseries 
[4]_.

We define the series of log returns as:

.. math::
    \begin{equation*}
        X_t = \log(S_{t+1}) - \log(S_t)
    \end{equation*}

where :math:`S_t` is the value of the underlying at time :math:`t`. 
In the GARCH(1,1) model, the log return :math:`X_t` is expressed as the time-varying variance parameter :math:`\sigma_t^2` 
multiplied by a i.i.d random innovation :math:`z_t`, the random shock. 

The variance parameter itself is a weighted combination of the previous variance and the previous return. It therefore 
is called conditional variance, since it depends on all previous returns. 

The GARCH(1,1) model has the following form:

.. math::
    \begin{align}
        X_t &= z_t \sigma_t\\
        \sigma_t^2 &= \omega + \alpha X_{t-1}^2 + \beta \sigma_{t-1}^2
    \end{align}\label{equation:GARCH}

After fitting the model to the data, the model is used to simulate many potential paths of 
the underlying. The HD is obtained by Kernel Density Estimation on these paths. 

Trading on Differences between RND and HD
----------------------------------------------------------------

A Pricing Kernel can be calculated, which is used to determine trading strategies. 

The Kernel is the quotient of Risk Neutral Density divided by Historical Density and therefore represents 
discrepancies in the future's price distribution by the market (RND - red curve) vs. the historical data (HD - blue curve) [5]_, [6]_:
    
.. math::
    \begin{align*}
        K = \frac{RND}{HD}, \,\, &K > 1 \text{ option overpriced} \rightarrow \text{sell options (blue area)} \\
                                &K < 1 \text{ option underpriced} \rightarrow \text{buy options (red area)}
    \end{align*}

.. image:: ../_static/example_kernel.png
    :height: 250
    :align: center

Resources:
----------

.. [1] The Valuation of Options for Alternative Stochastic Processes - Cox, John C. and Ross, Stephen A. - 1976 
.. [2] Prices of State-Contingent Claims Implicit in Option Prices - Breeden, Douglas T. and Litzenberger, Robert H. - 1978 
.. [3] Option Pricing and Risk Management - Rookley, Cameron - 1997
.. [4] Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series: An Extreme Value Approach - McNeil, Alexander J. and Frey, Ruediger - 2000
.. [5] Do Option Markets Correctly Price the Probabilities of Movement of the Underlying Asset? - Aı̈t-Sahalia, Y. Wang, and F. Yared - 2001
.. [6] “Trading on Deviations of Implied and Historical Densities” in Applied Quantitative Finance  - K. W. Haerdle, O. J. Blaskowitz and P. Schmidt - 2002
.. _documentation: https://localpoly.readthedocs.io/en/latest/index.html