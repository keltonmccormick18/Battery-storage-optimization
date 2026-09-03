We apply the concepts of stochastic optimal control to a simple battery storage problem, with the goal of building a flexible system that works in two very different markets -- CISO and NYISO.

With the system being as follows:
$$dX_t = \theta(\mu-X_t)dt + \sigma dW_t$$, and
$$dS_t =(-u_t^+ + \eta u_t^-)dt$$,
running reward:
$$w = u_t [f(t) + X_t] - c_{deg}|u_t|$$ 
$u_t > 0$ selling(discharging)
$f(t)$ seasonal component + $X_t$ current (non-seasonal) price  = total price.., $c_{deg}$ per-mwh deg cost.
Terminal cost $z(X_T) = avg(P) \cdot S_T$

$\mathbb{U} = [\alpha,\beta]$

Gives us the HJB equation: 
$$0 = \frac{\partial V}{\partial t} + \theta (\mu-X) \frac{\partial V} {\partial X} + \frac{1}{2} \sigma^2 \frac {\partial ^2 V}{\partial X^2} + \sup_{u\in[\alpha,\beta]}H(t,X,S,u)$$

where the Hamiltonian $H$ is 
$$H = w + \frac{\partial V}{\partial S}(-u^+ + \eta u^-)$$

And now, we can see that the Hamiltonian is piecewise linear in $u$, so the optimal control at each point is bang-bang. Hence, we can numerically solve this dynamic programming loop according the the three possible decisions at each state:

If $u > 0$ and $H > 0$; $u^* = \beta$;
If $u < 0$ and $H < 0$ (??); $u^* = \alpha$;
Otherwise, 0.

And the model assumes no market impact from our actions, which is reasonable in our personal situation but may be susceptible to scaling.

Details:
