import numpy as np

def battery_step(soc, u, price, params):
    """
    used for all battery physics & applications

    u > 0 : discharge
    u < 0 : charge
    eta applies when charging only

    """
    eta = params["eta"]
    S_max = params["S_max"]
    dt = params["dt"]

    revenue = u * price * dt

    if u > 0:
        soc_next = soc - u * dt
    elif u < 0:
        soc_next = soc + eta * abs(u) * dt
    else:
        soc_next = soc

    soc_next = np.clip(soc_next, 0, S_max)

    return soc_next, revenue
    