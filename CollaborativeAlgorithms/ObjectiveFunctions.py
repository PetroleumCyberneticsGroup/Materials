import math
import autograd.numpy as np
from autograd import grad


# Modified Rastrigin function
def ModRas(vars, args):
    A = args["A"]
    mR = args["mR"]
    mv = args["mv"]
    sv = args["sv"]
    nv = len(vars)
    val = A * nv
    for i in range(nv):
        val = val + ((vars[i] + sv[i]) ** 2 - A * np.cos(2 * math.pi * (vars[i] + sv[i])))
    val = mR * (- val)
    for i in range(nv):
        val = val + mv[i] * (vars[i] + sv[i])
    return val


# Derivatives of Modified Rastrigin function
ModRas_d = grad(ModRas)


# Linear function
def Linear(vars, args):
    mv = args["mv"]
    nv = len(vars)
    val = 0
    for i in range(nv):
        val = val + mv[i] * vars[i]
    return val
