import sys
import numpy
import pyDOE
from ObjectiveFunctions import *
from copy import deepcopy
from itertools import permutations
import os
import shutil
import json


out = "/home/igangga/CollaborativeAlgorithms/ProblemSet1/GD"
if not os.path.isdir(out):
    os.makedirs(out)
    os.mkdir(out + "/C-GD-I")
    os.mkdir(out + "/C-GD-II")
    os.mkdir(out + "/NC-GA")

coop = str(sys.argv[1]) == "True"   # This argument will decide to run either the collaborative or the non-collaborative GD.
coop_str = str(sys.argv[2])         # This argument will decide which version of collaborative GD to run.
run = int(sys.argv[3])

if coop:
    if coop_str == "jump":
        out = out + "/C-GD-I"
    elif coop_str == "swap":
        out = out + "/C-GD-II"
else:
    out = out + "/NC-GD"
out = out + "/Run" + str(run)

nv = 2
np = 9
list_OF = [ModRas,
           ModRas,
           ModRas,
           ModRas,
           ModRas,
           ModRas,
           ModRas,
           ModRas,
           ModRas]
list_OF_d = [ModRas_d,
             ModRas_d,
             ModRas_d,
             ModRas_d,
             ModRas_d,
             ModRas_d,
             ModRas_d,
             ModRas_d,
             ModRas_d]
A = 10.0
mR = 1.0
mv = [[10.0,  10.0],
      [10.0,   7.5],
      [10.0,   5.0],
      [10.0,   2.5],
      [10.0,   0.0],
      [10.0,  -2.5],
      [10.0,  -5.0],
      [10.0,  -7.5],
      [10.0, -10.0]]
sv = [[0.000, 0.000],
      [0.125, 0.125],
      [0.250, 0.250],
      [0.375, 0.375],
      [0.500, 0.500],
      [0.625, 0.625],
      [0.750, 0.750],
      [0.875, 0.875],
      [1.000, 1.000]]
list_args = []
for i in range(np):
    args = {}
    args["A"] = A
    args["mR"] = mR
    args["mv"] = mv[i]
    args["sv"] = sv[i]
    list_args.append(args)
lb = [-5.12, -5.12]
ub = [ 5.12,  5.12]
max_lf = 0.1
max_itr = 10

numpy.random.seed(run)
init_vars = []
for i in range(nv):
    lhs = pyDOE.lhs(1, samples=np)
    lhs = lhs.flatten()
    init = lhs * (ub[i] - lb[i]) + lb[i]
    init = init.tolist()
    init_vars.append(init)


class Indv(object):
    def __init__(self, itr, cid, vars, ofvs, stat, src):
        self.itr = itr
        self.cid = cid
        self.vars = vars
        self.ofvs = ofvs
        self.stat = stat
        self.src = src


# Initialization
wrld = []
for i in range(np):
    itr = 0
    cid = i
    vars = []
    for j in range(nv):
        vars.append(init_vars[j][i])
    ofvs = []
    for j in range(np):
        OF = list_OF[j]
        args = list_args[j]
        ofvs.append(OF(vars, args))
    stat = "init"
    src = -1
    wrld.append(Indv(itr, cid, vars, ofvs, stat, src))
wrld_hst = []
for i in range(np):
    wrld_hst.append([deepcopy(wrld[i])])
best_hst = deepcopy(wrld_hst)
shrng_hst = []
for i in range(np):
    shrng_hst.append([])


def update_itr(itr):
    for i in range(np):
        indv = wrld[i]
        indv.itr = itr


def check_feasibility(vars):
    decision = True
    for i in range(nv):
        if lb[i] <= vars[i] <= ub[i]:
            decision = decision and True
        else:
            decision = decision and False
    return decision


def update_vars(indv, p):
    OF = list_OF[p]
    OF_d = list_OF_d[p]
    args = list_args[p]
    grad = OF_d(indv.vars, args)
    list_lf = numpy.linspace(0, max_lf, num=10001)
    opt_lf = 0
    opt_ofv = OF(indv.vars, args)
    for lf in list_lf:
        vars = (numpy.array(indv.vars) + lf * numpy.array(grad)).tolist()
        if check_feasibility(vars) and OF(vars, args) > opt_ofv:
            opt_lf = lf
            opt_ofv = OF(vars, args)
    indv.vars = (numpy.array(indv.vars) + opt_lf * numpy.array(grad)).tolist()


def update_ofvs(indv):
    ofvs = []
    for i in range(np):
        OF = list_OF[i]
        args = list_args[i]
        ofvs.append(OF(indv.vars, args))
    indv.ofvs = ofvs


def update_stat(indv, stat):
    indv.stat = stat


def record_wrld():
    for i in range(np):
        indv = deepcopy(wrld[i])
        wrld_hst[i].append(indv)


def move():
    for i in range(np):
        indv = wrld[i]
        update_vars(indv, i)
        update_ofvs(indv)
        update_stat(indv, "move")
    record_wrld()


def update_src(indv):
    indv.src = indv.cid


def update_cid(indv, cid):
    indv.cid = cid


def record_shrng():
    for i in range(np):
        indv = deepcopy(wrld[i])
        if indv.src != -1:
            shrng_hst[i].append([indv])
        else:
            shrng_hst[i].append([])


def reset_src():
    for i in range(np):
        indv = wrld[i]
        indv.src = -1


def jump():
    temp = []
    for i in range(np):
        best = deepcopy(wrld[i])
        for j in range(np):
            indv = deepcopy(wrld[j])
            if indv.ofvs[i] > best.ofvs[i]:
                best = indv
        temp.append(best)
    for i in range(np):
        wrld[i] = temp[i]
    for i in range(np):
        indv = wrld[i]
        update_stat(indv, "jump")
        if indv.cid != i:
            update_src(indv)
            update_cid(indv, i)
    record_shrng()
    reset_src()
    record_wrld()


def swap():
    list_scn = list(permutations(range(np)))
    list_tofv = []
    for scn in list_scn:
        tofv = 0
        for i in range(np):
            indv = wrld[scn[i]]
            tofv = tofv + indv.ofvs[i]
        list_tofv.append(tofv)
    best_scn = list_scn[list_tofv.index(max(list_tofv))]
    temp = []
    for i in range(np):
        best = deepcopy(wrld[best_scn[i]])
        temp.append(best)
    for i in range(np):
        wrld[i] = temp[i]
    for i in range(np):
        indv = wrld[i]
        update_stat(indv, "swap")
        if indv.cid != i:
            update_src(indv)
            update_cid(indv, i)
    record_shrng()
    reset_src()
    record_wrld()


def record_best():
    for i in range(np):
        best = deepcopy(wrld[i])
        best_hst[i].append(best)


# Iterative process
for i in range(max_itr):
    update_itr(i + 1)

    if coop and coop_str == "jump":
        jump()

    if coop and coop_str == "swap":
        swap()

    move()

    record_best()


if os.path.isdir(out):
    shutil.rmtree(out)
os.mkdir(out)

log_input = {}
log_input["out"] = out
log_input["coop"] = coop
log_input["coop_str"] = coop_str
log_input["run"] = run
log_input["nv"] = nv
log_input["np"] = np
log_input["list_OF"] = [i.__name__ for i in list_OF]
log_input["list_OF_d"] = [i.__name__ for i in list_OF_d]
log_input["list_args"] = list_args
log_input["lb"] = lb
log_input["ub"] = ub
log_input["max_lf"] = max_lf
log_input["max_itr"] = max_itr
file = open(out + "/log_input.json", "w")
json.dump(log_input, file, indent=4, separators=(",", ": "))
file.close()

log_cases = []
for i in range(len(wrld_hst[0])):
    line = []
    for j in range(np):
        indv_hst = wrld_hst[j]
        indv = indv_hst[i]
        itr = indv.itr
        cid = indv.cid
        vars = indv.vars
        ofv = indv.ofvs[j]
        stat = indv.stat
        src = indv.src
        line.extend([itr, cid])
        line.extend(vars)
        line.extend([ofv, stat, src])
    log_cases.append(line)
pre = "itr,cid"
for i in range(nv):
    pre = pre + ",vars[" + str(i) + "]"
post = "stat,src"
for i in range(np):
    if i == 0:
        hdr = pre + ",ofvs[" + str(i) + "]," + post
    else:
        hdr = hdr + "," + pre + ",ofvs[" + str(i) + "]," + post
file = open(out + "/log_cases.txt", "w")
file.write(hdr + "\n")
for line in log_cases:
    for i in range(len(line)):
        if i == 0:
            file.write(str(line[i]))
        else:
            file.write("," + str(line[i]))
    file.write("\n")
file.close()

log_optimization = []
for i in range(len(best_hst[0])):
    line = []
    for j in range(np):
        indv_hst = best_hst[j]
        indv = indv_hst[i]
        itr = indv.itr
        cid = indv.cid
        vars = indv.vars
        ofv = indv.ofvs[j]
        stat = indv.stat
        src = indv.src
        line.extend([itr, cid])
        line.extend(vars)
        line.extend([ofv, stat, src])
    log_optimization.append(line)
file = open(out + "/log_optimization.txt", "w")
file.write(hdr + "\n")
for line in log_optimization:
    for i in range(len(line)):
        if i == 0:
            file.write(str(line[i]))
        else:
            file.write("," + str(line[i]))
    file.write("\n")
file.close()


def generate_sharing_data(shared_pop):
    sharing_data = [0] * np
    for i in range(len(shared_pop)):
        indv = shared_pop[i]
        src = indv.src
        sharing_data[src] = sharing_data[src] + 1
    return sharing_data


log_sharing = []
for i in range(len(shrng_hst[0])):
    line = []
    for j in range(np):
        indv_hst = shrng_hst[j]
        shared_pop = indv_hst[i]
        itr = i + 1
        cid = j
        sharing_data = generate_sharing_data(shared_pop)
        totl = len(shared_pop)
        line.extend([itr, cid])
        line.extend(sharing_data)
        line.append(totl)
    log_sharing.append(line)
pre = "itr,cid"
for i in range(np):
    if i == 0:
        mid = "src=" + str(i)
    else:
        mid = mid + "," + "src=" + str(i)
post = "totl"
for i in range(np):
    if i == 0:
        hdr = pre + "," + mid + "," + post
    else:
        hdr = hdr + "," + pre + "," + mid + "," + post
file = open(out + "/log_sharing.txt", "w")
file.write(hdr + "\n")
for line in log_sharing:
    for i in range(len(line)):
        if i == 0:
            file.write(str(line[i]))
        else:
            file.write("," + str(line[i]))
    file.write("\n")
file.close()
