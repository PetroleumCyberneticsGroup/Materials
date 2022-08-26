import sys
import numpy
import pyDOE
from ObjectiveFunctions import *
from copy import deepcopy
import os
import shutil
import json


out = "/home/igangga/CollaborativeAlgorithms/ProblemSet1/PSO"
if not os.path.isdir(out):
    os.makedirs(out)
    os.mkdir(out + "/C-PSO")
    os.mkdir(out + "/NC-PSO")

coop = str(sys.argv[1]) == "True"   # This argument will decide to run either the collaborative or the non-collaborative PSO.
run = int(sys.argv[2])

if coop:
    out = out + "/C-PSO"
else:
    out = out + "/NC-PSO"
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
fb = 0.5
N = 2 * nv
w = 0.5
cp = 2.0
cg = 2.0
vs = 0.5
max_itr = 10

numpy.random.seed(run)
init_vars = []
for i in range(nv):
    lhs = pyDOE.lhs(1, samples=(np * N))
    lhs = lhs.flatten()
    lhs = lhs.tolist()
    init = []
    for j in range(np):
        cntry = []
        for k in range(N):
            for l in range(len(lhs)):
                rand = lhs[l]
                if k / N <= rand < (k + 1) / N:
                    cntry.append(rand * (ub[i] - lb[i]) + lb[i])
                    lhs.remove(rand)
                    break
        numpy.random.shuffle(cntry)
        init.append(cntry)
    init_vars.append(init)

vmax = vs * (numpy.array(ub) - numpy.array(lb))
vmax = vmax.tolist()
init_vels = []
for i in range(nv):
    init = []
    for j in range(np):
        cntry = []
        for k in range(N):
            rand = numpy.random.rand()
            cntry.append(- vmax[i] + rand * 2 * vmax[i])
        init.append(cntry)
    init_vels.append(init)


class Indv(object):
    def __init__(self, itr, cid, vels, vars, ofvs, stat, src):
        self.itr = itr
        self.cid = cid
        self.vels = vels
        self.vars = vars
        self.ofvs = ofvs
        self.stat = stat
        self.src = src


# Initialization
wrld = []
for i in range(np):
    cntry = []
    for j in range(N):
        itr = 0
        cid = i
        vels = []
        for k in range(nv):
            vels.append(init_vels[k][i][j])
        vars = []
        for k in range(nv):
            vars.append(init_vars[k][i][j])
        ofvs = []
        for k in range(np):
            OF = list_OF[k]
            args = list_args[k]
            ofvs.append(OF(vars, args))
        stat = "init"
        src = -1
        cntry.append(Indv(itr, cid, vels, vars, ofvs, stat, src))
    wrld.append(cntry)
pbest = deepcopy(wrld)
gbest = []
for i in range(np):
    cntry = wrld[i]
    best = deepcopy(cntry[0])
    for j in range(N):
        indv = deepcopy(cntry[j])
        if indv.ofvs[i] > best.ofvs[i]:
            best = indv
    gbest.append(best)
wrld_hst = deepcopy(wrld)
best_hst = []
for i in range(np):
    best = deepcopy(gbest[i])
    best_hst.append([best])
shrng_hst = []
for i in range(np):
    shrng_hst.append([])


def update_pbest():
    for i in range(np):
        for j in range(N):
            if wrld[i][j].ofvs[i] > pbest[i][j].ofvs[i]:
                pbest[i][j] = deepcopy(wrld[i][j])


def list_cndts(p):
    cndts = []
    for i in range(np):
        if i != p:
            cntry = wrld[i]
            for j in range(N):
                indv = cntry[j]
                cndts.append(deepcopy(indv))
    return cndts


def sort(pop, p):
    tuples = []
    for i in range(len(pop)):
        indv = pop[i]
        ofv = indv.ofvs[p]
        tuples.append((indv, ofv))
    sorted_tuples = sorted(tuples, key=lambda item: item[1], reverse=True)
    sorted_pop = []
    for i in range(len(pop)):
        indv = sorted_tuples[i][0]
        sorted_pop.append(deepcopy(indv))
    return sorted_pop


def improve_gbest():
    for i in range(np):
        cndts = list_cndts(i)
        sorted_cndts = sort(cndts, i)
        best = sorted_cndts[0]
        if best.ofvs[i] > gbest[i].ofvs[i]:
            gbest[i] = deepcopy(best)


def update_src():
    for i in range(np):
        indv = gbest[i]
        if indv.cid != i:
            indv.src = indv.cid


def update_cid():
    for i in range(np):
        indv = gbest[i]
        if indv.cid != i:
            indv.cid = i


def record_shrng():
    for i in range(np):
        indv = deepcopy(gbest[i])
        if indv.src != -1:
            shrng_hst[i].append([indv])
        else:
            shrng_hst[i].append([])


def reset_src():
    for i in range(np):
        indv = gbest[i]
        indv.src = -1


def update_gbest():
    for i in range(np):
        cntry = wrld[i]
        best = deepcopy(cntry[0])
        for j in range(N):
            indv = deepcopy(cntry[j])
            if indv.ofvs[i] > best.ofvs[i]:
                best = indv
        if best.ofvs[i] > gbest[i].ofvs[i]:
            gbest[i] = deepcopy(best)
    if coop:
        improve_gbest()
    update_src()
    update_cid()
    record_shrng()
    reset_src()


def update_itr(itr):
    for i in range(np):
        cntry = wrld[i]
        for j in range(N):
            indv = cntry[j]
            indv.itr = itr


def update_vels():
    for i in range(np):
        cntry = wrld[i]
        for j in range(N):
            indv = cntry[j]
            indv.vels = w * numpy.array(indv.vels)
            r = []
            for k in range(nv):
                r.append(numpy.random.rand())
            indv.vels = indv.vels + cp * numpy.array(r) * (numpy.array(pbest[i][j].vars) - numpy.array(indv.vars))
            r = []
            for k in range(nv):
                r.append(numpy.random.rand())
            indv.vels = indv.vels + cg * numpy.array(r) * (numpy.array(gbest[i].vars) - numpy.array(indv.vars))
            indv.vels = indv.vels.tolist()


def ensure_vels_feasibility():
    for i in range(np):
        cntry = wrld[i]
        for j in range(N):
            indv = cntry[j]
            for k in range(nv):
                if indv.vels[k] < - vmax[k]:
                    indv.vels[k] = - vmax[k]
                elif indv.vels[k] > vmax[k]:
                    indv.vels[k] = vmax[k]


def update_vars():
    for i in range(np):
        cntry = wrld[i]
        for j in range(N):
            indv = cntry[j]
            indv.vars = numpy.array(indv.vars) + numpy.array(indv.vels)
            indv.vars = indv.vars.tolist()


def ensure_vars_feasibility():
    for i in range(np):
        cntry = wrld[i]
        for j in range(N):
            indv = cntry[j]
            for k in range(nv):
                if indv.vars[k] < lb[k]:
                    indv.vars[k] = lb[k] + fb * abs(indv.vars[k] - lb[k])
                elif indv.vars[k] > ub[k]:
                    indv.vars[k] = ub[k] - fb * abs(indv.vars[k] - ub[k])


def update_stat(stat):
    for i in range(np):
        cntry = wrld[i]
        for j in range(N):
            indv = cntry[j]
            indv.stat = stat


def update_ofvs():
    for i in range(np):
        cntry = wrld[i]
        for j in range(N):
            indv = cntry[j]
            ofvs = []
            for k in range(np):
                OF = list_OF[k]
                args = list_args[k]
                ofvs.append(OF(indv.vars, args))
            indv.ofvs = ofvs


def record_wrld():
    for i in range(np):
        cntry = deepcopy(wrld[i])
        wrld_hst[i].extend(cntry)


def record_best():
    for i in range(np):
        cntry = wrld[i]
        best = deepcopy(cntry[0])
        for j in range(N):
            indv = deepcopy(cntry[j])
            if indv.ofvs[i] > best.ofvs[i]:
                best = indv
        if gbest[i].ofvs[i] > best.ofvs[i]:
            best = deepcopy(gbest[i])
        best_hst[i].append(best)


def move():
    update_vels()
    ensure_vels_feasibility()
    update_vars()
    ensure_vars_feasibility()
    update_stat("move")
    update_ofvs()
    record_wrld()
    record_best()


# Iterative process
for i in range(max_itr):
    update_pbest()

    update_gbest()

    update_itr(i + 1)

    move()


if os.path.isdir(out):
    shutil.rmtree(out)
os.mkdir(out)

log_input = {}
log_input["out"] = out
log_input["coop"] = coop
log_input["run"] = run
log_input["nv"] = nv
log_input["np"] = np
log_input["list_OF"] = [i.__name__ for i in list_OF]
log_input["list_args"] = list_args
log_input["lb"] = lb
log_input["ub"] = ub
log_input["fb"] = fb
log_input["N"] = N
log_input["w"] = w
log_input["cp"] = cp
log_input["cg"] = cg
log_input["vs"] = vs
log_input["max_itr"] = max_itr
file = open(out + "/log_input.json", "w")
json.dump(log_input, file, indent=4, separators=(",", ": "))
file.close()

log_cases = []
for i in range(len(wrld_hst[0])):
    line = []
    for j in range(np):
        cntry_hst = wrld_hst[j]
        indv = cntry_hst[i]
        itr = indv.itr
        cid = indv.cid
        vels = indv.vels
        vars = indv.vars
        ofv = indv.ofvs[j]
        stat = indv.stat
        src = indv.src
        line.extend([itr, cid])
        line.extend(vels)
        line.extend(vars)
        line.extend([ofv, stat, src])
    log_cases.append(line)
pre = "itr,cid"
for i in range(nv):
    pre = pre + ",vels[" + str(i) + "]"
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
        cntry_hst = best_hst[j]
        best = cntry_hst[i]
        itr = i
        cid = best.cid
        vels = best.vels
        vars = best.vars
        ofv = best.ofvs[j]
        stat = best.stat
        src = best.src
        line.extend([itr, cid])
        line.extend(vels)
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
        cntry_hst = shrng_hst[j]
        shared_pop = cntry_hst[i]
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
