import sys
import numpy
import pyDOE
from ObjectiveFunctions import *
from copy import deepcopy
import os
import shutil
import json


out = "/home/igangga/CollaborativeAlgorithms/ProblemSet1/GA"
if !os.path.isdir(out):
    os.mkdir(out)
    os.mkdir(out + "/C-GA")
    os.mkdir(out + "/NC-GA")

coop = str(sys.argv[1]) == "True"   # This argument will decide to run either the collaborative or the non-collaborative GA.
coop_str = str(sys.argv[2])
run = int(sys.argv[3])

if coop:
    if coop_str == "clone":
        out = out + "/C-GA"
else:
    out = out + "/NC-GA"
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
if N % 2 != 0:
    N = N + 1
p = 1 / N
M = int(N / 2)
lmbd = 0.1
phi0 = 0.25
b = 4.0
c = 1 / N
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
    cntry = []
    for j in range(N):
        itr = 0
        cid = i
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
        cntry.append(Indv(itr, cid, vars, ofvs, stat, src))
    wrld.append(cntry)
wrld_hst = deepcopy(wrld)
best_hst = []
for i in range(np):
    cntry = wrld[i]
    best = deepcopy(cntry[0])
    for j in range(N):
        indv = deepcopy(cntry[j])
        if indv.ofvs[i] > best.ofvs[i]:
            best = indv
    best_hst.append([best])
cloned_hst = []
for i in range(np):
    cloned_hst.append([])
dvrst_hst = []
for i in range(np):
    cntry = wrld[i]
    cntry_ofv = []
    for j in range(N):
        indv = cntry[j]
        cntry_ofv.append(indv.ofvs[i])
    std = numpy.std(cntry_ofv)
    dvrst_hst.append([std])


def update_itr(itr):
    for i in range(np):
        cntry = wrld[i]
        for j in range(N):
            indv = cntry[j]
            indv.itr = itr


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


def choose(sorted_pop, qty, start=0):
    chosen_pop = []
    for i in range(start, start + qty):
        indv = sorted_pop[i]
        chosen_pop.append(deepcopy(indv))
    return chosen_pop


def update_src(pop):
    for i in range(len(pop)):
        indv = pop[i]
        indv.src = indv.cid


def update_cid(pop, cid):
    for i in range(len(pop)):
        indv = pop[i]
        indv.cid = cid


def update_stat(pop, stat):
    for i in range(len(pop)):
        indv = pop[i]
        indv.stat = stat


def record(wrld):
    for i in range(np):
        cntry = deepcopy(wrld[i])
        wrld_hst[i].extend(cntry)


def record_cloned():
    for i in range(np):
        cntry = wrld[i]
        cloned_ppl = []
        for j in range(N):
            indv = deepcopy(cntry[j])
            if indv.src != -1:
                cloned_ppl.append(indv)
        cloned_hst[i].append(cloned_ppl)


def reset_src():
    for i in range(np):
        cntry = wrld[i]
        for j in range(N):
            indv = cntry[j]
            indv.src = -1


def clone():
    set_chosen_cndts = []
    for i in range(np):
        cndts = list_cndts(i)
        sorted_cndts = sort(cndts, i)
        chosen_cndts = choose(sorted_cndts, int(c * N))
        update_src(chosen_cndts)
        update_cid(chosen_cndts, i)
        set_chosen_cndts.append(chosen_cndts)
    for i in range(np):
        ppl = deepcopy(wrld[i])
        chosen_cndts = deepcopy(set_chosen_cndts[i])
        ppl.extend(chosen_cndts)
        sorted_ppl = sort(ppl, i)
        chosen_ppl = choose(sorted_ppl, N)
        update_stat(chosen_ppl, "clone")
        wrld[i] = deepcopy(chosen_ppl)
    record(wrld)
    record_cloned()
    reset_src()


def select():
    for i in range(np):
        ppl = deepcopy(wrld[i])
        sorted_ppl = sort(ppl, i)
        chosen_ppl = choose(sorted_ppl, int((1 - p) * N))
        pN_best_ppl = choose(sorted_ppl, int(p * N))
        chosen_ppl.extend(pN_best_ppl)
        update_stat(chosen_ppl, "select")
        wrld[i] = deepcopy(chosen_ppl)
    record(wrld)


def split():
    wrld_hi = []
    wrld_lo = []
    for i in range(np):
        ppl = deepcopy(wrld[i])
        sorted_ppl = sort(ppl, i)
        cntry_hi = choose(sorted_ppl, M)
        cntry_lo = choose(sorted_ppl, M, M)
        update_stat(cntry_hi, "split")
        update_stat(cntry_lo, "split")
        wrld_hi.append(cntry_hi)
        wrld_lo.append(cntry_lo)
    record(wrld_hi)
    record(wrld_lo)
    return wrld_hi, wrld_lo


def compute_D(indv_hi, indv_lo):
    D = numpy.array(indv_hi.vars) - numpy.array(indv_lo.vars)
    r = []
    for i in range(nv):
        r.append(numpy.random.rand())
    binary = []
    for i in range(nv):
        if r[i] < 0.5:
            binary.append(0)
        else:
            binary.append(1)
    binary = numpy.array(binary)
    while numpy.linalg.norm(D * binary) == 0:
        id = numpy.random.randint(nv)
        binary[id] = 1
    D = (D * binary).tolist()
    return D


def ensure_feasibility(indv):
    for i in range(nv):
        if indv.vars[i] < lb[i]:
            indv.vars[i] = lb[i] + fb * abs(indv.vars[i] - lb[i])
        elif indv.vars[i] > ub[i]:
            indv.vars[i] = ub[i] - fb * abs(indv.vars[i] - ub[i])


def mate(indv_hi, indv_lo, sc, D):
    offs_hi = deepcopy(indv_hi)
    offs_hi.vars = (numpy.array(indv_hi.vars) + sc * numpy.array(D)).tolist()
    ensure_feasibility(offs_hi)
    offs_lo = deepcopy(indv_lo)
    offs_lo.vars = (numpy.array(indv_lo.vars) + sc * numpy.array(D)).tolist()
    ensure_feasibility(offs_lo)
    return offs_hi, offs_lo


def mutate(indv_hi, indv_lo, sm):
    r = []
    for i in range(nv):
        r.append(numpy.random.rand())
    Phi0 = []
    for i in range(nv):
        Phi0.append(- phi0 + 2 * r[i] * phi0)
    offs_hi = deepcopy(indv_hi)
    offs_hi.vars = (numpy.array(indv_hi.vars) + sm * numpy.array(Phi0) * (numpy.array(ub) - numpy.array(lb))).tolist()
    ensure_feasibility(offs_hi)
    offs_lo = deepcopy(indv_lo)
    offs_lo.vars = (numpy.array(indv_lo.vars) + sm * numpy.array(Phi0) * (numpy.array(ub) - numpy.array(lb))).tolist()
    ensure_feasibility(offs_lo)
    return offs_hi, offs_lo


def update_ofvs(pop):
    for i in range(len(pop)):
        indv = pop[i]
        ofvs = []
        for j in range(np):
            OF = list_OF[j]
            args = list_args[j]
            ofvs.append(OF(indv.vars, args))
        indv.ofvs = ofvs


def crossover():
    wrld_offs_hi = []
    wrld_offs_lo = []
    for i in range(np):
        cntry_hi = wrld_hi[i]
        cntry_lo = wrld_lo[i]
        cntry_offs_hi = []
        cntry_offs_lo = []
        for j in range(M):
            indv_hi = deepcopy(cntry_hi[j])
            indv_lo = deepcopy(cntry_lo[j])
            r = numpy.random.rand()
            if r >= lmbd and indv_hi.vars != indv_lo.vars and indv_hi.ofvs[i] != indv_lo.ofvs[i]:
                sc = abs(indv_hi.ofvs[i] - indv_lo.ofvs[i]) / abs(cntry_hi[0].ofvs[i] - cntry_lo[-1].ofvs[i])
                D = compute_D(indv_hi, indv_lo)
                offs_hi, offs_lo = mate(indv_hi, indv_lo, sc, D)
            else:
                sm = (1 - indv_hi.itr / max_itr) ** b
                offs_hi, offs_lo = mutate(indv_hi, indv_lo, sm)
            cntry_offs_hi.append(offs_hi)
            cntry_offs_lo.append(offs_lo)
        update_stat(cntry_offs_hi, "crossover")
        update_stat(cntry_offs_lo, "crossover")
        update_ofvs(cntry_offs_hi)
        update_ofvs(cntry_offs_lo)
        wrld_offs_hi.append(cntry_offs_hi)
        wrld_offs_lo.append(cntry_offs_lo)
    record(wrld_offs_hi)
    record(wrld_offs_lo)
    return wrld_offs_hi, wrld_offs_lo


def record_best():
    for i in range(np):
        cntry = wrld[i]
        best = deepcopy(cntry[0])
        for j in range(N):
            indv = deepcopy(cntry[j])
            if indv.ofvs[i] > best.ofvs[i]:
                best = indv
        best_hst[i].append(best)


def record_dvrst():
    for i in range(np):
        cntry = wrld[i]
        cntry_ofv = []
        for j in range(N):
            indv = cntry[j]
            cntry_ofv.append(indv.ofvs[i])
        std = numpy.std(cntry_ofv)
        dvrst_hst[i].append(std)


def replace():
    for i in range(np):
        cntry = []
        cntry_hi = wrld_hi[i]
        cntry_lo = wrld_lo[i]
        cntry_offs_hi = wrld_offs_hi[i]
        cntry_offs_lo = wrld_offs_lo[i]
        for j in range(M):
            indv_hi = deepcopy(cntry_hi[j])
            indv_lo = deepcopy(cntry_lo[j])
            offs_hi = deepcopy(cntry_offs_hi[j])
            offs_lo = deepcopy(cntry_offs_lo[j])
            if offs_hi.ofvs[i] > indv_hi.ofvs[i]:
                cntry.append(offs_hi)
            else:
                cntry.append(indv_hi)
            if offs_lo.ofvs[i] > indv_lo.ofvs[i]:
                cntry.append(offs_lo)
            else:
                cntry.append(indv_lo)
        update_stat(cntry, "replace")
        wrld[i] = deepcopy(cntry)
    record(wrld)
    record_best()
    record_dvrst()


# Iterative process
for i in range(max_itr):
    update_itr(i + 1)

    if coop and coop_str == "clone":
        clone()

    select()

    wrld_hi, wrld_lo = split()
    wrld_offs_hi, wrld_offs_lo = crossover()

    replace()


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
log_input["list_args"] = list_args
log_input["lb"] = lb
log_input["ub"] = ub
log_input["fb"] = fb
log_input["N"] = N
log_input["p"] = p
log_input["M"] = M
log_input["lmbd"] = lmbd
log_input["phi0"] = phi0
log_input["b"] = b
log_input["c"] = c
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
        cntry_hst = best_hst[j]
        best = cntry_hst[i]
        itr = best.itr
        cid = best.cid
        vars = best.vars
        ofv = best.ofvs[j]
        stat = best.stat
        src = best.src
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


def generate_cloning_data(cloned_ppl):
    cloning_data = [0] * np
    for i in range(len(cloned_ppl)):
        indv = cloned_ppl[i]
        src = indv.src
        cloning_data[src] = cloning_data[src] + 1
    return cloning_data


log_cloning = []
for i in range(len(cloned_hst[0])):
    line = []
    for j in range(np):
        cntry_hst = cloned_hst[j]
        cloned_ppl = cntry_hst[i]
        itr = i + 1
        cid = j
        cloning_data = generate_cloning_data(cloned_ppl)
        totl = len(cloned_ppl)
        line.extend([itr, cid])
        line.extend(cloning_data)
        line.append(totl)
    log_cloning.append(line)
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
file = open(out + "/log_cloning.txt", "w")
file.write(hdr + "\n")
for line in log_cloning:
    for i in range(len(line)):
        if i == 0:
            file.write(str(line[i]))
        else:
            file.write("," + str(line[i]))
    file.write("\n")
file.close()

log_diversity = []
for i in range(len(dvrst_hst[0])):
    line = []
    for j in range(np):
        itr = i
        cid = j
        std = dvrst_hst[j][i]
        line.extend([itr, cid, std])
    log_diversity.append(line)
hdr = "itr,cid,std"
for i in range(1, np):
    hdr = hdr + "," + "itr,cid,std"
file = open(out + "/log_diversity.txt", "w")
file.write(hdr + "\n")
for line in log_diversity:
    for i in range(len(line)):
        if i == 0:
            file.write(str(line[i]))
        else:
            file.write("," + str(line[i]))
    file.write("\n")
file.close()
