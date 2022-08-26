import os


coop = [True, False]
nr = 1000
list_args = []

for i in range(len(coop)):
    for j in range(nr):
        args = " " + str(coop[i]) + " " + str(j + 1)
        list_args.append(args)

for i in range(len(list_args)):
    os.system("python3.8 C-PSO.py" + list_args[i])
