import os


coop = [True, False]
coop_str = ["jump", "swap"]
nr = 1000
list_args = []

for i in range(len(coop)):
    if coop[i]:
        for j in range(len(coop_str)):
            for k in range(nr):
                args = " " + str(coop[i]) + " " + str(coop_str[j]) + " " + str(k + 1)
                list_args.append(args)
    else:
        for k in range(nr):
            args = " " + str(coop[i]) + " " + "-" + " " + str(k + 1)
            list_args.append(args)

for i in range(len(list_args)):
    os.system("python3.8 C-GD.py" + list_args[i])
