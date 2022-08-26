import os


coop = [True, False]
coop_str = ["jump_before_move", "swap_before_move", "jump_after_move", "swap_after_move"]
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
