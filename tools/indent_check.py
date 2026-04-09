import re
p='main27_train_final.py'
lines=open(p).read().splitlines()
for i in range(1900,2135):
    if i-1 < len(lines):
        l=lines[i-1]
        lead=len(l)-len(l.lstrip(' '))
        print(f"{i:4d} lead={lead} |{l}")
