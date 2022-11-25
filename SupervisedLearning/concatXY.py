import numpy as np
x=[]
with open ("X.txt","r")as f:
    for line in f:
        x.append(line.strip())
y=[]
with open ("Y.txt","r")as f:
    for line in f:
        y.append(line.strip())
print(len(x),len(y))
with open ("XY.txt","w") as f:
    for i in range(len(x)):
        f.write(x[i]+","+y[i]+"\n")


