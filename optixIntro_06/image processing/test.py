import numpy as np

a = np.array(([1,2,3],[4,5,6]))

mean = np.mean(a, axis=1).reshape((1,2))
std = np.std(a, axis=1).reshape((1,2))

ret = np.append(mean,std,axis=0)

print(ret)

np.savetxt("a.txt", ret, header = "a\nb\nc",delimiter = ' ')

for i in range(5,0,-1):
    print(i)
