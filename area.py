import numpy as np

p = np.asarray([101, 201, 301, 401, 501])

for i in range(len(p)):
    data = np.load("X"+str(p[i])+".npy")
    data = np.reshape(data, (p[i]+1, p[i]+1))
    area = np.trapz(np.trapz(data))
    print area



