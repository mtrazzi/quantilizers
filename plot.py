import sys
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
data = np.load(filename)
ep_rets = data['ep_rets']
plt.plot(ep_rets)
plt.show()