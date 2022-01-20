with open("res0_acc.txt") as f:
    res0 = list(map(float,f.read().splitlines()))
with open("res1_acc.txt") as f:
    res1 = list(map(float,f.read().splitlines()))
with open("res2_acc.txt") as f:
    res2 = list(map(float,f.read().splitlines()))

import matplotlib.pyplot as plt

plt.plot(res0, label="res0")
plt.plot(res1, label="res1")
plt.plot(res2, label="res2")
plt.legend(loc="lower right")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()