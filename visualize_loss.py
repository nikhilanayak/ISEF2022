import matplotlib.pyplot as plt
import re


file = open("out.log", "r").readlines()

valid = [i for i in file if "\x08" in i]
nums = {}

for i, line in enumerate(valid):
    try:
        line = line.replace("\x08", "")
        result = re.search("loss: (.*)$", line)
        nums[i] = float(result.group(1))
    except:
        pass

for i in range(5):
    #plt.plot([i * 12500, i * 12500], [0, 5], "k-", lw=1)
    #plt.text(i * 12500, 10, f"epoch {i}")
    pass
plt.plot(nums.keys(), nums.values())
plt.show()