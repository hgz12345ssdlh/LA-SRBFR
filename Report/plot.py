import seaborn as sns, matplotlib.pyplot as plt
import numpy as np

sns.set_style('darkgrid')

x = np.array([5, 10, 20, 30, 40])
y = np.array([0.6484, 0.8130, 0.9140, 0.9421, 0.9614])

ax = sns.regplot(x = x, y = y, order = 2, color = 'm')
ax.set_xlim(3, 45)
ax.set_ylim(0.5, 1.)
ax.set_xticks(x)
ax.set_xlabel('numTrainee')
ax.set_ylabel('Accuracy')

for i in range(5):
	plt.annotate(str(y[i]), xy = (x[i], y[i]), xytext = (x[i] - 1.7, y[i] + 0.007))

plt.show()