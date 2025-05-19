import matplotlib.pyplot as plt
import numpy as np
r2=[ 0.6405,0.6415,0.6526]
models=['pure python','numpy optimised','scikit-learn model']
x = np.arange(len(models))
width = 0.25
plt.figure(figsize=(8, 10))
plt.bar(x, r2, width, label='R²')
plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Comparison of R2 score")
plt.legend()
plt.savefig('Comparison1.png')
plt.grid(axis='y')
plt.yticks(np.arange(0, 0.75, 0.05))
plt.show()