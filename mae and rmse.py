import matplotlib.pyplot as plt
import numpy as np
mae=[50519.6950,50390.6575,49964.0128]
rmse=[69375.6927,69279.9713,68201.5212]
models=['Pure Python','Numpy Optimised','Scikit-Learn Model']
x = np.arange(len(models))
width = 0.25
plt.figure(figsize=(10, 6))
plt.bar(x+width, mae, width, label='MAE')
plt.bar(x, rmse, width, label='RMSE')
plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Comparison of Regression Metrics: MAE and RMSE")
plt.legend()
plt.savefig('Comparison.png')
plt.grid(axis='y')
plt.show()