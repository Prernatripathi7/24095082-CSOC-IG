import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import time
df= pd.read_csv(r"C:\Users\bibha\Downloads\housing.csv\housing.csv")
df=df.dropna(axis=0)
y=df.median_house_value
features=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']
x=df[features]
x=pd.get_dummies(x,drop_first=True)
from sklearn.model_selection import train_test_split
trainx, valx, trainy, valy = train_test_split(x, y,test_size=0.2, random_state = 0)
mean= trainx.mean()
std= trainx.std()
trainx_norm = (trainx - mean) / std
valx_norm = (valx - mean) / std
trainy_list = trainy.tolist()
valy_list = valy.tolist()
weights = [0.0] * (len(features) + 1) 
learning_rate = 0.01
epochs = 500
costs=[]
m = len(trainx_norm)
start_time=time.time()
for iterations in range(epochs):
    gradient=[0.0]*len(weights)
    for i in range(m):
        x_i=[1.0]+trainx_norm.iloc[i].tolist()
        y_i=trainy_list[i]
        y_pred = sum(weights[j] * x_i[j] for j in range(len(weights)))
        error = y_pred - y_i
        for j in range(len(weights)):
            gradient[j] += (2/m) * error * x_i[j]
    for j in range(len(weights)):
        weights[j] -= learning_rate * gradient[j]
    if iterations % 10 == 0:
        loss = 0.0
        for i in range(m):
            x_i = [1.0] + trainx_norm.iloc[i].tolist()
            y_i = trainy_list[i]
            y_pred = sum(weights[j] * x_i[j] for j in range(len(weights)))
            loss += (y_pred - y_i) ** 2
        loss /= m
        costs.append(loss)
        print(f"Iteration {iterations}: Loss = {loss:.6f}") 
end_time= time.time()
convergence_time=end_time-start_time
def predict(data_norm):
    predictions = []
    for i in range(len(data_norm)):
        x_i = [1.0] + data_norm.iloc[i].tolist()
        y_pred = sum(weights[j] * x_i[j] for j in range(len(weights)))
        predictions.append(y_pred)
    return predictions
train_preds = predict(trainx_norm)
val_preds = predict(valx_norm)           
for i, pred in enumerate(val_preds):
    print(f"Prediction for sample {i}: {pred}")
mae_train = mean_absolute_error(trainy, train_preds)
rmse_train = np.sqrt(mean_squared_error(trainy, train_preds))
r2_train = r2_score(trainy, train_preds)
mae_val = mean_absolute_error(valy, val_preds)
rmse_val = np.sqrt(mean_squared_error(valy, val_preds))
r2_val = r2_score(valy, val_preds)
print(f"Training MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
print(f"Validation MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}")
iterations=list(range(0,500,10))
print(f"Convergence time: {convergence_time:.4f} seconds")
plt.figure(figsize=(8,5))
plt.plot(iterations,costs)
plt.title("Costs vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("Costs")
plt.savefig('covergence vs iteration.png')
plt.show()


