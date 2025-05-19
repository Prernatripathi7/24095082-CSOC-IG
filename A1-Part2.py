import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
df= pd.read_csv(r"C:\Users\bibha\Downloads\housing.csv\housing.csv")
df=df.dropna(axis=0)
y=df.median_house_value.values.reshape(-1, 1)
features=['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']
x=df[features]
x=pd.get_dummies(x,drop_first=True)
x = pd.get_dummies(x, drop_first=True)
x = x.astype(float).values
from sklearn.model_selection import train_test_split
trainx, valx, trainy, valy = train_test_split(x, y, test_size=0.2, random_state = 0)
trainy = trainy.reshape(-1, 1) 
valy = valy.reshape(-1, 1)
mean= trainx.mean(axis=0)
std= trainx.std(axis=0)
trainx_norm = (trainx - mean) / std
valx_norm = (valx - mean) / std
m_train = trainx_norm.shape[0]
m_val = valx_norm.shape[0]
trainx_norm = np.hstack((np.ones((m_train, 1)), trainx_norm))
valx_norm = np.hstack((np.ones((m_val, 1)), valx_norm))
weights = np.zeros((trainx_norm.shape[1], 1))  
learning_rate = 0.01
epochs = 500
m = m_train
costs=[]
start_time=time.time()
for iteration in range(epochs):
    y_pred = trainx_norm.dot(weights)
    error = y_pred - trainy 
    gradients = (2/m) * (trainx_norm.T.dot(error))
    weights -= learning_rate * gradients
    if iteration % 10 == 0:
        loss = np.mean(error ** 2)
        print(f"Iteration {iteration}: Loss = {loss:.4f}")
        costs.append(loss)
end_time=time.time()
convergence_time=end_time-start_time
def predict(data_with_bias):
    preds = data_with_bias.dot(weights)
    return preds.flatten()
train_preds=predict(trainx_norm)
val_preds = predict(valx_norm)
for idx, pred in zip(valx_norm[:, 1:], val_preds):
    print(f"Prediction: {pred}")
mae_train = mean_absolute_error(trainy, train_preds)
rmse_train = np.sqrt(mean_squared_error(trainy, train_preds))
r2_train = r2_score(trainy, train_preds)
mae_val = mean_absolute_error(valy, val_preds)
rmse_val = np.sqrt(mean_squared_error(valy, val_preds))
r2_val = r2_score(valy, val_preds)
print(f"Training MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
print(f"Validation MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}")
print(f"Convergence time: {convergence_time:.4f} seconds")
iterations=list(range(0,500,10))
plt.figure(figsize=(8,5))
plt.plot(iterations,costs)
plt.title("Costs vs Iteration")
plt.xlabel("Iterations")
plt.ylabel("Costs")
plt.savefig('covergence vs iteration1.png')
plt.show()
