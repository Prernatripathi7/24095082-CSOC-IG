import pandas as pd
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
from sklearn import linear_model
df_model = linear_model.LinearRegression()
start_time=time.time()
df_model.fit(trainx_norm, trainy)
end_time=time.time()
fitting_duration= end_time-start_time
train_preds=df_model.predict(trainx_norm)
val_preds = df_model.predict(valx_norm)
mae_train = mean_absolute_error(trainy, train_preds)
rmse_train = np.sqrt(mean_squared_error(trainy, train_preds))
r2_train = r2_score(trainy, train_preds)
mae_val = mean_absolute_error(valy, val_preds)
rmse_val = np.sqrt(mean_squared_error(valy, val_preds))
r2_val = r2_score(valy, val_preds)
print(val_preds)
print(f"Fitting duration: {fitting_duration:.4f} seconds")
print(f"Training MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}, R2: {r2_train:.4f}")
print(f"Validation MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}")


