import pickle
import numpy as np

model = pickle.load(open('C:/Users/pouru/Desktop/nifty50_ModelDeployment/nifty50_model.pkl', 'rb'))


sample_data =  [594.750000, 594.750000, 585.049988, 593.000000, 489.003723, 5302564.0, 590.933329, 343.331185]
sample_data_arr = np.asarray(sample_data)
sample_data_arr = sample_data_arr.reshape(1,-1)
prediction = model.predict(sample_data_arr)
print(f'Predicted label: {prediction}')
if(prediction[0]==0):
    print('Stable Stock')
else: 
    print('Unstable stock')
