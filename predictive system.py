
import numpy as np 
import pickle 

loaded_model = pickle.load(open('/Userslydiacharif/PycharmProjects/diab_pred/trained_model.sav/trained_model.sav', 'rb'))
input_data = (4,110,92,0,0,37.6,0.191,30)

input_data_npr = np.asarray(input_data)

input_data_reshaped = input_data_npr.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0) :
    print ('the person doesnt have diabetes')
else :
    print("the person has diabtes sadly")