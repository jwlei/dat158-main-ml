from pycaret.classification import *#load_model, predict_model
import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
import os

class StreamlitApp:
    
    def __init__(self):
        self.model = load_model('model/final_model_heart_failure') 
        self.save_fn = 'path.csv'     
        
    def predict(self, input_data): 
        return predict_model(self.model, data=input_data)
    
    def store_prediction(self, output_df): 
        if os.path.exists(self.save_fn):
            save_df = pd.read_csv(self.save_fn)
            save_df = save_df.append(output_df, ignore_index=True)
            save_df.to_csv(self.save_fn, index=False)
            
        else: 
            output_df.to_csv(self.save_fn, index=False)  
            
    
    def run(self):
        image = Image.open('assets/human-heart.jpg')
        st.image(image, use_column_width=False)
    
    
        add_selectbox = st.sidebar.selectbox('How would you like to predict?', ('Online', 'Batch')) #bruke batch for aa predikere paa alle bildene. 
        st.sidebar.info('This app is created to predict heart failure' )
        st.sidebar.success('DAT158')
        st.title('Heart failure prediction')
        
       
        if add_selectbox == 'Online': 
            age = st.number_input('Age', min_value=18, max_value=100, value=25) #remember trained on age between ...
            sex = st.selectbox('Sex', ['M', 'F'])
            chest_pain_type = st.selectbox('ChestPainType', ['ASY', 'NAP', 'ATA', 'TA'])
            resting_bp = st.number_input('RestingBP', min_value=0, max_value=210, value=60)
            cholesterol = st.number_input('Cholesterol', min_value=0, max_value=605, value=200)            
            resting_ECG = st.selectbox('RestingECG', ['Normal', 'LVH', 'ST'])
            max_hr = st.number_input('MaxHR', min_value=0, max_value=210, value=140)
            oldpeak = st.slider('Oldpeak',min_value=0.0, max_value=7.0, value=1.0, step=0.1)  
            st_slope = st.selectbox('ST_Slope', ['Flat', 'Up', 'Down'])
            
            
            fasting_bs, exercise_angina= False, False
            if st.checkbox('FastingBS'): fasting_bs=True

            if st.checkbox('ExerciseAngina'): exercise_angina=True
            

            
            output=''
            input_dict = {'Age':age, 'Sex':sex, 'ChestPainType':chest_pain_type, 'RestingBP':resting_bp, 'Cholesterol':cholesterol, 'is_FastingBS':fasting_bs, 
                          'RestingECG':resting_ECG, 'MaxHR':max_hr, 'is_ExerciseAngina':exercise_angina, 'Oldpeak':oldpeak, 'ST_Slope':st_slope}
            input_df = pd.DataFrame(input_dict, index=[0])
        
            if st.button('Predict'): 
                output = self.predict(input_df)
                self.store_prediction(output)
                
                output = 'Heart disease' if output['Label'][0] == 1 else 'Normal'
                #output = str(output['Label'])
                
            
            st.success('Predicted output: {}'.format(output))
            
        if add_selectbox == 'Batch': 
            fn = st.file_uploader("Upload csv file for predictions") #st.file_uploader('Upload csv file for predictions, type=["csv"]')
            if fn is not None: 
                input_df = pd.read_csv(fn)
                predictions = self.predict(input_df)
                st.write(predictions)
            
sa = StreamlitApp()
sa.run()