import streamlit as st
import requests
import pandas as pd
import streamlit.components.v1 as components
#import joblib
#import pickle
#import dill 
import numpy as np
#import plotly
#import plotly.graph_objects as go
#import shap
st.set_option('deprecation.showPyplotGlobalUse', False)

# cd C:/Users/marat/Downloads/Py-DS-ML-Bootcamp-master/OCR/P7/dashboard/
#streamlit run dashboard.py

 
# loading the trained model
#pickle_in = open('lgbm.pkl', 'rb') # quel est le dossier courant?
#model = pickle.load(pickle_in)




# Titres de l'application
title_text = 'Prédiction de défaut de payment d\'un client à partir de son ID'
subheader_text = '''Classe 0: Crédit accordé, Classe 1: Crédit refusé '''
st.markdown(f"<h4 style='text-align: center;'><b>{title_text}</b></h4>", unsafe_allow_html=True)
st.markdown(f"<h7 style='text-align: center;'>{subheader_text}</h7>", unsafe_allow_html=True)


# Load test data
X = pd.read_csv('X_test_sample.csv')
X.set_index('sk_id_curr',inplace=True)
st.write(X)

# Sélectionner l'id client à partir d'une liste:
id_client = st.sidebar.selectbox('Quel user id chosir?', list(map(int, X.index.to_list())))

    
#Faire la prédiction et donner sa probabilité
predict_btn = st.sidebar.button('Donner la prédiction')
if predict_btn:
    resultat= requests.post(url='https://myappy.herokuapp.com/predict',json= {'user_id': id_client})
    
if uploaded_file_test:
    predict_btn = st.button('Prédire après sélection du user id ')
    if predict_btn:
        if uploaded_file_test:
            resultat= requests.post(url='http://127.0.0.1:8000/predict',json= {'user_id': id_client})
            st.write(resultat.json())   
            #st.write(resultat.json()) 
            st.write( 'Résultat de la prédiction:', int(resultat.json()['prediction']))
            # Visualiser la probabilité du résultat sous forme de jauge
            fig = go.Figure(go.Indicator(mode = "gauge+number",value = round(resultat.json()['probability'],2), domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text':    "Probabilité de la prédiction"},  gauge = {'axis': {'range': [0, 1]}, 
    'steps' : [{'range': [0, round(resultat.json()['probability'],2)], 'color': "green"}, {'range': [round(resultat.json()['probability'],2), 1], 'color': "red"}]}))
    
            # Plot
            st.plotly_chart(fig, use_container_width=True)
     
    

# Définition d'une fonction pour visualisation shap
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
#Interprétabilité des résultats avec SHAP:   

if uploaded_file_test:
    predict_btn_res = st.button("Expliquer les résultats")
    if predict_btn_res:
            #data_in = X.loc[[id_client]]
            # explain the model's predictions using SHAP         
            explainer = dill.load('explainer.pkl')
             #explain the model's predictions using SHAP values
            shap_values = dill.load('shap_values.pkl')    
            
            # visualize the prediction's explanation:
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X.loc[[id_client]]), 200)
            # ALL prédictions
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], X),400)
            
            st.pyplot(shap.summary_plot(shap_values, X))
            
            
            
            
            
    


        
