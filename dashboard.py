import streamlit as st
import requests
import pandas as pd
import streamlit.components.v1 as components
#import joblib
#import pickle
#import dill 
import numpy as np
import plotly
import plotly.graph_objects as pgo
import plotly.express as px
import shap
import dill
#import matplotlib.pyplot as plt
#import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

# cd C:/Users/marat/Downloads/Py-DS-ML-Bootcamp-master/OCR/P7/dashboard/
#streamlit run dashboard.py


# Titres de l'application
title_text = 'Prédiction du défaut de payment d\'un client à partir de son ID'
subheader_text = '''Classe 0: Crédit accordé, Classe 1: Crédit refusé '''
st.markdown(f"<h4 style='text-align: center;'><b>{title_text}</b></h4>", unsafe_allow_html=True)
st.markdown(f"<h7 style='text-align: center;'>{subheader_text}</h7>", unsafe_allow_html=True)


# Load test data
X = pd.read_csv('X_test_sample.csv')
X.set_index('sk_id_curr',inplace=True)
st.write(X)

# Sélectionner l'id client à partir d'une liste:
id_client = st.sidebar.selectbox('Séléctionner un ID client', list(map(int, X.index.to_list())))

    
#Faire la prédiction et donner sa probabilité
predict_btn = st.sidebar.button('Prédiction')
if predict_btn:
  resultat= requests.post(url='https://myappy.herokuapp.com/predict',json= {'user_id': id_client})
  #st.write(resultat.json())   
  st.write( 'Résultat de la prédiction:', int(resultat.json()['prediction']))
  # Visualiser la probabilité du résultat sous forme de jauge
  fig = pgo.Figure(pgo.Indicator(mode = "gauge+number",value = round(resultat.json()['probability'],2), domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text':    "Probabilité de la prédiction"},  gauge = {'axis': {'range': [0, 1]}, 
   'steps' : [{'range': [0, round(resultat.json()['probability'],2)], 'color': "green"}, {'range': [round(resultat.json()['probability'],2), 1], 'color': "red"}]}))
  st.plotly_chart(fig, use_container_width=True)
   
 
       

# Définition d'une fonction pour visualisation shap
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    
#Interprétabilité des résultats avec SHAP:   
predict_btn_res = st.sidebar.button("Analyse de la prédiction")
if predict_btn_res:
  #data_in = X.loc[[id_client]]
  
  # explain the model's predictions using SHAP
  # python v7 required for dill (in streamlitshare)
  # Loading explainer
  explainer_file = open("explainer.pkl", "rb")
  explainer = dill.load(explainer_file)
  explainer_file.close()
  
  #Loading shap_values
  shap_values_file = open("shap_values.pkl", "rb")
  shap_values = dill.load(shap_values_file)
  shap_values_file.close()   

  # visualize the prediction's explanation:
  st_shap(shap.force_plot(explainer.expected_value, shap_values[1][0,:], X.loc[[id_client]]), 200)
  # ALL prédictions
  st_shap(shap.force_plot(explainer.expected_value, shap_values[1], X),400)
  st.pyplot(shap.summary_plot(shap_values, X))
            
            
  
# Distribition des top features importance:
dist_btn_res = st.sidebar.button("Positionnement du client (Top Features)")
if dist_btn_res:
  # Feature: code_gender
  st.write('{code_gender} du client: ', X.loc[[id_client]]['code_gender'].values[0])
  fig= px.histogram(X, x="code_gender", color="code_gender")
  st.plotly_chart(fig)
  # Feature: payment_rate
  st.write('{payment_rate} du client: ', X.loc[[id_client]]['payment_rate'].values[0])
  fig= px.box(X, y="payment_rate")
  st.plotly_chart(fig)
  # Feature: region_rating_client_w_city
  st.write('{region_rating_client_w_city} du client: ', X.loc[[id_client]]['region_rating_client_w_city'].values[0])
  fig= px.histogram(X, x="region_rating_client_w_city")
  st.plotly_chart(fig)

  
            
            
    


        
