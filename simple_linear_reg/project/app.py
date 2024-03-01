import numpy as np
import pickle
import pandas as pd
import streamlit as st 
pickle_in = open(r"C:\Users\SVI\Desktop\DS-ML\ML_learn\simple_linear_reg\project\model.pkl","rb")
lr=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_note_authentication(cgpa):
    
    prediction=lr.predict([cgpa[0]])
    print(prediction)
    return prediction

def main():
    cg=[]
    st.title("Package predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Package predictor App</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    cgp = st.number_input('type here')
    cg.append(cgp)
    cgpa = np.array(cg).reshape(-1,1)
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(cgpa)
    st.success('The output is {} LPA'.format(result,))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    