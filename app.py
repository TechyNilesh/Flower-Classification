import streamlit as st
import numpy as np
import PIL.Image
import pandas as pd
import os
import svg
from fastai.vision.all import Path,load_learner,Image

path = Path('export.pkl')

learn = load_learner(path)

df = pd.read_csv("Flowers.csv",index_col=['Index'])

def get_name(cat_num):
    return df[df.index == cat_num].reset_index(drop=True)['Cat_Name'][0]
def get_details(cat_num):
    temp = df[df.index == cat_num].T.reset_index()[1:]
    temp.columns = ['Major','Description']
    return temp

def predict_img(img):
    #pil_img = PIL.Image.open(img)
    img = np.asarray(img) # Image to display   
    return get_name(int(learn.predict(img)[0])),round(np.max(np.array(learn.predict(img)[2]))*100,2),get_details(int(learn.predict(img)[0]))

html_temp = """
    <div style="background-color:#f63366;padding:10px;margin-bottom: 25px">
    <h2 style="color:white;text-align:center;">Flower Prediction App</h2>
    <p style="color:white;text-align:center;" >This is a <b>Streamlit</b> app use for prediction of the <b>102 types of flower</b>.</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

#image = PIL.Image.open('bg.jpg')
#st.image(image, use_column_width=True)

option = st.radio('', ['Choose a test image', 'Choose your own image'])
if option == 'Choose your own image':
    uploaded_file = st.file_uploader("Choose an image...", type="jpg") #file upload
    if uploaded_file is not None:
        img = PIL.Image.open(uploaded_file)
        pred_class, prob, details = predict_img(img)
        col1, col2 = st.beta_columns(2)
        with col1:
            st.image(img, width=200)
        with col2:
            st.success("Flower Name:  [" + str(pred_class) + "] ")
            st.info("Probability: [" + str(prob) + '%]')
        st.subheader("Flower Details:")
        st.table(details)
else:
    test_images = os.listdir('sample_images')
    test_image = st.selectbox('Please select a test image:', test_images)
    file_path = 'sample_images/' + test_image
    img = PIL.Image.open(file_path)
    pred_class, prob, details = predict_img(img)
    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(img, width=200)
    with col2:
        st.success("Flower Name:  [" + str(pred_class) + "] ")
        st.info("Probability: [" + str(prob) + '%]')
    st.subheader("Flower Details:")
    st.table(details)

