from tensorflow.keras.models import load_model
import streamlit as st
from fastapi import FastAPI
import numpy as np
from keras.utils.image_utils import load_img
from keras.utils.image_utils import img_to_array

model = load_model('C:/Users/jains/PycharmProjects/breastcancer/cancer_model.hdf5')
classes = ["Benign", "Malignant"]

cancer_app = FastAPI()

def cancer_UI():
    Title = '''
                    <div style=" color: #8A2BE2; padding: 20px; border-radius: 10px; text-align: center;">
                      <h1>Breast Cancer Risk Prediction</h1>
                    </div>
                '''
    st.write(Title, unsafe_allow_html=True)

    def add_bg_from_url():
        st.markdown(
            f"""
                 <style>
                 .stApp {{
                     background-image: url("https://img.freepik.com/free-photo/copy-space-medicine-sign_23-2148533010.jpg");
                     background-attachment: fixed;
                     background-size: cover
                 }}
                 </style>
                 """,
            unsafe_allow_html=True
        )

    add_bg_from_url()

    radius_mean = st.text_input("Enter radius_mean",key="a")
    texture_mean = st.text_input("Enter texture_mean",key="b")
    perimeter_mean = st.text_input("Enter perimeter_mean", key="c")
    area_mean = st.text_input("Enter area_mean", key="d")
    smoothness_mean = st.text_input("Enter smoothness_mean", key="e")
    compactness_mean = st.text_input("Enter compactness_mean", key="f")
    concavity_mean = st.text_input("Enter concavity_mean", key="g")
    concave_points_mean = st.text_input("Enter concave points_mean", key="h")
    symmetry_mean = st.text_input("Enter symmetry_mean", key="i")
    fractal_dimension_mean = st.text_input("Enter fractal_dimension_mean", key="j")
    radius_se = st.text_input("Enter radius_se", key="k")
    texture_se = st.text_input("Enter texture_se", key="l")
    perimeter_se = st.text_input("Enter perimeter_se", key="m")
    area_se = st.text_input("Enter area_se", key="n")
    smoothness_se = st.text_input("Enter smoothness_se", key="o")
    compactness_se = st.text_input("Enter compactness_se", key="p")
    concavity_se = st.text_input("Enter concavity_se", key="q")
    concave_points_se = st.text_input("Enter concave points_se", key="r")
    symmetry_se = st.text_input("Enter symmetry_se", key="s")
    fractal_dimension_se = st.text_input("Enter fractal_dimension_se", key="t")
    radius_worst = st.text_input("Enter radius_worst", key="u")
    texture_worst = st.text_input("Enter texture_worst", key="v")
    perimeter_worst = st.text_input("Enter perimeter_worst", key="w")
    area_worst = st.text_input("Enter area_worst", key="x")
    smoothness_worst = st.text_input("Enter smoothness_worst", key="y")
    compactness_worst = st.text_input("Enter compactness_worst", key="z")
    concavity_worst = st.text_input("Enter concavity_worst", key="ab")
    concave_points_worst = st.text_input("Enter concave points_worst", key="ac")
    symmetry_worst = st.text_input("Enter symmetry_worst", key="ad")
    fractal_dimension_worst = st.text_input("Enter fractal_dimension_worst", key="ae")

    ok = st.button("Predict Type of Breast Cancer")

    try:
        if ok == True:  # if user pressed ok button then True passed
            radius_mean = float(radius_mean)
            texture_mean = float(texture_mean)
            perimeter_mean = float(perimeter_mean)
            area_mean = float(area_mean)
            smoothness_mean = float(smoothness_mean)
            compactness_mean = float(compactness_mean)
            concavity_mean = float(concavity_mean)
            concave_points_mean = float(concave_points_mean)
            symmetry_mean = float(symmetry_mean)
            fractal_dimension_mean = float(fractal_dimension_mean)
            radius_se = float(radius_se)
            texture_se = float(texture_se)
            perimeter_se = float(perimeter_se)
            area_se = float(area_se)
            smoothness_se = float(smoothness_se)
            compactness_se = float(compactness_se)
            concavity_se = float(concavity_se)
            concave_points_se = float(concave_points_se)
            symmetry_se = float(symmetry_se)
            fractal_dimension_se = float(fractal_dimension_se)
            radius_worst = float(radius_worst)
            texture_worst = float(texture_worst)
            perimeter_worst = float(perimeter_worst)
            area_worst = float(area_worst)
            smoothness_worst = float(smoothness_worst)
            compactness_worst = float(compactness_worst)
            concavity_worst = float(concavity_worst)
            concave_points_worst = float(concave_points_worst)
            symmetry_worst = float(symmetry_worst)
            fractal_dimension_worst = float(fractal_dimension_worst)

            testdata = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean,
                                  smoothness_mean, compactness_mean, concavity_mean,
                                  concave_points_mean, symmetry_mean, fractal_dimension_mean,
                                  radius_se, texture_se, perimeter_se, area_se, smoothness_se,
                                  compactness_se, concavity_se, concave_points_se, symmetry_se,
                                  fractal_dimension_se, radius_worst, texture_worst,
                                  perimeter_worst, area_worst, smoothness_worst,
                                  compactness_worst, concavity_worst, concave_points_worst,
                                  symmetry_worst, fractal_dimension_worst]])

            prob = model.predict(testdata)
            print(prob)

            if prob[0][0] > 0.5:
                st.write("Cancer type is Benign with probability of", prob[0][0])
            if prob[0][1] > 0.5:
                st.write("Cancer type is Malignant with probability of", prob[0][1])

    except Exception as e: # all error
            st.info(e)

if __name__ == "__main__":
    from main1 import cancer_UI
    cancer_UI()



