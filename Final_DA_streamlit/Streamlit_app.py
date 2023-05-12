import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
from pickle import load
from PIL import Image

#Set up pages
st.set_page_config(page_title="House Sales Project",
                   page_icon="https://i.pinimg.com/originals/39/f7/48/39f7481d939f788cb3ff5d04716f96b3.jpg",
                   layout="wide"
)
st.snow()

#Header
st.title("**:green[House Sales Prediction Project]** :christmas_tree:") #blue, green, orange, red, violet.
image = Image.open('ML_Final/House_background.jpg')
st.image(image)
st.subheader("Please enter your name :snowflake:")
st.text_input("*Name:*", key="name")
st.subheader("Our dataset	:rainbow:")
data_new = pd.read_csv("ML_Final/House_sale_final.csv")
data_new

#Encoder
encoder = LabelEncoder()
encoder.classes_ = np.load("ML_Final/classes.npy",allow_pickle=True)

best_gbt_model = xgb.XGBRegressor()
best_gbt_model.load_model('ML_Final/gbt_model_weight.py')

n = round(len(data_new["price"])*0.8)
train_data = data_new.iloc[1:n,]
if st.checkbox('**Show Training Dataframe**'):
    train_data

#Map average price by city
st.subheader("Map: Average price by city :world_map:")
image1 = Image.open('ML_Final/Map avg price by city.jpg')
st.image(image1, caption = 'USA')

#House price visualization
st.subheader("House price visualization :ocean:")
image2 = Image.open('ML_Final/House_sale_visualization.png')
st.image(image2, caption = 'More info') #ML_Final/House price heatmap.png

#Side bar
st.sidebar.title("About project")
st.sidebar.subheader("Web created by Nguyen Tien Duc, thanks you for your supporting!")
#image3 = Image.open('ML_Final/House_background.jpg')
#st.sidebar.image(image3)

#Choose your city
st.subheader('Choose a city you want :night_with_stars:')
left_column, right_column = st.columns(2)
with left_column:
    inp_city = st.selectbox('*City name:*', np.unique(data_new['city']))
label_encoder = load(open("ML_Final/city_encoder.pkl", 'rb'))

st.subheader("Please choose the house characteristics :cityscape:")
input_bedrooms = st.slider('**Bedrooms amount(int)**', min(data_new["bedrooms"]), max(data_new["bedrooms"]), 1)
input_bathrooms = st.slider('**Bathrooms amount(int)**', min(data_new["bathrooms"]), max(data_new["bathrooms"]), 1)
input_sqft_living = st.number_input('**Size of the house (square footage)**', 1, max(data_new["sqft_living"]), 1)
input_sqft_lot = st.number_input('**Size of the lot (square footage)**', 1, max(data_new["sqft_lot"]), 1)
input_floors = st.slider('**Floors amount(int)**', min(data_new["floors"]), max(data_new["floors"]), 1)
input_view = st.slider('**View amount(int)**', min(data_new["view"]), max(data_new["view"]), 1)
input_waterfront = st.slider('**Waterfront(int)**', min(data_new["waterfront"]), max(data_new["waterfront"]), 1)
input_condition = st.slider('**Condition quality(int)**', min(data_new["condition"]), max(data_new["condition"]), 1)
input_sqft_above = st.number_input('**Sqft above(int)**', 1, max(data_new["sqft_above"]), 1)
input_sqft_basement = st.number_input('**Size of the basement(float)**', 1, max(data_new["sqft_basement"]), 1)
input_yr_built = st.number_input('**Year house built(int)**', 1, max(data_new["yr_built"]), 1)
input_yr_renovated = st.number_input('**Year house renovated(int)**', 1, max(data_new["yr_renovated"]), 1)

button_style = """
        <style>
        .stButton > button {
            color: black;
            background: white;
            Markdown: Bold;
            width: 200px;
            height: 75px;
        }
        </style>
        """
if st.button("**House price prediction** :moneybag:"):
    #st.snow()
    input_city = np.expand_dims(inp_city, -1)
    input_city = label_encoder.transform(input_city)
    inputs = np.expand_dims(
        [int(input_city), input_bedrooms, input_bathrooms, input_sqft_living, input_sqft_lot, input_floors, input_waterfront, input_condition, input_sqft_above, input_sqft_basement, input_yr_built, input_yr_renovated, input_view], 0)
    prediction = best_gbt_model.predict(inputs)
    print("final predition: ", np.squeeze(prediction, -1))
    
    #st.write(input_city)
    st.write(f"**The house price prediction is: {np.squeeze(prediction, -1):.2f}$** :house_with_garden:")
    st.write(f"*Thank you {st.session_state.name} for using our project! We hope you enjoyed it ^^*")
