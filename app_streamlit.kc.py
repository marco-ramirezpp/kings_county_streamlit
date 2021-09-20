import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model


model = load_model('kc_model-091321')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('kings_county.jpg')


    st.image(image, use_column_width=False)
    add_selectbox = st.sidebar.selectbox('Tipo de predicción', ('Online', 'csv'))

    st.sidebar.info('App para predecir viviendas en Kings County')
    st.title('Kings County Predicciones')

    if add_selectbox == 'Online':
        price = st.number_input('price', min_value=1, value=100000)
        bedrooms = st.number_input('bedrooms', min_value=1, value=2)
        bathrooms = st.number_input('bathrooms', min_value=1, value=1)
        sqft_living = st.number_input('sqft_living', min_value=100, value=100)
        floors = st.number_input('floors', min_value=1, value=1)
        sqft_lot = st.number_input('sqft_lot', min_value=150, value=150)
        view = st.selectbox('view', [0,1])
        waterfront = st.number_input('waterfront', min_value=1, value=2)
        condition = st.number_input('condition', min_value=1, max_value=5 ,value=3)
        grade = st.number_input('grade', min_value=1, max_value=8 ,value=4)
        lat = st.number_input('lat', min_value=47.1559, max_value=47.7776, value=47.4667)
        long = st.number_input('long', min_value=-122.519, max_value= -121.315 ,value=-121.917)
        recency = st.number_input('recency', min_value=0, max_value=0 ,value=0)

        output= ""

        input_dict = {'price': price, 'bedrooms': bedrooms, 'bathrooms' : bathrooms,
                      'sqft_living': sqft_living, 'floors': floors, 'sqft_lot': sqft_lot,
                      'view': view, 'condition': condition, 'grade': grade, 'lat':lat,
                      'long':long, 'waterfront': waterfront ,'recency': recency}

        input_df = pd.DataFrame([input_dict])

        if st.button('Predicción'):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('La predicción es: {}'.format(output))

    if add_selectbox == 'csv':

        file_upload = st.file_uploader('subir csv', type=['csv'])

        if file_upload is not None:
            data= pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()


