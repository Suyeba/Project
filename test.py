import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the trained model
pickle_in = open('modellog.pkl', 'rb')
model = pickle.load(pickle_in)

# Define the HTML content with color styling for the title
title_html = """
    <div style="background-color: #025246; padding: 10px;">
        <h1 style="color: white; text-align: center;">Terror Attack Prediction</h1>
    </div>
"""

# Render the title using st.markdown() with unsafe_allow_html=True
st.markdown(title_html, unsafe_allow_html=True)

# Streamlit app title and introduction
st.write("This app predicts whether a terrorist attack will be minor or major based on user inputs.")

# Define the input fields for the user
day_of_week = st.text_input('Day of Week', '')
location = st.text_input('Location', '')
attack_type = st.selectbox('Attack Type', ['shooting', 'bombing', 'kidnapping', 'hijacking', 'stabbing', 'arson', 'assassination'])
perpetrator = st.selectbox('Perpetrator', ['Group A', 'Group B', 'Group C', 'Group D', 'Group E', 'Group F', 'Group G',
                       'Group H', 'Group I', 'Group J', 'Group K', 'Group L', 'Group M', 'Group N',
                       'Group O', 'Group P', 'Group Q', 'Group R', 'Group S', 'Group T', 'Group U',
                       'Group V', 'Group W', 'Group X', 'Group Y', 'Group Z'])
victims_injured = st.number_input('Victims Injured', min_value=0)
victims_deceased = st.number_input('Victims Deceased', min_value=0)
target_type = st.selectbox('Target Type', ['Civilians', 'Government Officials', 'Infrastructure', 'Police', 'Tourists'])
weapon_used = st.selectbox('Weapon Used', ['Chemical', 'Bladed Weapon', 'Incendiary', 'Firearms', 'Explosives', 'Melee'])
claimed_by = st.selectbox('Claimed By', ['Group A', 'Group B', 'Group C', 'Group D', 'Group E', 'Group F', 'Group G',
                       'Group H', 'Group I', 'Group J', 'Group K', 'Group L', 'Group M', 'Group N',
                       'Group O', 'Group P', 'Group Q', 'Group R', 'Group S', 'Group T', 'Group U',
                       'Group V', 'Group W', 'Group X', 'Group Y', 'Group Z'])
motive = st.selectbox('Motive', ['Religious', 'Ethnic', 'Retaliation Motive', 'Political', 'Unknown'])
operational_success = st.selectbox('Operational Success', ['Yes', 'No'])
financial_support = st.selectbox('Financial Support', ['International', 'Local', 'Unknown'])
country = st.text_input('Country', '')


# Define a function to make predictions
def predict_major_incident(day_of_week, location, attack_type, perpetrator, victims_injured, victims_deceased,
                           target_type, weapon_used, claimed_by, motive, operational_success, financial_support,
                           country):
    # Create a dictionary to map categorical values to their one-hot encoded representations
    category_mapping = {
        'day_of_week': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
        'location': ['mumbai', 'athens', 'paris', 'new york', 'sydney', 'seoul', 'tokyo', 'berlin', 'mexico city', 'bangkok', 'moscow',
                     'beijing', 'toronto', 'lima', 'rio de janeiro', 'cape town', 'madrid', 'istanbul', 'dubai', 'nairobi', 'rome',
                     'london', 'cairo', 'buenos aires', 'jakarta'],
        'attack_type': ['shooting', 'bombing', 'kidnapping', 'hijacking', 'stabbing', 'arson', 'assassination'],
        'perpetrator': ['Group A', 'Group B', 'Group C', 'Group D', 'Group E', 'Group F', 'Group G',
                       'Group H', 'Group I', 'Group J', 'Group K', 'Group L', 'Group M', 'Group N',
                       'Group O', 'Group P', 'Group Q', 'Group R', 'Group S', 'Group T', 'Group U',
                       'Group V', 'Group W', 'Group X', 'Group Y', 'Group Z'],
        'target_type': ['Civilians', 'Government Officials', 'Infrastructure', 'Police', 'Tourists'],
        'weapon_used': ['Chemical', 'Bladed Weapon', 'Incendiary', 'Firearms', 'Explosives', 'Melee'],
        'claimed_by': ['Group A', 'Group B', 'Group C', 'Group D', 'Group E', 'Group F', 'Group G',
                       'Group H', 'Group I', 'Group J', 'Group K', 'Group L', 'Group M', 'Group N',
                       'Group O', 'Group P', 'Group Q', 'Group R', 'Group S', 'Group T', 'Group U',
                       'Group V', 'Group W', 'Group X', 'Group Y', 'Group Z'],
        'motive': ['Religious', 'Ethnic', 'Retaliation Motive', 'Political', 'Unknown'],
        'operational_success': ['Yes', 'No'],
        'financial_support': ['International', 'Local', 'Unknown'],
        'country': ['turkey', 'kenya', 'peru', 'japan', 'france', 'usa', 'brazil', 'spain',
                    'south africa', 'thailand', 'mexico', 'egypt', 'germany', 'australia', 'india',
                    'uk', 'indonesia', 'argentina', 'south korea', 'russia', 'canada', 'italy',
                    'uae', 'greece', 'china']
    }

    # Initialize the input_data with zeros for all features
    input_data = np.zeros(205)

    # Set the corresponding one-hot encoded feature values to 1
    input_data[category_mapping['day_of_week'].index(day_of_week)] = 1

    input_data[len(category_mapping['day_of_week']) + category_mapping['location'].index(location)] = 1

    input_data[len(category_mapping['day_of_week']) + len(category_mapping['location']) + category_mapping[
        'attack_type'].index(attack_type)] = 1

    input_data[len(category_mapping['day_of_week']) + len(category_mapping['location']) + len(
        category_mapping['attack_type']) + category_mapping['perpetrator'].index(perpetrator)] = 1

    input_data[len(category_mapping['day_of_week']) + len(category_mapping['location']) + len(
        category_mapping['attack_type']) + len(category_mapping['perpetrator']) + category_mapping['target_type'].index(
        target_type)] = 1

    input_data[len(category_mapping['day_of_week']) + len(category_mapping['location']) + len(
        category_mapping['attack_type']) + len(category_mapping['perpetrator']) + len(category_mapping['target_type']) +
        category_mapping['weapon_used'].index(weapon_used)] = 1

    input_data[len(category_mapping['day_of_week']) + len(category_mapping['location']) + len(
        category_mapping['attack_type']) + len(category_mapping['perpetrator']) + len(
        category_mapping['target_type']) + len(category_mapping['weapon_used']) + category_mapping['claimed_by'].index(
        claimed_by)] = 1

    input_data[len(category_mapping['day_of_week']) + len(category_mapping['location']) + len(
        category_mapping['attack_type']) + len(category_mapping['perpetrator']) + len(
        category_mapping['target_type']) + len(category_mapping['weapon_used']) + len(category_mapping['claimed_by']) +
        category_mapping['motive'].index(motive)] = 1

    input_data[len(category_mapping['day_of_week']) + len(category_mapping['location']) + len(
        category_mapping['attack_type']) + len(category_mapping['perpetrator']) + len(
        category_mapping['target_type']) + len(category_mapping['weapon_used']) + len(
        category_mapping['claimed_by']) + len(category_mapping['motive']) + category_mapping[
        'operational_success'].index(operational_success)] = 1

    input_data[len(category_mapping['day_of_week']) + len(category_mapping['location']) + len(
        category_mapping['attack_type']) + len(category_mapping['perpetrator']) + len(
        category_mapping['target_type']) + len(category_mapping['weapon_used']) + len(
        category_mapping['claimed_by']) + len(category_mapping['motive']) + len(
        category_mapping['operational_success']) + category_mapping['financial_support'].index(financial_support)] = 1

    input_data[len(category_mapping['day_of_week']) + len(category_mapping['location']) + len(
        category_mapping['attack_type']) + len(category_mapping['perpetrator']) + len(
        category_mapping['target_type']) + len(category_mapping['weapon_used']) + len(
        category_mapping['claimed_by']) + len(category_mapping['motive']) + len(
        category_mapping['operational_success']) + len(category_mapping['financial_support']) +
        category_mapping['country'].index(country)] = 1

    # Add other numerical features to the input_data
    input_data[196] = victims_injured
    input_data[197] = victims_deceased

    # Make a prediction using the model
    prediction = model.predict([input_data])

    return prediction[0]


# Make a prediction when the user clicks the "Predict" button
if st.button("Predict"):
    prediction_result = predict_major_incident(day_of_week, location, attack_type, perpetrator, victims_injured,
                                               victims_deceased, target_type, weapon_used, claimed_by, motive,
                                               operational_success, financial_support, country)

    if prediction_result == 1:
        st.success('The prediction is that it will be a major terrorist attack.')
    else:
        st.success('The prediction is that it will be a minor terrorist attack.')

# Optionally, you can add additional information or explanations to the app using st.write or st.markdown.
