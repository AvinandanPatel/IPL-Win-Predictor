import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import streamlit as st

# title of the Page
st.title('IPL Win Predictor')

teams = pkl.load(open('teams.pkl', 'rb'))
city = pkl.load(open('city.pkl', 'rb'))
model = pkl.load(open('model_pipe.pkl', 'rb'))

col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox('Select Batting Team:', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team:', sorted(teams))
with col3:
    city = st.selectbox('Select Venue:', sorted(city))

col4, col5, col6, col7 = st.columns(4)
with col4:
    target = st.number_input('Target Score', min_value = 0, max_value = 720, step = 1)
with col5:
    score = st.number_input('Live Score', min_value = 0, max_value = 720, step = 1)
with col6:
    overs = st.number_input('Overs Completed', min_value = 0, max_value = 20, step = 1)
with col7:
    wickets = st.number_input('Wickets Fell', min_value = 0, max_value = 10, step = 1)

if st.button('Predict the Win Probabilities'):
    target_left = target - score
    remaining_balls = 120 - (overs * 6)
    crr = 0 if overs == 0 else score/overs
    rrr = (target_left/remaining_balls)*6
    
    cols = ['batting_team', 'bowling_team', 'city', 'score', 'wickets', 
            'remaining_balls', 'target_left', 'crr', 'rrr']
    obs = [batting_team, bowling_team, city, score, wickets, remaining_balls, target_left, crr, rrr]
    test_obs = pd.DataFrame(obs, index = cols).T
    # st.write(test_obs)

    # Predicting the result
    result = model.predict_proba(test_obs)[0]
    win_chase = result[1]
    loss_chase = result[0]
    st.subheader(f'{batting_team} - **:blue[{win_chase*100:.0f}%]**')
    st.progress(win_chase)
    st.subheader(f'{bowling_team} - :blue[{loss_chase*100:.0f}%]')
    st.progress(loss_chase)
    












    