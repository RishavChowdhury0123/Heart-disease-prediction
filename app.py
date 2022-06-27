import pandas as pd
import numpy as np
import warnings
import pickle
import streamlit as st
import catboost
warnings.filterwarnings('ignore')

st.set_page_config(page_title= 'Heart attack predictor', layout= 'wide')

def find_outliers(series):
    q1= series.quantile(0.25)
    q3= series.quantile(0.75)
    iqr= q3-q1
    li= q1 - (1.5*iqr)
    ri= q3 + (1.5*iqr)
    return (series < li) | (series > ri)
    
class CustomEncoder:
    def __init__(self, columns=None):
        self.columns= columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        from sklearn.preprocessing import LabelEncoder
        if self.columns is not None:
            X[self.columns]= X[self.columns].apply(LabelEncoder().fit_transform)
        else:
            X= X.apply(LabelEncoder().fit_transform)
        return X
    
    def fit_transform(self, X,y=None):
        return self.fit(X).transform(X)

def predict_heart_disease(bmi, smoking, alcohol, stroke, physicalhealth, mentalhealth, diffwalking, sex, age, race, diabetic, physicalactivity, general_health, sleeptime, asthma, kidneydisease, cancer): 
    X= df.copy()
    features= [bmi, smoking, alcohol, stroke, physicalhealth, mentalhealth, diffwalking, sex, age, race, diabetic, physicalactivity,general_health, sleeptime, asthma, kidneydisease, cancer]
    X= X.append(dict(zip(df.columns, features)), ignore_index=True)
    X.mentalhealth= np.log(X.mentalhealth)
    X.mentalhealth.where(X.mentalhealth != -np.inf, 0, inplace=True)
    X.physicalhealth= np.log(X.physicalhealth)
    X.physicalhealth.where(X.physicalhealth != -np.inf, 0, inplace=True)
    X= pd.get_dummies(X, columns= ['race','genhealth','agecategory'], drop_first=True)
    return pipe.predict_proba(X.iloc[-2:,:])[-1,1]

def convert_age(x):
    if (x >=18) & (x <=24):
        return '18-24'
    elif (x >=25) & (x <=29):
        return '25-29'
    elif (x >=30) & (x <=34):
        return '30-34'
    elif (x >=35) & (x <=39):
        return '35-39'
    elif (x >=40) & (x <=44):
        return '40-44'
    elif (x >=45) & (x <=49):
        return '45-49'
    elif (x >=50) & (x <=54):
        return '50-54'
    elif (x >=55) & (x <=59):
        return '55-59'
    elif (x >=60) & (x <=64):
        return '60-64'
    elif (x >=65) & (x <=69):
        return '65-69'
    elif (x >=70) & (x <=74):
        return '70-74'
    elif (x >=75) & (x <=79):
        return '75-79'
    elif (x >=80):
        return '80 or older'

def in_to_m(x):
    return x/39.37

def cal_bmi(hieght, weight):
    return weight/(height**2)


@st.cache
def load_data():
    path= 'df.pkl'
    with open(path, 'rb') as ref:
        df= pickle.load(ref)
    path= 'model.pkl'
    with open(path, 'rb') as ref:
        pipe= pickle.load(ref)
    return df, pipe

def main():
    df, pipe= load_data()

    st.title('Heart disease predictor')
    cols= st.columns(4)
    sex= cols[0].radio('Select your gender', ['Male','Female'])
    age= cols[1].number_input('Age',value=18, min_value= 18, max_value= 90)
    height= cols[2].number_input('Height ( in inch.)', value= 60, min_value= 50, max_value=120)
    weight= cols[3].number_input('Weight ( in KG.)', value= 60, min_value=30, max_value= 150)

    cols= st.columns(3)
    race=cols[0].selectbox('Which race do you belong to?', df.race.unique())
    gen_health= cols[1].selectbox('How is your general health?', df.genhealth.unique())
    stroke= cols[2].radio('Ever had a stroke before?', ['Yes','No'])

    cols= st.columns(4)
    phy_active= cols[0].radio('Are you physically active?', ['Yes','No'])
    smoker= cols[1].radio('Has smoked more than 100 cigarettes?', ['Yes','No'])
    drinker= cols[2].radio('Are you a heavy drinker?', ['Yes','No'])
    diabetic= cols[3].radio('Are you diabetic?', ['Yes','No'])

    cols= st.columns(5)
    diff_walking = cols[0].radio('Do you have difficulty in walking?', ['Yes','No'])
    asthma= cols[1].radio('Do you have asthma?', ['Yes','No'])
    kidney= cols[2].radio('Do you have kidney disease?', ['Yes','No'])
    cancer= cols[3].radio('Do you have cancer', ['Yes','No'])


    cols= st.columns(3)
    sleep_time= cols[0].slider('On an average, how long do you sleep?',value=8, min_value= 1, max_value=20)
    phy_fit= cols[1].slider('How many days in a month were you physically unhealthy?', min_value= 0, max_value= 30)
    mental_fit= cols[2].slider('How many days in a month were you mentally unhealthy?', min_value= 0, max_value= 30)

    age= convert_age(age)
    bmi= cal_bmi(in_to_m(height), weight)

    btn= st.button('Predict')

    if btn:
        pred= predict_heart_disease(bmi, smoker, drinker, stroke, phy_fit, mental_fit, diff_walking, sex, age, 
                            race, diabetic,phy_active,gen_health,sleep_time, asthma, kidney, cancer)
        st.markdown('You have {:.0%} chance of having heart attack.'.format(pred))


if __name__=='__main__':
    main()
