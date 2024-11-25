import streamlit as st
import joblib
import pandas as pd

def main():
    st.title('mia prima app')
    st.text('ciao a rewrewrewrewrewrewr')
    model_pipe = joblib.load('titanic_pipe.pkl')


    Age= st.number_input('inserisci età', 0, 34, 1)
    Embarked = st.selectbox('inserisci imbarco', ('Q','S','C'))
    Fare = st.number_input('inserisci ticket', 0, 100, 1)
    Parch = st.number_input('inserisci figli', 0, 10, 1)
    Pclass = st.number_input('inserisci classe', 0, 3, 1)
    Sex = st.selectbox('inserisci sesso', ('female','male'))
    SibSp = st.number_input('inserisci SibSp', 0, 10, 1)

    data = {
            "Pclass": [Pclass],
            "Sex": [Sex],
            "Age": [Age],
            "SibSp": [SibSp],
            "Parch": [Parch],
            "Fare": [Fare],
            "Embarked": [Embarked]
            }
    classes = {0:'died',
            1:'survived',
            }

    res = model_pipe.predict(pd.DataFrame(data)).astype(int)[0]
    print(res)

    y_pred = classes[res]
    st.success(f'la predizione è {y_pred}')

if __name__=="__main__":
    main()