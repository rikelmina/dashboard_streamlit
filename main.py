import streamlit as st

siteHeader=st.beta_container()
with siteHeader:
    st.title('Modelo de Evaluación de ingresos')
    st.markdown("""En este proyecto se busca encontrar cuáles son las variables principales que pueden predecir""")
    
    
newFeatures = st.beta_container()
with newFeatures:
    st.header( 'Nuevas variables: '   )
    st.text('Demo')
    

from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('./income.csv')
data=df[['age','workclass','education']]
st.write(df.sample(5))

st.subheader('Distribuciones: ')
st.text('Esta es la gráfica de distribución por edad')

distribution_age=pd.DataFrame(df['age'].value_counts())
st.bar_chart(distribution_age)

distribution_sex=pd.DataFrame(df['sex'].value_counts())
st.bar_chart(distribution_sex)

modelTraining = st.beta_container()
with modelTraining:
    st.header('Entrenamiento del modelo')
    st.text('En esta sección puedes seleccionar la profundidad del modelo')
    
df = df.drop(['Unnamed: 0','income','fnlwgt','capital-gain','capital-loss','native-country'], axis=1)
Y = df['income_bi']
df = df.drop(['income_bi'], axis=1)

X = pd.get_dummies(df, columns = ['race','sex','workclass','education','education-num','marital-status','occupation','relationship'])


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state= 15)

max_depth_in = st.slider ('¿Cuál debería ser el valor de max_depth para el modelo?', min_value=1, max_value=10, value=2, step=1)


t=tree.DecisionTreeClassifier(max_depth=max_depth_in)
model=t.fit(x_train,y_train)

prediction=model.predict(x_test)
score=model.score(x_train,y_train)
st.header('Performance del Modelo')
st.text('Score:') 
st.text(score)


st.markdown( """ <style>
 .main {
 background-color: #AF9EC;
}
</style>""", unsafe_allow_html=True )
