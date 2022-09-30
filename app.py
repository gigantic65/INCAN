import streamlit as st

# Custom imports 
from multipage import MultiPage

from pages import home, Prediction_app


# rest of the code

st.set_page_config(
     page_title="색상 배합 최적화 App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
)

# Create an instance of the app 

app = MultiPage()

#st.set_page_config(layout="wide")
    
# Title of the main page
st.markdown("<h2 style='text-align: center; background-color:#0e4194; color: white;'> 색상 배합 최적화 프로그램 </h2>", unsafe_allow_html=True)



# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("Incan 색상 배합 최적화 Model", Prediction_app.app)
#app.add_page("PCM 색상 배합 최적화 Model", Prediction_app2.app)

#app.add_page("Predict New Conditions", Prediction_app.app)
#app.add_page("Monitoring", Monitor_app.app)


# The main app
app.run()