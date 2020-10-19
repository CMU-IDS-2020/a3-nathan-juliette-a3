import streamlit as st
import pandas as pd
import altair as alt

################## CSS Stuff ##################

# load in css file (from: https://discuss.streamlit.io/t/colored-boxes-around-sections-of-a-sentence/3201/2) 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style.css")

################## Intro ##################

st.title("Analyzing Credit Card Defaults")
st.markdown("<h2>What is a Default?</h2>", unsafe_allow_html=True)

defaultDescription ="""
                        <p class='important-container'>
                            A <b>default</b> is a failure to make a payment on a credit card bill
                            by the due date. The usual consequence for a default is a raise in 
                            interest rates to the default, or decrease the line of credit.
                        </p>
                        <p>
                            Our Data: <a href='https://www.kaggle.com/mishra5001/credit-card?select=application_data.csv&fbclid=IwAR1BFzFdio_1DgfBYb_tc7uf6sCKYB4Ajz3aqUeqrEmkn41-J0hpX5HWFNk'>Source</a>
                        </p>
                    """
st.markdown(defaultDescription, unsafe_allow_html=True)
@st.cache
def load_data(url):
    return pd.read_csv(url)

df = load_data("./application_data_min.csv")

if st.checkbox("Show Raw Data"):
    st.write(df)


################## Sidebar ##################

def createRadioInput(inputName, inputSelections):
    field = st.sidebar.radio("What's your " + inputName + "?",
    inputSelections)

    return field

def createNumInput(inputName):
    inputField = st.sidebar.number_input('Whats your ' + inputName + ' ?')

    return inputField

st.sidebar.title("Enter Your Own Data!")
contractType = createRadioInput("Contract Type", ('Cash Loans', 'Revolving Loans'))
gender = createRadioInput("Gender", ('M', 'F'))
income = createNumInput('income')
creditAmount = createNumInput('Credit Amount')
annuityAmount = createNumInput('Annuity Amount')
famMembers = st.sidebar.slider('How Many Family Members do you Have?', 0, 20, 1)


################## Input Section ##################

st.markdown("<h2>Where Do You Fit In?</h2>", unsafe_allow_html=True)
st.markdown("<p>Enter you data in the sidebar to see if our model will predict whether you will default or not.</p>", unsafe_allow_html=True)


# target = (output from model using input cols)
target = 0

if target == 0:
    st.markdown("<h1 class='success'>No Default! ✅</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 class='default'>Default! ❌</h1>", unsafe_allow_html=True)


userData = (contractType, gender, income, creditAmount, annuityAmount, famMembers)


################## Visualizations ##################

st.markdown("<h2>Data Exploration</h2>", unsafe_allow_html=True)

st.subheader('Distribution of Contract Types')
chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("NAME_CONTRACT_TYPE", scale=alt.Scale(zero=False)),
    y=alt.Y("count(NAME_CONTRACT_TYPE)", scale=alt.Scale(zero=False)),
    color=alt.Y("TARGET")
)
st.write(chart)

# Append the new data to the existing chart.
# chart.add_rows(userData)

# income = st.number_input('Whats your income?')
# st.write('The current income is ', income)
# scatter = alt.Chart(df).mark_point().encode(
#     x=alt.X("AMT_INCOME_TOTAL", scale=alt.Scale(zero=False)),
#     y=alt.Y("NAME_EDUCATION_TYPE", scale=alt.Scale(zero=False)),
#     color=alt.Y("TARGET")
# )
# ).properties(
#     width=600, height=400
# ).interactive()

# st.write(scatter)
