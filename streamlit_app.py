import streamlit as st
import pandas as pd
import altair as alt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

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
df.loc[df['CNT_FAM_MEMBERS'].isnull()] = 0
df = df.dropna(subset = ['AMT_ANNUITY'])

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

################## Classification Model ##################

X = df.drop(columns = ['TARGET'])
X = pd.get_dummies(X, drop_first = True)
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)
oversample = RandomOverSampler(sampling_strategy='minority', random_state = 42)
X_train_o, y_train_o = oversample.fit_resample(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train_o, y_train_o)


################## Visualizations ##################

### Sample of Data
defaults = df[df['TARGET'] == 1]
nondefaults = df[df['TARGET'] == 0]
num_defaults = int(5000 * 0.080732)
num_nondefaults = 5000 - num_defaults
default_sample = defaults.sample(n = num_defaults)
nondefault_sample = nondefaults.sample(n = num_nondefaults)
sample = pd.concat([default_sample, nondefault_sample])

st.markdown("<h2>Data Exploration</h2>", unsafe_allow_html=True)

st.subheader('Distribution of Contract Types')
chart = alt.Chart(sample).mark_bar().encode(
    x=alt.X("NAME_CONTRACT_TYPE", scale=alt.Scale(zero=False)),
    y=alt.Y("count()", scale=alt.Scale(zero=False)),
    color=alt.Y("TARGET:O")
)
st.write(chart)

# Append the new data to the existing chart.
# chart.add_rows(userData)

st.write('The current income is ', income)
scatter = alt.Chart(sample).mark_point().encode(
     x=alt.X("AMT_INCOME_TOTAL", scale=alt.Scale(zero=False)),
     y=alt.Y("NAME_EDUCATION_TYPE", scale=alt.Scale(zero=False)),
     color=alt.Y("TARGET:O")
).properties(
     width=600, height=400
 ).interactive()

st.write(scatter)

heatmap = alt.Chart(sample).mark_rect().encode(
    alt.X("AMT_CREDIT:Q", bin=True),
    alt.Y("AMT_ANNUITY:Q", bin=True),
    color='count()'
)

st.write(heatmap)