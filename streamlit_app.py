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
st.markdown("<p class='subtitle'>By Juliette Wong & Nathan Jen</p>", unsafe_allow_html=True)
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
    inputField = st.sidebar.number_input("What's your " + inputName + ' ?')

    return inputField

st.sidebar.title("Enter Your Own Data!")
income = createNumInput('Income')
creditAmount = createNumInput('Credit Amount')
annuityAmount = createNumInput('Annuity Amount')
famMembers = st.sidebar.slider('How Many Family Members do you Have?', 0, 20, 1)
contractType = createRadioInput("Contract Type", ('Cash Loans', 'Revolving Loans'))
gender = createRadioInput("Gender", ('M', 'F'))
educationLevel = createRadioInput("Highest level of education completed", ("Graduate", "Undergraduate", "Some undergraduate", "High School", "Less than high school"))


################## Classification Model ##################

X = df.drop(columns = ["SK_ID_CURR", "TARGET"])
X = pd.get_dummies(X, drop_first = True)
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42)
oversample = RandomOverSampler(sampling_strategy="minority", random_state = 42)
X_train_o, y_train_o = oversample.fit_resample(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train_o, y_train_o)

userList = [income, creditAmount, annuityAmount, famMembers]

# ContractType
if contractType == 'Cash Loans': userList += [1, 0]
else: userList += [0, 1]

# Gender
if gender == 'M': userList += [0, 1, 0]
else: userList += [1, 0, 0]
    
# Education section?
if educationLevel == 'Graduate': userList += [1, 0, 0, 0, 0]
elif educationLevel == 'Undergraduate': userList += [0, 1, 0, 0, 0]
elif educationLevel == 'Some undergraduate': userList += [0, 0, 1, 0, 0]
elif educationLevel == 'High School': userList +=[0, 0, 0, 0, 1]
else: userList += [0, 0, 0, 1, 0]
 

userList = [userList]

column_names = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CNT_FAM_MEMBERS", "NAME_CONTRACT_TYPE_Cash loans", \
               "NAME_CONTRACT_TYPE_Revolving loans", "CODE_GENDER_F", "CODE_GENDER_M", "CODE_GENDER_XNA", 
               "NAME_EDUCATION_TYPE_Academic_degree", "NAME_EDUCATION_TYPE_Higher education", \
               "NAME_EDUCATION_TYPE_Incomplete higher", "NAME_EDUCATION_TYPE_Lower secondary", \
               "NAME_EDUCATION_TYPE_Secondary / secondary special"]
userData = pd.DataFrame(userList, columns = column_names)

# target = output from model using input cols
target = dt.predict(userData)


################## Input Section ##################

st.markdown("<h2>Where Do You Fit In?</h2>", unsafe_allow_html=True)
st.markdown("<p><i>Enter your own data to see what our model predicts for you!</i></p>", unsafe_allow_html=True)
st.markdown("<p>Here is the data currently given to our model. </p>", unsafe_allow_html=True)
st.write(userData)


if target == 0:
    st.markdown("<h1>Prediction: <span class='success'>No Default! ✅<span></h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 class='default'>Default! ❌</h1>", unsafe_allow_html=True)

targetBar = alt.Chart(df).mark_bar().encode(
                x=alt.X("TARGET", axis=alt.Axis(labelAngle = 0)),
                y='count()',
            )

st.write(targetBar)

################## Univariate Visualizations ##################

st.markdown("<h2>Univariate Exploration</h2>", unsafe_allow_html=True)

### Sample of Data
defaults = df[df['TARGET'] == 1]
nondefaults = df[df['TARGET'] == 0]
num_defaults = int(5000 * 0.080732)
num_nondefaults = 5000 - num_defaults
default_sample = defaults.sample(n = num_defaults, random_state = 40)
nondefault_sample = nondefaults.sample(n = num_nondefaults, random_state = 40)
sample = pd.concat([default_sample, nondefault_sample])

# Fix names
sample['TARGET'] = sample['TARGET'].replace([0],'No Default')
sample['TARGET'] = sample['TARGET'].replace([1],'Default')
for df in [sample, default_sample, nondefault_sample]:
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(['Academic degree'], 'Graduate')
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(['Higher education'], 'Undergrad')
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(['Incomplete Higher'], 'Some Undergrad')
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(['Lower secondary'], 'High School')
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].replace(['Secondary / secondary special'], '< High School')
    #df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].astype('category')
    #df['NAME_EDUCATION_TYPE'].cat.reorder_categories(['Graduate', 'Undergrad', 'Some Undergrad', 'High School', '< High School'])
    

# append user data to col
# sample.append({"SK_ID_CURR": 0, "AMT_INCOME_TOTAL": income, "AMT_CREDIT": creditAmount, "AMT_ANNUITY": annuityAmount, "CNT_FAM_MEMBERS": famMembers, "NAME_CONTRACT_TYPE": contractType, "CODE_GENDER": gender, "NAME_EDUCATION_TYPE": educationLevel})

# TODO: Figure out the colors?
def createSideBySideHistogram(col):   
    hist1 = alt.Chart(default_sample).transform_joinaggregate(
        total='count(*)').transform_calculate(
        pct='1 / datum.total').mark_bar().encode(
        x=alt.X(col, bin = True),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'), title = 'Percent of Total Observations'),
        ).properties(title='Distribution of ' + col + ' - Default')
    
    hist2 = alt.Chart(nondefault_sample).transform_joinaggregate(
        total='count(*)').transform_calculate(
        pct='1 / datum.total').mark_bar().encode(
        x=alt.X(col, bin = True),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'), title = 'Percent of Total Observations'),
        ).properties(title='Distribution of ' + col + ' - Nondefault') 
    st.write(hist1 | hist2)

numericalCols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]
for col in numericalCols:
    createSideBySideHistogram(col)
    
# TODO: Figure out the colors?
def createSideBySideBar(col):
    bar1 = alt.Chart(default_sample).transform_joinaggregate(
        total='count(*)').transform_calculate(
        pct='1 / datum.total').mark_bar().encode(
        x=alt.X(col, axis=alt.Axis(labelAngle = 0)),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'), title = 'Percent of Total Observations'),
        ).properties(width = 415, title='Distribution of ' + col + ' - Default')
    
    bar2 = alt.Chart(nondefault_sample).transform_joinaggregate(
        total='count(*)').transform_calculate(
        pct='1 / datum.total').mark_bar().encode(
        x=alt.X(col, axis=alt.Axis(labelAngle = 0)),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'), title = 'Percent of Total Observations'),
        ).properties(width = 415, title='Distribution of ' + col + ' - Nondefault')
    st.write(bar1 | bar2)
    
categoricalCols = ["CNT_FAM_MEMBERS:O", "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_EDUCATION_TYPE"]
for col in categoricalCols:
    createSideBySideBar(col)

################## Data Exploration ##################

st.markdown("<h2>Multivariate Exploration</h2>", unsafe_allow_html=True)

brush = alt.selection_interval()
brush_scatter = alt.Chart(sample).mark_point().encode(
    y = alt.Y('AMT_INCOME_TOTAL'),
    color=alt.condition(brush,'TARGET:O', alt.value('lightgray'), scale=alt.Scale(scheme='dark2'))
).properties(
    width=400,
    height=400
).add_selection(
    brush
)

st.write(brush_scatter.encode(x='AMT_CREDIT') | brush_scatter.encode(x = 'AMT_ANNUITY'))