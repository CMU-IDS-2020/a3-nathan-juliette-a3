import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix

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

if st.checkbox("Show First 5 Rows of Data"):
    st.write(df.head())


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

targetVal = 'No Default'
if target == 1: targetVal = 'Default'

targetChart = alt.Chart(sample).mark_bar().encode(
    y='TARGET:O',
    x='count(TARGET)',
    color=alt.condition(
        alt.datum.TARGET == targetVal,
        alt.value('#f43666'),     # same hex as color for radio buttons
        alt.value('gray'))
).properties(width = 600, height = 200)

st.write(targetChart)

default_sample['INCOME_VAL'] = income
default_sample['CREDIT_VAL'] = creditAmount
default_sample['ANNUITY_VAL'] = annuityAmount
nondefault_sample['INCOME_VAL'] = income
nondefault_sample['CREDIT_VAL'] = creditAmount
nondefault_sample['ANNUITY_VAL'] = annuityAmount

def createSideBySideHistogram(col, colValue):   
    title = col.replace("_", " ").title()
    
    base1 = alt.Chart(default_sample)
    hist1 = base1.transform_joinaggregate(
        total='count(*)').transform_calculate(
        pct='1 / datum.total').mark_bar().encode(
        x=alt.X(col, bin = True),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'), title = 'Percent of Total Observations'),
        color=alt.value('gray')
        ).properties(width = 300, title='Distribution of ' + title + ' - Default')
    rule1 = base1.mark_rule(color='#f43666').encode(
    x=alt.X(colValue),
    size=alt.value(4))
 
    base2 = alt.Chart(nondefault_sample)
    hist2 = base2.transform_joinaggregate(
        total='count(*)').transform_calculate(
        pct='1 / datum.total').mark_bar().encode(
        x=alt.X(col, bin = True),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'), title = 'Percent of Total Observations'),
        color=alt.value('gray')
        ).properties(width = 300, title='Distribution of ' + title + ' - Nondefault') 
    rule2 = base2.mark_rule(color='#f43666').encode(
    x=alt.X(colValue),
    size=alt.value(4))
    st.write(hist1 + rule1 | hist2 + rule2)

numericalCols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]
numericalVals = ["INCOME_VAL", "CREDIT_VAL", "ANNUITY_VAL"]
for i in range(len(numericalCols)):
    createSideBySideHistogram(numericalCols[i], numericalVals[i])


    
educationCode = educationLevel
if educationLevel == 'Undergraduate': educationCode = 'Undergrad'
elif educationLevel == 'Some undergraduate': educationCode = 'Some Undergrad'
elif educationLevel == 'Less than high school': educationCode = '< High School'
    
    
def createSideBySideBar(col):
    title = col.split(':')[0].replace("_", " ").title()
    condition = (alt.datum.NAME_CONTRACT_TYPE == contractType) | (alt.datum.CODE_GENDER == gender) | \
                (alt.datum.NAME_EDUCATION_TYPE == educationCode) | (alt.datum.CNT_FAM_MEMBERS == famMembers)
    bar1 = alt.Chart(default_sample).transform_joinaggregate(
        total='count(*)').transform_calculate(
        pct='1 / datum.total').mark_bar().encode(
        x=alt.X(col, axis=alt.Axis(labelAngle = 0)),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'), title = 'Percent of Total Observations'),
        color=alt.condition(
        condition,
        alt.value('#f43666'),     # same hex as color for radio buttons
        alt.value('gray'))
        ).properties(width = 300, title='Distribution of ' + title + ' - Default')
    
    bar2 = alt.Chart(nondefault_sample).transform_joinaggregate(
        total='count(*)').transform_calculate(
        pct='1 / datum.total').mark_bar().encode(
        x=alt.X(col, axis=alt.Axis(labelAngle = 0)),
        y=alt.Y('sum(pct):Q', axis=alt.Axis(format='%'), title = 'Percent of Total Observations'),
        color=alt.condition(
        condition,
        alt.value('#f43666'),     # same hex as color for radio buttons
        alt.value('gray'))
        ).properties(width = 300, title='Distribution of ' + title + ' - Nondefault')
    st.write(bar1 | bar2)

    
categoricalCols = ["CNT_FAM_MEMBERS:O", "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_EDUCATION_TYPE"]
for col in categoricalCols:
    createSideBySideBar(col)

################## Data Exploration ##################

st.markdown("<h2>Multivariate Exploration</h2>", unsafe_allow_html=True)


domain = ['Default', 'No Default']
range_ = ['#800080', 'steelblue']

brush = alt.selection_interval()
brush_scatter = alt.Chart(sample).mark_circle(opacity = 0.5).encode(
    y = alt.Y('AMT_INCOME_TOTAL'),
    color=alt.condition(brush,'TARGET:N', alt.value('lightgray'), scale=alt.Scale(domain=domain, range=range_))
).properties(
    width=350,
    height=350
).add_selection(
    brush
)

bars = alt.Chart(sample).mark_bar().encode(
    y='TARGET:O',
    color='TARGET:O',
    x='count(TARGET)'
).transform_filter(
    brush
).properties(width = 700)

st.write(bars & (brush_scatter.encode(x='AMT_CREDIT') | brush_scatter.encode(x = 'AMT_ANNUITY')))

################## Model Exploration ##################

st.markdown("<h2>Model Exploration</h2>", unsafe_allow_html=True)

feature_names = X.columns
feature_importances = dt.feature_importances_
features = pd.DataFrame({'Feature Name':feature_names, 'Feature importance':feature_importances})

feature_chart = alt.Chart(features).mark_bar().encode(
    y=alt.Y('Feature Name:O', sort='-x'),
    x='Feature importance:Q',
    color=alt.value('#f43666')
).properties(width = 600)
st.write(feature_chart)

pred = dt.predict(X_test)

x_val, y_val = np.meshgrid(range(0, 2), range(0, 2))
z = confusion_matrix(pred, y_test)
source = pd.DataFrame({'Predicted': x_val.ravel(),
                     'Actual': y_val.ravel(),
                     'z': z.ravel()})

confusion_mat = alt.Chart(source).mark_rect().encode(
    x='Predicted:O',
    y='Actual:O',
    color=alt.Color('z:Q', scale=alt.Scale(scheme='purplered'))
).properties(title = 'Confusion matrix - test values', width = 300, height = 300)
st.write(confusion_mat)