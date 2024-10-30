#import some necessary librairies
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
import math
from sklearn.metrics import r2_score, mean_squared_error

#Reading Data
df = pd.read_csv('/content/data.csv')
df.head()
df.shape
df.columns
df.info()

for col in df.columns:
    print( col,':', df[col].nunique() )
    print(df[col].value_counts().nlargest(5))
    print('\n' + '*' * 20 + '\n')
    
# Some Cleaning
df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)
print(string_columns)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
    
df.columns
df.rename(columns = {'msrp': 'price'}, inplace = True)
df.head()

#Exploraty data analysis
pd.options.display.float_format = '{:,.2f}'.format
df.describe().T

df.describe(include=['O'])

#Target Variable Analysis (Price)
plt.figure(figsize=(15, 7))

sns.histplot(df.price, bins=40)
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.title('Distribution of prices')

plt.show()

plt.figure(figsize=(6, 4))

sns.histplot(df.price[df.price < 80000], bins=40)
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.title('Distribution of prices')

plt.show()

# Log Transformation
df['log_price'] = np.log1p(df.price)

plt.figure(figsize=(6, 4))

sns.histplot(df.log_price, bins=40)
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Distribution of prices after log tranformation')

plt.show()
df.price.skew()
df.log_price.skew()

#Check Missing Value
df.isnull().sum()
df.drop('market_category', axis = 1, inplace = True)
null_values = df[df.isnull().any(axis = 1)]
null_values

df['engine_fuel_type'] = df['engine_fuel_type'].fillna('regular unleaded')

df['engine_hp'] = df['engine_hp'].fillna(0)

df['engine_cylinders'] = df['engine_cylinders'].fillna(0)

df['number_of_doors'] = df['number_of_doors'].fillna(df['number_of_doors'].mean())

# Now let's seperate the numerical and categorical columns for using later.
num_col = df.select_dtypes(include = [np.number])
cat_col = df.select_dtypes(exclude = [np.number])

df.drop(df[df['transmission_type']=='unknown'].index, axis='index', inplace = True)

# Handling Outlier
for i in num_col:
    fig = px.box(df, x = df[i])
    fig.update_traces(fillcolor = '#C0A56C')
    fig.show()
    
s1 = df.shape
clean = df[['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'price']]
for i in clean.columns:
    qt1 = df[i].quantile(0.25)
    qt3 = df[i].quantile(0.75)
    iqr =  qt3 - qt1
    lower = qt1-(1.5*iqr)
    upper = qt3+(1.5*iqr)
    min_in = df[df[i]<lower].index
    max_in = df[df[i]>upper].index
    df.drop(min_in, inplace = True)
    df.drop(max_in, inplace = True)
s2 = df.shape
outliers = s1[0] - s2[0]
print("Deleted outliers are : ", outliers)

fig = px.box(df, x = df['engine_hp'])
fig.update_traces()

df.describe().T

for i in df:
    fig = px.histogram(df, x= i, color_discrete_sequence = ["#13f707"])
    fig.show()

fig = px.scatter(df, x='year', y='price', color='engine_cylinders')
fig.show()

df.columns

fig = px.scatter(df, x = 'engine_hp', y = 'price', color = 'engine_cylinders')
fig.show()

fig = px.scatter(df, x = 'engine_cylinders', y = 'price', color = 'engine_cylinders')
fig.show()

fig = px.scatter(df, x = 'city_mpg', y = 'price', color = 'engine_cylinders')
fig.show()

fig = px.scatter(df, x = 'highway_mpg', y = 'price', color = 'engine_cylinders')
fig.show()

fig = px.scatter(df, x = 'popularity', y = 'price', color = 'engine_cylinders')
fig.show()

plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(numeric_only = True), annot=True)
plt.show()

df.columns

# Preprocessing
label_encoder = LabelEncoder()

for column in cat_col:
    df[column] = label_encoder.fit_transform(df[column])
df

#Splitting the dataset into train and test split. I'll keep the test size of 0.2.

X = df.drop('price', axis = 1)   #features
y = df['price']        #target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Modeling
rfr = RandomForestRegressor(n_estimators = 40)
rfr_algo = make_pipeline(rfr)

rfr_algo.fit(X_train, y_train)
rfr_pred = rfr_algo.predict(X_test)

print('R2 Score is : ', r2_score(y_test, rfr_pred))
print('Mean squared error is : ', math.sqrt(mean_squared_error(y_test, rfr_pred)))

plt.figure(figsize=(10,10))
plt.ylabel("Predicted Value")

# Pass y_test as x and rfr_pred as y
sns.regplot(x=y_test, y=rfr_pred, fit_reg=True, scatter_kws={"s": 100})

plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#Applying Linear Regression Model
LinearRegressionModel = LinearRegression(fit_intercept=True,copy_X=True,n_jobs=-1)
LinearRegressionModel.fit(X_train, y_train)

#Calculating Prediction
y_pred = LinearRegressionModel.predict(X_test)
print('Predicted Value for Linear Regression is : ' , y_pred[:10])


#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
print('Mean Squared Error Value is : ', MSEValue)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared Score: {r2}")

plt.figure(figsize=(10,10))
plt.ylabel("Predicted Value")

# Pass y_test as x and rfr_pred as y
sns.regplot(x=y_test, y=y_pred, fit_reg=True, scatter_kws={"s": 100})

plt.show()
