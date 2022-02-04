"""
Data Handling - Cheat Sheet
1. Data Loading & Storage
    A. Read Files into Python (+Index and rename)
    B. Reshape/Pivoting DataSets
    C. Merge Datasets
2. Exploratory Data Analysis
3. Data Cleaning & Preperation (Feature Engineering)
    A. Handle Missing Values
    B. Remove Duplicates
    C. Filtering Outlier
    D. Handle Inconsistent Entries
    E. Transform Values in Numerical Form
    F. Convert date object to datetime
    G. Standardize/Normalize (Feature Scaling)
    H. Create Pipeline
4. Descriptive Statistics
5. Split into Training & Test Set
6. Prediction & Classification (supervised/unsupervised learning)
    A. Create Model 
    B. Model Fitting
    C. Model Prediction
    D. Model Evaluation
    E. Hyperparameter Tuning
    F. Deep Learning
7. Data Visualization
    A. Plotting with Matplotlib
    B. Plotting with Seaborn
"""

######################### 1. Data Loading & Storage ##########################
#Import Packages
import os
import pandas as pd
import numpy as np
import datetime 

###############A. Read Files into Python
#Create Workspace
os.chdir("C:/Users/yate9397/Desktop/Interview Preperation/data/")
base_dir = "C:/Users/yate9397/Desktop/Interview Preperation/data/"
datafile = "C:/Users/yate9397/Desktop/Interview Preperation/data/ex5.csv"

#Read data
df = pd.read_csv('ex6.csv', parse_dates=['']/True, header=None, index_col='', na_values='' )
df = pd.read_excel("ex6.xlsx")

#Reading Files in Pieces
chunker = pd.read_csv('ex6.csv', chunksize=1000)
for count, chunk in enumerate(chunker):
    print(count)
    df = chunk 
    
###Rename Columns, Indexes    
#Indexing
df.set_index(['a','b'])
df.reset_index()

#Rename
df.index.names = ['Ind_Name']   #Renaming Axis Indexes
data.rename(index={'X': 'Y'},
            columns={'A': 'B'}
            inplace=True)       #Rename Index and Columns (inplace change in actual data)


###############B. Reshape/Pivoting DataSets
#Reshaping from 'Long' to 'Wide' Format
data.unstack('index0')

#Reshaping from 'Wide' to 'Long' Format
wide.stack(1)

#Reshaping “Wide” to “Long” Format
df.melt(var_name['Name_Var'], value_name=['Value_Name'])

###############C. Merge DataSets
#Database-Style DataFrame Joins
df1.merge(df2, on='key')                         #Merges Database df1 and df2 on 'key' (inner join)
df1.merge(df2, left_on='lkey', right_on='rkey')  #Merges Database df1 and df2 on 'lkey' and 'rkey'
df1.merge(df2, how='outer')                      #Outer join
df1.merge(df2, how='left')                       #Left join
df1.merge(df2, how='inner')                      #Inner join
df1.merge(df2, on=['key1', 'key2'], suffixes=('_left', '_right'))   #Merge on 2 keys

#Merging on Index
df1.merge(df2, left_on='key', right_index=True)    #Right value is index
df1.merge(df2, left_index=True, right_index=True)  #Both merged on index
df1.join(df2, how='outer')                         #Join combines df based on index
df1.join(df2, on='key')
df1.join([df2, df3])

#Concatenating Along an Axis
np.concatenate([arr, arr])                      #Concatenate arr (vertically)
np.concatenate([arr, arr], axis=1)              #Concatenate arr along axis1 (horizontally)
np.concatenate([arr, arr], join='inner')        #Only elements that are contained in both datasets
np.hstack([arr, arr])                           #Join horizontally

#Combining Data with Overlap
np.where(pd.isnull(a), b, a)                    #When true, yield b, otherwise a
b[:-2].combine_first(a[2:]
                     
#Save Final Preperad Data
df.to_csv('data.csv', index=False)
    
    
####################### 2. Exploratory Data Analysis ##########################
###Data Eploration
#View head/tail of data 
df.head()
df.tail()
#Check for data type of columns
df.info()
#Check shape of data 
df.shape
#Check for null values
df.isnull().sum()
# Get unique count for each variable
df.nunique()
#View Values Count of Features
df['feature1'].value_counts() 
#View Unique Categorical Features
df['feature1'].unique()                
#Descriptive Statistic of Numerical Data
df.describe()

#Check Names
df.index
df.values
df.columns                  

###Plot Numerical Distribution
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.show()

###Plot Correlations
corr_matrix = df.corr()
df.plot(kind='scatter', x='X1', y='y', alpha=.1)

##Pairplots to get an intuition of potential correlations
import seaborn as sns
sns.pairplot(df[["Feature1", "Feature2"]], diag_kind="kde")

################## 3. Data Cleaning & Preperation ##############################
##############A. Handle Missing Values
###Handling Missing Data 
df.isnull().sum()            #View Number of Missing Data
df.isnull().sum()/len(df.index) *100    #Percentage Missing
data[data.notnull()]            #Shows data that is Not Missing as Boolean

#Split into Categorical and Numerical Variables
df_cat = df.select_dtypes("object")
df_num = df.select_dtypes("number")

#Replacing Values
df.replace('?', np.nan)       #Replace variables to Missing-format (NaN)
df.replace([A,B], [X,Y])      #Replace Variable A, B with X,Y 

#Filtering Out Missing Data 
df.dropna()                   #Drops Missing (nan) data (rows) (option 1)
df.dropna(axis=1, how='all')  #Drops Missing Data columns (option 2)
df.dropna(thresh=2)           #Keep only the rows with at least 2 non-NA values.

#Filling in Numerical Missing Data
df.fillna(0, inplace=True)      #Fills in Missing Values with 0 (inplace: changes in DataFrame)
df.fillna({1: 0.5, 2: 0})       #Fills in Missing Val in Column 1 (2) with .5 (0)
df.fillna(method='ffill')       #Propagate last valid observation forward; limit=X
df.fillna(method='bfill')       #Propagate last valid observation backward
df.fillna(df.mean())            #Replace Missing Data with Mean/Median (option 3)
median = housing["feature1"].median() 
housing["feature1"].fillna(median, inplace=True)

###Automize Imputation 
from sklearn.impute import SimpleImputer
df_num = df.drop("feat_cat", axis=1)
df_cat = df[["feat_cat"]]
#Fill in Numerical Features
imputer = SimpleImputer(strategy="median")
imputer.fit(df_num)
df_num = imputer.transform(df_num)
df_num = pd.DataFrame(df_num, columns=df_num.columns)

#Filling in Categorical Missing Data
# Iterate over each column of data
for col in df_cat.columns:
    # Check if the column is of object type
    if df_cat[col].dtypes == 'object':
        # Impute with the most frequent value
        df_cat = df_cat.fillna(df[col].value_counts().index[0])

#Combine Numerical and Categorical Feature Again
df = housing_cat.join(housing_num)
        
#Fill in with group mean
fill_mean = lambda g: g.fillna(g.mean())
data.groupby(group_key).apply(fill_mean)

##############B. Remove Duplicates
#Removing Duplicates
data.duplicated()               #Displays Data that is Duplicated
data.drop_duplicates()          #Keeps Non-Duplicated Data (based on Index)
data.drop_duplicates(['f1'])    #Keeps Non-Duplicated Data per Column
data.drop_duplicates(['f1', 'f2'], keep='last')     #keep last of Duplicates

##############C.Filtering Outliers
#Distribution of data
data.describe()                         #Gives range of values
#Boxplot
import seaborn as sns
sns.boxplot(x=boston_df['DIS'])         #Displays outlier in boxplot
#Calculate z-score to find outlier (z-score>3)
from scipy import stats
z = np.abs(stats.zscore(df['feature1']))
print(np.where(z > 3))
#Drop Outlier Values
df = df[(z < 3).all(axis=1)]            #Keep data that does not exceed z>3

#Replace Outlier with Max/Min of IQR
upper_limit = df['f1'].mean() + 3*df['f1'].std())
lower_limit = df['f1'].mean() - 3*df['f1'].std())
df['f1'] = np.where(
    df['f1']>upper_limit,
    upper_limit,
    np.where(
        df['f1']<lower_limit,
        lower_limit,
        df['f1']
    )
)

##############D. Handle Inconsistent Entries
###Change Inconsistent Entries Manually
df['feature1'].unique().sort()                # get all the unique values in the 'feature' column
df['feature1'] = df['feature1'].str.lower()   # convert to lower case
df['feature1'] = df['feature1'].str.strip()   # remove trailing white spaces

###Match column entries with top 10 closest matches
# get the top 10 closest matches to "south korea"
matches = fuzzywuzzy.process.extract("feature1", feature1, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)


##############E. Transform Categorical Values in Numerical Form
######Converting Features
###1.Drop Categorical Variables
df.select_dtypes(exclude=['object'])

###2.Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
df_cat = ordinal_encoder.fit_transform(df_cat)

###3.One-Hot-Encoder
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
df_cat = OH_encoder.fit_transform(df_cat)

###4.Dummy Variable Encoding (contains C-1 categories)
dummies = pd.get_dummies(df_cat['f1'], prefix='f1')   #Create Column with Dummies
df_cat = df_cat[['data1']].join(dummies)         #Join Dummy and Data

#Dummy Encoding for bins
pd.get_dummies(pd.cut(values, bins))

###Loop through all non-numeric data and convert into numeric.
# Instantiate LabelEncoder
le = LabelEncoder()
# Iterate over all the values of each column and extract their dtypes
for col in df_cat.columns.to_numpy:
    # Compare if the dtype is object
    if df_cat[col].dtypes =='object':
    # Use LabelEncoder to do the numeric transformation
        df_cat[col]=le.fit_transform(df_cat[col])

###Converting target variables
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)              # ordinal encode target variable

###Transforming Data Using a Function or Mapping
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                              'pastrami', 'corned beef', 'bacon',
                              'pastrami', 'honey ham', 'nova lox'],
                     'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
meat_to_animal = {
  'bacon': 'pig',
  'pulled pork': 'pig',
  'pastrami': 'cow',
  'corned beef': 'cow',
  'honey ham': 'pig',
  'nova lox': 'salmon'
}
data['animal'] = data['food'].map(meat_to_animal)           #Maps Function to Data
data['food'].map(lambda x: meat_to_animal[x.lower()])       

###Convert Integer Class to String
df['feature1'] = df['feature1'].map({1: 'NewFeature1', 2: 'NewFeature2', 3 : 'NewFeature3'})

###Discretization and Binning
df = pd.DataFrame(pd.cut(df['f1'], bins))                   #cut into bins
df = pd.DataFrame(pd.qcut(data, 4))                         #Cut into quartiles
df = pd.DataFrame(pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.]))    #Cut into given intervales

###String Manipulations
#String Object Method
df[['f2','f3']] = df['f1'].str.split(',', expand=True)                                  
df[['f2','f3']] = pd.DataFrame([x.split(',') for x in df_comma['f1'].tolist() ])

#Partition
df['feature1'].str.partition("")[?]            #Search for word "" and returns tuple with 3 elements


###############F. Convert date object to datetime
#Check data type of data column
df['date'].dtype

#Convert from String/Integer to Datetime
from datetime import datetime
value = '2011-01-03'
dt_value = datetime.strptime(value, '%Y-%m-%d')           #string to datetime object
datestrs = ['7/6/2011', '8/6/2011']
dt_date = [datetime.strptime(x, '%m/%d/%Y') for x in datestrs]  
pd.to_datetime(df['Date'], infer_datetime_format=True)
df['Date'] = df['Date'].astype('datetime64[ns]')          #Datetime conversion using astype

#Date and Time Data Types and Tools
now = datetime.now()
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta.days
delta.seconds
from datetime import timedelta
start = datetime(2011, 1, 7)
start + timedelta(12)
start - 2 * timedelta(12)

###Time Series Basic
df[df['Date'].dt.year == 2017]                  #Returns time series in 2017
df[df['Date'].dt.month == 10]                   #Returns time series in October
df['2001-05']
df['1/6/2011':'1/11/2011']
df.truncate(after='1/9/2011')

#Time Series with Duplicate Indices
df.set_index('Date')
df.index.is_unique                      #Check if index in unique
df.groupby(level=0)           #group by first index (level=0)

###Shifting (Leading and Lagging) Data
ts.shift(2)                                 #Shift index by desired number of periods with an optional time freq.
ts.shift(-2)
ts.shift(2, freq='M')
ts.shift(3, freq='D')
df = df.reindex(columns=['Hardcover', 'Lag_1'])     #place NaN for values where there is no value
X.dropna(inplace=True)                              # drop missing values in the feature set
y, X = y.align(X, join='inner')                     # drop corresponding values in target

#Shifting dates with offsetsts
from pandas.tseries.offsets import Day, MonthEnd
offset = MonthEnd()
ts.groupby(offset.rollforward).mean()


###############G. Standardization/Normalization
df_col = df
###Normilazation
from sklearn.preprocessing import MinMaxScaler          # Import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))             # Instantiate MinMaxScaler 
df = pd.DataFrame(scaler.fit_transform(df_num, index=df.index))

###Standardization
from sklearn.preprocessing import StandardScaler        # Import StandardScaler
scaler = StandardScaler()                               # Instantiate StandardScaler
df = pd.DataFrame(scaler.fit_transform(df, index=df.index))
df.columns = df_col.columns

#Manual standardizzation
for col in df.columns:
  df[col] = (df[col] - df[col].mean())/df[col].std() 


###############H. Create Pipelines
#Only relevent if train/test split before data transformation#
###1.Write Function to automize data processing
def data_transformation(data):
    #A. Handle Missing Values
    #B. Remove Duplicates
    #C. Filtering Outlier
    #D. Handle Inconsistent Entries
    #E. Transform Values in Numerical Form
    #F. Convert date object to datetime
    #G. Standardize/Normalize (Feature Scaling)
    
    #Concatening all Data
    df_processed = np.hstack([df_num, df_cat])
    
    return df_processed


###2.Transforamtion Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, df_num),
        ("cat", OneHotEncoder(), df_cat),
    ])




################### 4. Descriptive Statistics #################################
###Data Aggregation and Group Operations
#Group By
df.groupby(['key1', 'key2'])
df.groupby(['key1', 'key2']).mean()        #Mean per Group key1 and key2
df.groupby('key1').quantile(0.9)           #Compute Quantiles per Group key1
def peak_to_peak(arr):
    return arr.max() - arr.min()
df.groupby('key1').agg(['mean', 'std', peak_to_peak]) #Compute Mean, StDev, Peak by Group  
df.groupby('key1').np.percentile(data['feature'], [2,97.5])
df.groupby('key1').apply(lambda df: df['feature1'].iloc[0]) #first Observation per Group
df.groupby(['key1', 'key2']).apply(lambda df: df.loc[df['feature1'].idxmax()])   #best wine by country and province
def top(df, n=5, column='feature1'):
    return df.sort_values(by=column)[-n:]
tips.groupby(['key1', 'key2']).apply(top, n=5, column='feature1')   #Top N values per Group
grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
grouped.apply(get_wavg)
spx_corr = lambda x: x.corrwith(x['SPX'])
get_year = lambda x: x.year
by_year = rets.groupby(get_year)
by_year.apply(spx_corr)

#Iterating Over Groups
for name, group in df.groupby('key1'):
    print(name)
    print(group)

###Pivot Tables (summarize one or more numeric variable)
pd.pivot_table(data=df, index='key1', values='feature1', aggfunc='sum', fill_value=0)

###Cross-Tabulation
pd.crosstab(data['feature1', data['feature2'], margins=True) #computes a frequency table of the factor

###Group Transforms
df.transform(lambda x: x.mean())
df.transform(lambda x: x.rank(ascending=False))
def normalize(x):
    return (x - x.mean()) / x.std()
df.transform(normalize)
df.apply(normalize)

###Sorting
df.sort_values(by='feature', ascending=False)




################## 5. Split into Training & Test Set #########################
###Split into Predictors and Target Variables
X = df.iloc[:,0:-1].copy()                  #X = df.copy()
y = df.iloc[:,-1].copy()                    #Y = X.pop("target")

###Split into training and test data for Features and Target 
#Split Based on Target Var
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=1, stratify=y)

#Split Based on Specific Feature (Stratified Shuffle Split)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["feature1"]):
     df_train = df.loc[train_index]
     df_test = df.loc[test_index]




###################### 6. Prediction & Classifications ########################
###A.Create Model
###Supervised Learning Estimators
#Linear Regression
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()

#Logistic Regression
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
from sklearn.linear_model import LogisticRegressionCV   #k-fold stratified cross-validation
log_cv_model = LogisticRegressionCV(10)

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
dtr_model = DecisionTreeRegressor()

#Random Forrest Regressor
from sklearn.ensemble import RandomForestRegressor
rfr_model = RandomForestRegressor(random_state=1)

#XGBoost
from xgboost import XGBRegressor
xgbr_model = XGBRegressor(n_estimators=?, learning_rate=0.05)

#Support Vector Machines
from sklearn.svm import SVC
svc = SVC(kernel='linaer')

#Naive Bayes
from sklearn.naive_bayes importGaussianNB
gnb =GaussianNB()

#K-Nearest Neighbors
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

###Unsupervised Learning Estimators
#K-Means Clustering
kmeans = KMeans(n_clusters=6)                   # Create cluster feature
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

#Principal Component Analysis
from sklearn.decomposition import PCA
# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X)
# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)


###B.Model Fitting
#Supervised Learning
lin_model.fit(X_train, y_train)             #Linear Model
# Get estimated intercept and coefficient values 
lin_model.intercept_                       
lin_model.coef_
knn.fit(X_train, y_train)                   #k-nearest neighbor
svc.fit(X_train, y_train)                   #Support Vector Machines

#Unsupervised Learning
kmeans.fit(X_train)
pca = pca.fit_transform(X_train)

###C.Model Prediction
y_pred = lin_model.predict(X_test)
y_pred = scv.predict(X_test)
y_pred = knn.predict(X_test)


###D.Model Evaluation
#Regression Metrics
#Mean Absolute error
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

#Mean Squared Error
from sklearn.metrics import mean_squareed_error
mean_squared_error(y_test, y_pred)

#r2 Score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#Clustering Metrics
#Confusion Matrix
confusion_matrix(y_test,y_pred)

#ROC-AUC curves
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
fpr_train = dict()
tpr_train = dict()
roc_auc = dict()
fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
fpr_train[i], tpr_train[i], _ = roc_curve(y_train[:, i], y_pred_prob_train[:, i])
plt.figure()
plt.plot(fpr[i], tpr[i],  color='black',  label='Main Sample (Test)')
plt.plot(fpr_train[i], tpr_train[i],  color='black',  ls='dotted',  label='Training Data')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

#Cross validation scores
from sklearn.model_selection import cross_val_score
cross_val_score(lin_model, X_train, y_train, cv=4)

#Feature Importance
import matplotlib.pyplot as plt
coefs = np.abs(lin_model.coef_)
indices = np.argsort(coefs)[::-1]
plt.figure()
plt.title("Feature importances (Logistic Regression)")
plt.bar(range(5), coefs[indices[:5]], color="r", align="center")
plt.xticks(range(5), X.columns[indices[:5]], rotation=45, ha='right')


###E.Hyperparameter Tuning
#Grid Search
from sklearn.grid_search import GridSearchCV
params = { "n_neighbors": np.arange(1,3),
          "metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator=knn, param_grid=params)
V(estimator=knn,
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)

#Randomized Parameter Optimization
from sklearn.model_selection import RandomizedSearchCV      # Import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X, y)
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
final_model = grid_search_best_estimator
print("Best: %f using %s" % (best_score, best_params))


###F. Deep Learning
#Import packages
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense

#1. Create a Network
#Binary Classification
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8,kernel_initializer= ,activation='relu'))
model.add(Dense(1,kernel_initializer= ,activation= 'sigmoid'))

#Multi-Class Classification
model = keras.Sequential([layers.Dense(units=1, input_shape=[3])])
model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.BatchNormalization(),    
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])

#Regression
model.add(Dense(64,activation='relu',input_dim=train_data.shape[1]))
model.add(Dense(1))

#Inspect Model
model.output_shape
model.summary()
model.get_config/()
model.get_weights()

#2. Compile Model
#Binary Classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Multiclass Classification
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#Regression
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#Recurrent Neural Network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#3. Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,                         #feed the optimizer 256 rows of the training data
    epochs=10,                              #do that 10 times all the way through the dataset (the epochs)
    callbacks=[early_stopping]             # put your callbacks in a list
)

#4.Model Evaluation
score = model.evaluate(X_test, y_test)

# Show the learning curves
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();

#Model Fine-Tuning
#Add early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)


########################## 7. Data Visualization #############################
############A.Plotting with Matplotlib
import matplotlib.pyplot as plt

###Plot Workflow (Note: Need to run it in one command)
#1.Create Plot
fig = plt.figure()
fig2 = plt.figure(figsize=(8,6))
#2.Plot
ax = fig.add_subplot(1,1,1)         #row-column-number
#3.Costumized Plot
ax.plot(df['feature1'], df['feature2'])
ax.scatter(df['feature1'], df['feature2'])
ax.hist(df['feature1'])
ax.boxplot(df['feature1'])
ax.violinplot(df['feature1'])
#4.Save Plot
plt.savefig('') 
#5.Plot
plt.show()

#Close and Clear
plt.cla()           #Clear an axis
plt.clf()           #Clear the entire figure
plt.close()         #Close a window

###Costumize Plot
#Linestyles
plt.plot(x,y,linewidth=4.0, ls='--'/'-.'/'solid', color=)

#Markers
fig, ax = plt.subplots()
ax.scatter(x,y,marker='o')

#Text & Annotations
ax.text(1,-2.1,'Example Graph', style='italic')
plt.title(r'$sigma_i=15$', fontsize=20)

#Limits & Autoscaling
ax.margins(x=.1,y=.1)                   #Add padding to a plot
ax.axis('equal')                        #Set the aspect ratio of the plot to 1
ax.set(xlim=[0,10.5],ylim=[-1.5,1.5])   #Set limits for x-and y-axis
ax.set_xlim(0,10.5)                     #Set limits for x-axis

#Legends
ax.set(title='An Example Axes', ylabel='Y-Axis', xlabel='X-Axis')
ax.legend(loc='best')                   #No overlapping plot elements

#Ticks
ax.xaxis.set(ticks=range(1,5), ticklabels=[3,100,-12,"foo"])    #Manually set x-ticks
ax.tick_params(axis='y',  direction='inout',length=10)          #Make y-ticks longer and go in and out

#Subplot Spacing
fig3.subplots_adjust(wspace=0.5, hspace=0.3,left=0.125,right=0.9,top=0.9,bottom=0.1)    #Adjust the spacing between subplots
fig.tight_layout() #Fit subplot(s) in to the figure area

#Axis Spines
ax1.spines['top'].set_visible(False)  #Make the top axis line for a plot invisible
ax1.spines['bottom'].set_position(('outward',10)) #Move the bottom axis line outward



############B.Plotting with Seaborn
import seaborn as sns
#Numerical Plots
#Regression Plots
sns.regplot(x=df['feature1'], y=df['feature2'])
sns.lmplot(x='feature1', y='feature2', hue='feature3', data=df)         #seperate regression lines

#Line Charts (trends over timee)
sns.lineplot(data=df['feature'], label="Label_Name")
sns.relplot(x='feature1', y='feature2', kind='line'/'scatter', hue='feature_cat3', data=df)  

#Distributions
sns.distplot(a=df['feature1'], kde=False)       #histogram 
sns.kdeplot(data=data['feature1'])              #Probability Density Function

#Matrix Plots
sns.heatmap(data=df, annot=True)

#Categorical Plots
#Scatter Plots
sns.scatterplot(x=df['feature1'], y=df['feature2'], hue=df['feature3']) #hue is color coding
sns.stripplot(x='feature_cat1', y='feature2', data=df) 
sns.swarmplot(x=df['feature_cat1'], y=df['feature2']) 

#Bar Charts
sns.barplot(x=df['feature_cat1'], y=df['feature1'])      #categorical scatter plot   

#Count Plot
sns.countplot(x='feature_cat1', data=df)                 #Show count of observations 

#Point Plot
sns.pointplot(x='feature1', y='feature_cat2', hue='feature_cat3', data=df)

#Violinplot
sns.pointplot(x='feature1', y='feature_cat2', hue='feature_cat3', data=df) #Draw a vertical violinplot grouped by a categorical variable:

###Further Costumizations
#Axisgrid Objects
plt.title("")
plt.xlabel("")
plt.ylabel("")
plt.xlim(0,10)                  #Adjust the limits of the x-axis
plt.ylim(0,100)                 #Adjust the limits of the y-axis
plt.legend()                    #force legend to appear

###Show or Save Plot
plt.show()                      #Show the plot
plt.savefig()                   #Save plot as a figure

plt.cla()                       #Clear on axis
plt.clf()                       #Clear on entire figure
plt.close()                     #Close window


