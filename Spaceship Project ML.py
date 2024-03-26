#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries and tools

# In[1]:


#Basic Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Dictionary for reference:
# 
# * train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# 
# * PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# 
# * HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# 
# * CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# 
# * Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# 
# * Destination - The planet the passenger will be debarking to.
# 
# * Age - The age of the passenger.
# 
# * VIP - Whether the passenger has paid for special VIP service during the voyage.
# 
# * RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# 
# * Name - The first and last names of the passenger.
#         
# * Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# In[3]:


df = pd.read_csv(r'D:\OneDrive\Desktop\Datasets\spaceship-titanic\train.csv')


# # Exploring our data

# In[4]:


df.head()


# In[7]:


df.shape


# In[8]:


df.tail()


# In[13]:


df.isna().sum()


# In[15]:


df.HomePlanet.value_counts()


# In[17]:


df.Destination.value_counts()


# In[18]:


df.info()


# In[19]:


df.describe()


# In[27]:


df.Age.plot.hist()
plt.xlabel('Age')


# In[28]:


df['Transported'].value_counts().plot(kind = 'bar', color = ['salmon','lightblue'])


# In[31]:


df.Transported.value_counts()


# In[33]:


pd.crosstab(df.Transported,df.HomePlanet)


# In[34]:


pd.crosstab(df.Transported,df.HomePlanet).plot(kind = 'bar')


# In[35]:


df.CryoSleep.value_counts()


# In[36]:


pd.crosstab(df.Transported,df.CryoSleep)


# In[37]:


pd.crosstab(df.Transported,df.CryoSleep).plot(kind = 'bar')


# In[38]:


pd.crosstab(df.Transported,df.VIP)


# In[39]:


df.VIP.value_counts()


# # Checking for non-numeric columns

# In[43]:


for label,content in df.items():
    
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[50]:


pd.api.types.is_object_dtype('HomePlanet')


# In[49]:


for label,content in df.items():
    
    if pd.api.types.is_object_dtype(content):
        print(label)


# In[48]:


df.info()


# In[5]:


df_tmp = df.copy()


# # Data Transformation into numeric values

# In[6]:


for label,content in df_tmp.items():
    if pd.api.types.is_object_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()


# In[54]:


df_tmp.info()


# In[55]:


df_tmp.HomePlanet.cat.categories


# In[57]:


df_tmp.HomePlanet.cat.codes


# In[58]:


df.isna().sum()


# In[59]:


for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[7]:


for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_tmp[label+'_is_missing'] = pd.isnull(content)
            
            df_tmp[label] = content.fillna(content.median())


# In[8]:


df_tmp.head()


# In[62]:


df_tmp.isna().sum()


# In[9]:


for label,content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_tmp[label+'_is_missing'] = pd.isnull(content)
        
        df_tmp[label] = pd.Categorical(content).codes+1


# In[10]:


df_tmp.head()


# # Plotting the Correlation Matrix

# In[65]:


corr_matrix = df_tmp.corr()
fig, ax = plt.subplots(figsize = (15,10))
ax = sns.heatmap(corr_matrix, annot = True, linewidth = 0.5, fmt = '.2f', cmap = 'YlGnBu')


# In[66]:


df_tmp.isna().sum()


# In[7]:


df_tmp.head()


# In[9]:


df_tmp.HomePlanet_is_missing.value_counts()


# # Modelling
# 
# For modelling, We will use: 
# 
# 
# * Support Vector Machine(SVM)
# 
# * Logistic Regression
# 
# * RandomForest Classifier
# 
# * Gradient Boosting

# In[11]:


from sklearn import svm


# In[12]:


X = df_tmp.drop('Transported',axis = 1)
y = df_tmp['Transported']


# In[13]:


model = svm.SVC()


# In[14]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[15]:


np.random.seed(42)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[16]:


X_train


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


# In[18]:


from sklearn.svm import SVC  # Import Support Vector Classification (SVC)


# In[19]:


models = {'Logistic Regression': LogisticRegression(),
          'KNearest Neighbors Classifier': KNeighborsClassifier(),
          'RandomForest Classifier': RandomForestClassifier(),
          'Support Vector Machine': SVC()}


# In[20]:


def fit_and_score(models,X_train,X_test,y_train,y_test):
    
    np.random.seed(42)
    
    model_scores = {}
    
    for name,model in models.items():
        
        model.fit(X_train,y_train)
        model_scores[name] = model.score(X_test,y_test)
        
    return model_scores


# In[21]:


model_scores = fit_and_score(models = models,
                             X_train = X_train,
                             X_test = X_test,
                             y_train = y_train,
                             y_test = y_test)

model_scores


# In[22]:


model_compare = pd.DataFrame(model_scores,index = ['Accuracy'])
model_compare.T.plot(kind = 'bar',figsize = (10,6))


# # HyperParameter Tuning

# In[24]:


#Trying to tune KNN 

train_scores = []
test_scores = []

neighbors = range(1,21)

knn = KNeighborsClassifier()

for i in neighbors:
    
    knn.set_params(n_neighbors = i)
    
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    
    test_scores.append(knn.score(X_test,y_test))
    


# In[25]:


train_scores


# In[26]:


test_scores


# In[27]:


plt.plot(neighbors,train_scores,label = 'Train Scores')
plt.plot(neighbors,test_scores,label = 'Test Scores')
plt.xticks(np.arange(1,21,1))
plt.xlabel('Neighbors')
plt.ylabel('Model Scores')
plt.legend()
print(f"Maximum score on KNN model test data: {max(test_scores)*100: 2f}%")
print(f"Max score on train data:{max(train_scores)*100: 2f}%")


# In[28]:


log_reg_grid = {'C': np.logspace(-4,4,20),
                'solver': ['liblinear']}

rf_grid = {'n_estimators':np.arange(10,100,50),
           'max_depth': [None,3,5,10],
           'min_samples_split': np.arange(2,20,2),
           'min_samples_leaf': np.arange(1,20,2)}


# In[47]:


from scipy.stats import randint


# In[52]:


gb_grid = {'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01],
           'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
           'max_depth' : np.linspace(1, 32, 32, endpoint=True),
           'min_samples_split' : np.linspace(0.1, 1.0, 10, endpoint=True),
           'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
           'max_features': list(range(1,X_train.shape[1]))}


# In[49]:


gb_rs = RandomizedSearchCV(GradientBoostingClassifier(),
                           param_distributions=gb_grid,
                           cv = 5,
                           n_iter = 20,
                           verbose=True)
gb_rs.fit(X_train,y_train)


# In[50]:


gb_rs.score(X_test,y_test)


# In[29]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import RocCurveDisplay 


# In[30]:


#Tune Logistic Regression

np.random.seed(42)

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions= log_reg_grid,
                                cv = 5,
                                n_iter=20,
                                verbose=True)

rs_log_reg.fit(X_train,y_train)


# In[32]:


rs_log_reg.score(X_test,y_test)


# In[34]:


#Tuning Random Forest Classifier

np.random.seed(42)

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv = 5,
                           n_iter=20,
                           verbose= True)

rs_rf.fit(X_train,y_train)


# #  Best Score
# 
# * The best score we get is 78.6% with Random Forest Classifier Model

# In[35]:


rs_rf.score(X_test,y_test)


# In[38]:


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier


# In[39]:


clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)


# In[55]:


rs_rf.best_params_


# In[58]:


best_model = RandomForestClassifier(n_estimators= 60,
                                     min_samples_split= 12,
                                     min_samples_leaf= 7,
                                     max_depth= None,
                                     n_jobs = -1)


# In[59]:


best_model.fit(X_train,y_train)


# In[61]:


best_model.score(X_test,y_test)


# # Importing the test set

# In[62]:


df_test = pd.read_csv(r'D:\OneDrive\Desktop\Datasets\spaceship-titanic\test.csv')


# In[63]:


df_test


# # Pre Processing the Test set to match the Training set

# In[64]:


def preprocess_data(df):
    
    for label,content in df.items():
        if pd.api.types.is_object_dtype(content):
            df[label] = content.astype('category').cat.as_ordered()
        
        
    
    for label,content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label+'_is_missing'] = pd.isnull(content)
                df[label] = content.fillna(content.median())
            
    for label,content in df.items():
        if not pd.api.types.is_numeric_dtype(content):
            df[label+'_is_missing'] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes+1
    


# In[65]:


preprocess_data(df_test)


# In[67]:


df_tmp.head()


# In[66]:


df_test.head()


# In[68]:


test_preds = best_model.predict(df_test)


# In[69]:


test_preds


# In[70]:


df_preds = pd.DataFrame()


df_preds['PassengerId'] = df_test['PassengerId']
df_preds['Transported'] = test_preds

df_preds


# In[71]:


df_preds.to_csv(r'D:\OneDrive\Desktop\Datasets\spaceship-titanic\test_preds.csv', index=False)


# In[72]:


# Assuming df is your DataFrame containing the dataset
# Convert 'PassengerId' column to string type
df_preds['PassengerId'] = df_preds['PassengerId'].astype(str)
df_preds['Transported'] = df_preds['Transported'].astype(str)


# In[75]:


passenger_id_type = df_preds['PassengerId'].dtype
print(passenger_id_type)


# In[74]:


df_preds.to_csv(r'D:\OneDrive\Desktop\Datasets\spaceship-titanic\test_predss.csv', index=False)


# # Feature Importance

# In[76]:


best_model.feature_importances_


# In[78]:


def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({'Features': columns,
                        'Feature_Importances': importances})
          .sort_values('Feature_Importances', ascending=False)  # Corrected column name
          .reset_index(drop=True))

    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df['Features'][:n], df['Feature_Importances'][:n])
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Features')
    ax.invert_yaxis()  # Invert y-axis for better visualization
    ax.set_title('Top {} Feature Importances'.format(n))
    plt.show()


# In[79]:


plot_features(X_train.columns,best_model.feature_importances_)

