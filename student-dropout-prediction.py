#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[80]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


# In[2]:


get_ipython().run_line_magic('pip', 'install scikit-learn')


# In[ ]:





# # Load Data

# In[3]:


df = pd.read_csv("./dataset.csv")
df


# # Data Cleaning, Information & Visualization

# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.shape


# In[7]:


df.size


# In[8]:


df.describe().T


# In[9]:


df['Target'].value_counts()


# In[10]:


df['Target'] = LabelEncoder().fit_transform(df['Target'])


# In[11]:


df['Target'].value_counts()


# In[12]:


plt.figure(figsize=(5, 10))
sns.distplot(df['Target'], color = "Blue")


# In[13]:


plt.figure(figsize=(5, 10))
sns.countplot(data = df, x="Target").set_title('Target')


# In[14]:


new_data = df.copy()
new_data = new_data.drop(columns=['Nacionality', 
                                  'Mother\'s qualification', 
                                  'Father\'s qualification', 
                                  'Educational special needs', 
                                  'International', 
                                  'Curricular units 1st sem (without evaluations)',
                                  'Unemployment rate', 
                                  'Inflation rate'], axis=1)


# In[15]:


data_corr_descending = new_data.corr()['Target'].sort_values(ascending=False)
data_corr_descending = pd.DataFrame(data_corr_descending)


# In[16]:


plt.bar(data_corr_descending.index, data_corr_descending['Target'], color='skyblue')

# Add labels and title
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.title('Updated Table and Correlation with Target Variable')

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=90)

# Show the plot
plt.show()


# In[17]:


plt.figure(figsize=(8, 8))
plt.title("Education Status")
plt.pie(df['Target'].value_counts(), labels = ['Graduate', 'Dropout', 'Enrolled'], explode = (0.1, 0.1, 0.0), autopct='%1.2f%%', shadow = True)
plt.legend( loc = 'lower right')


# In[18]:


plt.figure(figsize=(8, 8))
plt.title("Gender")
plt.pie(df['Gender'].value_counts(), labels = ['Male', 'Female'], explode = (0.1, 0.0), autopct='%1.2f%%', shadow = True)
plt.legend( loc = 'lower right')


# In[19]:


demo_data = df[["Marital status", "Nacionality", "Displaced", "Gender", "Age at enrollment", "International", "Target"]]

# Socio-economic data
se_data = df[["Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", "Educational special needs", "Debtor", "Tuition fees up to date", "Scholarship holder", "Target"]]

# Macro-economic Enrollment Data
me_data = df[['Unemployment rate', 'Inflation rate', 'GDP', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance', 'Previous qualification', 'Target']]

# Academic data
academic_data = df[['Curricular units 1st sem (credited)',
       'Curricular units 1st sem (enrolled)',
       'Curricular units 1st sem (evaluations)',
       'Curricular units 1st sem (approved)',
       'Curricular units 1st sem (grade)',
       'Curricular units 1st sem (without evaluations)',
       'Curricular units 2nd sem (credited)',
       'Curricular units 2nd sem (enrolled)',
       'Curricular units 2nd sem (evaluations)',
       'Curricular units 2nd sem (approved)',
       'Curricular units 2nd sem (grade)',
       'Curricular units 2nd sem (without evaluations)', 'Target']]


# In[20]:



plt.figure(figsize=(7,5))
sns.heatmap(demo_data.corr(), annot=True, cmap='coolwarm', fmt=' .2f')
plt.title('Correlation Heatmap - Demographic Data')
plt.show()


# In[21]:


# Correlation Matrix for socio-economic data
plt.figure(figsize=(7,5))
sns.heatmap(se_data.corr(), annot=True, fmt=' .2f')
plt.title('Correlation Matrix - Socio-Economic Data')
plt.show()


# In[22]:


# Correlation Matrix for Macro-Economic Data
plt.figure(figsize=(7,5))
sns.heatmap(me_data.corr(), annot=True, cmap='YlGnBu', fmt=' .2f')
plt.title('Correlation Matrix - Macro-Economic Data')
plt.show()


# In[23]:


# Correlation Matrix for Academic Data
plt.figure(figsize=(10,7))
sns.heatmap(academic_data.corr(), annot=True, cmap='coolwarm', fmt=' .2f')
plt.title('Correlation Matrix - Academic Data')
plt.show()


# In[24]:


plt.figure(figsize=(20, 45))

for i in range(0, 35):
    plt.subplot(12,3,i+1)
    sns.distplot(df.iloc[:, i], color='blue')
    plt.grid()


# In[25]:


#feature selection
corr_matrix = df.corr(method="pearson")
plt.figure(figsize=(10, 10)) 
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=False, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Pearson correlation")
plt.show()


# In[26]:


# Filter the data
filtered_data = new_data[new_data['Target'] == 0]

# Sort the filtered data by 'Curricular units 2nd sem (approved)' in descending order
sorted_data = filtered_data.sort_values(by='Gender', ascending=False)

# Count the occurrences of each value in the sorted column
value_counts = sorted_data['Gender'].value_counts()

# Create a bar chart
value_counts.plot(kind='bar', color='red')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender for Dropout Students')
plt.xticks(rotation=90)
plt.show()


# In[27]:


# Filter the data
filtered_data = new_data[new_data['Target'] == 0]

# Sort the filtered data by 'Curricular units 2nd sem (approved)' in descending order
sorted_data = filtered_data.sort_values(by='Debtor', ascending=False)

# Count the occurrences of each value in the sorted column
value_counts = sorted_data['Debtor'].value_counts()

# Create a bar chart
value_counts.plot(kind='bar', color='orange')
plt.xlabel('Debtor')
plt.ylabel('Count')
plt.title('Debtor for Dropout Students')
plt.xticks


# In[28]:


# Filter the data
filtered_data = new_data[new_data['Target'] == 0]

# Sort the filtered data by 'Curricular units 2nd sem (approved)' in descending order
sorted_data = filtered_data.sort_values(by='Scholarship holder', ascending=False)

# Count the occurrences of each value in the sorted column
value_counts = sorted_data['Scholarship holder'].value_counts()

# Create a bar chart
value_counts.plot(kind='bar', color='blue')
plt.xlabel('Scholarship holder')
plt.ylabel('Count')
plt.title('Scholarship holder for Dropout Students')
plt.xticks


# In[29]:


# Filter the data
filtered_data = new_data[new_data['Target'] == 0]

# Sort the filtered data by 'Curricular units 2nd sem (approved)' in descending order
sorted_data = filtered_data.sort_values(by='Application mode', ascending=False)

# Count the occurrences of each value in the sorted column
value_counts = sorted_data['Application mode'].value_counts()

# Create a bar chart
value_counts.plot(kind='bar', color='magenta')
plt.xlabel('Application mode')
plt.ylabel('Count')
plt.title('Application mode for Dropout Students')
plt.xticks(rotation=90)
plt.show()


# In[30]:


# Filter the data
filtered_data = new_data[new_data['Target'] == 0]

# Sort the filtered data by 'Curricular units 2nd sem (approved)' in descending order
sorted_data = filtered_data.sort_values(by='Tuition fees up to date', ascending=False)

# Count the occurrences of each value in the sorted column
value_counts = sorted_data['Tuition fees up to date'].value_counts()

# Create a bar chart
value_counts.plot(kind='bar', color='green')
plt.xlabel('Tuition fees up to date')
plt.ylabel('Count')
plt.title('Tuition fees up to date for Dropout Students')
plt.xticks(rotation=90)
plt.show()


# In[31]:


["Tuition fees up to date","Curricular units 1st sem (approved)","Curricular units 1st sem (grade)","Curricular units 2nd sem (approved)","Curricular units 2nd sem (grade)"]
corr_matrix["Target"]


# ## **Assertion**
# 
# ### As we are predicting whether a student will dropout or not so, the number of "Enrolled" student is irrelevant. We only need to know whether a student graduated or dropedout. So, we are dropping the "Enrolled" values and going forward with "Graduate" & "Dropout" values.

# In[32]:


df.drop(df[df['Target'] == 1].index, inplace = True)
df


# In[33]:


df['Dropout'] = df['Target'].apply(lambda x: 1 if x==0 else 0)
df


# In[34]:


plt.figure(figsize=(5, 10))
sns.distplot(df['Dropout'], color = "red")


# In[35]:


plt.figure(figsize=(8, 8))
plt.title("Dropout Status")
plt.pie(df['Dropout'].value_counts(),  labels = ['Non-Dropout', 'Dropout'], explode = (0.2, 0.0), autopct='%1.2f%%', shadow = True)
plt.legend( loc = 'lower right')


# # Standard Scaling the Data

# In[36]:


x = df.iloc[:, :36].values
#x = df[["Tuition fees up to date","Curricular units 1st sem (approved)","Curricular units 1st sem (grade)","Curricular units 2nd sem (approved)","Curricular units 2nd sem (grade)"]].values
print(x)
x = StandardScaler().fit_transform(x)
x


# In[37]:


y = df['Dropout'].values
y


# # Train & Test Splitting the Data

# In[38]:


features = ['Marital status', 'Application mode', 'Application order', 'Course', 'Previous qualification','Mother\'s occupation','Father\'s occupation', 'Displaced','Tuition fees up to date','Scholarship holder','Age at enrollment']
X = new_data[features]
y = new_data.Target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:





# # Function to Measure Performance

# In[39]:


def perform(y_pred):
    print("Precision : ", precision_score(y_test, y_pred, average = 'micro'))
    print("Recall : ", recall_score(y_test, y_pred, average = 'micro'))
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("F1 Score : ", f1_score(y_test, y_pred, average = 'micro'))
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    print("\n")
    print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
    print(classification_report(y_test, y_pred))
    print("**"*27+"\n")
    
    cm = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['Non-Dropout', 'Dropout'])
    cm.plot()


# # Gaussian Naive Bayes

# In[40]:


model_nb = GaussianNB()
model_nb.fit(x_train, y_train)


# In[41]:


y_pred_nb = model_nb.predict(x_test)


# In[1]:


#perform(y_pred_nb)
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Assuming 'cm' is the confusion matrix
cm = confusion_matrix(y_pred_nb, y_pred_nb)
print(cm)
display_labels = ['Non-Dropout', 'Dropout']
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
cm_display.plot()


# # Logistic Regression

# In[44]:


model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)


# In[45]:


y_pred_lr = model_lr.predict(x_test)


# In[2]:


perform(y_pred_lr)


# # Random Forest

# In[47]:


model_rf = RandomForestClassifier(n_estimators=500,criterion='entropy')
model_rf.fit(x_train, y_train)


# In[48]:


y_pred_rf = model_rf.predict(x_test)


# In[3]:


perform(y_pred_rf)


# # Support Vector Classifier

# In[50]:


model_svc = SVC(C=0.1,kernel='linear')
model_svc.fit(x_train, y_train)


# In[51]:


y_pred_svc = model_svc.predict(x_test)


# In[4]:


perform(y_pred_svc)


# # Perceptron

# In[53]:


model_mlp = Perceptron(alpha=0.001,l1_ratio=0.5,max_iter=100)
model_mlp.fit(x_train, y_train)


# In[54]:


y_pred_mlp = model_mlp.predict(x_test)


# In[55]:


perform(y_pred_mlp)


# # KNN Classifier

# In[64]:


error = []

# Calculating MAE error for K values between 1 and 39
for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    accuracy = accuracy_score(y_test, pred_i)
    error.append(accuracy)


# In[65]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 100), error, color='red', 
         linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
         
plt.title('K Value accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')


# In[66]:


model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train, y_train)


# In[67]:


y_pred_knn = model_knn.predict(x_test)


# In[5]:


perform(y_pred_knn)


# # Comparison

# In[68]:


pred=[y_pred_nb,y_pred_lr,y_pred_rf,y_pred_svc,y_pred_mlp,y_pred_knn]
acc=[]
classifiers=["NaiveBayes","Logistic Regression","RandomForest","Support Vector Classier","Perceptron","KNN"]
for i in pred:
    temp=accuracy_score(y_test, i)
    acc.append(temp)

plt.barh(classifiers, acc)

# Add labels and title
plt.ylabel('classifiers')
plt.xlabel('Accuracy')
plt.title('Comparison')
plt.show()

    


# In[81]:


X2 = df.drop('Target', axis=1)
y2 = df['Target']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
scaler = preprocessing.MinMaxScaler()
X2 = scaler.fit_transform(X)


# In[83]:


# Create a function to build our models
def models(X_train2, y_train2):

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train2, y_train2)

    # Decision Trees
    dt = DecisionTreeClassifier()
    dt.fit(X_train2,y_train2)

    # K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn.fit(X_train2, y_train2)

    return lr,dt,knn

lr,dt,knn = models(X_train2, y_train2)

# prediction
y_pred_lr = lr.predict(X_test2)
y_pred_knn = knn.predict(X_test2)


# In[105]:


def evaluate_models(X_test2, y_test2, models):
    results = [] # A list to store the evaluation results of each model
    for name, model in models.items():

        # make prediction on the test data
        y_pred = model.predict(X_test2)

        # calculate the evaluation metric scores
        accuracy = accuracy_score(y_test2, y_pred)
        precision = precision_score(y_test2, y_pred, pos_label=0)
        recall = recall_score(y_test2, y_pred, pos_label=0 )

        # store the results in a list of dictionary
        results.append({'Model': name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall})
    
    # Convert the lit of dictionaries to a pandas DataFrame
    return pd.DataFrame(results)

# define a dict containing the trained models for each algorithm
models = {'Logistic Regression': lr,'Decision Tree': dt, 'KNN': knn}

# call the evaluate models function on the test data and models dictionary
results_df = evaluate_models(X_test2, y_test2, models)

print(results_df)


# In[93]:


y_test2.describe()


# In[ ]:




