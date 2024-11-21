import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('salary.CSV')
data.head()
data.dtypes
data.shape
data.isna().sum()
inputs = data.drop('salary morethan 10,000 birr', axis = 'columns')
target = data['salary more than 10,000 birr']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree'] = le_degree.fit_transform(inputs['degree'])

inputs_n = inputs.drop(['company','job','degree'], axis = 'columns')
inputs_n

model = DecisionTreeClassifier()
model.fit(inputs_n, target)
model.score(inputs_n, target)
model.pridict([[2,1,0]])
model.pridict([[2,1,1]])