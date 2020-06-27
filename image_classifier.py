# importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline

# using pandas to read the database stored in the same folder
data = pd.read_csv('mnist_test.csv')

# viewing column heads
data.head()

#this it the output for the uploaded data
#label	1x1	1x2	1x3	1x4	1x5	1x6	1x7	1x8	1x9	...	28x19	28x20	28x21	28x22	28x23	28x24	28x25	28x26	28x27	28x28
#0	7	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
#1	2	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
#2	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
#3	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
#4	4	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
#5 rows × 785 columns

# extracting data from the dataset and viewing them up close
a=data.iloc[3,1:].values

# extracting data from the dataset and viewing them up close
a=data.iloc[3,1:].values

# reshaping the extracted data into a reasonable size
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)

# the handwritten image is shown
#matplotlib.image.AxesImage at 0x7fd3bdeab550>

# preparing the data
# separating labels and data values
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

# creating test and train sizes / batches
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)

# check data
y_train.head()

# this is the output
#6    4
#2    1
#8    5
#0    7
#1    2
#Name: label, dtype: int64

# call rf classifier
rf = RandomForestClassifier(n_estimators=100)

# fit the model
rf.fit(x_train, y_train)

#output is
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
           # max_depth=None, max_features='auto', max_leaf_nodes=None,
           # min_impurity_decrease=0.0, min_impurity_split=None,
           # min_samples_leaf=1, min_samples_split=2,
           # min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
           # oob_score=False, random_state=None, verbose=0,
           # warm_start=False)
           
# prediction on test data
pred = rf.predict(x_test)

pred
# output is array([2, 4])

# check prediction accuracy
s = y_test.values

# calculate number of correctly predicted values
count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count=count+1
count
# output is 1

#total values that the prediction code was run on 
len(pred)

#output is 2

# accuracy value
1/2

#output is 0.5
