## PREDICTING CAT3 STUDENTS GRADES
*linear regression*

###  Attributes
1. Studytime
2. Absensce
3. Grade1 as "G1"
4. Grade2 as "G2"

###  Label / The prediction
5. Grade3 as "G3"

### Requirements
1. pandas
   
    ```python import pandas as pd ```
    Helps read our dataset easily, a better visualization.

2. Numpy
   
    ```python import numpy as np ```
    Helps to deal with arrays, since python is not as effective.

3. matplotlib
    ```python ```
    - *responsible for the graphical representantion of our data*

4. sklearn
   
    ```python import sklearn ```
    ```python from sklearn import linear_model ```
    ```python from sklearn.utils import shuffle ```


## Steps of the project:

STEP 1:
We have to read in our dataset. Using panda.
    ```python data = pd.read_csv("student-mat.csv"), sep=";" ```
    sep -> refers to what separates each row, in our case ';'
<br>

STEP 2:
How to trim the data set, This allows us to get specific columns that we need.
```python data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]] ```
In our case we pass in the the attributes we listed above.
NB: preferably interger fields.
<br>

STEP 3: 
Here we define what we want to predict. The label.
```python predict = "G3" ```
```python x = np.array(data.drop([predict], 1)) ```
This line defines attributes of our predicted label 'G3'.
```python y = np.array(data[predict]) ```
This line gives only the 'G3' value.
NB: in 'x', the 'G3' label is avoided.
NOTE: in short the python snippets X and Y are similar only that in X-> "G3" is dropped/ IT WON'T BE DISPLAYED INCLUDED.
And in Y -> "G3" IS THE ONLY ONE THAT WILL BE DISPLAYED.
<br>

STEP 4:
We divide x & y into four('x train', 'y train', 'x test', 'y test'). Using sklearn!
```python x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) ```
This line splits our data X and Y, This is for TRAINING & TESTING.
The line 'test_size=0.1', means that from our dataset 10% will be used for testing.

    ```python ```
    ```python ```
    ```python ```



