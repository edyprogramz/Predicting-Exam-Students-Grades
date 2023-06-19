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
   
    ```python 
    import matplotlib.pyplot as pyplot
    from matplotlib import style
    ```

    - *responsible for the graphical representantion of our data*

4. sklearn
   
    ```python 
    import sklearn 
    from sklearn import linear_model
    from sklearn.utils import shuffle 
    ```

5. pickle
   
    ```python import pickle ```

    Helps to save our files after prediction.


## Steps of the project:

STEP 1:

- We have to read in our dataset. Using panda.

    ```python 
    data = pd.read_csv("student-mat.csv"), sep=";"
    ```

    **sep**, refers to what separates each row, in our case ';'
<br>

STEP 2:

- How to trim the data set, This allows us to get specific columns that we need.

    ```python 
    data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]] 
    ```

- In our case we pass in the the attributes we listed above.
NB: preferably interger fields.
<br>

STEP 3: 

- Here we define what we want to predict. The label.

    ```python
    predict = "G3"
    ```

    ```python
    x = np.array(data.drop([predict], 1)) 
    ```

- This line defines attributes of our predicted label 'G3'.
  
    ```python 
    y = np.array(data[predict]) 
    ```
- This line gives only the 'G3' value.
  
NB: in 'x', the 'G3' label is avoided.

NOTE: in short the python snippets X and Y are similar only
that:
- in X-> "G3" is dropped/ IT WON'T BE DISPLAYED INCLUDED.

- And in Y -> "G3" IS THE ONLY ONE THAT WILL BE DISPLAYED.
<br>

STEP 4:

- We divide x & y into four('x train', 'y train', 'x test', 'y test'). Using sklearn!
  
    ```python
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) 
    ```

- This line splits our data X and Y, This is for TRAINING & TESTING.
  
- The line 'test_size=0.1', means that from our dataset 10% will be used for testing.

STEP 5:
- create the training model.
  
    ```python 
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    ```
- **linear-fit** , finds the best fit line 📈
  
![Web capture_19-6-2023_02130_www bing com](https://github.com/edyprogramz/Predicting-Exam-Students-Grades/assets/116636391/8f867b95-49f7-44e3-b7be-1ad684fbe1f7)
<br>

```python 
accuracy = linear.score(x_test, y_test)
```

- **linear-score**, finds how accurate the model is!
  
- To view the **Coefficient**, **Intercept** & **Accuracy**:
   
    ```python
    print(accuracy)
    print('Coefficient:', linear.coef_)
    print('Intercept: ', linear.intercept_)
    ```
NOTE: You will have 5 coefficient or more depending on the number of you attributes.

STEP 6:

    ```python 
    predictions = linear.predict(x_test)

    for x in range(len(predictions)):
        print(predictions[x], x_test[x], y_test[x])
        
    ```

- On the first line our model makes a prediction.
- A for loop to iterate through each prediction!
  
**predictions[x]**  *gives what grade the model predicted*

**x_test[x]**    *gives the original attributes eg 'studytime' etc.*

**y_test[x]**    *gives the actual grade of CAT 3*

The results will look like this:

| Model Prediction | original attributes | Actual Score |
|:---------------- |:-------------------:|-------------:|
|5.870112162153946 | [7 7 2 2 1] | 7 |
|5.452722721882618 |[ 7  6  2 10  0]| 6|
|8.491827220328435 | [10  9  2  0  0] | 9 |
|9.219306681367364 |[11  9  1 19  1] | 10|
|15.633495531267252 | [16 15  2 10  0] | 15|
|10.192528017953217  |[ 9 11  3  2  0] | 11|

STEP 7:

- Saving our model
- Finding the most accurate model

```python
with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
```


    ```python ```
    ```python ```
    ```python ```
    ```python ```

    ```python ```



