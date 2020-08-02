import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:/Deep_Learning/Artificial_Neural_Networks/Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#labelencoder_X_1 = LabelEncoder()
#X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
#labelencoder_X_2 = LabelEncoder()
#X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
#onehotencoder = OneHotEncoder(categories_ = [1])
#X = onehotencoder.fit_transform(X).toarray()
label_encoder_x_1 = LabelEncoder()
X[: , 2] = label_encoder_x_1.fit_transform(X[:,2])
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())
X = X.astype('float64')
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#new_pred = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 1, 1, 50000]])))
#new_pred = (new_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size' : [25, 32], 
              'epochs' : [100, 500],
              'optimizer' : ['adam', 'rmsprop'] }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_paramters = grid_search.best_params_
best_accuracy = grid_search.best_score_

    



