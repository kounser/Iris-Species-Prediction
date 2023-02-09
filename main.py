from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#1.Data procurement
iris = load_iris()

#2.Basic data processing
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=2006)

#3.Feature preprocessing
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#4.Machine Learning -- KNN
## 4.1 Instantiate an estimator
estimator = KNeighborsClassifier(n_neighbors=5)
## 4.2 Train modol
estimator.fit(x_train, y_train)

#5. Model evaluation
##5.1 Forecast result output
y_pre = estimator.predict(x_test)
print("Predicted value is:\n", y_pre, "\n")
print("Whether the predicted value is true:\n", y_pre==y_test,"\n")
#5.2 Calculation accuracy
score = estimator.score(x_test, y_test)
print("Accuracy is:\n", score)

