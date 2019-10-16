import numpy
import pandas
import csv
from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy import stats
from datetime import datetime
from sklearn.base import TransformerMixin
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

#Files
FileRoot = "ML/Individual Competition/"
TrainingWithLabels = "tcd ml 2019-20 income prediction training (with labels).csv"
SubmissionFile = "tcd ml 2019-20 income prediction submission file.csv"
TestFile = "tcd ml 2019-20 income prediction test (without labels).csv"

#Column names
InstanceColumn = "Instance"
YearOfRecordColumn = "Year of Record"
GenderColumn = "Gender"
AgeColumn = "Age"
CountryColumn = "Country"
SizeOfCityColumn = "Size of City"
ProfessionColumn = "Profession"
UniversityDegreeColumn = "University Degree"
WearsGlassesColumn = "Wears Glasses"
HairColourColumn = "Hair Color"
BodyHeightColumn = "Body Height [cm]"
IncomeColumn = "Income in EUR"
IncomeColumnTest = "Income"

#Columns to drop
GenderDrop = GenderColumn + "_male"
CountryDrop = CountryColumn + "_Cuba"
ProfessionDrop = ProfessionColumn + "_painter"
UniversityDegreeDrop = UniversityDegreeColumn + "_PhD"
HairColourDrop = HairColourColumn + "_Black"


# Crefit to https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
# for this implementation of a categorical and numeric imputer
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pandas.Series([X[c].value_counts().index[0]
            if X[c].dtype == numpy.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def main():
    #Read data from csvs
    data = pandas.read_csv(FileRoot + TrainingWithLabels)
    test = pandas.read_csv(FileRoot + TestFile)


    #Impute Missing Data
    data = DataFrameImputer().fit_transform(data)
    test = DataFrameImputer().fit_transform(test)


    #One hot encocde categorical data
    res = oneHotEncode(data.append(test))
    data = res.head(111993)
    test = res.tail(73230)


    # Add polynomial features
    data = addPolynomialFeature(data, 8)
    test = addPolynomialFeature(test, 8)


    # Split into training and test data
    size = (data.shape)[0]

    dataA = data.head(100000-size)
    dataB = data.tail(11993)


    # Remove outliers from training data
    dataA = removeOutliersZScore(dataA)


    # Setup the training, check and test data, removing the undeeded columns
    training = dataA.drop([InstanceColumn, IncomeColumn], axis=1)
    target = dataA.filter([IncomeColumn], axis=1)

    checker = dataB.drop([InstanceColumn, IncomeColumn], axis=1)
    checkerTarget = dataB.filter([IncomeColumn], axis=1)

    test = test.drop([InstanceColumn, IncomeColumn], axis=1)


    # Scale the data
    trainingcols = training.columns
    checkercols = checker.columns
    testcols = test.columns

    scaler = StandardScaler()
    trainingSc = scaler.fit_transform(training)
    checkerSc = scaler.transform(checker)
    testSc = scaler.transform(test)

    training = pandas.DataFrame(trainingSc, columns = trainingcols)
    checker = pandas.DataFrame(checkerSc, columns = checkercols)
    test = pandas.DataFrame(testSc, columns = testcols)


    # Fit the model
    model = doMLPRegression(training, target[IncomeColumn])


    # Predict and score the trainging
    predictions = predictRegression(model, training)
    mse = math.sqrt(mean_squared_error(target[IncomeColumn], predictions))
    print(mse)


    # Predict and score the unseen data
    predictions = predictRegression(model, checker)
    mse = math.sqrt(mean_squared_error(checkerTarget[IncomeColumn], predictions))
    print(mse)


    # Predict the unknown data
    predictions = predictRegression(model, test)
    printToCsv(predictions)


# Prints a prediction to a csv, using timestamp as the filename
def printToCsv(predictions):
     now = datetime.now()
     time = now.strftime("%d_%m_%Y_%H_%M_%S")

     with open(FileRoot + time + ".csv", mode='w', newline='') as predictionFile:
          predictionWriter = csv.writer(predictionFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

          x = predictions[0]

          predictionWriter.writerow([InstanceColumn, IncomeColumnTest])

          for i in range(111994,185224):
               predictionWriter.writerow([i, predictions[i-111994]])


# Fits a linear regression model
def doLinearRegression(X, y):
    model = linear_model.LinearRegression()
    return model.fit(X, y)


# Fits an SGD regression model
def doSGDRegression(X, y):
     model = linear_model.SGDRegressor()
     return model.fit(X, y)


# Fits a ridge regression model
def doRidgeRegression(X, y):
     model = linear_model.Ridge()
     return model.fit(X, y)


# Fits a lasso regression model
def doLassoRegression(X, y):
     model = linear_model.Lasso()
     return model.fit(X, y)


# Fits an MLP regression model
def doMLPRegression(X,y):
    model = MLPRegressor(hidden_layer_sizes=(200,), max_iter=1000, verbose=True)
    return model.fit(X, y)


# Fits a random forrest regression
def doRandomForrestRegression(X,y):
    model = RandomForestRegressor()
    return model.fit(X,y)


# Make a prediction using a regression model and data
def predictRegression(model, X):
    return model.predict(X)


# One hot encode given data, and drop one column from each
def oneHotEncode(data):
    encodedData = pandas.get_dummies(data,
    columns=[
    GenderColumn,
    CountryColumn,
    ProfessionColumn,
    UniversityDegreeColumn,
    HairColourColumn
    ],
    prefix=[
    GenderColumn,
    CountryColumn,
    ProfessionColumn,
    UniversityDegreeColumn,
    HairColourColumn
    ])

    encodedData = encodedData.drop(columns=[
    GenderDrop,
    CountryDrop,
    ProfessionDrop,
    UniversityDegreeDrop,
    HairColourDrop
    ])

    return encodedData

# Remove outliers from given data by z-score
def removeOutliersZScore(data):
    outlierColumns = data[[IncomeColumn]].copy()
    z = numpy.abs(stats.zscore(outlierColumns))


    newData = data[(z < 12).all(axis=1)]
    return newData


def addPolynomialFeature(data, degree):
    poly_data = PolynomialFeatures(degree=degree)

    dataPoly = poly_data.fit_transform(data[[YearOfRecordColumn, AgeColumn, SizeOfCityColumn, BodyHeightColumn]])
    # testP = poly_test.fit_transform(test[[YearOfRecordColumn, AgeColumn, SizeOfCityColumn, BodyHeightColumn]])

    dataPoly_cols = poly_data.get_feature_names(data[[YearOfRecordColumn, AgeColumn, SizeOfCityColumn, BodyHeightColumn]].columns)
    # testP_cols = poly_test.get_feature_names(test[[YearOfRecordColumn, AgeColumn, SizeOfCityColumn, BodyHeightColumn]].columns)

    dataPoly_df = pandas.DataFrame(dataPoly, columns = dataPoly_cols)
    # testPdf = pandas.DataFrame(testP, columns = testP_cols)

    dataPoly_df = dataPoly_df.drop([YearOfRecordColumn, AgeColumn, SizeOfCityColumn, BodyHeightColumn], axis=1)
    # testPdf = testPdf.drop([YearOfRecordColumn, AgeColumn, SizeOfCityColumn, BodyHeightColumn], axis=1)

    data = pandas.concat([data, dataPoly_df], axis=1)
    # test = pandas.concat([test, testPdf], axis=1)

    return data

if __name__ == "__main__": main()