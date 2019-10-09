import numpy
import pandas
import csv
from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from datetime import datetime

#GLOBALS

MaxRows = 1119949

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

def getTrainingData(filePathTrain, filePathTest):
    dataFrame = pandas.read_csv(filePathTrain)
    dataFrame2 = pandas.read_csv(filePathTest)
    dataFrame = dataFrame.append(dataFrame2)

#     dataFrame = dataFrame.drop([WearsGlassesColumn, HairColourColumn, GenderColumn, SizeOfCityColumn, UniversityDegreeColumn, BodyHeightColumn], axis=1)

    dataFrame = normalizeData(dataFrame)

    dataFrame = oneHotEncode(dataFrame)

    top = dataFrame.head(111993)
    training = top.head(100000)
    errorCheck = top.tail(11993)
    test = dataFrame.tail(73230)

    training = removeNaN(training)
    errorCheckTemp = removeNaN(errorCheck)
    test = meanNan(test)

    data = training.drop([InstanceColumn, IncomeColumn], axis=1)
    errorCheck = errorCheckTemp.drop([InstanceColumn, IncomeColumn], axis=1)
    target = training.filter([IncomeColumn], axis=1)
    errorCheckTarget = errorCheckTemp.filter([IncomeColumn], axis=1)
    test = test.drop([InstanceColumn, IncomeColumn], axis=1)

    return data, target, errorCheck, errorCheckTarget, test

def getPredictData(filePath):
    dataFrame = pandas.read_csv(filePath)
    data = dataFrame.drop([InstanceColumn, IncomeColumnTest], axis=1)
    print(data[data.isnull().any(axis=1)])
    data = data.fillna(data.mean())
    return data

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

    encodedData.drop(columns=[
    GenderDrop, 
    CountryDrop, 
    ProfessionDrop, 
    UniversityDegreeDrop, 
    HairColourDrop
    ])

    return encodedData

def normalizeData(data):
     year = data[[YearOfRecordColumn]].values.astype(float)
     age = data[[AgeColumn]].values.astype(float)
     city = data[[SizeOfCityColumn]].values.astype(float)
     height = data[[BodyHeightColumn]].values.astype(float)

     min_max_scaler = preprocessing.MinMaxScaler()

     yearScaled = min_max_scaler.fit_transform(year)
     ageScaled = min_max_scaler.fit_transform(age)
     cityScaled = min_max_scaler.fit_transform(city)
     heightScaled = min_max_scaler.fit_transform(height)

     data[YearOfRecordColumn] = pandas.DataFrame(yearScaled)
     data[AgeColumn] = pandas.DataFrame(ageScaled)
     data[SizeOfCityColumn] = pandas.DataFrame(cityScaled)
     data[BodyHeightColumn] = pandas.DataFrame(heightScaled)

     return data

def removeNaN(data):
    return data.dropna()

def meanNan(data):
     return data.fillna(data.mean())

def doLinearRegression(X, y):
    model = linear_model.LinearRegression()
    return model.fit(X, y)

def doSGDRegression(X, y):
     model = linear_model.SGDRegressor(loss="huber", alpha=0.1)
     return model.fit(X, y)

def doRidgeRegression(X, y, alpha):
     model = linear_model.Ridge(alpha=alpha)
     return model.fit(X, y)

def doLassoRegression(X, y, alpha):
     model = linear_model.Lasso(alpha=alpha, max_iter=1000, tol=0.1)
     return model.fit(X, y)

def predictRegression(model, X):
    return model.predict(X)

def printToCsv(predictions):
     now = datetime.now()
     time = now.strftime("%d_%m_%Y_%H_%M_%S")

     with open(FileRoot + time + ".csv", mode='w', newline='') as predictionFile:
          predictionWriter = csv.writer(predictionFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

          x = predictions[0]

          predictionWriter.writerow([InstanceColumn, IncomeColumnTest])

          for i in range(111994,185224):
               predictionWriter.writerow([i, predictions[i-111994]])

def main():
    training, target, errorCheck, errorCheckTarget, test = getTrainingData(FileRoot + TrainingWithLabels, FileRoot + TestFile)
#     data = oneHotEncode(data)
#     print(data.shape)

#     training = data.head(111993)
#     test = data.tail(73230)

#     training = removeNaN(training)
#     test = meanNan(test)

# Linear Regression
#     model = doLinearRegression(training, target[IncomeColumn])

# SGD Regression
#     model = doSGDRegression(training, target[IncomeColumn])

# Ridge Regression
    model = doRidgeRegression(training, target[IncomeColumn], 0.1)

# Lasso Regression
#     model = doLassoRegression(training, target[IncomeColumn], 1)

    predictions = predictRegression(model, training)

    mse = mean_squared_error(target[IncomeColumn], predictions)

    print(mse)

    predictions = predictRegression(model, errorCheck)
    mse = mean_squared_error(errorCheckTarget[IncomeColumn], predictions)

    print(mse)

#     toPredict = getPredictData(FileRoot + TestFile)
#     toPredict = oneHotEncode(toPredict)
    predictions = predictRegression(model, test)
#     print(predictions)

    printToCsv(predictions)


if __name__ == "__main__": main()