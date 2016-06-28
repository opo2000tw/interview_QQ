import csv
import random

#1 open csv file
# csvlist [sepal_length,sepal_width,petal_length,petal_width,species],150
with open('iris.csv', mode='r') as infile:
    reader = csv.reader(infile)
    csvlist = list(reader)
    print '[sepal_length,sepal_width,petal_length,petal_width,species] total member:'+str(len(csvlist));
#2 create sample: [[answer, [feature 1, ...]]]
maxFeature = [0, 0, 0, 0];
samples = list()
for rows in csvlist:
    features = [rows[0], rows[1] , rows[2], rows[3]]    
# features[sepal_length,sepal_width,petal_length,petal_width],4
# maxFeature[sepal_length,sepal_width,petal_length,petal_width],4
    maxFeature[0] = max([maxFeature[0], rows[0]]);
    maxFeature[1] = max([maxFeature[1], rows[1]]);
    maxFeature[2] = max([maxFeature[2], rows[2]]);
    maxFeature[3] = max([maxFeature[3], rows[3]]);
    if rows[4] == "setosa":
        samples.append([0.333333, features])
    elif rows[4] == "versicolor":
        samples.append([0.666666, features])
    else:
        samples.append([1.000000, features])

# normalization
print samples[1]
print samples[1][2]
