import csv
import random

# open csv file
# csvlist [sepal_length,sepal_width,petal_length,petal_width,species],150
samples=list()
maxFeature = [0, 0, 0, 0];
with open('iris.csv', mode='r') as infile:
    csvlist = list(csv.reader(infile))

# maxFeature[a,b,c,d],4
for rows in csvlist:
    for x in range(4):
        maxFeature[x]=max(maxFeature[x],rows[x])

# samples[(0.3,0.6,1),features],150
    if rows[4] == "setosa":
        samples.append([0.333333, [rows[0], rows[1] , rows[2], rows[3]]])
    elif rows[4] == "versicolor":
        samples.append([0.666666, [rows[0], rows[1] , rows[2], rows[3]]])
    else:
        samples.append([1.000000, [rows[0], rows[1] , rows[2], rows[3]]])
print 'maxFeature'+str(maxFeature)

# normalization samples[(0.3,0.6,1),features],150
i=0
for rows in samples:
    j=0
    for feature in samples[i][1]:
        samples[i][1][j]=float(feature)/float(maxFeature[j])
        j+=1
    i+=1

# Sampling random candidates,row(0~len(sample)=samples[row,features],150-x->predictList[row,feature],x
predictList = list()
i = 0
while i < 10:
    r = random.randint(0, len(samples))
    predictList.append((samples[r][0], samples[r][1]))
    del samples[r]
    i += 1
print 'Sample Length  = ' + str(len(samples))
print 'Predict Length = ' + str(len(predictList))

# Spark lib node,node.conf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark import SparkContext

# http://spark.apache.org/docs/latest/cluster-overview.html
# SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext("local", "Simple App")

# create labels;labelpoint(sample
labels = list()
i = 0
for row in samples:
    labels.append(LabeledPoint(rows[0], rows[1]))
    i += 1

# training model 
# http://enginebai.logdown.com/posts/241677,bayes-classification
# http://www.ramlinbird.com/slm-%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF/#more-640 
# http://molecular-service-science.com/2012/06/30/statistical-estimation/
# NB(data,var.)var.=1(Laplace smoothing)var.=0 (ML)
data = sc.parallelize(labels)
print '-------------------------------'
print 'RDD(cluster):'+str(data)
print '-------------------------------'
model = NaiveBayes.train(data, 1.0)

# test
correct=0
i = 1
for predict in predictList:
    answer = model.predict(predict[1])
    print str(i) + ' -> ' + str(predict[0]) + ' = ' + str(answer)
    if answer == predict[0]:
        correct += 1
    i += 1
print 'Accuracy = ' + str(float(correct) / float(len(predictList)) * 100) + '%'
