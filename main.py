import datetime
import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def bayes(train_x, train_y, data, trainWordCount, testing_data, unigram, vector, indexes2=None, ):
    indexes = []
    for row in train_y.index:
        indexes.append(row)

    if (indexes2 == None):
        for a, value in enumerate(train_x.toarray()):
            category = data["Category"][indexes[a]]
            trainWordCount[category] = np.sum([trainWordCount[category], value], axis=0)
        trainwordCountCopy = trainWordCount.copy()
    else:
        for a, value in enumerate(train_x.toarray()):
            category = data["Category"][indexes[a]]
            temp = np.zeros(len(indexes2))
            for count, b in enumerate(indexes2):
                temp[count] = value[b]
            trainWordCount[category] = np.sum([trainWordCount[category], temp], axis=0)

        trainwordCountCopy = trainWordCount.copy()

    bow = {}
    if indexes2 == None:
        for a, value in enumerate(trainwordCountCopy):
            bow[categories[a]] = {}
            temp = {}
            for b, val in enumerate(vector.get_feature_names_out()):
                temp.update({val: trainwordCountCopy[a][b]})
            bow[categories[a]] = temp
    else:
        for a, value in enumerate(trainwordCountCopy):
            bow[categories[a]] = {}
            temp = {}
            countx = 0
            for val in (indexes2):
                temp.update({indexes2[val]: trainwordCountCopy[a][countx]})
                countx += 1
            bow[categories[a]] = temp
    results = testing_data["Category"]

    (true, false) = 0, 0
    if (indexes2 == None):
        constx = len(vector.get_feature_names_out())
    else:
        constx = len(indexes2)

    time = datetime.datetime.now()
    for count, a in enumerate(testing_data["Text"]):
        mx = -math.inf
        maxindex = 0
        for b in bow:
            extend = 0
            if len(bow) == 0:
                print("bow:", bow)
            cons = math.log2(len(bow[b]) / len(train_y))
            tr = np.sum(trainwordCountCopy[cat[b]])
            if (unigram == False):
                for c in range(len(a.split(" ")) - 1):
                    word = str(a.split(" ")[c]) + " " + str(a.split(" ")[c + 1])
                    try:
                        if bow[b][word] == 0:
                            extend = constx
                            break
                    except Exception:
                        extend = constx
                        break

                for c in range(len(a.split(" ")) - 1):
                    word = str(a.split(" ")[c]) + " " + str(a.split(" ")[c + 1])
                    try:
                        number = bow[b][word]
                        if (extend != 0):
                            if bow[b][word] != 0:
                                cons += math.log2((number + 1) / (tr + extend))
                            else:
                                cons += math.log2((1) / (tr + extend))

                        else:
                            cons += math.log2((number) / (tr))
                    except Exception:
                        cons += math.log2((1) / (tr + extend))
            else:
                for c in a.split(" "):
                    try:
                        if bow[b][c] == 0:
                            extend = constx
                            break
                    except Exception:
                        extend = constx
                        break
                for c in a.split(" "):
                    try:
                        if c not in my_stop_words:
                            if (extend != 0):
                                if (bow[b][c] != 0):
                                    cons += math.log2((bow[b][c] + 1) / (tr + extend))
                                else:
                                    cons += math.log2((1) / (tr + extend))
                            else:
                                cons += math.log2((number) / (tr))
                    except Exception:
                        cons += math.log2((1) / (tr + extend))

            if (cons > mx):
                mx = cons
                maxindex = cat[b]
        if (categories[maxindex] == categories[results[results.index[count]]]):
            true += 1
        else:
            false += 1

    return true / (true + false)


def tfidf(vector):
    specwords = []
    df4 = pd.DataFrame()
    for a in range(5):
        arr = []
        corpus = training_data[training_data["Category"] == a]["Text"]
        Z = vector.fit_transform(corpus)
        vocabulary = vector.get_feature_names_out()
        corpus = corpus.tolist()
        pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)), ('tfid', TfidfTransformer())]).fit(corpus)
        idx = np.argpartition(pipe['tfid'].idf_, 10)
        ls = pipe['tfid'].idf_
        ls2 = pipe['tfid'].idf_
        for b in range(10):
            ind = np.argmin(ls)
            value = ls[ind]
            ls[ind] = math.inf
            temp = [vocabulary[ind], value]
            specwords.append(vocabulary[ind])
            arr.append(temp)
        df = pd.DataFrame(data=arr,
                          columns=["Word", "TFID"])
        arr = []

        for b in range(10):
            ind = np.argmax(ls2)
            value = ls2[ind]
            ls2[ind] = -math.inf
            temp = [vocabulary[ind], value]
            arr.append(temp)
        df2 = pd.DataFrame(data=arr,
                           columns=["Word", "TFID"])

        res = []
        for m in specwords:
            if m not in res:
                res.append(m)

        df3 = pd.concat([df, df2], axis=1)
        df3.columns = pd.MultiIndex.from_product(
            [[categories[a].upper()], ["Effect of Presence", "Effect of Absence"], ["Word", "TF-IDF"]],
            names=["Category", "Effect", " "])

        df4 = pd.concat([df4, df3], axis=1)

    return res, df4


data = pd.read_csv("English Dataset.csv")
dataFrame = data.drop(["ArticleId"], axis=1)
categories = {0: "sport", 1: "business", 2: "politics", 3: "entertainment", 4: "tech"}
cat = {"sport": 0, "business": 1, "politics": 2, "entertainment": 3, "tech": 4}
dataFrame["Category"] = dataFrame["Category"].map(cat).astype(int)
my_stop_words = text.ENGLISH_STOP_WORDS

data = dataFrame.iloc[:, :]
vectorizer = CountVectorizer(stop_words="english")
vectorizerwithStop = CountVectorizer()
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
vectorizer2withStop = CountVectorizer(stop_words="english", analyzer='word', ngram_range=(2, 2))
X = vectorizer.fit_transform(data["Text"])

wordCount = np.zeros((5, (X.toarray()).shape[1]))

for a, value in enumerate(X.toarray()):
    category = data["Category"][a]
    wordCount[category] = np.sum([wordCount[category], value], axis=0)

wordCountCopy = wordCount.copy()
wordnumberdf = pd.DataFrame()
for count, a in enumerate(wordCountCopy):
    dict = []
    for b in range(4):
        ind = np.argmax(a)
        word = vectorizer.get_feature_names_out()[ind]
        amount = int(a[ind])
        temp = [word, amount]
        dict.append(temp)
        a[ind] = 0

    tempdf = pd.DataFrame(data=dict,
                          columns=pd.MultiIndex.from_product([[categories[count].upper()], ["Word", "Word Count"]]))

    wordnumberdf = pd.concat([tempdf, wordnumberdf], axis=1)

training_data, testing_data = train_test_split(dataFrame, test_size=0.2, shuffle=True)

train_x = vectorizer.fit_transform(training_data["Text"])
train_y = training_data["Category"]
train_x_withStop = vectorizerwithStop.fit_transform(training_data["Text"])
trainWordCount = np.zeros((5,(train_x.toarray()).shape[1]))
trainWordCountwithStop = np.zeros((5,(train_x_withStop.toarray()).shape[1]))

a1=bayes(train_x,train_y,data,trainWordCount,testing_data,True,vectorizer)

b1=bayes(train_x_withStop,train_y,data,trainWordCountwithStop,testing_data,True,vectorizerwithStop)


train_x_bigram = vectorizer2.fit_transform(training_data["Text"])
train_y_bigram = training_data["Category"]
train_x_bigram_stop=vectorizer2withStop.fit_transform(training_data["Text"])
trainWordCount_bigram = np.zeros((5,(train_x_bigram.toarray()).shape[1]))
trainWordCount_bigram_stop=np.zeros((5,(train_x_bigram_stop.toarray()).shape[1]))

accdf=pd.DataFrame(0.0,index=range(2), columns=range(2))


c1=bayes(train_x_bigram,train_y_bigram,data,trainWordCount_bigram,testing_data,False,vectorizer2)
d1=bayes(train_x_bigram_stop,train_y_bigram,data,trainWordCount_bigram_stop,testing_data,False,vectorizerwithStop)



vectorizer3 = CountVectorizer()
newdata,tfidfWithStopword = tfidf(vectorizer3)
train_x_updated_1 = vectorizer3.fit_transform(training_data["Text"])
train_y_updated_1 = training_data["Category"]

vectorizer4 = CountVectorizer(stop_words="english")

newdata2,tfidfWithoutStopword = tfidf(vectorizer4)
train_x_updated_2 = vectorizer4.fit_transform(training_data["Text"])
train_y_updated_2 = training_data["Category"]

(indexes3,indexes4)  = ({},{})
for count,a in enumerate(vectorizer3.get_feature_names_out()):
    if a in newdata:
        indexes3[count] = a


for count,a in enumerate(vectorizer4.get_feature_names_out()):
    if a in newdata2:
        indexes4[count] = a

newCountArr = np.zeros((5,(len(indexes3))))
newCountArr2 = np.zeros((5,(len(indexes4))))
e1=bayes(train_x_updated_1,train_y_updated_1,data,newCountArr,testing_data,True,vectorizer3,indexes3)
f1=bayes(train_x_updated_2,train_y_updated_2,data,newCountArr2,testing_data,True,vectorizer4,indexes4)

vectorizer5 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
vectorizer6withStop = CountVectorizer(stop_words="english",analyzer='word', ngram_range=(2, 2))

accdf[0][0]=a1
accdf[1][0]=b1
accdf[0][1]=c1
accdf[1][1]=d1

accdf.index =["Unigram","Bigram"]
accdf.columns=pd.MultiIndex.from_product([["Included","Not Included"]],names = ["Stop-word"])

