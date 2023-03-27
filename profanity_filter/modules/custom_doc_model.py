from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
import pickle
from os import path
from concurrent import futures
#from bert_serving.client import BertClient
from concurrent.futures import ThreadPoolExecutor
import json
import shutil

class svm_classifier:

    global vectorizer
    global clf
    global bClient
    global cfg

    def __init__(self,data=None):

        #cfg = json.loads(open('config/config.json')) 
        if data is None:
            #if model is already trained, load the model
            if path.exists('model/model.sav') and path.exists('model/vectorizer.pk'):
                self.clf = pickle.load(open('model/model.sav', 'rb'))
                self.vectorizer = pickle.load(open('model/vectorizer.pk', 'rb'))
            #otherwise, train new model
            else:
                csv = pd.read_csv('data/consolidated_data.csv')
                csv = shuffle(csv)
                csv = csv.head(20000)
                data = pd.DataFrame()
                data['is_offensive'] = csv['is_offensive']
                data['text'] = csv['text']
                self.train_model(data)
                #self.bClient = BertClient()
        else:
            self.train_model(data)

    def train_model(self, data):
        result = pd.DataFrame()
        data = data.dropna()
        result['is_offensive'] = data['is_offensive']

        #vectorize the text 
        self.vectorizer = TfidfVectorizer(stop_words='english').fit(data['text'])

        tf = self.vectorizer.transform(data['text'])


        #test train split
        tf_test = tf[int(result.shape[0] - result.shape[0]/5):]
        tf_train = tf[:int(result.shape[0] - result.shape[0]/5)]

        result_train = result.head(int(result.shape[0] - result.shape[0]/5))
        result_test  = result.tail(int(result.shape[0]/5) +1)

        #train the model
        self.clf = SVC(kernel='linear', probability=True)
        self.clf.fit(tf_train, result_train['is_offensive'].tolist())

        #test the trained model
        pred       = self.clf.predict(tf_test)
        result_test_arr = result_test['is_offensive'].tolist() 

        correctPred = 0
        wrongPred   = 0
        class0 = 0
        class1 = 0
        class2 = 0
        for indx, val in enumerate(pred):
            if val == result_test_arr[indx]:
                correctPred += 1
            else:
                wrongPred += 1
                if str(val) == '0':
                    class0 = class0 +1
                elif str(val) == '1':
                    class1 = class1 +1
                else:
                    class2 = class2 +1

        #print test results
        print('class0>>' + str(class0))
        print('class1>>' + str(class1))
        print('class2>>' + str(class2))
        print('model accuracy:' + str((correctPred/(wrongPred+correctPred))*100) )

        #take backup
        if path.exists('model/model.sav') and path.exists('model/vectorizer.pk'):
            shutil.move('model/model.sav', 'model/bkp/model.sav')
            shutil.move('model/vectorizer.pk', 'model/bkp/vectorizer.pk')
        
        #save model
        modelName = 'model/model.sav'
        pickle.dump(self.clf, open(modelName, 'wb'))
        vectorizerName = 'model/vectorizer.pk'
        pickle.dump(self.vectorizer, open(vectorizerName, 'wb'))
        
    def predict(self, text):
        text_df = pd.DataFrame(columns=['text'])
        text_df.loc[-1] = [text]

        #vectorize the text same vectorizer used for training data
        tf_ext = self.vectorizer.transform(text_df['text'])
        pred  = self.clf.predict(tf_ext)
        pred_proba = self.clf.predict_proba(tf_ext)

        #find the high probability class in prediction
        highest_proba = max(pred_proba[0])
        if pred[0] == 0:
            return {"classification" : "Not Offensive" , "probability" : highest_proba , 'text' : text }
        elif (pred[0] == 1):
            return {"classification" : "Offensive" , "probability" : highest_proba, 'text' : text }
       

    def line_analysis(self, data):
        jobs = []
        result = []
        if '.' in data:
            with ThreadPoolExecutor(5) as executor:
                for line in data.split('.'):
                    if line.strip():
                        jobs.append(executor.submit(self.predict, line))
                        #analysis = classifier.predict(line)
                for job in futures.as_completed(jobs):
                    analysis = job.result()
                    obj = {}
                    obj['text'] = analysis['text']
                    obj['classification'] = analysis['classification']
                    obj['probability'] = analysis['probability']
                    result.append(obj)
        return result

    def retrain_model(self):
        csv = pd.read_csv('data/consolidated_data.csv')
        csv = shuffle(csv)
        csv = csv.head(20000)
        data = pd.DataFrame()
        data['is_offensive'] = csv['is_offensive']
        data['text'] = csv['text']
        self.train_model(data)
        return