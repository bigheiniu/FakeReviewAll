
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
'''
Binary Classification model for the fake review
'''
class Classifier(object):
    def __init__(self, config_dic):
        self.config_dic = config_dic

    def build_model(self):
        raise NotImplementedError

    def train(self, train_data):
        raise NotImplementedError

    def test(self, test_data):
        raise NotImplementedError



class XgboostClassfier(Classifier):
    def __init__(self, config_dic):
        super(XgboostClassfier, self).__init__(config_dic)

    def build_model(self):
        self.model = XGBClassifier(**self.config_dic)

    def train(self, train_data):
        feature = train_data['feature']
        label = train_data['label']
        self.model.fit(feature, label)

    def test(self, test_data):
        feature = test_data['feature']
        label = test_data['label']
        pre = self.model.predict(feature)
        pre = [round(ele) for ele in pre]
        acc = accuracy_score(label, pre)
        print("[XGBoost] Accuracy: %.2f%%" % (acc * 100.0))

class SVMclassifier(Classifier):
    def __init__(self, config_dic):
        super(SVMclassifier, self).__init__(config_dic)


    def build_model(self):
        self.model = SVC(**self.config_dic)

    def train(self, train_data):
        feature = train_data['feature']
        label = train_data['label']
        self.model.fit(feature, label)

    def test(self, test_data):
        feature = test_data['feature']
        label = test_data['label']
        pre = self.model.predict(feature)
        pre = [round(ele) for ele in pre]
        acc = accuracy_score(label, pre)
        print("[SVM] Accuracy: %.2f%%" % (acc * 100.0))



class LogisticClassifier(Classifier):
    def __init__(self, config_dic):
        super(LogisticClassifier, self).__init__(config_dic)

    def build_model(self):
        self.model = LogisticRegression(**self.config_dic)

    def train(self, train_data):
        feature = train_data['feature']
        label = train_data['label']
        self.model.fit(feature, label)

    def test(self, test_data):
        feature = test_data['feature']
        label = test_data['label']
        pre = self.model.predict(feature)
        pre = [round(ele) for ele in pre]
        acc = accuracy_score(label, pre)
        print("[Logistic] Accuracy: %.2f%%" % (acc * 100.0))
