# Kütüphaneler

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score,roc_auc_score, roc_curve,auc

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

class mainContainer():
    
    def loadDataset(self):
        print("Course Review Data")
        print()
        # #conda install xlrd
        data = pd.read_excel("dataset2.xlsx")

        print("Veri setinin ilk 5 satırı")
        print("----------------------------")
        print(data.head()) # data.tail sondaki 5 satır için kullanılabilir. (5 sayısı default)
        print()
        
        print("Veri seti Özellikleri")
        print("----------------------------")
        print(data.describe())
        print()
        
        print("Veri seti Bilgisi")
        print("----------------------------")
        print(data.info())
        print()
        
        print("Veri setinin Sütunları")
        print("----------------------------")
        for i in range(len(data.columns)):
            print("{0} sütun : ".format(i+1),data.columns[i])
        print()
        
        data = self.cleaner(data)
        X = data['comment']
        y = data['label'] 
        
        labelEncoding = preprocessing.LabelEncoder()
        y = labelEncoding.fit_transform(y)
        
        return data,X,y
    
    def cleaner(self,data):
        # DATA CLEANING
        print("Data Cleaning")
        for idx in range(len(data.comment)):
            data.comment[idx] = self.cleanText(str(data.comment[idx]))
            if idx % 250 == 0:
                print(idx)
        
        print("\nEksik/Boş Veriler")
        print(data.isna().sum())
        print()
        #Boş değerler içeren satırlar bırakıldı. (Drop edildi.)
        data = data.dropna(how='any')
        
        return data
        
    def visualizer(self,data):
        fig , ax = plt.subplots(figsize = (10,10))
        ax = data['label'].value_counts().plot(kind = 'bar')      
        plt.xticks(rotation = 0, size = 14)
        plt.yticks(size = 14, color = 'white')
        plt.title('Distribution of Sentiment', size = 20)
 
#Bu kısımda normal şartlarda histogram grafiğinin üzerine veri sayılarımı yazdırmaya çalıştım. Neden çalışmadığı hakkında bir bilgim yok.
#Yorum satırına çevirip devam ediyorum.
# =============================================================================
#         ax.annotate(text = data['label'].value_counts().values[0], xy = (-0.13,259), size = 18)
# =============================================================================
# =============================================================================
#         ax.annotate(text = data['label'].value_counts().values[1], xy = (0.87,303),  size = 18)
# =============================================================================
# =============================================================================
#         ax.annotate(text = data['label'].value_counts().values[2], xy = (1.87,285),  size = 18)
# =============================================================================
# =============================================================================
#         ax.annotate(text = data['label'].value_counts().values[3], xy = (2.87,231),  size = 18)
# =============================================================================
# =============================================================================
#         ax.annotate(text = data['label'].value_counts().values[4], xy = (3.87,266),  size = 18)
# =============================================================================
# =============================================================================
#         plt.show()  
# =============================================================================

        
    def XGBoost(self,X,y): 
        print("\nXGBoost Classifier")
        print("----------------------------")
        # Veriler eğitim ve test olmak üzere bölünüldü.
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)
        # Pipeline sınıfından pipeline nesnesi oluşturuldu. 
        # TfidfVectorizer ile string veriler sayısal verilere dönüştürüldü.
        # Pipeline : Birden fazla işlemi seri olarak yapmak için kullanılır.
        pipeline = Pipeline([('vectorizer', TfidfVectorizer()), ('model', XGBClassifier())])
        # Eğitim işlemi
        pipeline.fit(X_train, y_train)
        # Model Performansı
        # Model performansı bu aşamada perfResult metodu ile yazdırıldı.
        self.perfResults(pipeline, X_test, y_test)
        return pipeline
    
    def SVC(self, X,y):
        print("\nSupport Vector Classifier")
        print("----------------------------")
        # Veriler eğitim ve test olmak üzere bölünüldü.
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)
        # Pipeline sınıfından pipeline nesnesi oluşturuldu. 
        # TfidfVectorizer ile string veriler sayısal verilere dönüştürüldü.
        # Pipeline : Birden fazla işlemi seri olarak yapmak için kullanılır.
        pipeline = Pipeline([('vectorizer', TfidfVectorizer()), ('model', LinearSVC())])
        # Eğitim işlemi
        pipeline.fit(X_train, y_train)
        # Model Performansı
        # Model performansı bu aşamada perfResult metodu ile yazdırıldı.
        self.perfResults(pipeline, X_test, y_test)
        return pipeline
 
    def perfResults(self, obj, X_test, y_test):
        y_pred=obj.predict(X_test) # Modele y verilmeden x ile y'yi bulması sağlanır. Bu durum test edilir
        cm = confusion_matrix(y_test, y_pred)
        print("Score : " + str(obj.score(X_test,y_test)))   
        print("\nAccuracy : "  + str(accuracy_score(y_test, y_pred)))
        print("\nConfusion Matrix\n" + str(cm))
        print("\nR2 Score : " + str(r2_score(y_test, y_pred)))
        print("\nMean Absolute Error : " + str(mean_absolute_error(y_test, y_pred)))
        print("\nMean Squared Error : " + str(mean_squared_error(y_test, y_pred)))
        print("\nClassification Report\n" + str(classification_report(y_test, y_pred)))

    def cleanText(self,string):
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        cleanedText = ' '.join(''.join([i for i in string if not i.isdigit()]).split())
        
        for i in punc:
            cleanedText = cleanedText.replace(i, '')
            
        a = [i for i in cleanedText if i.isalpha() or i == ' ']
                
        final_text = ' '.join(''.join(a).split())
        return final_text
            
    
    def prediction(self,obj):
        dsc = str(input("Oluşturulan model üzerinde prediction yapmak ister misiniz (E/H) ?"))
        if dsc == 'E':     
            txt = str(input("Metninizi giriniz (ENG) :\n"))
            state = obj.predict([txt])
            if state == 1:
                print("Prediction : STAR 1")
            if state == 2:
                print("Prediction : STAR 2")
            if state == 3:
                print("Prediction : STAR 3")
            if state == 4:
                print("Prediction : STAR 4")
            if state == 5:
                print("Prediction : STAR 5")

            print()
        
def main():  
    # Sınıf nesnesi oluştur.
    mc = mainContainer()
    
    # Veri seti yükleme fonksiyonuna git
    data1,X,y = mc.loadDataset()
    
    # Veri Görselleştirme
    mc.visualizer(data1)
    
    # Support Vector Classifier
    objSVC = mc.SVC(X,y)
    
    # XGBoost Classifier
    objXG = mc.XGBoost(X,y)

    print("\nEğitim işlemi tamamlandı.")

    #Prediction
    print("\nLinear Support Vector Classfier Prediction")
    print("--------------------------------------------") 
    mc.prediction(objSVC)
    
    print("\nXGBoost Classifier Prediction")
    print("--------------------------------------------") 
    mc.prediction(objXG)


if __name__== "__main__":
    # main fonksiyonu
    main()