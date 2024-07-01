import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
import math

class KNearestNeighborsClassifier:
    def __init__(self,n=1, k=1):
        self.n = n
        self.k = k
        self.categories=[]
        self.beta=1

    def fit(self,X,y):
        self.X=sorted(X, key=lambda x: x[1], reverse=True) #descending sequence
        self.y=y

    def predict(self, X):

        Y_pred = [None] * len(X)

        for i, list_X_test in enumerate(X):
            distance = {}

            k_most_frequency_n_grams = self.X[:self.k]

            for word, value in k_most_frequency_n_grams:
                parts = word.split("_")
                n_gram=parts[0]
                genre=parts[1]
                frequency_test = next((val for key, val in list_X_test if key == n_gram), 0)

                if genre not in distance:
                    distance[genre] = 0

                distance[genre] = math.sqrt(distance[genre]**2 + (value - frequency_test)**2)

            Y_pred[i] = min(distance, key=distance.get)

        return Y_pred
    
    def pretprocessing_data(self,X_train,Y_train):
        count_all_words={}
        count_words_genre={}
        total_n_grams=0
        unique_genres = Y_train.unique()
        self.categories=unique_genres
        for i in range(len(X_train)):
            text = X_train.iloc[i]['Title_Description']
            #text = X_train.iloc[i]['Text_cleaning']
            genre=Y_train.iloc[i]
            
            n_grams = [text[j:j+self.n] for j in range(len(text)-self.n+1)]
            total_n_grams+=len(text)-self.n+1

            for n_gram in n_grams:
                if n_gram in count_all_words.keys():
                    count_all_words[n_gram]+=1
                else:
                    count_all_words[n_gram]=1
                if n_gram+"_"+genre in count_words_genre.keys():
                    count_words_genre[n_gram+"_"+genre]+=1
                else:
                    count_words_genre[n_gram+"_"+genre]=1

        counter = Counter(count_all_words)
        most_common_n_grams = counter.most_common(len(count_all_words))
        most_common_n_grams_keys = [key for key, value in most_common_n_grams]

        frequency_dict={}

        for common_n_grams_keys in most_common_n_grams_keys:
            for genre in unique_genres:
                key = f"{common_n_grams_keys}_{genre}"
                if key in count_words_genre.keys():
                    frequency_dict[key]=count_words_genre[key]/total_n_grams
                else:
                    frequency_dict[key]=0
        return from_dict_to_list(frequency_dict)
    
    def pretprocessing_x_test(self,X_test):
        list_instances=[]
        for i in range(len(X_test)):
            count_all_words={}
            frequency={}
            text = X_train.iloc[i]['Title_Description']
            #text = X_train.iloc[i]['Text_cleaning']
            n_grams = [text[j:j+self.n] for j in range(len(text)-self.n+1)]
            total_n_grams=len(text)-self.n+1
            for n_gram in n_grams:
                if n_gram in count_all_words.keys():
                    count_all_words[n_gram]+=1
                else:
                    count_all_words[n_gram]=1
            for key,value in count_all_words.items():
                frequency[key]=value/total_n_grams
            
            list=from_dict_to_list(frequency)
            list_instances.append(list)
        return list_instances
    
    def calculate_f_measures(self,y_test,y_pred):

        confusion_matrices={}

        for category in self.categories:
            confusion_matrices[category] = {'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'true_negative': 0}

        for i in range(len(y_test)):                #creating matrix confusions
            true_class = y_test.iloc[i]
            predicted_class = y_pred[i]

            if true_class in confusion_matrices:
                matrix = confusion_matrices[true_class]
                if true_class == predicted_class:
                    matrix['true_positive'] += 1
                else:
                    matrix['false_negative'] += 1

            for cls in self.categories:
                if cls != true_class:
                    if cls == predicted_class:
                        confusion_matrices[cls]['false_positive'] += 1
                    else:
                        confusion_matrices[cls]['true_negative'] += 1

        macro_precision = 0                 #calculating macro F measure
        macro_recall = 0

        for cls, matrix in confusion_matrices.items():
            tp = matrix['true_positive']
            fn = matrix['false_negative']
            fp = matrix['false_positive']

            if tp + fp == 0:                    #Prevention of division by 0
                precision = 0
            else:
                precision = tp / (tp + fp)

            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            macro_precision += precision
            macro_recall += recall

        macro_precision =macro_precision/len(confusion_matrices)
        macro_recall = macro_recall/len(confusion_matrices)

        if macro_precision + macro_recall == 0:
            macro_f1 = 0
        else:
            macro_f1 = (1+self.beta**2) * macro_precision * macro_recall / ((self.beta**2)*macro_precision + macro_recall)

        tp_micro = 0                #Calculating micro F measure
        fp_micro = 0
        fn_micro = 0

        for cls, matrix in confusion_matrices.items():
            tp_micro += matrix['true_positive']
            fn_micro += matrix['false_negative']
            fp_micro += matrix['false_positive']

        if tp_micro + fp_micro == 0:
            micro_precision = 0
        else:
            micro_precision = tp_micro / (tp_micro + fp_micro)

        if tp_micro + fn_micro == 0:
            micro_recall = 0
        else:
            micro_recall = tp_micro / (tp_micro + fn_micro)

        if micro_precision + micro_recall == 0:
            micro_f1 = 0
        else:
            micro_f1 = (1+self.beta**2) * micro_precision * micro_recall / ((self.beta**2)*micro_precision + micro_recall)

        return macro_f1,micro_f1
    
def from_dict_to_list(frequency):
    list=[]
    for key, value in frequency.items():
        list.append((key,value))
    return list
    
if __name__=="__main__":

    data_set = pd.read_csv('data-set.csv')
    data_set['Title_Description'] = data_set['Title'] + " " + data_set['Description']
    grouped_data_set = data_set.groupby('Id').agg({'Title_Description': ' '.join, 'Genre': 'first','Text_cleaning': 'first'}).reset_index()     #obrisati posle Text_cleaning
    grouped_data_set.drop(['Id'],axis=1, inplace=True)

    #X_train, X_test, y_train, y_test = train_test_split(grouped_data_set[['Text_cleaning']], grouped_data_set['Genre'], test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(grouped_data_set[['Title_Description']], grouped_data_set['Genre'], test_size=0.2, random_state=42)

    classifier = KNearestNeighborsClassifier(3, 2000)
    
    frequencies=classifier.pretprocessing_data(X_train,y_train)

    proccessed_x_test=classifier.pretprocessing_x_test(X_test)

    classifier.fit(frequencies,y_train)

    y_pred=classifier.predict(proccessed_x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy*100)          #n=3, k=2000 54%

    macro_measure, micro_measure=classifier.calculate_f_measures(y_test,y_pred)

    #print(macro_measure*100)
    print(micro_measure*100)
