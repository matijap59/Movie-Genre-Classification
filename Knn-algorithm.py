import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
import math

class KNearestNeighborsClassifier():
    def __init__(self,n=1, k=1):
        self.n = n
        self.k = k

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
        for i in range(len(X_train)):
            text = X_train.iloc[i]['Title_Description']
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

    X_train, X_test, y_train, y_test = train_test_split(grouped_data_set[['Title_Description']], grouped_data_set['Genre'], test_size=0.2, random_state=42)

    classifier = KNearestNeighborsClassifier(3, 4000)
    
    frequencies=classifier.pretprocessing_data(X_train,y_train)

    proccessed_x_test=classifier.pretprocessing_x_test(X_test)

    classifier.fit(frequencies,y_train)

    y_pred=classifier.predict(proccessed_x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy*100)          #n=3, k=2000 54%
