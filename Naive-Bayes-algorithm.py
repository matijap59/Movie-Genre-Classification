import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

class TextProcessing:
    pass

def extract_unique_genres(all_reviews_genres):
        return set(all_reviews_genres)

class NB_Sentiment_Classifier:
    def __init__(self,alpha,beta):
        self.alpha=alpha
        self.beta=beta
        self.categories=[]
        self.count_words_dict = {}
        self.review_counts = {}
        self.prior={}

    def _preprocess(self, text:str) -> str:
        import re
        text = re.sub(r'[^\w\s]', '', text)
        words = text.lower().split()
        return words

    def fit(self,X,y):
        self.text=X
        self.genres=y
        self.categories=extract_unique_genres(self.genres)

        reviews={}
        for i in range(len(self.text)):
            text = self.text.iloc[i]['Text_cleaning']
            genre=self.genres.iloc[i]
            reviews[text]=genre

        for category in self.categories:
            self.count_words_dict[f'{category}'] = {}
            self.review_counts[f'{category}']=0

        for review, sentiment in reviews.items():
            words = review.split()
            for category in self.categories:
                for word in words:
                    if sentiment==category:
                        self.count_words_dict[f'{category}'][word] =self.count_words_dict[f'{category}'].get(word,0)+1
                if sentiment==category:
                    self.review_counts[f'{category}']+=1

        for category in self.categories:
            self.count_words_dict[category] = dict(sorted(self.count_words_dict[category].items(), key=lambda x: x[1], reverse=True))
            
        n_total_reviews = sum(self.review_counts.values())
        for category in self.categories:
            self.prior[category] = self.review_counts[category] / n_total_reviews
        print(self.prior)

    def predict(self, X):                           #Using Laplace add alpha smoothing
        y_pred = [None] * len(X)
        for i in range(len(X)):
            text = X.iloc[i]['Text_cleaning']
            words = self._preprocess(text)
            p_words_given_category={}
            for category in self.categories:        #for each category, we need to find probability that the word belongs to that category
                n_category_words=sum(self.count_words_dict[category].values())
                list_category=[]
                for word in words:
                    p_word_given_category =(self.count_words_dict[category].get(word, 0) + self.alpha) / (n_category_words + self.alpha*len(self.count_words_dict[category])) # Laplace Smoothing
                    list_category.append(p_word_given_category)
                p_words_given_category[category]=list_category
                #temp=np.prod(p_words_given_category[category])                             #way multiply all probability
                #p_text_given_category[category]=self.prior[category]*temp
                p_words_given_category[category]=np.log(self.prior[category])+np.sum(np.log(list_category))     #way ln whole expression

            predict_genre = max(p_words_given_category, key=p_words_given_category.get)
            y_pred[i]=predict_genre

        return y_pred
    
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

if __name__ == '__main__':
    data_set = pd.read_csv('data-set.csv')
    data_set['Title_Description'] = data_set['Title'] + " " + data_set['Description']
    grouped_data_set = data_set.groupby('Id').agg({'Title_Description': ' '.join, 'Genre': 'first','Text_cleaning': 'first'}).reset_index()
    grouped_data_set.drop(['Id'],axis=1, inplace=True)

    #print(data_set.groupby("Genre").count())

    X_train, X_test, y_train, y_test = train_test_split(grouped_data_set[['Text_cleaning']], grouped_data_set['Genre'], test_size=0.2, random_state=42)

    nb=NB_Sentiment_Classifier(1,1)

    nb.fit(X_train,y_train)

    y_pred=nb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy*100)

    macro_measure, micro_measure=nb.calculate_f_measures(y_test,y_pred)

    print(macro_measure*100)
    print(micro_measure*100)

    
