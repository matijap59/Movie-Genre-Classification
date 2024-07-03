import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import string
import matplotlib.pyplot as plt
import unicodedata

class WordProcessing:
    
    def __init__(self,genres):
        self.vocabulary=string.ascii_letters + " .,;!?~%$@#^&*()'"              #one-hot encoding for each caracter
        self.vocabulary_size=len(self.vocabulary)
        self.genres=genres

    def unicode_to_ascii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.vocabulary
        )
    
    def letter_to_index(self,letter):        #return index in vocabulary
        return self.vocabulary.find(letter)
    
    def letter_to_tensor(self,letter):
        tensor = torch.zeros(1, self.vocabulary_size)
        tensor[0][self.letter_to_index(letter)] = 1
        return tensor


    def line_to_tensor(self,line):
        tensor = torch.zeros(len(line), 1, self.vocabulary_size)
        for li, letter in enumerate(line):
            tensor[li][0][self.letter_to_index(letter)] = 1
        return tensor
    
    def genre_to_tensor(self,genre, genres):
        return torch.tensor([genres.index(genre)], dtype=torch.long)
    
    def compute_label_weights(self, df):                            #balance output class
        label_counts = df['Genre'].value_counts().to_dict()

        label_weight_map = {label: 1.0 / torch.log1p(torch.tensor(count, dtype=torch.float)) for label, count in label_counts.items()}      #weight=1/log(1+count(genre))

        self.genres = sorted(label_counts.keys())
        label_weights = torch.tensor([label_weight_map[l] for l in self.genres])
        return label_weights
    
    def train_valid_test_split(self,df, valid_split, test_split):
        train_size = int(len(df) * (1 - valid_split - test_split))
        valid_size = int(len(df) * valid_split)
        train = df[:train_size]
        valid = df[train_size:train_size + valid_size]
        test = df[train_size + valid_size:]

        X_train = train['Title_Description'].apply(lambda n: self.line_to_tensor(n))
        y_train = train['Genre'].apply(lambda l: self.genre_to_tensor(l, self.genres))
        X_valid = valid['Title_Description'].apply(lambda n:self.line_to_tensor(n))
        y_valid = valid['Genre'].apply(lambda l: self.genre_to_tensor(l, self.genres))
        X_test = test['Title_Description'].apply(lambda n:self.line_to_tensor(n))
        y_test = test['Genre'].apply(lambda l: self.genre_to_tensor(l, self.genres))

        return X_train, y_train, X_valid, y_valid, X_test, y_test
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size # dimension hidden layer
        self.i2h = nn.Linear(input_size, hidden_size, bias=True) # matrix U
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=True) # matrix W
        self.h2o = nn.Linear(hidden_size, output_size, bias=True) # matrix V
        self.softmax= nn.LogSoftmax(dim=1) # activation function used for classification genres
        #self.loss_fn = nn.NLLLoss(weight=label_weights)             #sum of logs
        #self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = torch.zeros(1, self.hidden_size)

        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden
    

def train(category_tensor: torch.Tensor, line_tensor: torch.Tensor):
    hidden = None
    model.train()
    optimizer.zero_grad()
    # line tensor
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss =loss_fn(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()


def predict(line_tensor):
    model.eval()
    hidden = None
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output

# def predict_test(model, X):
#     predictions = []
#     model.eval()
#     with torch.no_grad():
#         for idx, row in X.items():
#             hidden = None
#             for i in range(row.size()[0]):
#                 output, hidden = model(row[i], hidden)
#             predictions.append(output.argmax(1).item())
#     return predictions

def evaluate(X, y):
    model.eval()
    total_acc, total_count = 0, 0

    for idx, (text, label) in enumerate(zip(X,y)):
        predicted_label = predict(text)
        a = (predicted_label.argmax(1) == label).sum().item()
        total_acc += a
        total_count += label.size(0)
    return total_acc / total_count


def evaluate_confusion_matrix(X, y, genres):
    model.eval()
    confusion_matrix = {}

    for idx, (text, label) in enumerate(zip(X, y)):
        predicted_label = predict(text)
        predicted_genre_item = predicted_label.argmax(1).item()
        true_genre_item = label.item()

        true_genre=genres[true_genre_item]
        predicted_genre=genres[predicted_genre_item]

        if true_genre not in confusion_matrix:
            confusion_matrix[true_genre] = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}

        if true_genre == predicted_genre:
            confusion_matrix[true_genre]['tp'] += 1
        else:
            confusion_matrix[true_genre]['fn'] += 1
            if predicted_genre not in confusion_matrix:
                confusion_matrix[predicted_genre] = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
            confusion_matrix[predicted_genre]['fp'] += 1

        for cls in genres:
            if cls != true_genre and cls != predicted_genre:
                if cls not in confusion_matrix:
                    confusion_matrix[cls] = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
                confusion_matrix[cls]['tn'] += 1

    return confusion_matrix

def calculate_f_measures(confusion_matrix, beta):
        macro_precision = 0                 #calculating macro F measure
        macro_recall = 0

        for cls, matrix in confusion_matrix.items():
            tp = matrix['tp']
            fn = matrix['fn']
            fp = matrix['fp']

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

        macro_precision =macro_precision/len(matrix)
        macro_recall = macro_recall/len(matrix)

        if macro_precision + macro_recall == 0:
            macro_f1 = 0
        else:
            macro_f1 = (1+beta**2) * macro_precision * macro_recall / ((beta**2)*macro_precision + macro_recall)

        tp_micro = 0                #Calculating micro F measure
        fp_micro = 0
        fn_micro = 0

        for cls, matrix in confusion_matrix.items():
            tp_micro += matrix['tp']
            fn_micro += matrix['fn']
            fp_micro += matrix['fp']

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
            micro_f1 = (1+beta**2) * micro_precision * micro_recall / ((beta**2)*micro_precision + micro_recall)

        return macro_f1,micro_f1

    
if __name__=="__main__":

    data_set = pd.read_csv('data-set.csv')
    data_set['Title_Description'] = data_set['Title'] + " " + data_set['Description']
    grouped_data_set = data_set.groupby('Id').agg({'Title_Description': ' '.join, 'Genre': 'first','Text_cleaning': 'first'}).reset_index()     #obrisati posle Text_cleaning
    grouped_data_set.drop(['Id'],axis=1, inplace=True)

    genres = sorted(grouped_data_set['Genre'].unique())

    #print(grouped_data_set)

    #print(genres)

    word_processing=WordProcessing(genres)

    grouped_data_set['Title_Description'] = grouped_data_set['Title_Description'].apply(lambda n: word_processing.unicode_to_ascii(n))


    X_train, y_train, X_valid, y_valid, X_test, y_test = word_processing.train_valid_test_split(grouped_data_set, 0.15, 0.2)

    label_weights=word_processing.compute_label_weights(grouped_data_set)

    loss_fn = nn.NLLLoss(weight=label_weights)

    model=RNN(word_processing.vocabulary_size,128,len(genres))

    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.05)

    epochs = 5
    plot_every = 1000
    all_losses = []
    current_loss = 0
    iter=1

    for epoch in range(epochs):
        model.train()

        for idx, row in X_train.items():
            output, loss = train(y_train[idx], row)
            current_loss += loss
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                print(f"Epoch {epoch}, Training loss {current_loss/plot_every}")
                current_loss = 0
            iter+=1

        valid_loss = 0
        accuracy = 0
        total_count = 0

        model.eval()

        for idx, row in X_valid.items():
            hidden = None

            for i in range(row.size()[0]):
                output, hidden = model(row[i], hidden)

            loss = loss_fn(output, y_valid[idx])
            valid_loss += loss.item()

            accuracy += (output.argmax(1) == y_valid[idx]).sum().item()
            total_count += y_valid[idx].size(0)

        print(f"Epoch: {epoch}, validation loss: {valid_loss/len(X_valid)}. Accuracy: {accuracy/total_count}")

    plt.figure()
    plt.plot(all_losses)

    accuracy = evaluate(X_test, y_test)
    print(f"Accuracy on test set: {accuracy}")

    matrix=evaluate_confusion_matrix(X_test,y_test,genres)

    macro_measure, micro_measure=calculate_f_measures(matrix,1)

    print(macro_measure)
    print(micro_measure)



    
