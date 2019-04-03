""" This code splits the train data """

import json
import pickle

with open("data/train-v2.0.json") as data_json:
    data_j = json.load(data_json)
    data = data_j["data"]


# useful function

def char_to_word(cont, integer):
    context_start = cont[:integer + 1]
    return context_start.count(' ')


# splitting the data between questions, contexts and answers
X = []  # input
Y = []  # output
n = len(data)
for i in range(n):
    articles = data[i]["paragraphs"]
    for j in range(len(articles)):
        paragraph = articles[j]["qas"]
        context = articles[j]["context"]
        for k in range(len(paragraph)):
            if paragraph[k]["is_impossible"] == 0:
                question = paragraph[k]
                X.append([question["question"], context])
                Y.append([char_to_word(context, question["answers"][0]["answer_start"]),
                          question["answers"][0]["text"]])

questions = [x[0] for x in X]
contexts = [x[1] for x in X]
answers = [y[1] for y in Y]
pickle.dump(questions, open('pickled_data/pickle_questions', 'wb'))
pickle.dump(contexts, open('pickled_data/pickle_contexts', 'wb'))
pickle.dump(answers, open('pickled_data/pickle_answers', 'wb'))
