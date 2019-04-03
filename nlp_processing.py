import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import os

#os.chdir("/Users/adrienthomas/Documents/2018-question-answering/")
os.chdir("/home/charles1110/OneDrive/3eme annee ecole/OSY/Projet Illuin/projet_question_answering")


#%%

# Load the data

X_questions = pickle.load(open('pickled_data/pickle_questions', 'rb'))  # total of words in X_questions : 873973
X_contexts = pickle.load(open('pickled_data/pickle_contexts', 'rb'))   # total of words in X_contexts : 10400665
X_answers = pickle.load(open('pickled_data/pickle_answers', 'rb'))
f = open('data/glove.6B.50d.txt', encoding="utf8")

# Load the entire GloVe word embedding file into memory

glove_dict = dict()
for line in f:
    values = line.split()
    word = values[0]
    coefficients = np.asarray(values[1:], dtype='float32')
    glove_dict[word] = coefficients
f.close()  # Loaded 400 000 word vectors


#%% 

# Create our tokinizer

t = Tokenizer()
t.fit_on_texts(X_contexts)
t.fit_on_texts(X_questions)
vocab_size = len(t.word_index)


#%%

# Make a selection between the words of our vocabaluray in our Glove file, the digits and the unknown words during the creation of the embedding matrix

index_digit = vocab_size + 1
index_unknown = vocab_size + 2
embedding_matrix = np.zeros((vocab_size+3, 50))
for word, index in t.word_index.items():
    word_vectorised = glove_dict.get(word)
    if re.search(r"[0-9][0-9 /-_\.]{0,}", word) is not None:
        t.word_index[word] = index_digit
        embedding_matrix[-2] = (np.random.rand(1, 50)-0.5)*2
    elif word_vectorised is not None:
        embedding_matrix[index] = word_vectorised
    else:
        t.word_index[word] = index_unknown
        embedding_matrix[-1] = (np.random.rand(1, 50)-0.5)*2
        
#%%

# Put true random number for digit and unknow

random_coeffs=[]
embedding_permute = np.transpose(embedding_matrix[:vocab_size])
for k in range(50):
    random_coeffs.append([np.mean(embedding_permute[k]),np.std(embedding_permute[k])])

embedding_matrix[-1] = [np.random.normal(random_coeffs[k][0],random_coeffs[k][1],1) for k in range (50)]
embedding_matrix[-2] = [np.random.normal(random_coeffs[k][0],random_coeffs[k][1],1) for k in range (50)]

#%%
 
# Tokenising our questions and contexts
        
encoded_questions = t.texts_to_sequences(X_questions)
encoded_contexts = t.texts_to_sequences(X_contexts)
encoded_answers = t.texts_to_sequences(X_answers)


#%%

# Select the contexts with less than 150 words
len_context = 150
liste = []
for k in range(len(encoded_contexts)):
    if len(encoded_contexts[k]) > len_context:
        liste.append(k)
liste.reverse()
for e in liste:
    del encoded_questions[e]
    del encoded_contexts[e]
    del encoded_answers[e]
    del X_questions[e]
    del X_contexts[e]
    del X_answers[e]

for k in range(len(encoded_contexts)):
    assert(len(encoded_contexts[k]) <= len_context)


# Padding the documents (question and context) to a max length

max_length_questions = max([len(x) for x in encoded_questions])
padded_questions = pad_sequences(encoded_questions, maxlen=max_length_questions, padding='pre')

max_length_contexts = max([len(x) for x in encoded_contexts])
padded_contexts = pad_sequences(encoded_contexts, maxlen=max_length_contexts, padding='pre')


#%%

# Select the encoded answers that fit with the encoded contexts and get the start and end indexes

fails_index = []
answers_index = []
for i, answer in enumerate(encoded_answers):
    j = 0
    isHere = False
    j_start = None
    while j < max_length_contexts:
        if list(padded_contexts[i][j:j+len(answer)]) == answer:
            isHere = True
            j_start = j
        j += 1
    if isHere:
        answers_index.append((j_start, j_start + len(answer)-1))
    else:
        fails_index.append(i)

# Lets delete all the non corresponding QA sets
fails_index.reverse()
for e in fails_index:
    del encoded_questions[e]
    del encoded_contexts[e]
    del encoded_answers[e]
    padded_questions = np.delete(padded_questions,e,0)
    padded_contexts = np.delete(padded_contexts,e,0)
    del X_questions[e]
    del X_contexts[e]
    del X_answers[e]


#%%

# Building the p_starts and p_ends vectors

p_starts, p_ends = [], []
for indexes in answers_index:
    p_start, p_end = [0] * max_length_contexts, [0] * max_length_contexts
    p_start[indexes[0]] = 1
    p_end[indexes[1]] = 1
    p_starts.append(p_start)
    p_ends.append(p_end)


#%%

# Dumping all our files

vocab_data = {'vocab_size': vocab_size+3,
              'max_length_questions': max_length_questions,
              'max_length_contexts': max_length_contexts,
              'tokenizer': t}

pickle.dump(X_questions, open('pickled_data/pickle_filtered_questions', 'wb'))
pickle.dump(X_contexts, open('pickled_data/pickle_filtered_contexts', 'wb'))
pickle.dump(X_answers, open('pickled_data/pickle_filtered_answers', 'wb'))
pickle.dump(padded_questions, open('pickled_data/pickle_padded_questions', 'wb'))
pickle.dump(padded_contexts, open('pickled_data/pickle_padded_contexts', 'wb'))
pickle.dump(p_starts, open('pickled_data/pickle_padded_pstarts', 'wb'))
pickle.dump(p_ends, open('pickled_data/pickle_padded_pends', 'wb'))
pickle.dump(embedding_matrix, open('pickled_data/pickle_embedding_matrix', 'wb'))
pickle.dump(vocab_data, open('pickled_data/pickle_vocab_data', 'wb'))


