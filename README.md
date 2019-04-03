# 2019 Question Answering



## Dossier `data`

Dans ce dossier, vous trouverez toutes les données que nous avons utilisé pour ce projet.

- `common_words` : fichier contenant une liste de mots communs utilisée dans le preprocessing
- `train-v2.0.json` : fichier json contenant le dataset d'entrainement DataSquad sur lequel nous avons entrainé notre algorithme 
- `glove.6B.50d.txt` : fichier non présent sur le github car trop lourd. Il est nécessaire pour le preprocessing et il est téléchargable sur ce site : https://www.kaggle.com/watts2/glove6b50dtxt

## Dossier `pickled_data`

Dans ce dossier, nous avons enregistré toutes nos données sous le bon format, une fois qu'elles avaient passé les étapes de "split" et de "preprocessing".

## Dossier `saved_models`

Dans ce dossier, nous avons enregistré notre modèle qui marche actuellement le mieux sur notre problème, dans l'idée de le recharger à la demande. Il est possible d'enregister d'autres modèles tournant avec d'autres paramètres par exemple.

## Fichier python `split_train_data`

Dans ce fichier, nous séparons notre dataset d'entrainement `train-v2.0.json` en trois parties :
- les questions dans un fichier `pickle_questions`
- les réponses dans un fichier `pickle_answers`
- les contextes dans un fichier `pickle_contexts`

## Fichier python `nlp_preprocessing`

Dans ce fichier, nous traitons les questions, réponses et contextes précedemment séparés. <br>
Nous commencons par "tokeniser" chacun de ces datasets en retirant la ponctuation et nous créons un dictionnaire correspondant au vocabulaire de nos datasets. <br>
Ensuite, nous remplacons chaque mot par son index dans le dictionnaire vocabulaire. <br>
Puis, nous créons notre matrice d'embedding qui va associer à chaque mot (chaque index) un vecteur de 50 poids. Nous construisons cette matrice grâce au fichier `glove.6B.50d.txt`.<br>
Nous passons ensuite par une étape de "padding" afin que tous les contextes et les questions est la même longueur. A ce moment, nous en profitons pour ne préserver que les contextes qui font moins de 150 mots.<br>
Enfin, nous construisons nos indices de début et de fin de la réponse tirée du contexte.<br>
Nous enregistrons tous ces fichiers traités dans les fichiers `picke_padded_questions`, `picke_padded_contexts`, `picke_padded_pstarts`, `picke_padded_pends`, `picke_padded_embdedding_matrix` et `picke_padded_vocab_data`.


