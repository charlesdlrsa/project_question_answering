# 2019 Question Answering

L'ensemble du projet est détaillé et bien expliqué dans le fichier pdf `rapport_projet`.<br>
Ci-dessous, vous trouverez une description des rôles des différents dossiers et fichiers de ce repo github.<br>
Chacun des fichiers python est également bien commenté pour une meilleure compréhension.

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
Puis, nous créons notre matrice d'embedding qui va associer à chaque mot (chaque index) un vecteur de 50 poids. Nous construisons cette matrice grâce au fichier `glove.6B.50d.txt`.<br><br>
Nous passons ensuite par une étape de "padding" afin que tous les contextes et les questions aient la même longueur. A ce moment, nous en profitons pour ne préserver que les contextes qui font moins de 150 mots.<br>
Enfin, nous construisons nos indices de début et de fin pour chaque réponse tirée de son contexte.<br>
Nous enregistrons tous ces fichiers traités dans les fichiers `picke_padded_questions`, `picke_padded_contexts`, `picke_padded_pstarts`, `picke_padded_pends`, `picke_padded_embdedding_matrix` et `picke_padded_vocab_data`.

## Fichier python `qa_model_final`

C'est dans ce fichier que nous construisons notre réseau de neurones à l'aide des bibliothèques Keras et TensorFlow.<br>
Nous commencons par charger nos fichiers contenus dans `pickled_data`et par définir les paramètres de notre réseau : learning rate, nombre d'époques, la taille des batchs, le taux de drop-out ... <br>
Ensuite, deux choix s'offrent à l'utilisateur : soit il peut charger un modèle pré-existant contenu dans le dossier `saved_models`, soit il peut construire un nouveau modèle. S'il choisit de construire un nouveau modèle, nous avons implémenté toutes les couches de notre réseau de neurones qui prennent en paramètres ceux renseignés par l'utilisateur.<br>
Puis, nous avons implémenté la phase d'entrainement de l'algorithme avec la possibilité de choisir la taille du dataset désirée. Une fois le modèle entrainé, les performances sont enregistrées grâce à l'outil TensorBoard et le nouveau modèle est sauvegardé.<br>
Pour finir, l'utilisateur peut tester son modèle sur un dataset de validation et voir les résultats obtenus (vrais réponses et réponses obtenues avec les indices renvoyés par l'algorithme).

