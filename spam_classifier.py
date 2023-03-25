import os
import io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# chemin des dossiers contenant les fichiers
PATH_TO_HAM_DIR = "D:/naive-bayes-spam-classifier-machine-learning/emails/ham"
PATH_TO_SPAM_DIR = "D:/naive-bayes-spam-classifier-machine-learning/emails/spam"

SPAM_TYPE = "SPAM"
HAM_TYPE = "HAM"

#les tableaux X et Y seront de la meme taille et ordonnes
X = [] # represente l'input Data (ici les mails)
#indique s'il s'agit d'un mail ou non
Y = [] #les etiquettes (labels) pour le training set


def readFilesFromDirectory(path, classification):
    os.chdir(path)
    files_name = os.listdir(path)
    for current_file in files_name:
        message = extract_mail_body(current_file)
        X.append(message)
        Y.append(classification)
        
           
#fonction de lecture du contenu d'un fichier texte donne.
#ici, on fait un peu de traitement pour ne prendre en compte que le "corps du mail".
# On ignorer les en-tetes des mails

def extract_mail_body(file_name_str):
    inBody = False
    lines = []
    file_descriptor = io.open(file_name_str,'r', encoding='latin1')
    for line in file_descriptor:
        if inBody:
            lines.append(line)
        elif line == '\n':
            inBody = True
        message = '\n'.join(lines)
    file_descriptor.close()
    return message

#appel de la fonction de chargement des mails (charger les mail normaux ensuite les SPAM)
readFilesFromDirectory(PATH_TO_HAM_DIR, HAM_TYPE)
readFilesFromDirectory(PATH_TO_SPAM_DIR, SPAM_TYPE)

# Diviser les données en deux parties: une pour l'apprentissage et une pour le test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9)

# Convertir les données textuelles en vecteurs numériques
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

# Appliquer la méthode de Naive Bayes sur le jeu de formation
classifier = MultinomialNB()
classifier.fit(X_train_counts, Y_train)

# Convertir les exemples de test en vecteurs numériques et les utiliser pour prédire les étiquettes
X_test_counts = vectorizer.transform(X_test)
predictions = classifier.predict(X_test_counts)

# Afficher les prédictions pour chaque exemple de test
# for i in range(len(X_test)):
#     print("Example: ", X_test[i])
#     print("Prediction: ", predictions[i])
#     print("True Label: ", Y_test[i])


# Prédiction pour les exemples de test
test_counts = vectorizer.transform(X_test)
predictions = classifier.predict(test_counts)

# Création du tableau de prédictions
pred_df = pd.DataFrame({
    'Email': X_test,
    'Prediction': predictions,
    'True Label': Y_test
})

# Affichage du tableau de prédictions
print(pred_df)



# Prédiction pour les exemples de test
test_counts = vectorizer.transform(X_test)
predictions = classifier.predict(test_counts)

# Calcul du score de prédiction
score = classifier.score(test_counts, Y_test)

# Affichage du score
print("Score de prédiction : {:.2f}%".format(score * 100))




# Prédiction pour les exemples de test
test_counts = vectorizer.transform(X_test)
predictions = classifier.predict(test_counts)

# Création d'un dataframe pour les prédictions incorrectes
df_incorrect = pd.DataFrame(columns=["Email", "Prediction", "True Label"])

# Boucle pour remplir le dataframe avec les prédictions incorrectes
for i in range(len(X_test)):
    if predictions[i] != Y_test[i]:
        df_incorrect = df_incorrect.append({
            "Email": X_test[i],
            "Prediction": predictions[i],
            "True Label":  Y_test[i]
        }, ignore_index=True)

# Affichage du dataframe
print(df_incorrect)


# Prédiction pour les exemples de test
test_counts = vectorizer.transform(X_test)
predictions = classifier.predict(test_counts)

# Création du tableau de prédictions
pred_df = pd.DataFrame({
    'Email': X_test,
    'Prediction': predictions,
    'True Label': Y_test
})

# Affichage du tableau de prédictions
print(pred_df)

# Création de la matrice de confusion
cm = confusion_matrix(Y_test, predictions, labels=[SPAM_TYPE, HAM_TYPE])

# Création de la heatmap de la matrice de confusion
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[SPAM_TYPE, HAM_TYPE], yticklabels=[SPAM_TYPE, HAM_TYPE])
plt.xlabel('Prédiction')
plt.ylabel('Vraie étiquette')
plt.title('Matrice de confusion')
plt.show()



