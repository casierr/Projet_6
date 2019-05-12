import flask
from flask import request, jsonify, flash,render_template
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.tokenize import ToktokTokenizer
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import pickle

data = pd.read_csv("QueryResults.csv",sep=',',low_memory=False)

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class StackOverFlowForm(Form):
    titre_StackOverFlow = StringField('titre_StackOverFlow', validators=[validators.DataRequired()])
    question_stackOverFlow = StringField('question_StackOverFlow', validators=[validators.DataRequired()])
    


	
@app.route('/')
def index():
    form = StackOverFlowForm(request.form)
    return render_template('index.html',form=form)	

@app.route('/', methods=['POST'])
def prevision():
    resultat = StackOverFlowForm(request.form)
   
    titre_StackOverFlow=resultat.titre_StackOverFlow
    question_stackOverFlow=resultat.question_stackOverFlow
    print("Titre",titre_StackOverFlow)
    print("Body",question_stackOverFlow)
    input_texte = str(request.form['titre_StackOverFlow']) + str(request.form['question_StackOverFlow'])
    
    vecteur_tags = predict_Tags(input_texte,'SUPERVISEE')
    tags = get_Tag(vecteur_tags)
    mots_clefs = predict_Tags(input_texte,'NON_SUPERVISEE')
    print("Mots clefs",mots_clefs)
    print("rang 1",mots_clefs)
       
    return render_template('prevision_tag.html', form=request.form,Tags=tags,mots_clefs=mots_clefs)	
	
stopWordList=stopwords.words('english')
stopWordList.remove('no')
stopWordList.remove('not')

def removeTags(data):
    soup=BeautifulSoup(data,'html.parser')
    text=soup.get_text()
    return text

def removeCharDigit(text):
    str='`1234567890-=~@#$%^&*()_+[!{;":\'><.,/?"}]'
    for w in text:
        if w in str:
            text=text.replace(w,'')
    return text


lemma=WordNetLemmatizer()
token=ToktokTokenizer()

def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w,'v')
        listLemma.append(x)
    return text

def stopWordsRemove(text):
    
    wordList=[x.lower().strip() for x in token.tokenize(text)]
    
    removedList=[x for x in wordList if not x in stopWordList]
    text=' '.join(removedList)
    
    return text

def Tfidfsupervise(texte):
    print("""Debut fonction TFIDF""")
    filename = 'TFIDf_SVM.sav'
    vectorizer = pickle.load(open(filename, 'rb'))
     
    data_vectorized = vectorizer.transform(texte)
    print("""Fin fonction TFIDF""")
    return data_vectorized


def Idfnonsuprvisee(texte):
    print("""Debut fonction IDF""")
    filename = 'IDf_LDA.sav'
    vectorizer = pickle.load(open(filename, 'rb'))
    data_vectorized = vectorizer.transform(texte)
    print("end of the funtion")
    
    return data_vectorized


#topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)

# Topic - Keywords Dataframe
#df_topic_keywords = pd.DataFrame(topic_keywords)

def get_svm_prediction(tfidf):
    print("""Debut fonction svm prediction""")
    filename = 'SVM.sav'
    linear_svm = pickle.load(open(filename, 'rb'))
    print("""Fin fonction svm prediction""")
    return linear_svm.predict(tfidf)

def get_lda_prediction(idf):
    print("""Debut fonction lda prediction""")
    
    print("Chargement du modèle")
    filename = 'LDA.sav'
    Lda = pickle.load(open(filename, 'rb'))
    print("Chargement de la Matrice")
    filename = 'IDf_LDA.sav'
    vectorizer = pickle.load(open(filename, 'rb'))
    print("Chargement du dictionnaire")
    filename = 'Topics_LDA.sav'
    df_topic_keywords = pickle.load(open(filename, 'rb'))
    topic_probability_scores = (Lda.best_estimator_).transform(idf)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :13].values.tolist()
    return topic


def get_Tag(clf):
    filename = 'multilabel.sav'
    multilabel_binarizer = pickle.load(open(filename, 'rb'))
    Tags = multilabel_binarizer.inverse_transform(clf)
    return Tags
        
def print_LDA(topic):
    
    print(topic)


def PreProcessingText(texte):
    print("Step 0")
    print(texte)
    print("Step 1")
    texte_1=removeTags(texte)
    print("Step 2")
    texte_2=removeCharDigit(texte_1)
    print("Step 3")
    #Lemmization
    texte_3=lemitizeWords(texte_2)
    print("Step 4")
    texte_4=stopWordsRemove(texte_3)
    print("Finish")
    return(texte_4)      



def predict_Tags(texte,modele):
    
    #Pre-processing Text
    texte = PreProcessingText(texte)
   # print("modele :",modele,"texte:",texte)
    texte=[texte]
    #Feature Engineering + Modèle
    if (modele =='SUPERVISEE'):
        donnees_matrice = Tfidfsupervise(texte)
        return get_svm_prediction(donnees_matrice)   

        
    if (modele =='NON_SUPERVISEE'):
        donnees_matrice = Idfnonsuprvisee(texte)
        return get_lda_prediction(donnees_matrice)
             
    return "Erreur"
app.run()