from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
#import pattern3.web as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
import errno
import requests
#import itunes
import urllib
import bs4 as bs
import requests
import json
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import sys
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.svm import SVC
import time
import pickle
import warnings
#from pyvirtualdisplay import Display
#from xvfbwrapper import Xvfb
warnings.filterwarnings('ignore')



FLASK_PREFIX = 'flaskexample/static/'
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

filename = FLASK_PREFIX+'model.pkl'
ifdif_filename = FLASK_PREFIX+'tfidf.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
loaded_vectorizer = pickle.load(open(ifdif_filename, 'rb'))

DEFAULT = {'rating_level':'unknown', 'link':'-', 'score0':0, 'score1':0, 'score2':0, 'score3':0, 'error':'ERROR'}

Information = 'knowledge, data, news, database, intelligence, evidence, message, report, entropy, info, details, material, content, \
communication, update, know, advice, media, fact, secret, sequence, misinformation, instance, example, readout, \
information theory, selective information, datum, acquaintance, learn, communications, informative, documents, access, \
newspaper, documentation, understanding, reports, inform, services, document, reporting, materials, available, sign, \
messages, informatics, service, reference, notify, notification, uninformed, knowledgeable, advise, disclosure, \
publicize, feedback, awareness, question, observation, illustration, grounds, familiarity, aggregation, accumulation, \
collection, stuff, substance, background, condition, circumstance, informational, source, education, relevant, check, \
accounts, regarding, analysis, papers, related, provides, finding, useful, contents, detailed, address, relating, \
explaining, concerning, familiarize, unaware, revelation, disclosures, insights, dissemination, tips, publication, \
narrative, herald, resources, insight, article, explain, description, resource, overview, records, detail'


Emotion = 'emotion, fear, anger, sadness, disgust, love, sympathy, emotional, anxiety, feeling, joy, mood, awe, empathy, affection, \
frustration, passion, indignation, humility, jealousy, psychology, sociology, consciousness, happiness, surprise, \
evolution, contempt, ecstasy, hate, sentiment, hatred, excitement, curiosity, joyousness, emotional state, sense, \
sorrow, hunger, pathos, feel, compassion, fearfulness, feelings, shock, grief, anguish, laughter, motivation, \
cognition, tears, arousal, spontaneity, despair, enthusiasm, hysteria, bitterness, philosophy, remorse, subjectivity, \
bewilderment, bravado, nuance, pity, guilt, neuroscience, medicine, astonishment, history, physiological, \
shame, angst, nervousness, mammal, sentimentality, joyfulness, spirit, emotionful, emotions, boredom, emotionally, \
depression, sentimental, affective, doubt, dislike, impression, unconcern, delight, humor, felt, touching, smile, \
passions, smiles, passionate, laugh, feels, pain, confusion, loneliness, warmth, poignancy, jubilation, wistfulness, \
bashfulness, sarcasm, exultation, peevishness, hopefulness, positiveness, nostalgia, pathetic, romanticism, \
displeasure, amazement, affectionate, gratitude, emotionless, unhappy, dissapoint, frustrate, frustration, \
disillusion, delude, depression'

Advertisement = 'advertisement, advert, newspaper, promotion, ad, advertising, product, publicity, television, poster, service, \
brochure, magazines, promo, magazine, mailer, handbill, billboard, editorial, propaganda, pamphlet, postcard, \
website, direct mail, advertizing, commercial, advertise, media, sponsor, trailer, flier, flyer, teaser, advertorial, \
twitter, facebook, ads, billboards, mail, commercials, slogan, tabloid, airing, page, tagline, branding, tv, \
edward bernays, ticket, cartoon, publication, logo, photo, banner, campaign, booklet, mailings, parody, edition, \
photograph, placard, bulletin, article, signage, company, marketing, sales, advertizement, consumption, circular, \
bill, non-commercial, profit, promotional, advertised, blog, youtube, mailing, printed, coupon, audience, online, \
brochures, branded, posters, pamphlets, campaigns, blogs'


Audio = 'audio, sound, recording, video, soundtrack, acoustic, playback, television, audio frequency, disc, disk, \
audible, audiotape, audio recording, noise, cassette, sonic, stereophonic, tape, graphics, digital, \
stereo, music, beep, multimedia, radio, analog, uncompressed, streaming, prerecorded, ipod, \
broadcast, dolby, tv, frequency, record, constituent, gurgle, soundless, soundness, zing, decibel, \
twang, whizz, vroom, chirrup, resonate, whir, soundscape, clank, tinkle, loudness, frequence, \
formats, mp3, sound recording, loud, echo, jingle, vibration, devices, download, format, cassettes, \
instrumentation, voice, wav, loudspeaker, microphones, earphones, hi-fi, audiocassette, hifi, \
recordist, preamplifier, sound wave, wave propagation, loud sound, sound effects, speech, language, \
monologue, words, vocabulary, speaking, talk, rhetoric, speaker, spoken language, lexicon, preaching, \
verbalize, idiolect, pronunciation, tone, vowel'


def remove_punctuation(text):
    tmp_list = []
    for ch in text:
        if ch not in string.punctuation:
            tmp_list.append(ch)
    return "".join(tmp_list)

def tokenize(text):
    return re.split("\W+", text.lower()) # capital W means non-word character [^a-zA-Z0-9_]

def remove_stopWord(textList):
    res = []
    stopwords = nltk.corpus.stopwords.words('english')
    for word in textList:
        if word not in stopwords:
            res.append(word)
    return res

stopwords = nltk.corpus.stopwords.words('english')

def clean(text):
    res = remove_punctuation(text)
    res = tokenize(res)
    res = remove_stopWord(res)
    return res

def stemming(textList):
    ps = nltk.PorterStemmer()
    res = []
    for word in textList:
        word_stemmed = ps.stem(word)
        res.append(word_stemmed)
    return res

def clean_stem(text):
    if type(text) == float:
        return ''
    resList = clean(text)
    resList = stemming(resList)
    return ' '.join(resList)

def rating_categorize(ratingsSum):
    if ratingsSum < 3:
        return 'low'
    elif ratingsSum < 20:
        return 'fair'
    else:
        return 'good'

def rating_level(avg_rating):
    if avg_rating < 1:
        return 'low'
    elif avg_rating < 4.5:
        return 'fair'
    else:
        return 'good'

def star(avg_rating):
    if np.isnan(avg_rating):
        return '0'
    else:
        return str(round(avg_rating))

def categorize(text):
    doc0 = nlp(text)
    sm = -sys.maxsize
    categoryIdx = 0
    if doc0.similarity(doc1) > sm:
        categoryIdx = 0
        sm = doc0.similarity(doc1)
    if doc0.similarity(doc2) > sm:
        categoryIdx = 1
        sm = doc0.similarity(doc2)
    if doc0.similarity(doc3) > sm:
        categoryIdx = 2
        sm = doc0.similarity(doc3)
    if doc0.similarity(doc4) > sm:
        categoryIdx = 3
        sm = doc0.similarity(doc4)
    return categoryIdx

def add_score(text, categoryIdx, review_vect):
    score = analyzer.polarity_scores(text)['compound']
    review_vect[categoryIdx] += score

def vectorize(review):
    review_vect = [0.0, 0.0, 0.0, 0.0]
    categoryIdx = categorize(review)
    add_score(review, categoryIdx, review_vect)
    return np.array(review_vect)

def split_vect(review_string):
    vect = np.array([0.0, 0.0, 0.0, 0.0])
    if review_string is None:
        return vect
    if type(review_string) == float:
        return vect
    if review_string == '':
        return vect
    review_list = review_string.split('.')
    for token in review_list:
        token_clean_stem = clean_stem(token)
        vect += vectorize(token)
    #print(vect)
    return vect

def reviews_combine(review_list):
    if review_list == ['']:
        return ' '
    else:
        return ' '.join(review_list)

doc1 = nlp(clean_stem(Information))
doc2 = nlp(clean_stem(Emotion))
doc3 = nlp(clean_stem(Advertisement))
doc4 = nlp(clean_stem(Audio))



def classify(LINK):

    try:
        #display = Display(visible=0, size=(800, 600))
        #display.start()
        #vdisplay = Xvfb()
        #vdisplay.start()
        #chrome_options = Options()
        #chromes_options.set_headless()


        data_description_input = pd.DataFrame(columns = ('name', 'description'))
        url_list = [LINK]
        for i in range(len(url_list)):
            link = url_list[0]
            sauce = requests.get(link)
            soup = bs.BeautifulSoup(sauce.content,'html.parser')
            if soup.find("script", {"class":"ember-view"}):
                soup_json = json.loads(soup.find("script", {"class":"ember-view"}).text)
            else:
                name = ''
                description = ''
                data_description_input.loc[0] = [name, description]
                continue


            # name
            if soup_json['name'] is not None:
                name = soup_json['name']
            else:
                name = ''

            # description
            sub_soup = soup.find("div",{"class":"product-hero-desc"})
            description_tag = sub_soup.find('p')
            if description_tag is not None:
                description = description_tag.text
            else:
                description = ''

            # episode description
            for elem in soup_json['workExample']:
                if elem is not None:
                    if 'description' in elem:
                        description += ' ' + elem['description']

            data_description_input.loc[0] = [name, description]

        data_review_input = pd.DataFrame(columns = ('reviews', 'ratings'))
        #browser = webdriver.Chrome()
        #browser = webdriver.Chrome(FLASK_PREFIX+"chromedriver")

        link = url_list[0]
        sauce = requests.get(link)
        soup = bs.BeautifulSoup(sauce.content,'html.parser')

        # find review
        if soup.find('div',{'class':'we-customer-review'}) is None:
            reviews = ['']
            ratings = [0]
            data_review_input.loc[0] = [[''], [0]]
        else:
            review_link = link+'#see-all/reviews'
            sauce = requests.get(review_link)
            soup = bs.BeautifulSoup(sauce.content,'html.parser')
            reviews = []
            ratings = []
            headList = soup.find_all('h3')
            bodyList = soup.find_all('blockquote')
            for i in range(len(headList)):
                reviews.append(headList[i].text)
                reviews.append(bodyList[i].find('p').text)
                ratings.append(int(soup.find_all('figure')[i]['aria-label'][0]))

            data_review_input.loc[0] = [reviews, ratings]

        data_review_input['reviews_comb'] = data_review_input['reviews'].apply(lambda x:reviews_combine(x))
        data_review_input['ratings_sum'] = data_review_input['ratings'].apply(lambda x:sum(x))
        data_review_input['ratings_count'] = data_review_input['ratings'].apply(lambda x:len(x))
        data_review_input['avg_rating'] = 1.0*data_review_input['ratings_sum']/data_review_input['ratings_count']
        data_review_input['rating_level'] = data_review_input['avg_rating'].apply(lambda x : rating_level(x))

        data_description_input['description_clean_stem'] = data_description_input['description'].apply(lambda x : clean_stem(x))
        X_input = loaded_vectorizer.transform(data_description_input['description_clean_stem'])

        tfidf_vector_input = X_input.toarray()
        print(X_input.shape)


        data_review_input['vector'] = data_review_input['reviews_comb'].apply(lambda x : split_vect(x))
        review_vector_input = np.array(data_review_input['vector'].values.tolist())

        concat_vector_input = np.concatenate((tfidf_vector_input, review_vector_input), axis = 1)
        label_vector_input = np.array(data_review_input["rating_level"].values.tolist())
        print(loaded_model)
        print(concat_vector_input)
        y_pred = loaded_model.predict(concat_vector_input)
        print(1);

        print('The predicted rating level is: {}'.format(y_pred[0]))
        output_vector = data_review_input['vector'][0]
        print('Information : {:03.1f}'.format(output_vector[0]))
        print('Emotion : {:03.1f}'.format(output_vector[1]))
        print('Advertisement : {:03.1f}'.format(output_vector[2]))
        print('Audio : {:03.1f}'.format(output_vector[3]))

        if output_vector[0] > 2:
            print('The information is helpful.')
        else:
            print('Need more helpful information.')

        if output_vector[1] > 2:
            print('The emotion it conveys is quite positive.')
        else:
            print('Need more positive emotion.')

        if output_vector[2] > 2:
            print('The number of ads are reasonable.')
        else:
            print('Too many ads.')

        if output_vector[3] > 2:
            print('The audio quality if quite high.')
        else:
            print('Need to adjust the audio quality.')

        return {'rating_level':y_pred[0], 'link':LINK, 'name':data_description_input['name'][0], 'score0':round(output_vector[0],1), 'score1':round(output_vector[1],1), 'score2':round(output_vector[2],1), 'score3':round(output_vector[3],1)}

        #browser.quit()
        #vdisplay.stop()
    except:
        return {'error':'ERROR', 'link':LINK}

if __name__ == '__main__':
    results = classify('https://podcasts.apple.com/us/podcast/abundanceo/id1477784794')
    print(results['rating_level'], results['score0'], results['score1'], results['score2'], results['score3'])








