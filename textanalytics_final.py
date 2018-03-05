# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:14:12 2018

@author: jaa977
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:36:01 2018

@author: jakeatlas
"""
#%% CREATE SINGULAR TEXT FILE FROM AGGLOMERATION OF DAILY ARTICLE FILES

#Create string of all text
import glob
path = 'C:/Users/jaa977/Documents/textfiles308/*.txt'
files = glob.glob(path)
text = ''

for document in files:
    with open(document, 'r', errors='ignore') as single_document:
        read_document = single_document.read().replace('\n', '')
    text = text + ' ' + read_document
    
#Write string of all text to a text file
file = open('C:/Users/jaa977/Documents/textfiles308/corpus.txt','w')
file.write(text)
file.close()

#%% DATA PREPROCESSING
import nltk

#Sentence Segmentation
from nltk.tokenize import sent_tokenize
with open('C:/Users/jaa977/Documents/textfiles308/corpus.txt','r') as corpus:
    read_corpus = corpus.read()
sentences = sent_tokenize(read_corpus)

#Tokenization
from nltk.tokenize import word_tokenize
tokens = word_tokenize(read_corpus)

#Normalization
for index in range(0,16756013):
    punctuation = '?!\(\):,\''
    for symbol in punctuation:
        tokens[index] = tokens[index].replace(symbol,'')
                     
#Remove stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add('')
tokens_removed_stopwords = []
for word in tokens:
    if word not in stop_words:
        tokens_removed_stopwords.append(word)

#Determine part of speech for tokens
tokens_pos = nltk.pos_tag(tokens_removed_stopwords)

#Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

tokens_lemmatized = []
for element in range(0,9097595):
    try:
        tokens_lemmatized.append(lemmatizer.lemmatize(tokens_pos[element][0],tokens_pos[element][1]))
    except:
        tokens_lemmatized.append(lemmatizer.lemmatize(tokens_pos[element][0]))
        
##Stemming
#tokens_preprocessed = []        
#stemmer = nltk.PorterStemmer()
#for word in tokens_lemmatized:
#    tokens_preprocessed.append(stemmer.stem(word))

tokens_preprocessed = tokens_lemmatized.copy()
#%% READ IN AND SIMILARLY PREPROCESS TRAINING DATA FILES

import pandas as pd

ceo_train = pd.read_csv('C:/Users/jaa977/Documents/all/ceo.csv',header=None,encoding='cp1252')
percentage_train = pd.read_csv('C:/Users/jaa977/Documents/all/percentage.csv',header=None,encoding='cp1252')
company_train = pd.read_csv('C:/Users/jaa977/Documents/all/companies.csv',header=None)

ceo_train = list(ceo_train[0])
percentage_train = list(percentage_train[0])
company_train = list(company_train[0])

#Normalization 
training_data = [ceo_train, percentage_train, company_train]
#for training_file in training_data:
#    for index in range(0,len(training_file)):
#            training_file[index] = training_file[index].replace('.','')
                     
#Lemmatization  
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()          
ceo_lemmatized = []
for element in range(0,len(ceo_train)):
    ceo_lemmatized.append(lemmatizer.lemmatize(ceo_train[element]))

percentages_lemmatized = []
for element in range(0,len(percentage_train)-1):
    percentages_lemmatized.append(lemmatizer.lemmatize(percentage_train[element]))

companies_lemmatized = []    
for element in range(0,len(company_train)):
    companies_lemmatized.append(lemmatizer.lemmatize(company_train[element]))   
    
##Stemming
#ceo_preprocessed = []     
#percentages_preprocessed = []     
#companies_preprocessed = []     
#   
#for word in ceo_lemmatized:
#    ceo_preprocessed.append(stemmer.stem(word))
#
#for word in percentages_lemmatized:
#    percentages_preprocessed.append(stemmer.stem(word))
#    
#for word in companies_lemmatized:
#    companies_preprocessed.append(stemmer.stem(word))

ceo_preprocessed = ceo_lemmatized.copy()
percentages_preprocessed = percentages_lemmatized.copy()
companies_preprocessed = companies_lemmatized.copy()

#%% USE REGEX TO DETERMINE LISTS OF MATCHES BETWEEN TRAINING FILES AND CORPUS

import re
preprocessed_document = ' '.join(tokens_preprocessed)

ceo_matches = []
for expression in ceo_preprocessed:
    try:
        match = re.search(expression,preprocessed_document).group()
        ceo_matches.append(match)
    except: 
        pass

percentages_matches = []
for expression in percentages_preprocessed:
    try:
        match = re.search(expression,preprocessed_document).group()
        percentages_matches.append(match)
    except: 
        pass
    
companies_matches = []
for expression in companies_preprocessed:
    try:
        match = re.search(expression,preprocessed_document).group()
        companies_matches.append(match)
    except: 
        pass

#%% CREATE FEATURE TABLES FROM MATCHABLE TRAINING DATA

#Create CEO dataframe with training data
ceo_dataframe = pd.DataFrame(columns = ['ceo','ceo_nearby','is_ceo'])
for ceo in ceo_matches:
    match_span = re.search(ceo, preprocessed_document).span()
    if ('CEO' in preprocessed_document[match_span[0]-40:match_span[1]+20]):
           ceo_dataframe = pd.concat([ceo_dataframe, pd.DataFrame({'ceo':[ceo],'ceo_nearby':[1],'is_ceo':[1]})])
    else:
           ceo_dataframe = pd.concat([ceo_dataframe, pd.DataFrame({'ceo':[ceo],'ceo_nearby':[0],'is_ceo':[1]})])

#Create percentages dataframe with training data
percentages_dataframe = pd.DataFrame(columns= ['percent','percent_nearby','is_percent'])
for percent in percentages_matches:
    match_span = re.search(percent, preprocessed_document).span()
    if ('percent' in preprocessed_document[match_span[0]-10:match_span[1]+20]  or \
        '%' in preprocessed_document[match_span[0]-10:match_span[1]+20]):
            percentages_dataframe = pd.concat([percentages_dataframe, pd.DataFrame({'percent':[percent],'percent_nearby':[1],'is_percent':[1]})])
    else:
            percentages_dataframe = pd.concat([percentages_dataframe, pd.DataFrame({'percent':[percent],'percent_nearby':[0],'is_percent':[1]})])

#Create companies dataframe with training data
companies_dataframe = pd.DataFrame(columns = ['company','company_nearby','is_company'])
for company in companies_matches:
    match_span = re.search(company, preprocessed_document).span()
    if ('Company' in preprocessed_document[match_span[0]-40:match_span[1]+20]       or \
        'Inc' in preprocessed_document[match_span[0]-40:match_span[1]+20]           or \
        'Corp' in preprocessed_document[match_span[0]-40:match_span[1]+20]):
            companies_dataframe = pd.concat([companies_dataframe, pd.DataFrame({'company':[company],'company_nearby':[1],'is_company':[1]})])
    else:
        companies_dataframe = pd.concat([companies_dataframe, pd.DataFrame({'company':[company],'company_nearby':[0],'is_company':[1]})])

#%% NEGATIVE SAMPLING 

#Redetermine part of speech based on lemmatized tokens
tokens_pos = nltk.pos_tag(tokens_preprocessed)

#Identify PERSON and ORGANIZATION types with built-in NER
ner_tagged = nltk.chunk.ne_chunk(tokens_pos)
        
#Create CEO negative sample list
people_names = []
for item in range(0,8719517):
    if ((type(ner_tagged[item])==type(ner_tagged[3])) and (ner_tagged[item].label()=='PERSON')):
        tree_length = len(ner_tagged[item].leaves())
        name = ner_tagged[item].leaves()[0][0]        
        if tree_length>=2:
            for leaf in range(1,tree_length-1):
                name = name + ' ' + ner_tagged[item].leaves()[leaf][0]
        people_names.append(name)

#Create company negative sample list
company_names = []
for item in range(0,8719517):
    if ((type(ner_tagged[item])==type(ner_tagged[3])) and (ner_tagged[item].label()=='ORGANIZATION')):
        tree_length = len(ner_tagged[item].leaves())
        organization = ner_tagged[item].leaves()[0][0]        
        if tree_length>=2:
            for leaf in range(1,tree_length-1):
                organization = organization + ' ' + ner_tagged[item].leaves()[leaf][0]
        company_names.append(organization)           

unique_people_names = list(set(people_names))
unique_company_names = list(set(company_names))

#Refine negative samples using manual selection to remove possible positives
subset_people = list(set(unique_people_names[0:300])-set(['Jonah','Catalin VossCatalin','Yanai','Apthorps','Michael Tiffany Ash','Terry','Draghi Jackson','Fleming','Adobe CFO Murray','Marc Andreesen Scott','Bill Gates Microsofts']))
subset_companies  = list(set(unique_company_names[0:300]) - set(['PNC','BBC','AXP','HSBC Corporate','INET','WMT','United Airlines American','Newmark','BlackRocks','SinoPAc Solutions Services Ltd Hong','Bank Holding Company','AutoNavi Holdings','Microsoft Corp','Equifax','CNBC','Gallup','Spirit Airlines']))

#Create percentages negative sample list
subset_percentages = subset_people + subset_companies

#%% ADD NEGATIVE SAMPLES TO DATAFRAMES FOR CLASSIFICATION MODEL

#Completion of dataframe for CEOs by adding negative sample
for loop_count in range(0,4):
    for ceo in subset_people:
        match_span = re.search(ceo, preprocessed_document).span()
        if ('CEO' in preprocessed_document[match_span[0]-40:match_span[1]+20]):
               ceo_dataframe = pd.concat([ceo_dataframe, pd.DataFrame({'ceo':[ceo],'ceo_nearby':[1],'is_ceo':[0]})])
        else:
               ceo_dataframe = pd.concat([ceo_dataframe, pd.DataFrame({'ceo':[ceo],'ceo_nearby':[0],'is_ceo':[0]})])

#Completion of dataframe for companies by adding negative sample
for loop_count in range(0,6):           
    for company in subset_companies:
        match_span = re.search(company, preprocessed_document).span()
        if ('Company' in preprocessed_document[match_span[0]-30:match_span[1]+30]       or \
            'Inc' in preprocessed_document[match_span[0]-30:match_span[1]+30]           or \
            'Corp' in preprocessed_document[match_span[0]-30:match_span[1]+30]):
                companies_dataframe = pd.concat([companies_dataframe, pd.DataFrame({'company':[company],'company_nearby':[1],'is_company':[0]})])
        else:
            companies_dataframe = pd.concat([companies_dataframe, pd.DataFrame({'company':[company],'company_nearby':[0],'is_company':[0]})])           
   
#Completion of dataframe for percentages by adding negative sample      
for loop_count in range(0,2):
    for percent in subset_percentages:
        match_span = re.search(percent, preprocessed_document).span()
        if ('percent' in preprocessed_document[match_span[0]-10:match_span[1]+20]  or \
            '%' in preprocessed_document[match_span[0]-10:match_span[1]+20]):
                percentages_dataframe = pd.concat([percentages_dataframe, pd.DataFrame({'percent':[percent],'percent_nearby':[1],'is_percent':[0]})])
        else:
                percentages_dataframe = pd.concat([percentages_dataframe, pd.DataFrame({'percent':[percent],'percent_nearby':[0],'is_percent':[0]})])
           
#%% COMPLETE DATAFRAMES WITH REMAINING FEATURES FOR CLASSIFICATION MODEL

companies_dataframe['num_characters'] = companies_dataframe['company'].apply(len)
ceo_dataframe['num_characters'] = ceo_dataframe['ceo'].apply(len)

def num_capitals(phrase):
    cap_count = 0
    for letter in phrase:
        if letter.isupper():
            cap_count = cap_count + 1
    return(cap_count)
    
companies_dataframe['num_capitals'] = companies_dataframe['company'].apply(num_capitals)
ceo_dataframe['num_capitals'] = ceo_dataframe['ceo'].apply(num_capitals)

def num_words(phrase):
    num_spaces = 0
    for letter in phrase:
        if letter == ' ':
            num_spaces = num_spaces + 1
    return(num_spaces)

companies_dataframe['num_words'] = companies_dataframe['company'].apply(num_words)
ceo_dataframe['num_words'] = ceo_dataframe['ceo'].apply(num_words)

def name_contains_company_word(phrase):
    try:
        re.search(r'(Group|LTD|AirLL|Management|Capital|Advisors|Partner|LP|Associate|Co)',phrase).group()
        return(1)
    except: return(0)

companies_dataframe['name_contains_company_word'] = companies_dataframe['company'].apply(name_contains_company_word)     
    
#%% TRAINING CLASSIFICATION REGRESSIONS

import statsmodels.api as sm

#Initialize columns on which to regress
ceo_train_cols = ceo_dataframe[['ceo_nearby','num_characters','num_capitals','num_words']]
company_train_cols = companies_dataframe[['company_nearby','num_characters','num_capitals','num_words','name_contains_company_word']]

#CEO regression
logistic_model_ceo = sm.Logit(ceo_dataframe['is_ceo'].astype(float),ceo_train_cols.astype(float))
logistic_fit_ceo = logistic_model_ceo.fit()
    
#Company regression 
logistic_model_company = sm.Logit(companies_dataframe['is_company'].astype(float),company_train_cols.astype(float))
logistic_fit_company = logistic_model_company.fit()

#Percent regression
logistic_model_percent = sm.Logit(percentages_dataframe['is_percent'].astype(float),percentages_dataframe['percent_nearby'].astype(float))
logistic_fit_percent = logistic_model_percent.fit()
      
#%%  ---------------SUBSEQUENT CODE IS FOR THE TEST SET------------------------
#%% LOOP THROUGH PERCENTAGES USING REGULAR EXPRESSIONS & CONSTRUCT DATAFRAME

#Locate percent-style structures
percent_regex = '\s([0-9]+|[a-zA-Z]+-?[a-zA-Z]*|[0-9]+\.[0-9]+)\s?(%|percent)(age point)?'
all_percents = re.findall(percent_regex,preprocessed_document)

all_test_percents = []
for entry in all_percents:
    entry_as_string = ''
    for item in entry:
        entry_as_string = entry_as_string + item
        entry_as_string = entry_as_string.strip()
    all_test_percents = all_test_percents + [entry_as_string]

#Create dataframe
df_test_percents = pd.DataFrame(all_test_percents,columns=['percent'])
df_test_percents['percent_nearby'] = 1

#%% LOOPING THROUGH FOR CEOs AND COMPANIES TO MAKE FEATURE: CEO/COMPANY NEARBY

full_document = preprocessed_document

list_potential_ceo = []
list_potential_company = []
list_potential_CEO_company = re.findall(r'(?<!\.\s)[A-Z][a-z]+(?:\s?[A-Z]?\s?[A-Z][a-z]+)*',full_document)

for element in list_potential_CEO_company:
    potential_CEO_company_span = re.search(element,full_document).span()
    potential_CEO_company_start = potential_CEO_company_span[0]
    potential_CEO_company_end = potential_CEO_company_span[1]
    
    if potential_CEO_company_start-40<0:
        ceo_company_check_start = 0
    else:
        ceo_company_check_start = potential_CEO_company_start - 40

    if ('CEO' in full_document[ceo_company_check_start:potential_CEO_company_end+20]):
            if (('Company' in full_document[ceo_company_check_start:potential_CEO_company_end+20]) or \
                ('Inc' in full_document[ceo_company_check_start:potential_CEO_company_end+20])     or \
                ('Corp' in full_document[ceo_company_check_start:potential_CEO_company_end+20])):
                    list_potential_ceo = list_potential_ceo + [1]
                    list_potential_company = list_potential_company + [1]
            else:
                    list_potential_ceo = list_potential_ceo + [1]
                    list_potential_company = list_potential_company + [0]
    elif  (('Company' in full_document[ceo_company_check_start:potential_CEO_company_end+20]) or \
          ('Inc' in full_document[ceo_company_check_start:potential_CEO_company_end+20])     or \
          ('Corp' in full_document[ceo_company_check_start:potential_CEO_company_end+20])):
            list_potential_company = list_potential_company + [1]
            list_potential_ceo = list_potential_ceo + [0]
    else:
            list_potential_ceo = list_potential_ceo + [0]
            list_potential_company = list_potential_company + [0]
    full_document = full_document.replace(element, element.lower(),1)

#%% CEO & COMPANY DATAFRAME COMPLETION (ADDING REMAINING FEATURES)
    
df_test_ceo_company = pd.DataFrame(
        {'potential_ceo_company':list_potential_CEO_company,
         'ceo_nearby': list_potential_ceo,
         'company_nearby': list_potential_ceo})
df_test_ceo_company['num_characters'] = df_test_ceo_company['potential_ceo_company'].apply(len)
df_test_ceo_company['num_capitals'] = df_test_ceo_company['potential_ceo_company'].apply(num_capitals)
df_test_ceo_company['num_words'] = df_test_ceo_company['potential_ceo_company'].apply(num_words)
df_test_ceo_company['name_contains_company_word'] = df_test_ceo_company['potential_ceo_company'].apply(name_contains_company_word)

#%% APPLY CLASSIFICATION MODEL TO TEST DATA

test_columns_ceo = df_test_ceo_company[['ceo_nearby','num_characters','num_capitals','num_words']]
test_columns_company = df_test_ceo_company[['company_nearby','num_characters','num_capitals','num_words','name_contains_company_word']]

df_test_ceo_company['ceo_prediction'] = logistic_fit_ceo.predict(test_columns_ceo)
df_test_ceo_company['company_prediction'] = logistic_fit_company.predict(test_columns_company)
df_test_percents['percent_prediction'] = logistic_fit_percent.predict(df_test_percents['percent_nearby'])

#%% EXTRACT CEOs, COMPANIES, AND PERCENTS

#Eliminate rows that don't meet threshold likelihood 
full_df_extracted_ceo = df_test_ceo_company[df_test_ceo_company['ceo_prediction'] >= .7]
full_df_extracted_company = df_test_ceo_company[df_test_ceo_company['company_prediction'] >= .9]
full_df_extracted_percent = df_test_percents[df_test_percents['percent_prediction'] >= .5]

#Extract CEO names, company names, and percents
df_extracted_ceo = full_df_extracted_ceo['potential_ceo_company']
df_extracted_company = full_df_extracted_company['potential_ceo_company']
df_extracted_percent = full_df_extracted_percent['percent']

#%% CREATE CSV FILES WITH RESULTS

pd.DataFrame(df_extracted_ceo).to_csv('C:/Users/jaa977/Documents/Homework3_CEOs.csv')
pd.DataFrame(df_extracted_company).to_csv('C:/Users/jaa977/Documents/Homework3_companies.csv')
pd.DataFrame(df_extracted_percent).to_csv('C:/Users/jaa977/Documents/Homework3_percents.csv')