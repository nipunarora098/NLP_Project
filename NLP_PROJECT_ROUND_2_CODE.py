# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:54:33 2022

@author: hp
"""

# libraries
# for preprocessing
from nltk.stem import WordNetLemmatizer
# for word tokenization
import nltk
import matplotlib.pyplot as plt
# for frequency distribution
import pandas as pd
from wordcloud import WordCloud
from nltk.corpus import stopwords
#for stemming
from nltk.stem import PorterStemmer
#library for regular language
import re
# import wordnet library
from nltk.corpus import wordnet as wn
# import spacy library
import spacy
from sklearn.metrics import precision_recall_fscore_support
from nltk import ne_chunk
from nltk.sem.relextract import extract_rels
def lower_case(text: str):
    return text.lower()

def line_size(line):
    count = 0 
    for char in line:
        if ( ord(char) != 32):
            count+= 1
    return count  
      
def remove_heading():
    with open("last.txt" , "r" ,encoding='utf-8' ) as input , open("last2.txt" , "w",encoding='utf-8') as output :
        for line in input:
            if(line_size(line) >= 50):
                output.write(line)
                
def discard(text):
    # removing starting and ending portion that is unwanted
    start = text.find('Contents')
    end = text.rfind('aima.cs.berkeley.edu')
    text = text[start:end]
    # removing links from text
    re.sub(r'http\S+', '', text)
    return text

# removing punctuations 
def remove_punctuations(text):
    punc = '''!()-[]{};:'"\,<>./?’@”#$%^“&*_~+-=:|'''
    for ele in text:
        if ele in punc:
            text = text.replace(ele, "")
    return text

# remove_HTML_tags
def remove_tags(text):
    re.sub(r'http\S+', '', text)
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

#  Lemmatisation
def Lemmatization(text):
    arr = text.split()
    array = []
    lemmatizer = WordNetLemmatizer()
    for i in arr:
        array.append(lemmatizer.lemmatize(i))
    return ' '.join(array)

# stemming
def stemming(text):
    tokens=tokenization(text)
    arr = []
    ps = PorterStemmer()
    for i in tokens:
        arr.append(ps.stem(i))
    return ' '.join(arr)

def preprocessing(text):
    text=lower_case(text)
    text=remove_tags(text)
    text=Lemmatization(text)
    text=remove_punctuations(text)
    #text=stemming(text)
    return text

# Tokenize the text T1
def tokenization(text):
    return nltk.word_tokenize(text)

# Analyze the frequency distribution of tokens in T1
def frequency_distribution(tokenized_text):
    pd.Series(tokenized_text).value_counts()[:20].plot(kind='bar')
    plt.show()

def word_cloud_plot(token):
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color="black").generate(token)
    plt.figure(figsize=(12,10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def remove_stop_words(tokens):
    arr = []
    stop_words = set(stopwords.words('english'))
    for i in tokens:
        if i not in stop_words:
            arr.append(i)
    return ' '.join(arr)

def word_length_frequency(token):
    length = [len(i) for i in token]
    pd.Series(length).value_counts()[:20].plot(kind='bar')
    plt.show()

def pos_tagging(tokens):
    tag = nltk.pos_tag(tokens)
    return tag

def frequency_distribution_of_tags(tags):
    tags = [i[1] for i in tags]
    pd.Series(tags).value_counts()[:20].plot(kind='bar')
    plt.show()
def find_noun_categories(noun):
    categories = []
    # Get the WordNet synsets for the noun
    synsets = wn.synsets(noun, pos='n')

    # Print the categories for each synset
    for synset in synsets:
        categories.append(synset.lexname())
    unique_categories = set(categories)
    return list(unique_categories)

def find_verb_categories(verb):
    categories = []
    # Get the WordNet synsets for the noun
    synsets = wn.synsets(verb, pos='v')

    # Print the categories for each synset
    for synset in synsets:
        categories.append(synset.lexname())
    unique_categories = set(categories)
    return list(unique_categories)

def plot_histogram(count):
    unique_elements = list(count.keys())
    element_counts = list(count.values())
    plt.bar(unique_elements, element_counts)
    plt.xticks(rotation = 90)
    plt.show()
    
def ner(text):
     #Use the spacy model to recognize entities in the text
     doc = nlp(text)
    # Extract the named entities and their types from the spacy doc
     entities = [(ent.text, ent.label_) for ent in doc.ents]
    # Return the list of named entities
     return entities  
 
def cal_accuracy_fscore(sampleText , manual_labels):
    predictedEntities = ner(sampleText)
    print('\n'*3)
    print("The predicted entities of the random passages selected from the book are as follows:-")
    print('\n')
    print(predictedEntities)
    print('\n'*2)
    print("The manually labelled entities of the same random passages selected from the book are as follows:-")
    print('\n')
    print(manual_labels)
    # y_true is the list of manually labelled entity types
    # y_pred is the list of predicted entity types by our ner() function 
    y_pred = []
    y_true = []
    for pair in predictedEntities:
        y_pred.append(pair[1])
    for pair in manual_labels:
        y_true.append(pair[1])
    #calculating f1 score via precision and recall values
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    # Calculate accuracy as the percentage of predicted entity types that match the gold standard
    accuracy = sum([1 for i, j in zip(y_true, y_pred) if i == j]) / len(y_true)
    print('\n'*2)
    #printing accuracy
    print(f'accuracy = {accuracy}')
    print('\n')
    #printing precision accuracy and f1 score
    print(f'precision, recall, f1 score= {precision, recall, f1, _}')
long2short = dict(LOCATION="LOC", ORGANIZATION="ORG", PERSON="PER")
def class_abbrev(type):
    try:
        return long2short[type]
    except KeyError:
        return type

def rtuple(reldict, lcon=False, rcon=False):
    items = [
        class_abbrev(reldict["subjclass"]),
        reldict["subjtext"],
        reldict["filler"],
        class_abbrev(reldict["objclass"]),
        reldict["objtext"],
    ]
    format = "[%s: %r] %r [%s: %r]"
    if lcon:
        items = [reldict["lcon"]] + items
        format = "...%r)" + format
    if rcon:
        items.append(reldict["rcon"])
        format = format + "(%r..."
    printargs = tuple(items)
    return format % printargs
    
path = "book.txt"    
file1 = open(path , "r" , encoding='utf-8')
text = file1.read()
text = discard(text)
file2 = open("last.txt" , "w" , encoding = "utf-8")
file2.write(text) 
remove_heading()
file3 = open("last2.txt" , "r" , encoding = 'utf-8')
text = file3.read()
preprocessText = preprocessing(text) 
tokenized_text = tokenization(preprocessText)


frequency_distribution(tokenized_text)
word_cloud_plot(preprocessText)
updated_text = remove_stop_words(tokenized_text)
word_cloud_plot(updated_text)
updated_tokens = tokenization(updated_text)
frequency_distribution(updated_tokens)
word_length_frequency(tokenized_text)
pos_tagged_text = pos_tagging(tokenized_text)
frequency_distribution_of_tags(pos_tagged_text)

noun = []
verb = []
for pair in pos_tagged_text:
    if pair[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
        noun.append(pair[0])
    elif pair[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        verb.append(pair[0])
noun_categories = {}

for word in noun:
    categories = find_noun_categories(word)
    for ele in categories :
        if ele in noun_categories:
            noun_categories[ele] += 1
        else:
            noun_categories[ele] = 1
print(noun_categories)    
verb_categories = {}

for word in verb:
    categories = find_verb_categories(word)
    for ele in categories :
        if ele in verb_categories:
            verb_categories[ele] += 1
        else:
            verb_categories[ele] = 1
print(verb_categories)
plot_histogram(noun_categories)
plot_histogram(verb_categories)

#Round 2 part 2
# Load the spacy English model
nlp = spacy.load('en_core_web_sm')
#extracting the entities and their types out of the preprocessesed text
preprocessText = preprocessText[:1000000]    
entities = ner(preprocessText)
print(entities)
#taking random paragraphs from the book and applying ner() function on it     
sampleText = "AI is one of the newest fields in science and engineering. Work started in earnest soon after World War II, and the name itself was coined in 1956. Along with molecular biology, AI is regularly cited as the “field I would most like to be in” by scientists in other disciplines. A student in physics might reasonably feel that all the good ideas have already been taken by Galileo, Newton, Einstein, and the rest. AI, on the other hand, still has openings for several full-time Einsteins and Edisons. AI currently encompasses a huge variety of subfields, ranging from the general (learning and perception) to the specific, such as playing chess, proving mathematical theorems, writing poetry, driving a car on a crowded street, and diagnosing diseases. AI is relevant to any intellectual task; it is truly a universal field. The Turing Test, proposed by Alan Turing (1950), was designed to provide a satisfactory operational definition of intelligence. A computer passes the test if a human interrogator, after posing some written questions, cannot tell whether the written responses come from a person or from a computer. Chapter 26 discusses the details of the test and whether a computer would really be intelligent if it passed. For now, we note that programming a computer to pass a rigorously applied test provides plenty to work on. The computer would need to possess the following capabilities. The rational-agent approach has two advantages over the other approaches. First, it is more general than the “laws of thought” approach because correct inference is just one of several possible mechanisms for achieving rationality. Pascal wrote that the arithmetical machine produces effects which appear nearer to thought than all the actions of animals. Gottfried Wilhelm Leibniz (1646–1716) built a mechanical device intended to carry out operations on concepts rather than numbers, but its scope was rather limited. Leibniz did surpass Pascal by building a calculator that could add, subtract, multiply, and take roots, whereas the Pascaline could only add and subtract. Some speculated that machines might not just do calculations but actually be able to think and act on their own. In his 1651 book Leviathan, Thomas Hobbes suggested the idea of an artificial animal, arguing For what is the heart but a spring; and the nerves, but so many strings; and the joints, but so many wheels."
manual_labels = [('AI', 'LANGUAGE'), ('World War II', 'EVENT'), ('1956', 'DATE'), ('AI', 'lANGUAGE'), ('Galileo', 'PERSON'), ('Newton', 'PERSON'), ('Einstein', 'PERSON'), ('AI', 'LANGUAGE'), ('Einsteins', 'PERSON'), ('AI', 'LANGUAGE'), ('AI', 'LANGUAGE'), ('Alan Turing', 'PERSON'), ('1950', 'DATE'), ('Chapter 26', 'ORDINAL'), ('two', 'CARDINAL'), ('First', 'ORDINAL'), ('Wilhelm Leibniz', 'PERSON'), ('1646–1716', 'DATE'), ('Pascal', 'PERSON'), ('Pascaline', 'PRODUCT'), ('1651', 'DATE'), ('Leviathan,', 'PERSON'), ('Thomas Hobbes', 'PERSON')]
cal_accuracy_fscore(sampleText , manual_labels)

  




#3rd part

# =============================================================================
# Person - Location Relationship extraction
# =============================================================================
sentences = nltk.sent_tokenize(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
BELONG = re.compile(r'.*\bin|from|belonged|lived\b.*')
print()
for i, sent in enumerate(tagged_sentences):
    sent =ne_chunk(sent)
    rels = extract_rels('PER' , 'GPE', sent , corpus = 'ace' , pattern = BELONG, window = 10)
    for rel in rels:
        print(rtuple(rel))




# =============================================================================
# Person-Person Relationship Extraction 
# =============================================================================

RELATIONS = re.compile(r'.*\bmother|father|sister|brother|aunt|uncle\b.*')
for i, sent in enumerate(tagged_sentences):
    sent =ne_chunk(sent)
    rels = extract_rels('PER' , 'PER', sent , corpus = 'ace' , pattern = RELATIONS, window = 10)
    for rel in rels:
        print(rtuple(rel))
    
    
    
    
# =============================================================================
# Person organization Relationship extraction 
# =============================================================================
    
    
ORG = re.compile(r'.*\bwork|of|in\b.*')
for i, sent in enumerate(tagged_sentences):
    sent =ne_chunk(sent)
    rels = extract_rels('PER' , 'ORG', sent , corpus = 'ace' , pattern = ORG, window = 10)
    for rel in rels:
        print(rtuple(rel))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



