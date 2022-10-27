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
    
file1 = open("book.txt" , "r" , encoding='utf-8')
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