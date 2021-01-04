# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 23:27:48 2020

@author: Karthik
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:46:34 2020

@author: Karthik
"""

import pandas as pd
import re
import string
import pandas as pd
import re
from nltk.corpus import stopwords
import stanfordnlp
import nltk
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
####################################################################################################### DEFINING ALL REQUIRED FUNCTIONS


##########Lambda functions to clean text

remove_emojis_and_symbols_lambda = lambda x: remove_emojis_and_symbols(x)

remove_alphanumeric_words_lambda = lambda y: re.sub('\w*\d\w*', "", y)    

remove_double_spaces_lambda = lambda z: re.sub(' +', ' ', z)

remove_other_than_words_lambda = lambda v: re.sub('\W', ' ', v)

last_n_words_lambda = lambda x, y: ' '.join(x.split()[-y:])


def remove_emojis_and_symbols(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
                            "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)



##########Function to clean scraped trip advisor reviews: INPUT: data frame, OUTPUT: data frame
def normalize_rating_star_and_date(data_frame): 
#####the rating star data has to be normalized - for example, ui_bubble_rating bubble_50 is 5/5 and ui_bubble_rating bubble_10 is 1/5
#####date data has to be fixed - current month reviews are in DD MON format whereas others are in MON YY format
   

    replacement_dict = {
        'a week ago on Google' : ['Nov', '2020'], 
        '2 weeks ago on Google' : ['Nov', '2020'], 
    	'3 weeks ago on Google' : ['Nov', '2020'], 
    	'a month ago on Google'  : ['Nov', '2020'],
    	'2 months ago on Google'  : ['Oct', '2020'],
    	'3 months ago on Google' : ['Sep', '2020'],
    	'4 months ago on Google' : ['Aug', '2020'],
    	'5 months ago on Google' : ['Jul', '2020'],
    	'6 months ago on Google' : ['Jun', '2020'],
    	'7 months ago on Google' : ['May', '2020'],
    	'8 months ago on Google'  : ['Apr', '2020'],
    	'9 months ago on Google'  : ['Mar', '2020'],
    	'10 months ago on Google'  : ['Feb', '2020'],
    	'11 months ago on Google'  : ['Jan', '2020'],
    	'12 months ago on Google'  : ['Dec', '2019'],
    	'a year ago on Google' : ['' , '2019'],
    	'2 years ago on Google' : ['', '2018'],
    	'3 years ago on Google'  : ['', '2017'],
    	'4 years ago on Google' : ['', '2016'],
    	'5 years ago on Google'  : ['', '2015'],
    	'6 years ago on Google'  : ['', '2014'],
    	'7 years ago on Google' : ['', '2013'],
    	'8 years ago on Google' : ['', '2012'],
    	'9 years ago on Google' : ['', '2011'],
    	'10 years ago on Google' : ['', '2010'],
    	'11 years ago on Google' : ['', '2009'],
    	'12 years ago on Google' : ['', '2008'],
    	'13 years ago on Google' : ['', '2007'],
    	'14 years ago on Google' : ['', '2006'],
    	'15 years ago on Google' : ['', '2005']
    }

    month_lambda = lambda x: replacement_dict[x][0]
    year_lambda = lambda x: replacement_dict[x][1]

    data_frame['month'] = pd.DataFrame(data_frame["Review_date"].apply(month_lambda))    
    data_frame['year'] = pd.DataFrame(data_frame["Review_date"].apply(year_lambda))    

    return data_frame

##########Function to convert reviews (multiple statements) in a data frame column to a new data frame with single statements: INPUT: data frame, OUTPUT: data frame
def seperate_sentences_into_new_df(data_frame):
#####split multiple statement reviews into single statements using sentence tokenizer
    data_frame['sent_tokenized'] = data_frame.Review_Text.apply(lambda row: nltk.sent_tokenize(row))
    temp_list = []
   
    for i in range(0,len(data_frame['sent_tokenized'])):
        for sent in data_frame['sent_tokenized'].iloc[i]:
            temp_list.append([data_frame.Review_date.iloc[i], data_frame.month.iloc[i], data_frame.year.iloc[i], data_frame.Rating_Star.iloc[i], sent])
     
    review_stmts =  pd.DataFrame(temp_list, columns = ['Review_date', 'month', 'year', 'Rating_Star', 'statement'])
    return review_stmts


##########Function to perform sentiment analysis on a data frame column and add 2 new columns - sentiment and sentiment score
#####INPUT: data frame, column containing review statements 
#####OUTPUT: data frame with 2 new columns - sentiment and sentiment_score
def sentiment_analyzer(data_frame, statement):
    analyser = SentimentIntensityAnalyzer()
    
    sentiment_score_list = []
    sentiment_label_list = []
    
    for i in data_frame[f'{statement}'].values.tolist():
        sentiment_score = analyser.polarity_scores(i)
        if sentiment_score['compound'] >= 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Positive')
        elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Neutral')
        elif sentiment_score['compound'] <= -0.05:
            sentiment_score_list.append(sentiment_score['compound'])
            sentiment_label_list.append('Negative')

    data_frame['sentiment'] = sentiment_label_list
    data_frame['sentiment_score'] = sentiment_score_list
    
    return data_frame

##########Function to perform sentiment analysis on a data frame column and add 2 new columns - sentiment and sentiment score
#####INPUT: data frame, column containing review statements 
#####OUTPUT: data frame with 2 new columns - noun-subject/ noun-object and sentiment_score
def extract_subject_object_nouns_and_calculate_sentiments(data_frame, statement, column_with_sentiment_score):

    def nsubj_dobj(text):
        doc = nlp(text)
        ret_list = []
        for token in doc:
            if((token.dep_ == "nsubj" or token.dep_ == "dobj") and (token.tag_.startswith('NN'))):            
                ret_list.append(token.lemma_)
        return ret_list
       
    data_frame['subjects_and_objects'] = data_frame[f'{statement}'].apply(nsubj_dobj)

    noun_list = []
    k = -1
    for i in data_frame['subjects_and_objects']:
        k+=1
        for item in i:
            if item != "":
                noun_list.append([item, data_frame[f'{column_with_sentiment_score}'].iloc[k], data_frame['Rating_Star'].iloc[k], data_frame['Review_date'].iloc[k], data_frame['month'].iloc[k], data_frame['year'].iloc[k]])

    noun_df = pd.DataFrame(noun_list, columns=['noun', 'sentiment_score', 'Rating_Star', 'Review_date', 'month', 'year'])
    
    return noun_df

    
##########This function takes a sentiment score and computes if it is a positive/ negative/ neutral sentiment
#####INPUT: data frame, column containing review statements 
#####OUTPUT: data frame with 2 new columns - noun-subject/ noun-object and sentiment_score
def compute_sentiment(sentiment_score):
        if sentiment_score >= 0.05:
            return 'Positive'
        if sentiment_score <= -0.05:
            return 'Negative'
        if sentiment_score > -0.05 and sentiment_score < 0.05:
            return 'Neutral'
        else:
            return 'xxxx'


##########This function takes a sentence and populates a dictionary containing nouns as key and list of some dependent words like adjectives as values
#####INPUT: sentence, dictionary
#####OUTPUT: values will be added to the dictionary passed as inout
def nouns_with_wordlist(data_frame, statement_column):

    # print(data_frame[f'{statement_column}'])
        
    statement_list = data_frame[f'{statement_column}'].values.tolist()
    # print(statement_list)
    
    dictionary = {}
    
    for stmt in statement_list:
        # print("m here :\n\n", stmt)
        noun_list = []
        attribute_list = []
    
        doc = nlp(stmt)
        for token in doc:
            # print(token)
            if((token.dep_ == "nsubj" or token.dep_ == "dobj") and (token.tag_.startswith('NN')) and (token.lemma_ not in stop_words)):
                noun_list.append(token.lemma_) 
            elif ((token.dep_ in  ["acl", "amod", "acomp", "xcomp", "conj", "compound" ]) and (token.lemma_ not in stop_words)):
                attribute_list.append(token.lemma_)
            # print(noun_list, "----------\n-----------", attribute_list)
        
        
        for i in noun_list:
            # print(i)
            if i in dictionary:
                # print("it is in the dict::::::::\n", dictionary, "\n", attribute_list)
                for x in attribute_list:
                    if x not in dictionary[i]:
                        dictionary.setdefault(i,[]).append(x) 
            else:
                # print("it is NOTTTT in the dict::::::::::::\n", dictionary, attribute_list)
                dictionary[i] = attribute_list       
        # print("\n\n", dictionary, "\n\n")
    
    df =  pd.DataFrame(dict([ (x,pd.Series(y)) for x,y in dictionary.items() ])).melt().dropna()
    df.rename(columns = {'variable':'noun', 'value':'attribute'}, inplace = True) 
    
    polarity_lambda = lambda z: TextBlob(z).sentiment.polarity
    df['polarity'] = pd.DataFrame(df["attribute"].apply(polarity_lambda))
    
        # k = -1
        # for i in noun_list:
        #     for ix, val in enumerate(attribute_list):
        #         k+=1
        #         df['noun'].iloc[k] = i
        #         df['attribute'].ilok[ix] = val
        #         df['polarity'].ilok[ix] = TextBlob(val).sentiment.polarity

        
    return df

def get_ngrams(n, data_frame, column_with_statement):

    cv = CountVectorizer(ngram_range=(n,n))
    data_frame = data_frame.filter([f'{column_with_statement}', 'Rating_Star'])
    ngram_vector = cv.fit_transform(data_frame[f'{column_with_statement}'])
    # print(bigram_vector)
    
    ngram_vector_array = pd.DataFrame(ngram_vector.toarray(), columns = cv.get_feature_names())
    df1_ngram = data_frame.join(ngram_vector_array)
    # df1_ngram = df1_ngram.drop(['Customer', 'Review_date', 'Review_Text', 'Cleaned_Review_Text', 'Cleaned_Review_Text_2', 'Cleaned_Review_Text_3', 'tokenized_review', 'tokenized_and_stopwords_removed', 'stemmed_review', 'Lemmatized_review', 'Lemmatized_sentences'], axis=1)
    df1_ngram = df1_ngram.drop(columns=[f'{column_with_statement}'])
    df1_ngram = df1_ngram.set_index('Rating_Star').transpose()
    df1_ngram = df1_ngram.reset_index()
    df1_ngram = df1_ngram.rename(columns = {'index':'ngram'})
    df1_ngram = df1_ngram.groupby(level=0, axis=1).sum()
    
    nlppp = spacy.load("en_core_web_sm")
    
    df1_ngram['ngram_pos'] = ""
    df1_ngram['needed?'] = ""
    
    for i in range(0,len(df1_ngram.index)):
        doc = nlppp(df1_ngram['ngram'].iloc[i])
        val = ' '.join(tkn.pos_ for tkn in doc)
        df1_ngram['ngram_pos'].iloc[i] = val   
        if (('NOUN' in val) or ('ADJ'  in val) or ('PROPN' in val)  or ('VERB'  in val) or ('ADV' in val)):
            df1_ngram['needed?'] = "yes"
        else:
            df1_ngram['needed?'] = "no"
    
    final_df =  df1_ngram[df1_ngram['needed?'] == "yes"]
    final_df = final_df.drop(columns=['needed?'])
    
    return final_df


####################################################################################################### REQUIRED INITIALIZATIONS
####Below is a more complete stopword list
####As an additional step, we are also adding stop words with punctuations removed
stop_words = [ "travelodge", "travel", "lodge", "hotel", "room", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", 
              "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", 
              "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", 
              "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", 
              "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", 
              "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", 
              "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", 
              "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", 
              "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", 
              "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", 
              "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", 
              "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", 
              "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due",
              "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", 
              "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", 
              "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", 
              "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", 
              "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", 
              "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", 
              "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", 
              "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", 
              "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", 
              "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", 
              "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", 
              "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", 
              "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", 
              "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", 
              "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", 
              "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", 
              "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety",
              "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny",
              "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", 
              "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", 
              "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", 
              "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", 
              "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", 
              "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", 
              "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", 
              "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", 
              "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", 
              "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", 
              "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere",
              "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", 
              "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", 
              "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered",
              "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", 
              "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", 
              "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", 
              "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", 
              "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", 
              "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", 
              "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", 
              "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", 
              "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", 
              "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", 
              "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]

for x in stop_words:
    if "'" in x:
        stop_words.append(re.sub("[']", "", x))           

############# Spacy NLP
nlp = spacy.load("en_core_web_sm")

####################################################################################################### MAIN PROGRAM BLOCK

scraped_file_list = [
                    r"C:\Users\Karthik\Desktop\SMA Assignment\data files for further processing\Travelodge Cardiff M4.csv",
                    r"C:\Users\Karthik\Desktop\SMA Assignment\data files for further processing\Travelodge Cardiff Atlantic Wharf.csv",
                    r"C:\Users\Karthik\Desktop\SMA Assignment\data files for further processing\Travelodge Cardiff Llanedeyrn.csv",
                    r"C:\Users\Karthik\Desktop\SMA Assignment\data files for further processing\Travelodge Cardiff Central Queen Street.csv",
                    r"C:\Users\Karthik\Desktop\SMA Assignment\data files for further processing\Travelodge Cardiff Whitchurch.csv",
                    r"C:\Users\Karthik/Desktop\SMA Assignment\data files for further processing\Travelodge Cardiff Central.csv"
                	]

output_file_1_list = []
output_file_2_list = []
output_file_3_list = []
output_file_4_list = []
file_path_1 = r'C:\\Users\\Karthik\\Desktop\\SMA_Assignment_Files\\'

for scraped_file in scraped_file_list:
    # for i in range(0,1):
    #     scraped_file = scraped_file_list[i]
    specific_hotel_name = (re.search('\\\(Travelodge.*)\.' , scraped_file)).group(1)
    print(specific_hotel_name)
    
    #######################################     PART 1

    # #####       1. get data from csv
    df = pd.read_csv(f"{scraped_file}", sep='|', error_bad_lines=False)
    
    # #####       2. normalize the data and remove customer column since we do not need it
    df = normalize_rating_star_and_date(df)
    df = df.drop(['Customer'], axis=1)
    # print(df)
    # print(df.columns)
    
    
    remove_double_double_quotes = lambda l: re.sub('\"\"', "", l)
    remove_double_single_quotes = lambda m: re.sub("\'\'", "", m) 
    remove_spl_characters_lambda = lambda n: re.sub('[^A-Za-z0-9\.\, ]+', '', n)
    remove_period_infront_of_word_lambda = lambda o: re.sub('\.(?!\s)', '', o)
    count_of_words_lambda = lambda o: len(str(o).split(' '))
    
    df["Review_Text"] = pd.DataFrame(df["Review_Text"].apply(remove_emojis_and_symbols_lambda)) 
    df["Review_Text"] = pd.DataFrame(df["Review_Text"].apply(remove_alphanumeric_words_lambda)) 
    df["Review_Text"] = pd.DataFrame(df["Review_Text"].apply(remove_spl_characters_lambda))
    df["Review_Text"] = pd.DataFrame(df["Review_Text"].apply(remove_double_double_quotes))
    df["Review_Text"] = pd.DataFrame(df["Review_Text"].apply(remove_double_single_quotes))
    df["Review_Text"] = pd.DataFrame(df["Review_Text"].apply(remove_double_spaces_lambda))
    df["Review_Text"] = pd.DataFrame(df["Review_Text"].apply(remove_period_infront_of_word_lambda))
    df["Review_Text"] = df["Review_Text"].str.lower()
    
    df['count_of_words'] = pd.DataFrame(df["Review_Text"].apply(count_of_words_lambda))
    df['source'] = "google"
    df['hotel_name'] = f"{specific_hotel_name}".lower()
    dfx = df[['Rating_Star', 'year', 'Review_Text', 'count_of_words', 'source','hotel_name']]
    dfx.rename(columns = {'year':'Review_date'}, inplace=True)
    print(dfx.columns)
    print(dfx)


    file_path_1 = 'C:\\Users\\Karthik\\Desktop\\SMA_Assignment_Files\\'
    file_path_2 = f'google-{specific_hotel_name}'

    file_path = file_path_1 + file_path_2
    file_name = r'main_df_with_word_count.csv'
    output_file_1_list.append(f"{file_path}" + f"{file_name}")
    dfx.to_csv(f"{file_path}" + f"{file_name}", sep='|')
      
    #####       3. create a new dataframe with the reviews broken into one tokenized sentence per row - capture them in a column called statement
    df_1 = seperate_sentences_into_new_df(df)
    # print(df_1)
    ####        4. add sentiment and sentiment score for tokenized sentences in the statement column
    df_1 = sentiment_analyzer(df_1, 'statement') 
    # print(df_1.statement)
       
    # # #######################################     EXTRACTING N-GRAMS

    stop_words_lambda = lambda v: ' '.join(word for word in v.split() if word not in stop_words)
    lemmatizer_lambda = lambda x: " ".join([y.lemma_ for y in nlp(x)])
    
    df_1['stopwords_removed'] = pd.DataFrame(df_1['statement'].apply(stop_words_lambda))
    # print(df_1['stopwords_removed'])
    
    df_1["Lemmatized_sentences"] = pd.DataFrame(df_1['stopwords_removed'].apply(lemmatizer_lambda))
    
    # df_1['year'] = pd.DataFrame(df_1['Review_date'].apply(last_n_words_lambda, y=1))
        
    
    multigram_list = []
    
    for year in df_1.year.unique():
        filtered_df = df_1[df_1['year'] == f'{year}']
        filtered_df = df_1[df_1['Lemmatized_sentences'] != ""]

        unigram_df = get_ngrams(1, filtered_df, 'Lemmatized_sentences')
        unigram_df[r"year"] = f'{year}'
        unigram_df[r"ngram_number"] = '1'

        bigram_df = get_ngrams(2, filtered_df, 'Lemmatized_sentences')
        bigram_df[r"year"] = f'{year}'
        bigram_df[r"ngram_number"] = '2'

        trigram_df = get_ngrams(3, filtered_df, 'Lemmatized_sentences')
        trigram_df[r"year"] = f'{year}'
        trigram_df[r"ngram_number"] = '3'
    
        multigram_list.append(unigram_df)
        multigram_list.append(bigram_df)
        multigram_list.append(trigram_df)

    # multigram_df = pd.DataFrame(multigram_list, columns=['1/5', '2/5', '3/5', '4/5', '5/5', 'ngram', 'pos_tag', 'year', 'ngram_number'], ignore_index=True)
    multigram_df = pd.concat(multigram_list, axis = 0, ignore_index=True)
    multigram_df['source'] = "google"
    multigram_df['hotel_name'] = f"{specific_hotel_name}".lower()
    
    multigram_require_columns = ['1/5', '2/5', '3/5', '4/5', '5/5', 'ngram', 'ngram_pos', 'year', 'ngram_number', 'source', 'hotel_name']
    if len(multigram_df.columns) != 11:
            for i in multigram_require_columns:
                if i not in multigram_df.columns:
                    multigram_df[i]= 0
    
    multigram_df = multigram_df[['1/5', '2/5', '3/5', '4/5', '5/5', 'ngram', 'ngram_pos', 'year', 'ngram_number', 'source', 'hotel_name']]
    
    file_path = file_path_1 + file_path_2
    file_name = r'multigram_df.csv'
    output_file_2_list.append(f"{file_path}" + f"{file_name}")
    multigram_df.to_csv(f"{file_path}" + f"{file_name}", sep='|')
    
    #######################################     PART 2 - WORKING WITH NOUNS AND ADJECTIVES
    
    ####        5. get the lemmatized subject and object nouns in statement column if they exist along with the respective sentiment score of the statement
    df_1 = df_1.query("Rating_Star == '1/5' | Rating_Star == '2/5'")
    noun_df = extract_subject_object_nouns_and_calculate_sentiments(df_1, 'statement', 'sentiment_score')
    noun_df = noun_df.drop(columns=['Review_date'])
    noun_df.rename(columns = {'year':'Review_date'}, inplace=True)    
            
    noun_df['Review_date'] = pd.DataFrame(noun_df['Review_date'].apply(last_n_words_lambda, y=1))
    # print(noun_df)
    
    # ####        6. group by the noun column and compute average sentiment score, count for each noun
    noun_df_summarized = noun_df.groupby(['Review_date', 'noun'], as_index=False).agg(agg_count=('noun',"count") , mean_senti_score=('sentiment_score',"mean"))
    # print(noun_df_summarized)
    
    ####        7. compute sentiment from the computer average sentiment score for each subject/ object noun
    noun_df_summarized['sentiment'] = pd.DataFrame(noun_df_summarized["mean_senti_score"].apply(compute_sentiment))
    
    noun_df_summarized['source'] = "google"
    noun_df_summarized['hotel_name'] = f"{specific_hotel_name}".lower()
    noun_df_summarized = noun_df_summarized[['Review_date','noun','agg_count','mean_senti_score','sentiment','source','hotel_name']]

    file_path = file_path_1 + file_path_2
    file_name = r'summarized_noun_df.csv'
    output_file_3_list.append(f"{file_path}" + f"{file_name}")
    noun_df_summarized.to_csv(f"{file_path}" + f"{file_name}", sep='|')
    
    ####        8. from the statement column in data frames under STEP 1, get a list of subject and abject nouns and some adjectives linked to them
    df_1 = df_1.query("Review_date == '2020' | Review_date == '2019' | Review_date == '2018'")
    noun_and_features_dictionary = nouns_with_wordlist(df_1, 'statement')
    
    noun_and_features_dictionary.sort_values('polarity', ascending=False, inplace=True)
    noun_and_features_dictionary.reset_index()
    # print(noun_and_features_dictionary.groupby('noun').head(5))
    noun_and_features_dictionary.sort_values('polarity', ascending=True, inplace=True)
    noun_and_features_dictionary.reset_index()
    # print(noun_and_features_dictionary.groupby('noun').tail(5))
    
    final_df = noun_and_features_dictionary.groupby('noun').head(5)
    final_df.append(noun_and_features_dictionary.groupby('noun').tail(5))
    final_df.sort_values(['noun','polarity'], ascending=[True, False], inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    final_df['source'] = "google"
    final_df['hotel_name'] = f"{specific_hotel_name}".lower()
    final_df = final_df[['noun','attribute','polarity','source','hotel_name']]
   
    file_path = file_path_1 + file_path_2
    file_name = r'noun_feature_polairty.csv'
    output_file_4_list.append(f"{file_path}" + f"{file_name}")
    final_df.to_csv(f"{file_path}" + f"{file_name}", sep='|')


temp_list = []
for file in output_file_1_list:
    # print(file)
    df = pd.read_csv(file, sep='|', header=0)
    temp_list.append(df)

main_df_with_word_count = pd.concat(temp_list, axis=0, ignore_index=True)
file_name = r"google_main_df_with_word_count.csv"
main_df_with_word_count.to_csv(f"{file_path_1}" + f"{file_name}", sep='|')

temp_list = []
for file in output_file_2_list:
    df = pd.read_csv(file, sep='|', header=0)
    temp_list.append(df)

multigram_df = pd.concat(temp_list, axis=0, ignore_index=True)
file_name = r"google_multigram_df.csv"
multigram_df.to_csv(f"{file_path_1}" + f"{file_name}", sep='|')

temp_list = []
for file in output_file_3_list:
    df = pd.read_csv(file, sep='|', header=0)
    temp_list.append(df)

summarized_noun_df = pd.concat(temp_list, axis=0, ignore_index=True)
file_name = r"google_summarized_noun_df.csv"
summarized_noun_df.to_csv(f"{file_path_1}" + f"{file_name}", sep='|')

temp_list = []
for file in output_file_4_list:
    df = pd.read_csv(file, sep='|', header=0)
    temp_list.append(df)

noun_feature_polairty = pd.concat(temp_list, axis=0, ignore_index=True)
file_name = r"google_noun_feature_polairty.csv"
noun_feature_polairty.to_csv(f"{file_path_1}" + f"{file_name}", sep='|')
