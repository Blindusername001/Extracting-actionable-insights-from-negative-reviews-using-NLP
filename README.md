# Extracting-actionable-insights-from-negative-reviews-using-NLP
In this project we seek a method to ethically scrape web review data without using an API or breaking any ethical barriers set by the review websites. We then look at methods we can use to get actionable insights from the extracted negative reviews.


# STEP 1: ETHICAL REVIEW EXTRACTION FROM WEBSITES
## APIs:
The most effective way to extract web data from review sites is through the use of APIs. Using APIs is not only quicker but one can often get customized results using the various API methods provided by the websites. The downside however is that all the best things cost us and hence the use of APIs might not be best suited for students or for research purposes when montiory resources may be constrained.

## Web-crawlers:
The second best way (atleast when using python) is to use web crawlers - scrapy and beautiful soap are example modules in python which can crawl thorugh a given website. The way they achieve this is by getting the http response of a given URL and crawling the response to get the desired data.

The issue with using this method is when a user has to do an interaction with a webpage to get the entire data. For example, the website might have a 'Read More' button which has to be clicked before extracting this data. These usually take the form of a GET request or a POST request.
![ScreenShot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/2021-01-03%2020_16_24-Window.png)

<img src=https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/2021-01-03%2020_20_42-Window.png width=400 height=400>

When using scrapy, one can easily overcome this by sending the http request with a GET or POST request. If the host website accepts this - it is well and good. But most websites do not allow this. To subvert this, you can write your scrapy code to copy the request parameters from firefox or chrome and trick the host website system into believing the request os from a normal user and not an automated code. It works but it isn't exactly ethical!

## Selenium:
When both the above cases aren't satisfactory, the only option left is to use selenium. It is comapritively slower than the above methods - but when one has to balance (ethics + cost) vs performance, there isn't any other option. The good thing here is that selenium can be used to create an automation which will mimic the way a user would browse a website manually - opening a browser, entering a URL, clicking buttons, drop-downs, etc. The below video will explain why the previous two methods did not work for my case and also show how selenium works more like a manual user.

https://www.youtube.com/watch?v=5RxbfPjQKe8

A text explanation of how selenium code works can be found here - https://github.com/karthikkumar001/Web-Scraping-infinite-scroll-websites-with-Python-Selenium/blob/main/README.md


# STEP 2: TRY USING EXISTING NLP IMPLEMENTATIONS (We might be lucky)
Why reinvent the wheel? Instead, we can save time by trying out if any existing implementations shared by others can be modified and used for our specific case.

Before I tried to develop my own solutions, I tried some existing NLP techniques implemented by others on the internet. However, these did not product satisfactory results for my dataset.

## Attempt #1: Latent Dirichlet Allocation (LDA)
LDA is a popular method in text analysis when it comes to topic modelling. Our aim here is to pick actionable insights. Through LDA, we can categorize words in negative reviews under different topics. Our assumption here will be that the words categorized under a topic describe it (For example, if bugs, dirty, stain were under the topic bed).

The below article gives a step-by-step guide on how to implement LDA
https://towardsdatascience.com/%EF%B8%8F-sentiment-analysis-aspect-based-opinion-mining-72a75e8c8a6d

## Attempy #2: Guided LDA
Guided LDA can be thought of as a machine learning LDA. We can bias the LDA by giving it topics and some common words under each topic, and it will try to classify the negative review words accordingly.

https://github.com/vi3k6i5/GuidedLDA

## Attempt #3: Aspect based sentiment analysis using Dependency Parsing
This method is based on the following research,
Chockalingam, N., 2018. Simple and Effective Feature Based Sentiment Analysis on Product Reviews using Domain Specific Sentiment Scores. Polibits, Volume 57.

The below article outlines how the same can be achieved using Stanford NLP parser,
https://medium.com/analytics-vidhya/aspect-based-sentiment-analysis-a-practical-approach-8f51029bbc4a

The difference here is the use of dependency parsing and parts of speech tagging (parts of speech are nouns, verbs, etc and dependency parsing outlines the dependecnies  different POS elements have in a statement).


# STEP 3: CUSTOMIZED SOLUTION
Though the output of the above methods were not satisfactory, some aspects of text analysis worked well in each. For this solution, I used Spacy module in python because I was looking for a solution witha quick turnaround.

![screenshot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/Picture1.png)


### PRE-PROCESSING
First step is to pre-process/ clean the data we saved in the csv file. 
The following were done on the review text,
  •	remove emojis and symbols
  •	remove alphanumeric words
  •	remove special characters
  •	remove double-double and double-single quotes
  •	remove extra spaces
  •	remove periods in front of words (e.g. ‘.room’)
  •	make the text lowercase

Additionally, because of the differences in how google and tripadvisor present the data, the following data normalization steps are also performed,
  •	Tripadvisor ratings should be normalized to google’s format of [1/5, 2/5, etc.]
  •	Google’s review data contains text [a few weeks ago, a few months ago, a year ago, etc.] – the proper year must be populated for these


### TOKENIZATION
The reviews are split into separate sentences. This step is important because the aim is to implement a simple solution and hence dealing with single statements will provide a better result.
Example,
The below text will be split into two statements.
"The bed was very dirty. It has stains and bugs."

![screenshot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/tk1.png)

Checkpoint: output 1: This data is written as a csv file for further use

### computing n-grams
The next step is to populate unigrams, bigrams, and trigrams. This will help us check for frequency of single and grouped words which might give some evidence.
We remove stop words from Output 1 and lemmatize the sentence. We pass this data to a count vectorizer function to populate the n-grams. We use the count vectorizer range (1,1), (2,2) and (3,3) to get unigrams, bigrams, and trigrams, respectively.

Checkpoint: Output 2: The n-grams are appended into a single data frame and written to a csv file.

Unigram:

![screenshot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/unigram.png)

Bigram:

![screenshot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/bigram.png)

Trigram:

![screenshot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/trigram.png)


### Getting the noun data:
From output 1, the sentiment of each statement is calculated using Sentiment Intensity Analyser from VaderSentiment python module. 
The code by (Williams, 2020) was modified for this part since it worked well.
All 1/5 and 2/5 ratings were filtered and from each sentence, the subject and object nouns are extracted. This is done with the help of Spacy module in python. The words of interest are extracted using the below conditions,
  •	Pos tag should be NOUN (tag_ attribute of Spacy parsed text should start with NN)
  •	dependency should be nominal subject or direct object (dep_ attribute of Spacy parsed text should be nsubj or dobj)

## Since the sentiment of each statement in output 1 was populated, it can now be assumed that the sentiment of the statement is directed towards the extracted noun. This is the assumption in our research – most of the statements in reviews will talk about a noun subject or object and the attributes surrounding the same.

If a statement has both a noun subject and a noun object, the sentiment of the subject and object are placed in separate rows of the resulting dataframe and the sentiment of the original statement is applied to both.

Checkpoint: Output 3: The resulting dataset is grouped by the extracted noun column and the count, mean sentiment score and sentiment for each noun. This is written to a csv file.

For example, all the sentiment scores of statements where "bed" is the noun will be averaged

![screenshot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/noun%20ext.png)


### Feature extraction
Output 1 is again used to get the attributes associated with each noun. Output 1 data is filtered for 1/5 and 2/5 reviews and years (2020, 2019, 2018). Once again, the noun subjects and direct objects are extracted from the statements.
From the rest of the words in the statement, words with the following Spacy parsed dependency are extracted, 
•	"acl" 
•	"amod"
•	"acomp"
•	"xcomp"
•	"conj"
•	"compound"

These are considered the attributes that describe the noun subject/ object. This list was determined by randomly parsing spacy reviews and getting a list of dependencies which represented features in the tested sample. The online website (https://explosion.ai/demos/displacy) was used for this exercise.

Checkpoint: Output 4 - After extracting the words, the polarity of each word is computed using TextBlob module of python. The output is of the form [noun word – attribute word – polarity of attribute word]. A noun can have multiple attributes and hence in this dataset, noun-attribute combination is unique (care was taken not to add duplicate attributes).


![screenshot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/pol.png)



### combining output data
The above-mentioned process is done separately for tripadvisor scraped files and google scraped files due to some minor changes required in the code. All the 4 output files for each hotel are then combined to form a final list of output files where all the data for all hotels from both tripadvisor and google are consolidated.


## ANALYSIS
1. Using the unigram, bigram and trigram data, the frequncy of words used in negative reviews can be found
2. Using output 3 where we have nouns with their mean sentiment scores and freuqncy, we can find nouns (or aspects) that have the most negative scores
3. Using output 4 where we have attributes of a noun and their polarity, we can find what exactly customers complained about in their reviews regarding a particular noun

This method can also be used in a filtered manner. For example, if a hotel wants to find what is wrong with their beds, they can use a filtered approach in this code to get that.
