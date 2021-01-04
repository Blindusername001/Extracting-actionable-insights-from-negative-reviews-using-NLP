# A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP
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
Though the output of the above methods were not satisfactory, some aspects of text analysis worked well in each. 




