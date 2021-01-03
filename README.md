# A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP
In this project we seek a method to ethically scrape web review data without using an API or breaking any ethical barriers set by the review websites. We then look at methods we can use to get actionable insights from the extracted negative reviews.


STEP 1: ETHICAL REVIEW EXTRACTION FROM WEBSITES
The most effective way to extract web data from review sites is through the use of APIs. Using APIs is not only quicker but one can often get customized results using the various API methods provided by the websites. The downside however is that all the best things cost us and hence the use of APIs might not be best suited for students or for research purposes when montiory resources may be constrained.

The second best way (atleast when using python) is to use web crawlers - scrapy and beautiful soap are example modules in python which can crawl thorugh a given website. The way they achieve this is by getting the http response of a given URL and crawling the response to get the desired data.

