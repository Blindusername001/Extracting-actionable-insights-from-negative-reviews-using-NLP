# A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP
In this project we seek a method to ethically scrape web review data without using an API or breaking any ethical barriers set by the review websites. We then look at methods we can use to get actionable insights from the extracted negative reviews.


# STEP 1: ETHICAL REVIEW EXTRACTION FROM WEBSITES
## APIs:
The most effective way to extract web data from review sites is through the use of APIs. Using APIs is not only quicker but one can often get customized results using the various API methods provided by the websites. The downside however is that all the best things cost us and hence the use of APIs might not be best suited for students or for research purposes when montiory resources may be constrained.

## Web-crawlers:
The second best way (atleast when using python) is to use web crawlers - scrapy and beautiful soap are example modules in python which can crawl thorugh a given website. The way they achieve this is by getting the http response of a given URL and crawling the response to get the desired data.

The issue with using this method is when a user has to do an interaction with a webpage to get the entire data. For example, the website might have a 'Read More' button which has to be clicked before extracting this data. These usually take the form of a GET request or a POST request.
![ScreenShot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/2021-01-03%2020_16_24-Window.png)

![ScreenShot](https://github.com/karthikkumar001/A-method-to-extract-actionable-insights-from-negative-reviews-using-NLP/blob/main/Images/2021-01-03%2020_20_42-Window.png)

When using scrapy, one can easily overcome this by sending the http request with a GET or POST request. If the host website accepts this - it is well and good. But most websites do not allow this. To subvert this, you can write your scrapy code to copy the request parameters from firefox or chrome and trick the host website system into believing the request os from a normal user and not an automated code. It works but it isn't exactly ethical!

## Selenium:
When both the above cases aren't satisfactory, the only option left is to use selenium. It is comapritively slower than the above methods - but when one has to balance (ethics + cost) vs performance, there isn't any other option. The good thing here is that selenium can be used to create an automation which will mimic the way a user would browse a website manually - opening a browser, entering a URL, clicking buttons, drop-downs, etc. The below video will explain why the previous two methods did not work for my case and also show how selenium works more like a manual user.

https://www.youtube.com/watch?v=5RxbfPjQKe8
