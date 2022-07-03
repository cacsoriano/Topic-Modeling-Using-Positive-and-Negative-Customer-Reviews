# Yelp Me! Topic Modeling Using Positive and Negative Customer Reviews

Open `Technical_Report.ipynb` to view the full report.

### Executive Summary

Small businesses tend to have it hard in the digital age. With the proliferation of social media and review platforms, there is an abundance of customer reviews that a typical business owner cannot handle. Which reviews should they focus on? And which ones should be ignored? How can a small business owner keep up with the influx of information while keeping the business afloat?

We intend to answer these questions through Latent Semantic Analysis (LSA). Through LSA, we hope we will be able to understand the underlying characteristics of customer reviews. We utilized the Yelp website as the source of our data as it has accumulated millions of reviews since its inception in 2004. However, our study focuses on reviews made in 2017 alone while also excluding neutral reviews (which have a star rating of 3). As such, we will be using two subsets of the Yelp customer review data, positive (more than a star rating of 3) and negative (less than a star rating of 3).

Our methodology applied significant preprocessing which entailed 1) removing punctuations and symbols, 2) using a corpus of stop words from the NLTK library, and 3) implementing lemmatization using the same library. We then tokenize each row of text to create the Term Frequency-Inverse Document Frequency (TF-IDF) matrix, or the document term matrix. We then employ LSA to decompose the TF-IDF into latent topics which will be the basis of clustering each review.

As we clustered the reviews, we saw unique themes for both subsets. Positive reviews tend to focus on food and quality of service. In addition, we see a cluster where it only focuses on customers recommending the business. Curiously, we also found a cluster solely for nail salons. On the other hand, negative review clusters tend to be the opposite for both restaurants and service-based businesses. However, a unique cluster for car maintenance was found in negative customer reviews.

In partial fulfillment for the requirements of AIM MSDS 2022 Data Mining and Wrangling under Prof. Christian Alis.
