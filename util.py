# general libraries
import numpy as np
import pandas as pd
import datetime as dt
import pickle
import json
import re
import scipy.sparse

# display images
from IPython.display import Image, display
from wordcloud import WordCloud

# plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import plotly.graph_objects as go

# nlp
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import casual_tokenize
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# sklearn functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ar
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.base import clone

pd.set_option('display.max_rows', 120)
pd.set_option('display.max_colwidth', None)
stop_words = set(stopwords.words('english'))


def import_data():
    """Load Yelp review datas into pandas DataFrame."""
    path = '/mnt/data/public/yelp/challenge12/yelp_dataset/'
    file = 'yelp_academic_dataset_review.json'
    data = pd.DataFrame(columns=['stars', 'date', 'text'])
    with pd.read_json(path + file, chunksize=1e5, lines=True) as reader:
        for chunk in reader:
            data = data.append(chunk[['date', 'stars', 'text']])
    data = data.reset_index(drop=True)
    return data

def plot_ratings_dist(data):
    """Plot distribution of ratings.
    
    Parameters
    ----------
    data : pandas DataFrame
        DataFrame where one column is ratings.
    """
    dist = (data.stars.value_counts(normalize=True)
            .drop(labels=0, axis=0).sort_index(ascending=False))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax = dist.plot(kind='bar', color=['g', 'g', 'grey', 'r', 'r'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Distribution of Ratings')
    ax.set_xlabel('Ratings')
    ax.set_ylabel('Percentage')
    ax.set_xticklabels(range(1, 6)[::-1], rotation=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.legend(handles=[mpatches.Patch(color='r', label='Negative'), 
                       mpatches.Patch(color='g', label='Positive')], loc=1)
    for i, v in dist.reset_index(drop=True).items():
        ax.text(i, v, s='{:.2f}%'.format(v*100), ha='center', va='bottom')
    plt.show()

def plot_year_dist(data):
    """Plot distribution of year.
    
    Parameters
    ----------
    data : pandas DataFrame
        DataFrame where one column is date a review was posted.
    """
    dist = (data.date.dt.year.value_counts(normalize=True).sort_index())
    colormat = np.where(dist == dist.max(), 'red', 'grey')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax = dist.plot(kind='bar', color=colormat)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Distribution of Years')
    ax.set_xlabel('Years')
    ax.set_ylabel('Percentage')
    ax.set_xticklabels(range(2004, 2019), rotation=0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    for i, v in dist.reset_index(drop=True).items():
        if i > 9:
            ax.text(i, v, s='{:.0f}%'.format(v*100), ha='center', va='bottom')
    plt.show()
    
def separate_reviews(data):
    """
    Creates two separate data frames for positive and negative reviews and
    save into new csv file
    """
    df_pos = (data[(data.stars > 3) & (data.date.dt.year == 2017)]
              .reset_index(drop=True))
    df_neg = (data[(data.stars < 3) & (data.date.dt.year == 2017)]
              .reset_index(drop=True))
    df_pos.to_csv('positive_2017_reviews.csv', index=False)
    df_neg.to_csv('negative_2017_reviews.csv', index=False)
    return df_pos, df_neg
    
def preprocess_text(doc):
    """Remove unnecessary characters and lemmatize words."""
    doc = casual_tokenize(doc, preserve_case=False, reduce_len=True)
    doc = [re.sub(r"[^a-z']+", '', word) for word in doc]
    doc = [word for word in doc if word not in stop_words]
    doc = [WordNetLemmatizer().lemmatize(word) for word in doc]
    doc = [word for word in doc if len(word) > 2]
    return ' '.join(doc)

def preprocess_docs(df_pos, df_neg):
    """Apply preprocessing to each document and save as pkl"""
    docs_pos = list(map(preprocess_text, df_pos.text.tolist()))
    with open(f'{filename}.pkl', 'wb') as fp:
        pickle.dump(docs_pos, fp)
    docs_neg = list(map(preprocess_text, df_neg.text.tolist()))
    with open(f'{filename}.pkl', 'wb') as fp:
        pickle.dump(docs_neg, fp)
    return docs_pos, docs_neg

def tfidf():
    """Vectorize corpus then save vectorization and vocabulary in pkl"""
    with open ('corpus_pos.pkl', 'rb') as fp:
        corpus_pos = pickle.load(fp)
    vectorizer_pos = TfidfVectorizer(min_df=0.01, stop_words='english')
    vectors_pos = vectorizer_pos.fit_transform(corpus_pos)
    scipy.sparse.save_npz('vectorized_pos', vectors_pos)
    with open('vocabulary_pos.pkl', 'wb') as fp:
        pickle.dump(vectorizer_pos.get_feature_names(), fp)
    with open ('corpus_neg.pkl', 'rb') as fp:
        corpus_neg = pickle.load(fp)
    vectorizer_neg = TfidfVectorizer(min_df=0.01, stop_words='english')
    vectors_neg = vectorizer_neg.fit_transform(corpus_neg)
    scipy.sparse.save_npz('vectorized_neg', vectors_neg)
    with open('vocabulary_neg.pkl', 'wb') as fp:
        pickle.dump(vectorizer_neg.get_feature_names(), fp)

def lsa_100_pos():
    """Truncate SVD for positive reviews to 100 latent components"""
    vectors_pos = scipy.sparse.load_npz('vectorized_pos.npz')
    svd = TruncatedSVD(n_components=100)
    lsa = svd.fit_transform(vectors_pos)
    return svd, lsa[:2500]
    
def lsa_100_neg():
    """Truncate SVD for negative reviews to 100 latent components"""
    vectors_neg = scipy.sparse.load_npz('vectorized_neg.npz')
    svd = TruncatedSVD(n_components=100)
    lsa = svd.fit_transform(vectors_neg)
    return svd, lsa[:2500]

def plot_lsa_pos(lsa): 
    """Plot positive reviews against its top 2 singular vectors"""
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.scatter(lsa[:, 0], lsa[:, 1], c='g', alpha=0.5)
    plt.title('Plot of Positive Reviews')
    plt.xlabel('SV1')
    plt.ylabel('SV2')
    plt.show()
    
def plot_lsa_neg(lsa): 
    """Plot negative reviews against its top 2 singular vectors"""
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.scatter(lsa[:, 0], lsa[:, 1], c='g', alpha=0.5)
    plt.title('Plot of Negative Reviews')
    plt.xlabel('SV1')
    plt.ylabel('SV2')
    plt.show()
        
def lsa(docs):
    """Perform LSA on the preprocessed documents"""
    vectorizer = TfidfVectorizer(min_df=0.01, stop_words='english')
    vectors = vectorizer.fit_transform(docs)
    svd = TruncatedSVD(n_components=vectors.shape[1]-1)
    lsa = svd.fit_transform(vectors)
    word_topic = pd.DataFrame(svd.components_,
                              columns=vectorizer.get_feature_names()).T
    return vectorizer, svd, lsa, word_topic

def top_10_per_topic(topics, n=20):
    """Return the top words for each topic or component"""
    df = pd.DataFrame(columns=range(1, 11))
    for i in abs(topics[range(0, n)]):
        df.loc[f'Topic {i}'] = topics[i].nlargest(10).index.values
    return df

def plot_var_explained(svd):
    """Plot variance explained of singular vector decomposition."""
    var = svd.explained_variance_ratio_.cumsum()
    var90 = var[np.where(var >= 0.9)[0][0]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(svd.explained_variance_ratio_.cumsum(), c='blueviolet')
    ax.axhline(0.9, c='grey', linestyle='--')
    ax.axvline(np.where(var >= 0.9)[0][0], c='grey', linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Cumulative Explained Variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Percentage of Variance Explained')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.text(0, var90, s='{:.0f}%'.format(var90*100), ha='center', va='bottom')
    plt.show()
    print(np.where(var >= 0.9)[0][0], 'components are needed to explain 90%.')
    
def plot_pcas(vectorizer, lsa):
    """Plot the top terms word vectors"""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_title('Top Words')
    ax.set_xlabel('SV1')
    ax.set_ylabel('SV2')
    for feature, vec in sorted(zip(vectorizer.get_feature_names(),
            lsa[:, 0:3]), key=lambda x: abs(x[1][1]), reverse=True)[:25]:
        ax.arrow(0, 0, vec[0], vec[1], alpha=0.1)
        ax.text(vec[0], vec[1], feature, c='blueviolet', fontsize=9)
    plt.show()

def doc_topic_matrix(lsa_, negative, topics=3):
    """Return dataframe of topics for reviews"""
    df = pd.DataFrame(lsa_[:, 0:topics])
    df['text'] = negative
    df = df[[df.columns.tolist()[-1]] + df.columns.tolist()[:-1]]
    return df

def plot_docs_pca(df, columns= [1, 2]):
    """Plot the document onto the first two components or topics"""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x=df.iloc[:, columns[0]], y=df.iloc[:, columns[1]],
               color='blueviolet')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Reviews')
    ax.set_xlabel('SV1')
    ax.set_ylabel('SV2')
    plt.show()

def encoding_matrix(svd, neg_vec, abs_value=True, sort=0):
    """Return a table of topics vis-a-vis the dictionary"""
    if abs_value:
        encoding_matrix = (pd.DataFrame(abs(svd.components_[:3]), 
                                        columns=neg_vec.get_feature_names())
                           .T.sort_values(by=sort, ascending=False))
    else:
        encoding_matrix = (pd.DataFrame(svd.components_[:3], 
                                        columns=neg_vec.get_feature_names())
                           .T.sort_values(by=sort, ascending=False))
    return encoding_matrix

def plot_heat_map(encoding_matrix, top=5, bot=5, no_topics=5):
    """Plot a heat map of the top and least occuring words for each topic"""
    columns = []
    topics = encoding_matrix.columns[:no_topics]

    for i in topics:
        head = pd.DataFrame()
        tail = pd.DataFrame()

        head = (encoding_matrix.loc[:, topics]
                .sort_values(by=i, ascending=False).head(top))
        tail = (encoding_matrix.loc[:, topics]
                .sort_values(by=i, ascending=False).tail(bot))

        series = head.append(tail)
        columns.append(series)

    df_heat = pd.concat(columns).drop_duplicates()
    df_heat = df_heat.drop(df_heat.head(top).index)
    fig_heat = go.Figure(data=go.Heatmap(z=df_heat, 
                                         x=df_heat.columns,
                                         y=df_heat.index))
    fig_heat.update_layout(height=800, width=400,
                           title='Dictionary Heatmap per Topic')
    fig_heat.update_xaxes(side="top")
    fig_heat.show()

def pooled_within_ssd(X, y, centroids, dist):
    """Compute pooled within-cluster sum of squares around the cluster mean
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
        
    Returns
    -------
    float
        Pooled within-cluster sum of squares around the cluster mean
    """
    distance = 0
    for i, label in enumerate(set(y)):
        data = X[np.where(y == label)]
        distance += np.mean([dist(row, centroids[i]) ** 2
                             for row in data]) / 2
    return distance

def gap_statistic(X, y, centroids, dist, b, clusterer, random_state=None):
    """Compute the gap statistic
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
    b : int
        Number of realizations for the reference distribution
    clusterer : KMeans
        Clusterer object that will be used for clustering the reference 
        realizations
    random_state : int, default=None
        Determines random number generation for realizations
        
    Returns
    -------
    float
        Gap statistic
    float
        Standard deviation of gap statistic
    """
    Wk = np.log(pooled_within_ssd(X, y, centroids, dist))
    Wki = []
    clusterer.set_params(n_clusters=centroids.shape[0])
    rng = np.random.default_rng(random_state)
    for i in range(b):
        x1 = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
        y1 = clusterer.fit_predict(x1)
        Wki.append(np.log(pooled_within_ssd(x1, y1,
            clusterer.cluster_centers_, euclidean)) - Wk)
    return [np.mean(Wki), np.std(Wki)]

def cluster_range(X, clusterer, k_start, k_stop):
    """Returns the internal validation criteria, centers and labels for each 
    cluster from start to end k"""
    ys = []
    centers = []
    inertias = []
    chs = []
    scs = []
    gss = []
    gssds = []
    for k in range(k_start, k_stop+1):
        clusterer_k = clone(clusterer).set_params(n_clusters=k)
        y = clusterer_k.fit_predict(X)
        ys.append(y)
        centers.append(clusterer_k.cluster_centers_)
        inertias.append(clusterer_k.inertia_)
        chs.append(calinski_harabasz_score(X, y))
        scs.append(silhouette_score(X, y))
        gs = gap_statistic(X, y, clusterer_k.cluster_centers_, 
                           euclidean, 5, clusterer_k, random_state=1337)
        gss.append(gs[0]); gssds.append(gs[1])
            
    results = {'ys':ys, 'centers':centers, 'inertias':inertias, 'chs':chs,
               'scs':scs, 'gss':gss, 'gssds':gssds}
    return results

def plot_clusters(X, ys, centers, transformer):
    """Plot clusters given the design matrix and cluster labels"""
    k_max = len(ys) + 1
    k_mid = k_max//2 + 2
    fig, ax = plt.subplots(2, k_max//2, dpi=150, sharex=True, sharey=True, 
                           figsize=(7,4), subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01))
    for k,y,cs in zip(range(2, k_max+1), ys, centers):
        centroids_new = transformer.transform(cs)
        if k < k_mid:
            ax[0][k%k_mid-2].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[0][k%k_mid-2].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y)) + 1),
                marker='s',
                ec='k',
                lw=1
            );
            ax[0][k%k_mid-2].set_title('$k=%d$'%k)
        else:
            ax[1][k%k_mid].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[1][k%k_mid].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y))+1),
                marker='s',
                ec='k',
                lw=1
            );
            ax[1][k%k_mid].set_title('$k=%d$'%k)
    return ax

def plot_internal(inertias, chs, scs, gss, gssds):
    """Plot internal validation values"""
    fig, ax = plt.subplots()
    ks = np.arange(2, len(inertias)+2)
    ax.plot(ks, inertias, '-o', label='SSE')
    ax.plot(ks, chs, '-ro', label='CH')
    ax.set_xlabel('$k$')
    ax.set_ylabel('SSE/CH')
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.errorbar(ks, gss, gssds, fmt='-go', label='Gap statistic')
    ax2.plot(ks, scs, '-ko', label='Silhouette coefficient')
    ax2.set_ylabel('Gap statistic/Silhouette')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2)
    return ax

def plot_clusters(lsa):
    """Plot fitted clusters against the top 2 singular vectors"""
    plt.rcParams["figure.figsize"] = (15, 6)
    fig, axes = plt.subplots(2, 5)
    for idx, ax in enumerate(axes.flat):
        ax.scatter(lsa[:, 0], lsa[:, 1], c=results['ys'][idx])
        ax.set_title(f'$k={idx}$')
        ax.set_ylabel('SV2')
        ax.set_xlabel('SV1')
    plt.tight_layout()
    plt.show()

def word_cloud(k, lsa, filename):
    """Create a word cloud based on the clustered results"""
    kmeans = KMeans(k, random_state=1337)
    clusters = kmeans.fit_predict(lsa)
    
    # plot clustered
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.scatter(lsa[:,0], lsa[:,1], c=kmeans.labels_, cmap='Reds')
    plt.title(f'$k=5$')
    plt.ylabel('SV2')
    plt.xlabel('SV1')
    plt.show()
    
    # plot word cloud
    with open(filename, 'rb') as fp:
        corpus_pos = pickle.load(fp)
    results_df = pd.DataFrame(lsa)
    results_df['cluster'] = clusters
    results_df['reviews'] = corpus_pos[:2500]
    for i in range(k):
        text = ''.join(results_df[results_df['cluster']==i].reviews.tolist())
        wordcloud = WordCloud(4000, 800, background_color='white',
                              min_font_size=6, colormap='Reds').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    