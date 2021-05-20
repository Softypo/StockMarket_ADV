
# importing modules
import numpy as np
import pandas as pd


###ETL reddit data
def pq (names, subredits='allstocks', sort='relevance', date='all', comments=False):
    #importing reddit api
    from reddit_api import get_reddit
    reddit = get_reddit()

    #preparing the inputs to be search
    if isinstance(names, str):
        if names.isupper()==False:
            if names[0].isupper()==False:
                name1 = names
                name2 = names.capitalize()
                name3 = names.upper()
            else:
                name1 = names.lower()
                name2 = names
                name3 = names.upper()
        else:
            name1 = names.lower()
            name2 = names.lower().capitalize()
            name3 = names
        pnames = [[name1,name2,name3]]
    elif isinstance(names, list):
        pnames =[]
        for i, n in enumerate(names):
            if isinstance(n, str):
                n = str(n)
                if n.isupper()==False:
                    if n[0].isupper()==False:
                        name1 = n
                        name2 = n.capitalize()
                        name3 = n.upper()
                    else:
                        name1 = n.lower()
                        name2 = n
                        name3 = n.upper()
                else:
                    name1 = n.lower()
                    name2 = n.lower().capitalize()
                    name3 = n
                pnames.append([name1,name2,name3])
            else: pnames = []
    elif (isinstance(names, str)==False) or (isinstance(names, list)==False): pnames = []

    #scraping posts
    posts = []
    for n in pnames:
        if subredits=='allstocks':
            stocks = reddit.subreddit('stocks')
            for post in stocks.search(n[0] or n[1] or n[3], sort, 'lucene', date):
                posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
            stocks = reddit.subreddit('StocksAndTrading')
            for post in stocks.search(n[0] or n[1] or n[3], sort, 'lucene', date):
                posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
            stocks = reddit.subreddit('stockspiking')
            for post in stocks.search(n[0] or n[1] or n[3], sort, 'lucene', date):
                posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
            stocks = reddit.subreddit('Stocks_Picks')
            for post in stocks.search(n[0] or n[1] or n[3], sort, 'lucene', date):
                posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
            stocks = reddit.subreddit('wallstreetbets')
            for post in stocks.search(n[0] or n[1] or n[3], sort, 'lucene', date):
                posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
            stocks = reddit.subreddit('Wallstreetbetsnew')
            for post in stocks.search(n[0] or n[1] or n[3], sort, 'lucene', date):
                posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
            stocks = reddit.subreddit('WallStreetbetsELITE')
            for post in stocks.search(n[0] or n[1] or n[3], sort, 'lucene', date):
                posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
        else:
            hot_posts = reddit.subreddit(subredits)
            for post in hot_posts.search(n[0] or n[1] or n[3], sort, 'lucene', date):
                posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])

    posts = pd.DataFrame(posts,columns=['title', 'score', 'post_id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
    posts = posts.infer_objects()
    posts.drop_duplicates(subset ="post_id", keep = "first", inplace = True)
    posts.reset_index(drop=True, inplace=True)

    #scraping comments
    if comments==True:
        comments = []
        for index, row in posts.iterrows():
            submission = reddit.submission(id=row['post_id'])
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                comments.append([row['post_id'], row['title'], comment.score, comment.id, comment.body, comment.created])
        comments = pd.DataFrame(comments,columns=['post_id', 'post', 'score', 'comment_id', 'body', 'created'])
        comments = comments.infer_objects()
        return posts, comments

    return posts

###Sentiment analysis
def sentiment (posts, comments=pd.DataFrame([]), conf=0.9):
    #importing sentiment model flair
    import flair
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')

    #calculating sentiment on body
    sentiment = []
    confidence = []
    for sentence in posts['body']:
        if sentence.strip()=='':
            sentiment.append(np.nan)
            confidence.append(np.nan)
        else:
            sample = flair.data.Sentence(sentence)
            sentiment_model.predict(sample)
            sentiment.append(sample.labels[0].value)
            confidence.append(sample.labels[0].score)
    
    posts['sentiment'] = sentiment
    posts['confidence'] = confidence
    posts['sentiment'] = posts['sentiment'].astype('category')
    posts['confidence'] = pd.to_numeric(posts['confidence'])
    
    #calculating sentiment on tittle if body is nan
    for index in posts[posts["sentiment"].isna()].index:
        if posts.loc[index,"title"].strip()!='':
            sample = flair.data.Sentence(posts.loc[index,"title"])
            sentiment_model.predict(sample)
            posts.at[index,"sentiment"] = sample.labels[0].value
            posts.at[index,"confidence"] =  sample.labels[0].score

    #calculating sentiment on comments
    if comments.empty==False:
        sentiment = []
        confidence = []
        for sentence in comments['body']:
            if sentence.strip()=='':
                sentiment.append(np.nan)
                confidence.append(np.nan)
            else:
                sample = flair.data.Sentence(sentence)
                sentiment_model.predict(sample)
                sentiment.append(sample.labels[0].value)
                confidence.append(sample.labels[0].score)
        comments['sentiment'] = sentiment
        comments['confidence'] = confidence
        comments['sentiment'] = comments['sentiment'].astype('category')
        comments['confidence'] = pd.to_numeric(comments['confidence'])
    
    #mean sentiment of posts by comments
    posts["comments_sentiment"] = np.nan
    posts["comments_confidence"] = np.nan
    for post_id in comments["post_id"].unique():
        posts.at[posts.loc[posts["post_id"]==post_id].index,"comments_sentiment"] = comments["sentiment"].loc[(comments["post_id"]==post_id) & (comments["confidence"]>=conf)].mode()[0]
        posts.at[posts.loc[posts["post_id"]==post_id].index,"comments_confidence"] = comments["confidence"].loc[(comments["post_id"]==post_id) & (comments["confidence"]>=conf)].mean()
    
    #combined sentiment score
    posts["combined_sentiment"] = np.where (posts['sentiment'] == posts['comments_sentiment'], "POSITIVE", "NEGATIVE")
    posts["combined_confidence"] = (posts["confidence"]+posts["comments_confidence"])/2

###Stock data from yahoo finance
def y_stock_data (name, period, interval):
    import yfinance as yf
    stock_info = yf.Ticker(name).info
    stock_data = yf.Ticker(name).history(tickers=name, period=period, interval=interval, auto_adjust=True)
    stock_c = yf.Ticker(name).calendar
    stock_r = yf.Ticker(name).recommendations
    stock_a = yf.Ticker(name).actions
    stock_b = yf.Ticker(name).balance_sheet
    stock_qb = yf.Ticker(name).quarterly_balance_sheet
    return stock_info, stock_data, stock_c, stock_r, stock_a, stock_b, stock_qb

###look for both stock and redit data
def looker (stock_name, subredits='allstocks', sort='relevance', date='all', comments=False):
    #preparing user inputs
    if date=='all': period, interval = 'max', '1d'
    elif date=='year': period, interval = '1y', '1d'
    elif date=='month': period, interval = '1mo', '1d'
    elif date=='week': period, interval = '7d', '1h'
    elif date=='day': period, interval = '1d', '1m'
    elif date=='hour': period, interval = '1h', '1m'

    #yahoo stock data
    stock_info, stock_data, stock_c, stock_r, stock_a, stock_b, stock_qb = y_stock_data(stock_name, period, interval)
    
    #cheking stock dataframe
    if stock_data.empty: print('DataFrame is empty!')
    else:
        print('Stock data downloaded from Yahoo finance')
        #reddit scrapping  
        if comments==True:
            posts, comments = pq([stock_info['longName'],stock_name], subredits, sort, date, comments)
            #cheking dataframe
            if posts.empty: print('Posts DataFrame is empty!')
            else: print('Posts downloaded from Reddit')
            if comments.empty: print('Comments DataFrame is empty!')
            else: print('Comments downloaded from Reddit')
            return stock_info, stock_data, stock_c, stock_r, stock_a, stock_b, stock_qb, posts, comments
        else:
            posts = pq([stock_info['longName'],stock_name], subredits, sort, date, comments)
            #cheking dataframe
            if posts.empty: print('Posts DataFrame is empty!')
            else: print('Posts downloaded from Reddit')
            return stock_info, stock_data, stock_c, stock_r, stock_a, stock_b, stock_qb, posts

###Analyse sentiment outputs
def analyse (posts, comments=pd.DataFrame([]), conf=0.9):
    #sentiment calculation
    sentiment(posts, comments)

def main ():
    try:
        print ('hola')
    except Exception:
        print (Exception)

if __name__ == "__main__":
    main()
