
# importing modules
import numpy as np
import pandas as pd

class stock_adv:
    ###ETL reddit data
    #----------------------------------------------------------------------------------------------------------------------------------------
    def reddit_data (self, names, subredits='allstocks', sort='relevance', date='all', comments=False):
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
            
            #formating comments dataframe
            comments['created'] = pd.to_datetime(comments['created'], unit='s')
            comments.set_index('created', inplace=True)

            #passing optional comments argument
            self.reddit_comments = comments
        
        #formating posts dataframe
        posts['created'] = pd.to_datetime(posts['created'], unit='s')
        posts.set_index('created', inplace=True)
        posts['score'] = posts['score']+1

        #passing arguments
        self.reddit_posts = posts
        self.reddit_search, self.reddit_subredits, self.reddit_sort, self.reddit_date = names, subredits, sort, date


    ###Sentiment analysis
    #----------------------------------------------------------------------------------------------------------------------------------------
    def sentiment (self):
        #passing arguments in
        posts = self.reddit_posts
        comments = self.reddit_comments
        
        #importing sentiment model flair
        import flair
        sentiment_model = flair.models.TextClassifier.load('en-sentiment')

        #changing index for processing
        posts.reset_index (inplace=True)
        posts.set_index('post_id', inplace=True)

        #calculating sentiment on body
        sentiment = []
        score = []
        for sentence in posts['body']:
            if sentence.strip()=='':
                sentiment.append(np.nan)
                score.append(np.nan)
            else:
                sample = flair.data.Sentence(sentence)
                sentiment_model.predict(sample)
                sentiment.append(sample.labels[0].value)
                score.append(sample.labels[0].score)
        posts['sentiment'] = sentiment
        posts['sentiment_score'] = score
        
        #calculating sentiment on tittle if body is nan
        for index in posts[posts["sentiment"].isna()].index:
            if posts.loc[index,"title"].strip()!='':
                sample = flair.data.Sentence(posts.loc[index,"title"])
                sentiment_model.predict(sample)
                posts.at[index,"sentiment"] = sample.labels[0].value
                posts.at[index,"sentiment_score"] =  sample.labels[0].score

        #calculating sentiment on comments
        if isinstance(comments, pd.DataFrame):
            sentiment = []
            score = []
            for sentence in comments['body']:
                if sentence.strip()=='':
                    sentiment.append(np.nan)
                    score.append(np.nan)
                else:
                    sample = flair.data.Sentence(sentence)
                    sentiment_model.predict(sample)
                    sentiment.append(sample.labels[0].value)
                    score.append(sample.labels[0].score)
            comments['sentiment'] = sentiment
            comments['sentiment_score'] = score
            
            #mean sentiment of posts by comments
            posts["comments_sentiment"] = np.nan
            posts["comments_score"] = np.nan
            for post_id in comments["post_id"].unique():
                posts.at[posts[posts.index==post_id].index,"comments_sentiment"] = comments["sentiment"].loc[comments["post_id"]==post_id].mode()[0]
                posts.at[posts[posts.index==post_id].index,"comments_score"] =comments["sentiment_score"].loc[comments["post_id"]==post_id].mean()
            
            #combined sentiment score
            posts["combined_sentiment"] = np.where (posts['comments_sentiment'].isna(), posts['sentiment'],np.where (posts['sentiment'] == posts['comments_sentiment'], 'POSITIVE', 'NEGATIVE'))
            posts["combined_score"] = (posts["sentiment_score"]+posts["comments_score"])/2
            posts["combined_score"] = np.where(posts["combined_score"].notna()==True, posts["combined_score"], posts["sentiment_score"])
        else:
            posts["comments_sentiment"] = np.nan
            posts["comments_score"] = np.nan
            posts["combined_sentiment"] = np.nan
            posts["combined_score"] = np.nan
        
        #returning to original index
        posts.reset_index(inplace=True)
        posts.set_index('created', inplace=True)

        #formating new columns
        posts['sentiment'] = posts['sentiment'].astype('category')
        posts['sentiment_score'] = pd.to_numeric(posts['sentiment_score'])
        comments['sentiment'] = comments['sentiment'].astype('category')
        comments['sentiment_score'] = pd.to_numeric(comments['sentiment_score'])
        posts['comments_sentiment'] = posts['comments_sentiment'].astype('category')
        posts['comments_score'] = pd.to_numeric(posts['comments_score'])
        posts['combined_sentiment'] = posts['combined_sentiment'].astype('category')
        posts['combined_score'] = pd.to_numeric(posts['combined_score'])
        
        #passing arguments out
        self.reddit_posts = posts
        self.reddit_comments = comments

    ###Stock data from yahoo finance
    #----------------------------------------------------------------------------------------------------------------------------------------
    def y_stock_data (self, name, period, interval):
        import yfinance as yf
        self.stock_info = yf.Ticker(name).info
        self.stock_data = yf.Ticker(name).history(tickers=name, period=period, interval=interval, auto_adjust=True)
        self.stock_c = yf.Ticker(name).calendar
        self.stock_r = yf.Ticker(name).recommendations
        self.stock_a = yf.Ticker(name).actions
        self.stock_b = yf.Ticker(name).balance_sheet
        self.stock_qb = yf.Ticker(name).quarterly_balance_sheet

    ###look for both stock and redit data
    #----------------------------------------------------------------------------------------------------------------------------------------
    def looker (self, stock_name, subredits='allstocks', sort='relevance', date='all', comments=False):
        #preparing user inputs
        if date=='all': period, interval = 'max', '1d'
        elif date=='year': period, interval = '1y', '1d'
        elif date=='month': period, interval = '1mo', '1d'
        elif date=='week': period, interval = '1wk', '1h'
        elif date=='day': period, interval = '1d', '1m'
        elif date=='hour': period, interval = '1h', '1s'

        #yahoo stock data
        self.y_stock_data(stock_name, period, interval)
        
        #cheking stock dataframe
        if self.stock_data.empty: print('DataFrame is empty!')
        else:
            print('Stock data downloaded from Yahoo finance')
            #reddit scrapping 
            self.reddit_data ([self.stock_info['symbol']+' '+self.stock_info['longName']], subredits, sort, date, comments) 
            if comments==True:
                #cheking dataframe
                if self.reddit_posts.empty: print('Posts DataFrame is empty!')
                else: print('Posts downloaded from Reddit')
                if self.reddit_comments.empty: print('Comments DataFrame is empty!')
                else: print('Comments downloaded from Reddit')
                return self.stock_info, self.stock_data, self.stock_c, self.stock_r, self.stock_a, self.stock_b, self.stock_qb, self.reddit_posts, self.reddit_comments
            else:
                self.reddit_data([self.stock_info['symbol']+' '+self.stock_info['longName']], subredits, sort, date, comments)
                #cheking dataframe
                if self.reddit_posts.empty: print('Posts DataFrame is empty!')
                else: print('Posts downloaded from Reddit')
                return self.stock_info, self.stock_data, self.stock_c, self.stock_r, self.stock_a, self.stock_b, self.stock_qb, self.reddit_posts

    ###Analyse sentiment outputs
    #----------------------------------------------------------------------------------------------------------------------------------------
    def analyse (self, sentiment='combined_sentiment', period='1d'):
        #passing arguments in
        posts = self.reddit_posts
        stock_data = self.stock_data
        stock_info = self.stock_info

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        #idea based
        if sentiment=='combined_sentiment': sentiment_prob = 'combined_score'
        elif sentiment=='comments_sentiment': sentiment_prob = 'comments_score'
        elif sentiment=='sentiment': sentiment_prob = 'sentiment_score'
        posts_all_s = posts['score'].loc[posts[sentiment] != np.nan].groupby(posts.loc[posts[sentiment] != np.nan].index.round(period)).sum()

        pos_s = posts['score'].loc[posts[sentiment] == 'POSITIVE'].groupby(posts.loc[posts[sentiment] == 'POSITIVE'].index.round(period)).sum()
        pos_r = pos_s/posts_all_s
        pos_p = posts[sentiment].loc[posts[sentiment] == 'POSITIVE'].groupby(posts.loc[posts[sentiment] == 'POSITIVE'].index.round(period)).count().mean()

        neg_s = posts['score'].loc[posts[sentiment] == 'NEGATIVE'].groupby(posts.loc[posts[sentiment] == 'NEGATIVE'].index.round(period)).sum()
        neg_r = neg_s/posts_all_s*-1
        neg_p = posts[sentiment].loc[posts[sentiment] == 'NEGATIVE'].groupby(posts.loc[posts[sentiment] == 'NEGATIVE'].index.round(period)).count().mean()

        date = 'week'

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        #figures
        fig.add_trace(go.Candlestick(x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'], name = stock_info['symbol']), secondary_y=True)

        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name = 'Stock price',
                        line=dict(color="#ffffff", width=5)),
                        secondary_y=True)

        fig.add_trace(go.Bar(x=pos_r.index,
                        y=pos_r,
                        base=0,
                        marker_color='lightgreen',
                        name='Positive Sentiment',
                        text=round(pos_r, 2),
                        textposition='auto',
                        ), secondary_y=False)

        fig.add_trace(go.Bar(x=neg_r.index,
                        y=neg_r,
                        base=0,
                        marker_color='crimson',
                        name='Negative Sentiment',
                        text=round(neg_r, 2),
                        textposition='auto',
                        ), secondary_y=False)

        fig.update_traces(opacity=0.6)

        # Add titles
        fig.update_layout(
            title=str(round(stock_data['Close'][-1], 2))+' '+stock_info['currency']+', '+stock_info['longName']+', '+stock_info['symbol']+', '+stock_info['exchange'],
            yaxis_title='Stock Price',
            xaxis_rangeslider_visible=False,
            barmode='relative',
            template='plotly_dark')

        # Axes
        fig.update_yaxes(showgrid=False)

        #Show
        fig.show()

def main ():
    try:
        ticket = stock_adv('AAPL', 'all', 'relevance', 'month', True)
    except Exception:
        print (Exception)
    
    #sentiment(posts, comments)

if __name__ == "__main__":
    main()
