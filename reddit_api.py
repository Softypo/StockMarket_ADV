
# importing modules
from os import getenv
import praw

#reddit api module getting user and key from enviromental global variables  
def get_reddit ():
    reddit = praw.Reddit(client_id=getenv('reddit_id'), client_secret=getenv('reddit_key'), user_agent='WebScraping')
    return reddit

def main ():
    try:
        reddit = get_reddit()
    except Exception:
        print (Exception)
    return reddit

if __name__ == "__main__":
    main()
