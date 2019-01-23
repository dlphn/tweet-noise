# -*- coding: utf-8 -*-

import time
import logging
from datetime import datetime, timedelta, timezone
from twython import Twython, exceptions
from config import ACCESS, MONGODB
from pymongo import MongoClient


logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class FetchTweets:
    """
    Add latest numbers of retweets and favorites to the Mongo DB tweets.
    """

    def __init__(self, app_key, app_secret, oauth_token, oauth_token_secret):

        self.twitter = Twython(app_key, app_secret,
                               oauth_token, oauth_token_secret)
        client = MongoClient(
            "mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB[
                "DATABASE"] + "?retryWrites=true")
        self.db = client[MONGODB["DATABASE"]]
        self.count = 0

    def retrieve(self):
        """
        Retrieve more-than-one-week-old tweets from db
        and update retweet_count and favorite_count
        """
        start = time.time()
        logging.info("Retrieving data...")
        logging.info("Fetching tweets info...")
        for obj in self.db.tweets.find():
            try:
                obj['updated_at']
            except KeyError:
                last_week = datetime.now(timezone.utc) - timedelta(days=7)
                if datetime.strptime(obj['created_at'], '%a %b %d %H:%M:%S %z %Y') < last_week:
                    retweets, favorites = self.fetch_info(obj['id_str'])
                    print(retweets, favorites)
                    self.db.tweets.update_one({"_id": obj["_id"]}, {
                        "$set": {
                            "updated_at": datetime.now(timezone.utc),
                            "retweet_count": retweets,
                            "favorite_count": favorites
                        }
                    })
                    self.count += 1
                    if self.count % 100 == 0:
                        logging.info("{} elements updated".format(self.count))

        end = time.time()
        logging.info("Total of {0} elements updated in {1} seconds".format(self.count, end - start))

    def fetch_info(self, tweet_id):
        """
        Fetch tweet info
        """
        try:
            tweet_info = self.twitter.show_status(id=tweet_id)
            return tweet_info['retweet_count'], tweet_info['favorite_count']
        except exceptions.TwythonError:
            return 0, 0

    def fetch_test(self, tweet_id):
        """
        Fetch tweet info
        """
        try:
            tweet_info = self.twitter.show_status(id=tweet_id)
            print(tweet_info)
        except exceptions.TwythonError:
            print('Tweet no longer exists')


if __name__ == "__main__":
    twitter = FetchTweets(*ACCESS)
    # twitter.retrieve()
    # twitter.fetch_test("1055055713146494976")
