# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:31:33 2016

@author: bmazoyer, dshi, hbaud, vlefranc
"""
# import json
import logging
import requests
# from datetime import datetime
from twython import TwythonStreamer
from config import ACCESS, PROXY, LANG, MONGODB
from pymongo import MongoClient

# connect to MongoDB
# client = MongoClient("mongodb://localhost:27017/")
client = MongoClient("mongodb+srv://" + MONGODB["USER"] + ":" + MONGODB["PASSWORD"] + "@" + MONGODB["HOST"] + "/" + MONGODB["DATABASE"] + "?retryWrites=true")
db = client[MONGODB["DATABASE"]]

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)


class SampleStreamer(TwythonStreamer):
    """
    Retrieve data from the Twitter Streaming API.

    The streaming API requires
    `OAuth 1.0 <http://en.wikipedia.org/wiki/OAuth>`_ authentication.
    """

    def __init__(self, app_key, app_secret, oauth_token, oauth_token_secret):

        self.do_continue = True
        self.count = 0
        self.line_count = 0
        # self.current_file = FILEDIR + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".txt"
        if PROXY:
            client_args = {
                'proxies': PROXY
            }
        else:
            client_args = {}
        TwythonStreamer.__init__(self, app_key, app_secret, oauth_token,
                                 oauth_token_secret, timeout=100, chunk_size=200,
                                 client_args=client_args)

    def on_success(self, data):
        """
        :param data: response from Twitter API (one tweet in json format)
        """

        db.tweets.insert_one(data)
        self.count += 1
        logging.info("Total of {} elements added".format(self.count))

        # with open(self.current_file, "a+", encoding='utf-8') as f:
        #     json.dump(data, f)
        #     f.write("\n")
        # self.line_count += 1
        #
        # if self.line_count > FILEBREAK:
        #     logging.info("Closing file {}".format(self.current_file))
        #     self.current_file = FILEDIR + datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + ".txt"
        #     self.line_count = 0

        if not self.do_continue:
            logging.info("disconnect")
            self.disconnect()

    def on_error(self, status_code, data):
        """
        :param status_code: The status code returned by the Twitter API
        :param data: The response from Twitter API

        """

        logging.error(str(status_code) + ": " + str(data))

    def sample(self, lang=None):
        """
        Wrapper for 'statuses / sample' API call
        """
        while self.do_continue:

        # Stream in an endless loop
            try:
                self.statuses.sample(language=lang)
            except requests.exceptions.ChunkedEncodingError as e:
                if e is not None:
                    logging.info("Encoding error (stream will continue): {0}".format(e))


if __name__ == "__main__":
    twitter = SampleStreamer(*ACCESS)
    twitter.sample(lang=LANG)
