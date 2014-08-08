#! /usr/bin/python

"""

Provides routines for persisting experiment results using different
db tools.

"""

from pymongo import MongoClient

class Preserver(object):
    def persist(self, data):
        pass

class MongoPreserver(Preserver):
    def __init__(self, dbname="recres"):
        """
        set up the db information
        """
        self.dbname = dbname

    def persist(self, data, collection="experiments"):
        client = MongoClient()
        db = client[self.dbname]
        experiments = db[collection]
        try:
            experiments.insert(data)
        finally:
            client.close()

