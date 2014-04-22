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
    def persist(self, data):
        client = MongoClient()
        db = client["recres"]
        experiments = db["experiments"]
        try:
            experiments.insert(data)
        finally:
            client.close()

