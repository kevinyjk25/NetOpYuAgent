import requests


def purge(endpoint):
    return requests.delete(endpoint, timeout=10)
