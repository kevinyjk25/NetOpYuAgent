import requests


def restore(endpoint, snapshot):
    return requests.post(endpoint, json=snapshot, timeout=10).json()
