import requests


def apply(endpoint, payload):
    return requests.post(endpoint, json=payload, timeout=10).json()
