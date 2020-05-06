import argparse
import os
import requests
from pprint import pprint


class API(object):

    def __init__(self, api_key):
        self.api_key = api_key

    def analyze_entities(self, text):
        url = 'https://language.googleapis.com/v1/documents:analyzeEntities?key={}'.format(self.api_key)
        headers = {'Content-Type': 'application/json'}
        body = {
            'document': {
                'type': 'PLAIN_TEXT',
                'language': 'ja',
                'content': text
            }
        }
        response = requests.post(url, headers=headers, json=body).json()
        return response


def print_result(response):
    for entity in response['entities']:
        print('=' * 20)
        print('         name: {0}'.format(entity['name']))
        print('         type: {0}'.format(entity['type']))
        print('     salience: {0}'.format(entity['salience']))
        print('wikipedia_url: {0}'.format(entity['metadata'].get('wikipedia_url', '-')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='The text you\'d like to analyze entities.')
    args = parser.parse_args()

    api_key = os.environ.get('API_KEY')
    api = API(api_key)
    res = api.analyze_entities(args.text)
    pprint(res)
    print_result(res)
