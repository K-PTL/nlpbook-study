import logging
import re

from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer

import logging_dict
logger = logging.getLogger('preproLogging')

t = Tokenizer()

def clean_html(html, strip=False):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(strip=False)
    return text

def tokenize(text):
    return t.tokenize(text, wakati=True)

def tokenize_base_form(text):
    tokens = [token.base_form for token in t.tokenize(text)]
    return tokens

def normalize_number(text, reduce=False):
    if reduce:
        normalized_text = re.sub(r"\d+", "0", text)
    else:
        normalized_text = re.sub(r"\d",  "0", text)
    return normalized_text
