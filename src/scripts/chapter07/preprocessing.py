import logging
import re

from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer

import logging_dict
logger = logging.getLogger('preproLogging')

t = Tokenizer(wakati=True)

def clean_html(html, strip=False):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(strip=False)
    return text

def tokenize(text):
    return t.tokenize(text, wakati=True)
