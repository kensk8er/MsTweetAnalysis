"""
Preprocessing utils
"""
import re
from module.text.compoundword import compound_words

__author__ = 'kensk8er'


def clean_text(text):
    # TODO: a bit like black magic... simplify this.
    text = re.sub("(http(s)?://[A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]+)", '', text)  # URL
    text = re.sub("(www\.[A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]+)", '', text)  # URL
    text = re.sub("([A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]+\.\w+)", '', text)  # URL
    text = re.sub("(\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,6})", '', text)  # email
    text = re.sub("(-{2,})", '', text)  # hyphen
    text = re.sub("(\w+)(-)(\w+)", r'\1 \3', text)  # hyphen
    text = re.sub("(mailto:\w+)", r'\1 \3', text)  # hyphen
    text = re.sub("=\r\n", '', text)  # next-line
    text = re.sub("([A-Za-z0-9\'~+\-=_.,/%\?!;:@#\*&\(\)]{15,})", '', text)  # long characters
    text = re.sub("(\b)(\w)(\b)", r'\1 \3', text)  # short (single) characters
    text = re.sub("[0-9]", ' ', text)  # number
    text = re.sub("\w*([~+\-=_/%@#\*&\?!]+\w+)", '', text)  # special code
    return text


def convert_compound(document):
    doc_string = " ".join(document)
    for compound_word in compound_words:
        doc_string = re.sub(compound_word, re.sub('\s', '-', compound_word), doc_string)

    return doc_string.split()


