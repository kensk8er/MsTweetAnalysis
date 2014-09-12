"""
input module

Utility functions related to input are defined here.
"""
from glob import glob
import os
import json
from pyth.plugins.plaintext.writer import PlaintextWriter
from pyth.plugins.rtf15.reader import Rtf15Reader
import re
import sys
from util.string import strip_suffix
from subprocess import Popen, PIPE
from docx import opendocx, getdocumenttext
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

__author__ = 'kensk8er'


def unpickle(file):
    import cPickle

    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_json(dir_path):
    file_paths = glob(dir_path + '/' + '*.json')
    json_data = {}

    for file_path in file_paths:
        dir_name, file_name = os.path.split(file_path)
        file_name = strip_suffix(string=file_name, suffix='.json')
        json_file = open(file_path, 'r')
        json_data[file_name] = json.load(json_file)

    return json_data


def document_to_text(file_path):
    """
    Convert document file (.pdf, .doc, .docx, .odt, .rtf) into plain text.

    * Additional dependency 'antiword' and 'odt2txt' command is required to run this function.
    * Converting pdf file takes much more time than others

    :rtype : string
    :param file_path:
    :return: text-converted version of document file contents
    """
    dir_name, file_name = os.path.split(file_path)

    def convert_pdf_to_txt(path):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = file(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)

        fp.close()
        device.close()
        str = retstr.getvalue()
        retstr.close()
        return str

    if file_name[-4:] == ".doc":
        cmd = ['antiword', file_path]
        p = Popen(cmd, stdout=PIPE)
        stdout, stderr = p.communicate()
        if len(stdout) > 0:
            return stdout.decode('ascii', 'ignore')
        else:
            # try .rtf format when it's not .doc file
            try:
                doc = Rtf15Reader.read(open(file_path))
                return PlaintextWriter.write(doc).getvalue()
            except:
                pass
    elif file_name[-5:] == ".docx":
        document = opendocx(file_path)
        paratextlist = getdocumenttext(document)
        newparatextlist = []
        for paratext in paratextlist:
            newparatextlist.append(paratext.encode("utf-8"))
        return '\n\n'.join(newparatextlist)
    elif file_name[-4:] == ".odt":
        cmd = ['odt2txt', file_path]
        p = Popen(cmd, stdout=PIPE)
        stdout, stderr = p.communicate()
        return stdout.decode('ascii', 'ignore')
    elif file_name[-4:] == ".pdf":
        return convert_pdf_to_txt(file_path)
    elif file_name[-4:] == ".rtf":
        doc = Rtf15Reader.read(open(file_path))
        return PlaintextWriter.write(doc).getvalue()


# for debug
if __name__ == '__main__':
    text = document_to_text(file_path='data/original/sample/cvs/53722108421aa9f83b0015a3.pdf')
    print 'hoge'