"""
string module

Utility functions related to string are defined here.
"""
__author__ = 'kensk8er'


def strip_suffix(string, suffix):
    """
    Remove specified suffix from the string.

    :rtype : str
    :param string: str
    :param suffix: str OR list[str]
    :return: str
    """
    if type(suffix) is str:
        if string.endswith(suffix):
            return string[:-len(suffix)]
    elif type(suffix) is list:
        assert type(suffix[0]) is str, 'Elements of suffix need to be str!'
        for suffic in suffix:  # suffix -> suffics -> suffic (singular)
            if string.endswith(suffic):
                return string[:-len(suffic)]
    return string
