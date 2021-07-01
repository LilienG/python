import re


def search(input_string: str):
    res = []
    para = input_string.split(" ")
    p = re.compile('^[fFGgHh][a-zA-Z]*a[a-zA-Z]*g$')
    for word in para:
        if p.findall(word):
            res.append(word)
    return res
