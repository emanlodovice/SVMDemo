import json


def build_data(positive='data_1.pos', negative='data_2.neg'):
    pos = __read(positive, 1)
    neg = __read(negative, -1)
    return (pos, neg)


def __read(filename, c):
    data = []
    with open(filename) as f:
        while True:
            t = f.readline()
            if t is None or t == '':
                break
            try:
                t = t.decode('utf-8', 'ignore')
                data.append({'text': t, 'class': c})
            except:
                pass
    return data
