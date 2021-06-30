#encoding=utf-8

from typing import Dict, List


class ResponseData:
    data_list: List[object]
    total: int

    def __init__(self, data_list=None, total: int = 0):
        if data_list is None:
            data_list = []
        self.data_list = data_list
        self.total = total


# turn string to number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def fetch_data(input_file_path: str,
               page_size: int = 0, page: int = 1,
               sort_key: str = None, sort_order: str = 'asc',
               filter_dict: Dict = None) -> ResponseData:
    # read data from file
    # TODO: open the file and retrieve the content
    body = list()
    try:
        file = open(input_file_path, "r", encoding='utf-8')
    except IOError:
        response_data = ResponseData(body, len(body))
        return response_data
    else:
        lines = file.readlines()
        if len(lines) == 0:
            response_data = ResponseData(body, len(body))
            return response_data
        head = (lines[0].strip()).split(",")
        for i in range(1, len(lines)):
            temp = dict()
            t = (lines[i].strip()).split(",")
            if len(t) != len(head):
                continue
            for j in range(len(head)):
                if is_number(t[j]):
                    t[j] = eval(t[j])
                temp[head[j]] = t[j]
            body.append(temp)
        file.close()

    # process logic
    # TODO: achieve paginate/sort/filter/... logic according to the augments
    # paginate
    if page_size == 0:
        pass
    elif page > len(body) / page_size or page < 1 or page_size < 0:
        body = []
    elif page > 0:
        # min(page * page_size, len(body))
        body = body[(page - 1) * page_size: page * page_size]

    # sort
    if sort_key is None:
        pass
    elif (sort_order != 'asc' and sort_order != 'desc') or sort_key not in head:
        body = []
    elif sort_order == 'asc':
        body.sort(key=lambda ele: ele[sort_key])
    elif sort_order == 'desc':
        body.sort(key=lambda ele: ele[sort_key], reverse=True)

    # filter
    if filter_dict is None:
        pass
    else:
        for key in filter_dict.keys():
            if filter_dict[key] is None or key not in head:
                del filter_dict[key]

        for i in range(len(body)-1, -1, -1):
            for key in filter_dict.keys():
                if is_number(filter_dict[key]):
                    if filter_dict[key] != body[i][key]:
                        del body[i]
                elif body[i][key].find(filter_dict[key]) == -1:
                    del body[i]

    # return
    # TODO: modify codes below to return correct response
    response_data = ResponseData(body, len(body))
    return response_data
