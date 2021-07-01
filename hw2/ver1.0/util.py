import os
import xml.etree.ElementTree as ET
import json


def transfer_voc_to_json(voc_dir: str, json_path: str):
    # TODO write your code here
    fileList = os.listdir(voc_dir)

    res = {}
    rawdata = []
    analysisdata = {}
    filenum = 0
    objectnum = 0
    maxobject = []
    minobject = []

    # rawdata: list
    for file in fileList:
        if file.endswith('.xml'):
            fileInfo = {}
            max_obj_dict = {}
            min_obj_dict = {}
            sub = float('inf')
            sup = float('-inf')
            tree = ET.parse(voc_dir + '\\' + file)
            root = tree.getroot()

            # filename: str
            fileInfo["filename"] = root[1].text
            max_obj_dict["filename"] = root[1].text
            min_obj_dict["filename"] = root[1].text

            # size: dict
            size = {}
            for data in root[3]:
                size[data.tag] = eval(data.text)
            fileInfo["size"] = size

            # object: list
            objectList = []
            for label in root.findall('object'):
                obj = {}
                # name: str
                obj["name"] = label[0].text

                # bndbox: dict
                bndbox = {}
                for lab in label[4]:
                    bndbox[lab.tag] = eval(lab.text)
                area = (bndbox["xmax"] - bndbox["xmin"]) * (bndbox["ymax"] - bndbox["ymin"])
                if area > sup:
                    max_obj_dict["bndbox"] = bndbox
                    sup = area
                if area < sub:
                    min_obj_dict["bndbox"] = bndbox
                    sub = area
                obj["bndbox"] = bndbox
                objectList.append(obj)

                objectnum += 1
            fileInfo["object"] = objectList

            # rawdata: list
            rawdata.append(fileInfo)

            # analysisdata: dict
            maxobject.append(max_obj_dict)
            minobject.append(min_obj_dict)

            filenum += 1

    # analysisdata: dict
    # filenum: str
    analysisdata["filenum"] = filenum

    # objectnum: str
    analysisdata["objectnum"] = objectnum

    # maxobject & minobject: list
    analysisdata["maxobject"] = maxobject
    analysisdata["minobject"] = minobject

    # res: dict
    res["rawdata"] = rawdata
    res["analysisdata"] = analysisdata

    info_json = json.dumps(res, sort_keys=False, indent=4, separators=(',', ':'))
    file = open(json_path, 'w')
    file.write(info_json)

    return res

