import os
from xml.dom.minidom import *
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", help="The folder containing the supervisely generated annotation files.")
parser.add_argument("output_folder", help="The output folder where to generate the voc pascal formatted annotation files.")
args = parser.parse_args()

path = args.input_folder
output = args.output_folder

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.split('.')[-1] == 'json']

print(files)

for file in files:
    file_path = os.path.join(path, file)
    with open(file_path) as json_file:
        data = json.load(json_file)

        for image in data["images"]:

            doc = Document()

            e_root = doc.createElement('annotation')
            e_size = doc.createElement('size')
            e_width = doc.createElement('width')
            e_height = doc.createElement('height')
            e_depth = doc.createElement('depth')

            doc.appendChild(e_root)
            e_root.appendChild(e_size)
            e_size.appendChild(e_width)
            e_size.appendChild(e_height)
            e_size.appendChild(e_depth)

            e_width.appendChild(doc.createTextNode(str(image["width"])))
            e_height.appendChild(doc.createTextNode(str(image["height"])))
            e_depth.appendChild(doc.createTextNode("3"))

            for annotation in data["annotations"]:

                if annotation["image_id"]== image["id"]:

                    #bbox is defined as [xmin, ymin, w, h in coco
                    xmin = annotation["bbox"][0]
                    ymin = annotation["bbox"][1]
                    xmax = annotation["bbox"][0] + annotation["bbox"][2]
                    ymax = annotation["bbox"][1] + annotation["bbox"][3]

                    e_obj = doc.createElement('object')
                    e_name = doc.createElement('name')
                    e_bnd = doc.createElement('bndbox')
                    e_xmin = doc.createElement('xmin')
                    e_xmax = doc.createElement('xmax')
                    e_ymin = doc.createElement('ymin')
                    e_ymax = doc.createElement('ymax')

                    e_root.appendChild(e_obj)
                    e_obj.appendChild(e_name)
                    e_name.appendChild(doc.createTextNode("Fish"))#TODO: fix this...
                    e_obj.appendChild(e_bnd)
                    e_bnd.appendChild(e_xmin)
                    e_xmin.appendChild(doc.createTextNode(str(xmin)))
                    e_bnd.appendChild(e_xmax)
                    e_xmax.appendChild(doc.createTextNode(str(ymin)))
                    e_bnd.appendChild(e_ymin)
                    e_ymin.appendChild(doc.createTextNode(str(xmax)))
                    e_bnd.appendChild(e_ymax)
                    e_ymax.appendChild(doc.createTextNode(str(ymax)))

                elif annotation["image_id"] > image["id"]:
                    break

            xml_file_name = image["file_name"] + '.xml'
            xml_file_path = os.path.join(output, xml_file_name)

            with open(xml_file_path, "w") as xml_file:
                doc.writexml(xml_file)


