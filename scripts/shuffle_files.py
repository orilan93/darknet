import argparse
import os
import random
from shutil import copy2
from xml.dom import minidom

parser = argparse.ArgumentParser()
parser.add_argument("voc_folder", help="The folder containing the voc annotations.")
parser.add_argument("img_folder", help="The folder containing the image files.")
parser.add_argument("output_folder", help="The root output folder.")
parser.add_argument("test_split", help="The percentage of test data.")
args = parser.parse_args()

random.seed(0)

voc_folder = args.voc_folder
img_folder = args.img_folder
root_folder = args.output_folder
test_split = int(args.test_split)

voc_files = [f for f in os.listdir(voc_folder) if os.path.isfile(os.path.join(voc_folder, f))]
img_files = [os.path.splitext(f)[0] for f in voc_files]

n_records = len(voc_files)
n_train = (n_records*(100 - test_split))/100

shuffled_indices = list(range(n_records))
random.shuffle(shuffled_indices)

shuffled_voc_files = []
shuffled_img_files = []

for i in shuffled_indices:
    shuffled_voc_files.append(voc_files[i])
    shuffled_img_files.append(img_files[i])

for i, file in enumerate(shuffled_voc_files):
    path = os.path.join(voc_folder, file)
    subfolder = "train" if i < n_train else "validation"
    new_name = "img_" + str(i)

    doc = minidom.parse(path)
    e_filename = doc.getElementsByTagName("filename")[0]
    e_filename.firstChild.replaceWholeText(new_name + ".png")

    new_filename = new_name + ".xml"
    new_path = os.path.join(root_folder, subfolder, "annotations", new_filename)

    with open(new_path, "w") as xml_file:
        doc.writexml(xml_file)

for i, file in enumerate(shuffled_img_files):
    path = os.path.join(img_folder, file)
    subfolder = "train" if i < n_train else "validation"
    new_name = "img_" + str(i)
    new_filename = new_name + ".png"
    new_path = os.path.join(root_folder, subfolder, "images", new_filename)
    copy2(path, new_path)