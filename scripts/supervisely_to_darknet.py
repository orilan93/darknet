import os
import xml.etree.ElementTree as ET
import json
import argparse
from shutil import copy2
import random
import posixpath

random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("input_dataset_folder", help="The folder containing the supervisely generated projects.")
parser.add_argument("input_xml_tree", help="The xml tree file to convert.")
parser.add_argument("output_root", help="The root output folder.")
parser.add_argument("test_split", help="The percentage of test data.")
parser.add_argument("--no_copy", help="Do not copy images.", action="store_true")
parser.add_argument("--linux", help="Generate files for linux.", action="store_true")
args = parser.parse_args()

input_dataset_folder = args.input_dataset_folder
input_xml_tree = args.input_xml_tree
output_root = args.output_root
test_split = int(args.test_split)
copy_images = True
if args.no_copy:
    copy_images = False

if args.linux:
    joinpath = posixpath.join
else:
    joinpath = os.path.join

data_folder = os.path.join(output_root, "data")
image_folder = os.path.join(data_folder, "dataset")
label_folder = os.path.join(data_folder, "dataset")

try:
    os.mkdir(data_folder)
except FileExistsError:
    pass
try:
    os.mkdir(image_folder)
except FileExistsError:
    pass
try:
    os.mkdir(label_folder)
except FileExistsError:
    pass

tree = ET.parse(input_xml_tree)
root = tree.getroot()

nodes = []
parents = []
names = []
used_classes = set()

tree = root[0]


def breadth_first(subtree):
    visited = [(subtree, -1)]

    while visited:
        visited_node = visited.pop(0)
        parents.append(visited_node[1])
        nodes.append("n" + str(len(nodes)).zfill(8))
        names.append(visited_node[0].attrib["name"])

        for child in visited_node[0]:
            parent = len(nodes) - 1
            visited.append((child, parent))


breadth_first(tree)

project_dirs = next(os.walk(input_dataset_folder))[1]

name_counter = 0

for pjdir in project_dirs:

    project_dir = os.path.join(input_dataset_folder, pjdir)

    input_ann_folder = os.path.join(project_dir, "ann")
    annotation_files = [f for f in os.listdir(input_ann_folder) if os.path.isfile(os.path.join(input_ann_folder, f)) and f.split('.')[-1] == 'json']
    included_annotation_files = []

    for file in annotation_files:
        file_path = os.path.join(input_ann_folder, file)
        with open(file_path) as json_file:
            data = json.load(json_file)

            if "objects" not in data or data["objects"] == []:
                print("File {} is skipped because it does not have any annotated objects.".format(file))
                continue

            image_width = data["size"]["width"]
            image_height = data["size"]["height"]

            filename = str(name_counter) + ".txt"
            output_path = os.path.join(label_folder, filename)
            output_file = open(output_path, 'w')

            for obj in data["objects"]:

                if (len(obj["tags"]) > 0):  # Use tag as name if tag exists
                    class_name = obj["tags"][0]["name"]
                else:
                    class_name = obj["classTitle"]

                try:
                    class_index = names.index(class_name)
                except ValueError:
                    print("Class {} does not exist in the xml file.".format(class_name))
                    continue

                xmin = min(obj["points"]["exterior"][0][0], obj["points"]["exterior"][1][0])
                xmax = max(obj["points"]["exterior"][0][0], obj["points"]["exterior"][1][0])
                ymin = min(obj["points"]["exterior"][0][1], obj["points"]["exterior"][1][1])
                ymax = max(obj["points"]["exterior"][0][1], obj["points"]["exterior"][1][1])

                box_width = xmax - xmin
                box_height = ymax - ymin

                center_x = xmin + (box_width / 2)
                center_y = ymin + (box_height / 2)

                rel_x = center_x / image_width
                rel_y = center_y / image_height
                rel_w = box_width / image_width
                rel_h = box_height / image_height

                used_classes.add(class_name)

                output_file.write("{} {} {} {} {}\n".format(class_index, rel_x, rel_y, rel_w, rel_h))

            included_annotation_files.append((file, name_counter))
            name_counter += 1
            output_file.close()


    input_img_folder = os.path.join(project_dir, "img")
    included_image_files = [(os.path.splitext(f[0])[0], f[1]) for f in included_annotation_files]
    for image_file in included_image_files:
        image_path = os.path.join(input_img_folder, image_file[0])
        new_path = os.path.join(image_folder, str(image_file[1]) + os.path.splitext(image_file[0])[1])
        if copy_images:
            copy2(image_path, new_path)


unused_classes = list(set(names) - used_classes)
for unused in unused_classes:
    print("Class \"{}\" was included in the tree file, but there is no annotation with this class.".format(unused))


dataset_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.split('.')[-1] == 'png']


n_records = len(dataset_files)
n_train = int((n_records*(100 - test_split))/100)

random.shuffle(dataset_files)

with open(os.path.join(data_folder, "train.manifest"), "w") as train_manifest:
    train_manifest.write("\n".join([joinpath("data", "dataset", file) for file in dataset_files[:n_train]]))

with open(os.path.join(data_folder, "test.manifest"), "w") as test_manifest:
    test_manifest.write("\n".join([joinpath("data", "dataset", file) for file in dataset_files[n_train:]]))


with open(os.path.join(data_folder, "fish.data"), "w") as data_file:
    data_file.write("classes = {}\n".format(len(names)))
    data_file.write("train = {}\n".format(joinpath("data", "train.manifest")))
    data_file.write("valid = {}\n".format(joinpath("data", "test.manifest")))
    data_file.write("names = {}\n".format(joinpath("data", "fish.names")))
    data_file.write("backup = backup\n")


file_parents = open(os.path.join(data_folder, "fish.tree"), "w")
file_names = open(os.path.join(data_folder, "fish.names"), "w")
file_labels = open(os.path.join(data_folder, "fish.labels"), "w")
for i in range(len(nodes)):
    file_parents.write("{} {}\n".format(nodes[i], parents[i]))
    file_names.write("{}\n".format(names[i]))
    file_labels.write("{}\n".format(nodes[i]))
file_parents.close()
file_names.close()
file_labels.close()