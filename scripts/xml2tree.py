import os
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="The xml tree file to convert.")
parser.add_argument("output_file", help="The name of the output path and file for the generated yolo tree file.")
args = parser.parse_args()

path = args.input_file
output = args.output_file

tree = ET.parse(path)
root = tree.getroot()

nodes = []
parents = []
names = []

tree = root[0]

def depth_first(subtree, parent=-1):
    parents.append(parent)
    parent = len(nodes)
    nodes.append("n" + str(len(nodes)).zfill(8))
    names.append(subtree.attrib["norwegian"])
    for child in subtree:
        depth_first(child, parent=parent)


depth_first(tree)

file_parents = open(output+".tree", "w")
file_names = open(output+".names", "w")
file_labels = open(output+".labels", "w")
for i in range(len(nodes)):
    file_parents.write("{} {}\n".format(nodes[i], parents[i]))
    file_names.write("{}\n".format(names[i]))
    file_labels.write("{}\n".format(nodes[i]))
file_parents.close()
file_names.close()
file_labels.close()