import os
import xml.etree.ElementTree as ET
import argparse
import csv
import functools
import re
import cv2
from operator import add
import matplotlib.pyplot as plt
import pandas as pd


PROB_THRESHOLD = 0.25
IOU_THRESH = 0.25

parser = argparse.ArgumentParser()
parser.add_argument("tree_file", help="The xml tree file.")
parser.add_argument("results_folder", help="The folder with the results from yolo")
parser.add_argument("manifest_file", help="The manifest file. NB! must be same as used for calculating results!")
parser.add_argument("resolution", help="The resolution of the outputted images from YOLO, format: 1920x1080")
parser.add_argument("names_file", help="The YOLO names file")
args = parser.parse_args()

class Node:
    def __init__(self,name, parent = None):
        self.name=name
        self.namename = name
        self.children=[]
        self.parent = parent

    def __repr__(self):
        return self.name

class Prediction:
    label = None
    def __init__(self, image_id, species, prob, xmin, ymin, xmax, ymax):
        self.image_id = int(image_id)
        self.species = species_nodes[species]
        self.prob = float(prob)


        self.bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
        for i in range(len(self.bbox)):
            if self.bbox[i] < 0: self.bbox[i] = 0
        #print(self.bbox)

    def __str__(self):
        return f"image_id: {self.image_id}, species: {self.species}, prob: {self.prob}, bbox: {self.bbox} |"

    def __repr__(self):
        return f"image_id: {self.image_id}, species: {self.species}, prob: {self.prob}, bbox: {self.bbox} |"

class Label:
    prediction = None
    iou = -1
    predicted = False
    tree_offset = None
    def __init__(self, image_id, species_id, rel_x, rel_y, rel_w, rel_h):
        self.image_id = int(image_id)
        self.species_id = int(species_id)

        rel_x, rel_y, rel_w, rel_h =  float(rel_x), float(rel_y), float(rel_w), float(rel_h)
        xmin = (rel_x - rel_w / 2) * resolution[0]
        ymin = (rel_y - rel_h / 2) * resolution[1]
        xmax = (rel_x + rel_w / 2) * resolution[0]
        ymax = (rel_y + rel_h / 2) * resolution[1]

        self.bbox = (xmin, ymin, xmax, ymax)

        self.species = species_nodes[species_names[self.species_id]]



    def __str__(self):
        return f"image_id: {self.image_id}, species: {self.species}, bbox: {self.bbox} | "

    def __repr__(self):
        return f"image_id: {self.image_id}, species: {self.species}, bbox: {self.bbox} | "

def depth_first(subtree, parent):
    node = Node(subtree.attrib["name"], parent)
    parent.children.append(node)
    species_nodes[node.name]=node
    for child in subtree:
        depth_first(child, node)

def get_image_id(image_path):
    p = re.compile("\/(\d*).txt")
    result = p.search(image_path)
    return result.group(1)

def in_parents(label_species, pred_species):
    #label_species = species_nodes[label_species]
    parent = label_species.parent
    parent_offset = 0
    while parent != None:
        parent_offset-=1
        if parent.name == pred_species.name:
            return True, parent_offset
        parent = parent.parent
    return False, 0

def depth_first_in_children(node, offset, pred_species):
    offset+=1
    results = []
    if len(node.children) > 0:
        for child in node.children:
            results.append(depth_first_in_children(child, offset, pred_species))
    if node.name == pred_species: return (True, offset)
    return functools.reduce(lambda a,b: b if b[0] else a ,results, (False,0))



def in_children(label_species, pred_species):
    #species_node = species_nodes[label_species]
    return depth_first_in_children(label_species,0, pred_species.name )


def check_prediction(label):
    label.predicted=False
    if label.iou <  IOU_THRESH:
        return
    if label.species == label.prediction.species:
        label.predicted = True; label.tree_offset = 0
        return
    is_in_children, tree_offset = in_children(label.species, label.prediction.species)
    if is_in_children:
        label.predicted = True;
        label.tree_offset = tree_offset
        return
    is_in_parents, tree_offset = in_parents(label.species,label.prediction.species)
    if is_in_parents:
        label.predicted = False; label.tree_offset = tree_offset
        return


#stolen from here https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

#load some misc data
resolution = (int(args.resolution.split("x")[0]),int(args.resolution.split("x")[1]))
species_names=[]
with open(args.names_file) as names_file:
    csv_reader = csv.reader(names_file, delimiter=',')
    species_names.extend(name[0] for name in csv_reader)

#load tree
tree = ET.parse(args.tree_file)
root = tree.getroot()[0]
root_class = Node(root.attrib["name"])
species_nodes = {}
species_nodes[root_class.name]=root_class
for child in root:
    depth_first(child,root_class)

# Load results
files = [f for f in os.listdir(args.results_folder)
         if os.path.isfile(os.path.join(args.results_folder, f)) and f.split('.')[-1] == 'txt']
predictions={}

for file in files:
    file_path = os.path.join(args.results_folder, file)
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        species = functools.reduce(lambda a,b: a if a in file else b,species_names)
        for row in csv_reader:
            prediction = Prediction(row[0], species, row[1], row[2], row[3],row[4],row[5])
            if(float(row[1]) < PROB_THRESHOLD): continue
            predictions.setdefault(int(row[0]), [])
            predictions[int(row[0])].append(prediction)
print("Loaded predictions:")
print(predictions)

# Load Ground Truth
labels = {}

with open(args.manifest_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        label_file = row[0][:-3]+"txt"
        image_id = get_image_id(label_file)
        with open(label_file) as csv_label_file:
            csv_label_reader = csv.reader(csv_label_file, delimiter=' ')
            for row in csv_label_reader:
                labels.setdefault(int(image_id),[])
                labels[int(image_id)].append(Label(image_id, row[0],row[1], row[2], row[3],row[4]))

print("Loaded labels:")
print(labels)

#calculate IOU

def draw_dumb_box(labels, preds):

    # load the image
    image = cv2.imread("fish.png")
    # draw the ground-truth bounding box along with the predicted
    # bounding box
    for pred in preds:
        pred_bbox = pred.bbox
        cv2.rectangle(image, (int(pred_bbox[0]),int(pred_bbox[1])),
                      (int(pred_bbox[2]), int(pred_bbox[3])), (255, 255, 255), 2)
    for key, image_labels in labels.items():
        for label in image_labels:
            label_bbox = label.bbox
            cv2.rectangle(image, (int(label_bbox[0]),int(label_bbox[1])),
                          (int(label_bbox[2]), int(label_bbox[3])), (0, 0, 255), 2)

    cv2.imshow("Image", image)
    cv2.waitKey(0)



#Assosicate labels and predictions
for image_id, image_labels in labels.items():
    for label in image_labels:
        if predictions.get(image_id) == None: continue
        best_iou = -1
        best_pred = None
        for pred in predictions[image_id]:
            iou = bb_intersection_over_union(label.bbox, pred.bbox)
            if iou > best_iou and iou > IOU_THRESH:
                best_iou = iou
                best_pred = pred
        if(best_pred is not None):
            label.iou = best_iou
            label.prediction = best_pred

            #print(f"im: {image_id} iou: {best_iou} species: {label.species} ")



labels_flat = []
for image_id, image_labels in labels.items():
    for label in image_labels:
        check_prediction(label)
        labels_flat.append(label)
        try:
            predicted_species = label.prediction.species.name
        except:
            predicted_species = "n/a"
        #print(f"im: {label.image_id} predicted: {label.predicted} iou: {label.iou} offset: {label.tree_offset} real_species: {label.species} prediceted: {predicted_species}")

#Associate lables in from prediction
for label in labels_flat:
    if label.prediction is not None:
        label.prediction.label = label


preds_flat = functools.reduce(add, predictions.values())

true_positives = list(filter(lambda e: e.predicted,labels_flat))
false_negatives = list(filter(lambda e: not e.predicted, labels_flat))
false_positive = list(filter(lambda e: e.label is None or not e.label.predicted, preds_flat))

precision = len(true_positives)/(len(true_positives)+len(false_positive))
print(f"Precision {precision}")
recall = len(true_positives)/(len(true_positives)+len(false_negatives))
print(f"Recall {recall}")


#Calulate MAP
preds_flat.sort(key=lambda e: e.prob)
preds_flat.reverse()
precision_list = []

false_negatives = list(filter(lambda e: not e.predicted, labels_flat))


for i in range(len(preds_flat)):
    true_positives = list(filter(lambda e: e.label is not None and e.label.predicted,preds_flat[:i+1]))
    false_positives =  list(filter(lambda e: e.label is None or not e.label.predicted, preds_flat[:i+1]))

    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(true_positives) + len(false_negatives))

    predicted = preds_flat[i].label is not None and preds_flat[i].label.predicted

    row = {"predicted":predicted,"precision":precision, "recall": recall}
    precision_list.append(row)

#print(precision_list)

#do interpolation
for i in range(len(precision_list)):
    r = filter(lambda e: e["recall"] >= precision_list[i]["recall"], precision_list)
    r = max(r, key = lambda e: e["precision"])
    precision_list[i]["precision_inter"] = r["precision"]
#print(precision_list)


#calc auc
drop_list = []
last = 2
for i in range(len(precision_list)):
    if precision_list[i]["precision_inter"] < last:
        drop_list.append(precision_list[i])
        last = precision_list[i]["precision_inter"]

auc = 0
for i in range(1,len(drop_list)):
    r1 = drop_list[i]["recall"]
    r2 = drop_list[i-1]["recall"]
    p = drop_list[i-1]["precision_inter"] #Here we miht want to use just percision instead
    auc += (r1-r2)*p


print(f"AUC {auc}")

#Plot
pres=[]
inter=[]
rec = []
for e in precision_list:
    pres.append(e["precision"])
    inter.append(e["precision_inter"])
    rec.append(e["recall"])

plt.plot(rec,pres)
plt.plot(rec,inter)
plt.savefig('filename.pdf', dpi=800)
plt.show()



df = pd.DataFrame([o.__dict__ for o in labels_flat])

print(f"Species: , # correctly predicted : , #correctly predicted with more specific class: , # Predicted with less specific class")

for species in species_names:
    df_pred = df[df['predicted']==True] #& (df['species'].name==species)
    df_species = df_pred['species'].apply(lambda s: s.name == species)
    df_offset = [df_pred['tree_offset'].apply(lambda s: s > 0)][0]
    z = zip(list(df_species), list(df_offset))
    df_offset_child = [all(tup) for tup in z]

    df_offset = [df_pred['tree_offset'].apply(lambda s: s < 0)][0]
    z = zip(list(df_species), list(df_offset))
    df_offset_parent = [all(tup) for tup in z]

    count_correct= sum(df_species)
    count_offset_child = sum(df_offset_child)
    count_offset_parent = sum(df_offset_parent)


    print( f"{species} & {count_correct} & {count_offset_child} & {count_offset_parent} \\\\")

pass



'''

import numpy as np
rec = [0.0, 0.00, 0.10, 0.15, 0.15, 0.25, 0.25, 0.35, 0.45, 0.45, 0.55, 0.55]
pres = [1,0.6,0.75,0.9,0.75,0.8,0.85,0.8,0.7,0.75,0.6,0.65]
plt.plot(rec,pres)
plt.savefig('lol.png', dpi=800)
plt.show()
print(rec)
'''