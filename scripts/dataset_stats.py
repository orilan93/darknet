import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("dataset_folder", help="The dataset folder.")
parser.add_argument("names_file", help="The file containing the class names.")
args = parser.parse_args()

dataset_folder = args.dataset_folder
names_file = args.names_file

with open(names_file) as f:
    names = [line.rstrip() for line in f]

stats = dict()
total_labels = 0
image_number = 0
highest_label_amount = 0

annotation_files = [f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f)) and f.split('.')[-1] == 'txt']

for file in annotation_files:
    with open(os.path.join(dataset_folder, file)) as f:
        image_number += 1
        current_label_amount = 0
        content = f.readlines()
        for line in content:
            class_number = int(line.split(' ')[0])
            class_name = names[class_number]
            stats[class_name] = stats.get(class_name, 0) + 1
            total_labels += 1
            current_label_amount += 1
        if current_label_amount > highest_label_amount:
            highest_label_amount = current_label_amount

print("Number of images:", image_number)
print("Number of labels:", total_labels)
print("Highest amount of labels:", highest_label_amount)
print(stats)