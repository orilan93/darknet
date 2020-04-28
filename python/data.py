import csv

classes = []
with open("data/fish.names") as names_file:
    csv_reader = csv.reader(names_file, delimiter=',')
    classes.extend(name[0] for name in csv_reader)