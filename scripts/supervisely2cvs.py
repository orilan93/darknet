import os
import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", help="The folder containing the supervisely generated annotation files.")
parser.add_argument("output_folder", help="The output folder where to generate the voc pascal formatted annotation files.")
parser.add_argument("test_split", help="The percentage of test data.")
args = parser.parse_args()

path = args.input_folder
output = args.output_folder
test_split = int(args.test_split)

dataset = []

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.split('.')[-1] == 'json']

for file in files:
    file_path = os.path.join(path, file)
    with open(file_path) as json_file:
        data = json.load(json_file)

        if "objects" not in data or data["objects"] == []:
            print("File {} is skipped because it does not have any annotated objects.".format(file))
            continue

        for obj in data["objects"]:

            record = {}
            if (len(obj["tags"]) > 0):  # Use tag as name if tag exists
                record["class"] = obj["tags"][0]["name"]
            else:
                record["class"] = obj["classTitle"]

            record["fileName"] = os.path.splitext(file)[0]

            xmin = min(obj["points"]["exterior"][0][0], obj["points"]["exterior"][1][0])
            xmax = max(obj["points"]["exterior"][0][0], obj["points"]["exterior"][1][0])
            ymin = min(obj["points"]["exterior"][0][1], obj["points"]["exterior"][1][1])
            ymax = max(obj["points"]["exterior"][0][1], obj["points"]["exterior"][1][1])

            record["height"] = ymax - ymin
            record["width"] = xmax - xmin

            record["xmax"] = xmax
            record["xmin"] = xmin
            record["ymax"] = ymax
            record["ymin"] = ymin

            dataset.append(record)

df = pd.DataFrame(dataset)

n_records = len(df)
n_train = int((n_records*(100 - test_split))/100)

df = df.sample(frac=1, random_state=0).reset_index(drop=True)

train = df[:n_train]
test = df[n_train:]

train_file_path = os.path.join(output, "train.csv")
test_file_path = os.path.join(output, "test.csv")

train.to_csv(train_file_path, index=False)
test.to_csv(test_file_path, index=False)