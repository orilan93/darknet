import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import re

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", help="The csv file containing the validation results.")
args = parser.parse_args()


def epoch_column_converter(c):
    m = re.search('(\d+)(?!.*\d)', c)
    if m:
        return m.group(1)


df = pd.read_csv(args.csv_file,
                 converters={"epoch": epoch_column_converter})
df.fillna(1, inplace=True)
df.epoch = pd.to_numeric(df.epoch)

row_max_map = df["result"].idxmin()
map_max_x = df["epoch"][row_max_map]
df.set_index("epoch", inplace=True)
max_map = df["result"][map_max_x]

ax = plt.gca()
ax.set_ylim(0, 1)
plt.axvline(x=map_max_x, linestyle='--', color="black")
df.plot(kind='line', ax=ax)

plt.savefig(os.path.splitext(args.csv_file)[0] + ".png")
#plt.show()

print("Epoch ", map_max_x)
print("result ", max_map)
