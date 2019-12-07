import csv
import numpy as np
from glob import glob

for infile in glob("../raw/*.csv"):
    # infile = "raw/control_death.csv"

    data = []
    with open(infile, 'r') as ropen:
        reader = csv.reader(ropen)
        for row in reader:
            month, percent = row

            month = float(month)
            percent = float(percent) / 100

            data.append((month, percent))

    data = sorted(data)
    timestep = 0.5
    months = np.arange((0, timestep, data[-1][0] + timestep))
    print(months)
    exit(0)

    new_data = [(0, 1)]
    for idx in range(len(data) - 1):
        month, percent = data[idx]
        next_month, next_percent = data[idx + 1]
        if int(month) == int(next_month):
            continue

        x = int(next_month)
        if x == 0:
            continue

        y = percent + (x - month) * (next_percent - percent) / (next_month - month)

        new_data.append((x, y))

    outfile = infile.replace("raw", "interpolated")

    with open(outfile, 'w') as wopen:
        writer = csv.writer(wopen)
        for month, percent in new_data:
            writer.writerow([str(month), str(percent)])
