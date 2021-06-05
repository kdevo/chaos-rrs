import csv
from collections import OrderedDict
import json

d = OrderedDict()

# CSV source data from: https://www.ssa.gov/oact/babynames/limits.html
with open("name-gender.txt") as input_csv:
    reader = csv.reader(input_csv)
    for row in reader:
        name, gender, cnt = row[0], row[1], int(row[2])
        if name not in d:
            d[name] = {}
            d[name]['total'] = cnt
        else:
            d[name]['total'] += cnt
        d[name]['female' if gender == 'F' else 'male'] = cnt

output_d = OrderedDict()

for name in d:
    if 'female' not in d[name]:
        d[name]['female'] = 0
    elif 'male' not in d[name]:
        d[name]['male'] = 0
    d[name]['female_prob'] = (d[name]['female'] / d[name]['total']) * 100.
    d[name]['male_prob'] = (d[name]['male'] / d[name]['total']) * 100.
    d[name]['is_female'] = d[name]['female_prob'] > d[name]['male_prob']
    output_d[name] = {'male': round(d[name]['male_prob'], 1),
                      'female': round(d[name]['female_prob'], 1)}

with open("../data/extractor/name2gender.json", 'wt') as output_json:
    json.dump(output_d, output_json)
