import json
import requests
import math
import pandas as pd
import numpy as np
from mendeleev import element
from periodictable import formulas

materials_codes = open("Raw_dataset/full_mpid_list_cell_size_sort.txt")
# materials_codes = open("Raw_dataset/mpid_list_cell_size_sort_end.txt")

url_base = "https://www.materialsproject.org/rest/v1/materials/"
url_tail = "/vasp/pretty_formula?API_KEY="
api_key = "TsOqaHtJ0LRPqf5jL2Hn"

output_path = "material_average_data2.csv"
formula_output_path = "all_chemical_formulas2.csv"

features = ['MPID', 'Chemical formula']
all_formulas = []

i = 0
for code in materials_codes:
	material = code.rstrip()
	url = url_base + material + url_tail + api_key
	
	if i % 500 == 0:
		print("Iteration " + str(i + 1) + "/10000")

	response = requests.get(url)
	data = response.json()
	if "error" in data or len(data["response"]) == 0: # no data available for this material
		continue

	# Getting atoms from chemical formula
	chem_formula = data["response"][0]["pretty_formula"]
	all_formulas.append([material, chem_formula])	

	i += 1

	# if i > 10:
	# 	break

print(all_formulas)

# Write stuff to output file
# with open(formula_output_path, 'w') as f:
#     for formula in all_formulas:
#         f.write("%s\n" % formula)
df = pd.DataFrame(data=all_formulas, columns=features)
print(df)
df.to_csv(path_or_buf=formula_output_path, index=False)