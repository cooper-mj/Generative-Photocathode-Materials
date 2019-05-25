import json
import requests
import pandas as pd
import numpy as np
from mendeleev import element

UNIT_CELL_PARAMS = 7
MAX_ATOMS = 10

materials_codes = open("Raw_dataset/full_mpid_list_cell_size_sort.txt")

url_base = "https://www.materialsproject.org/rest/v1/materials/"
url_tail = "/vasp/final_structure?API_KEY="
api_key = "TsOqaHtJ0LRPqf5jL2Hn"

output_path = "data/unit_cell_data.csv"

features = ["MPID" , 
			"a", "b", "c", "alpha", "beta", "gamma", "volume",	  # unit cell params
			"elem0", "a0", "b0", "c0", "elem1", "a1", "b1", "c1", # species params
			"elem2", "a2", "b2", "c2", "elem3", "a3", "b3", "c3",
			"elem4", "a4", "b4", "c4", "elem5", "a5", "b5", "c5",
			"elem6", "a6", "b6", "c6", "elem7", "a7", "b7", "c7",
			"elem8", "a8", "b8", "c8", "elem9", "a9", "b9", "c9"] 

# features = ["MPID", "a", "b", "c", "alpha", "beta", "gamma", "volume",	  # unit cell params
# 			"elem0", "a0", "b0", "c0", "elem1", "a1", "b1", "c1"] 

features_all_elements = []

i = 0
for code in materials_codes:
	material = code.rstrip() # Add a string for it
	url = url_base + material + url_tail + api_key

	response = requests.get(url)
	data = response.json()
	if len(data["response"]) == 0: # No data for this material available
		continue
	
	material_features = np.zeros(len(features))
	material_features[0] = float(code[3:])

	# Collecting unit cell features (dimensions, angles, volume)
	curr_data = data["response"][0]["final_structure"]
	feat = 1

	for unit_cell_param in features[1:UNIT_CELL_PARAMS + 1]:
		material_features[feat] = curr_data["lattice"][unit_cell_param]
		feat += 1
	
	# Collecting sites-related features (atom type, coordinates)
	curr_sites_data = curr_data["sites"]
	spec = 0

	for site in curr_sites_data[:MAX_ATOMS]:
		print(site)
		elem_symb = site["species"][0]["element"]
		material_features[feat] = element(elem_symb).atomic_number
		feat += 1

		coords = site["abc"]
		for coord in coords:
			material_features[feat] = coord
			feat += 1

	# print(material_features)
	features_all_elements.append(material_features)

	i += 1

	#if i > 5:
	#	break

# print(features_all_elements)

df = pd.DataFrame(data=features_all_elements, columns = features)
print(df)
df.to_csv(path_or_buf=output_path, index=False)
