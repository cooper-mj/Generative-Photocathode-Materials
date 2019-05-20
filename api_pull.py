import json
import requests
import pandas as pd
import numpy as np

materials_codes = open("Raw_dataset/full_mpid_list_cell_size_sort.txt")

url_base = "https://www.materialsproject.org/rest/v1/materials/"
url_tail = "/vasp/final_structure?API_KEY="
api_key = "TsOqaHtJ0LRPqf5jL2Hn"

features = ["a", "b", "c", "alpha", "beta", "gamma", "volume"]

features_all_elements = []

i = 0
for code in materials_codes:
	material = code.rstrip() #Add a string for it
	url = url_base + material + url_tail + api_key
	print(url)

	response = requests.get(url)
	print(response)
	data = response.json()
	
	material_features = np.zeros(len(features))
	for j, feature in enumerate(features):
		material_features[j] = data["response"][0]["final_structure"]["lattice"][feature]

	features_all_elements.append(material_features)

	if i > 5:
		break
	i += 1
print(features_all_elements)

df = pd.DataFrame(data=features_all_elements, columns = features)
print(df)


# Dataframe to csv