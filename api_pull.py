import json
import requests
import pandas as pd

materials_codes = open("Raw_dataset/full_mpid_list_cell_size_sort.txt")

url_base = "https://www.materialsproject.org/rest/v1/materials/"
url_tail = "/vasp/initial_structure?API_KEY="
api_key = "TsOqaHtJ0LRPqf5jL2Hn"

for code in materials_codes:
	print(code)
	material = code.rstrip() #Add a string for it

	url = url_base + material + url_tail + api_key
	print(url)

	response = requests.get(url)
	print(response)
	data = response.json()
	print(data)


# Dataframe to csv