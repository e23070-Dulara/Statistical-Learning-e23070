import pandas as pd, requests, io

url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
raw_df = pd.read_csv(io.StringIO(requests.get(url).text),
                     na_values=["","N/A","NA","NULL","null","?"], low_memory=False)
