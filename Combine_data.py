import pandas as pd
import os

def transform(path):    
    header_list = ["Date", "Price"]
    df1 = pd.read_csv(path+"/Y1.csv", names=header_list)
    df3 = pd.read_csv(path+"/Y3.csv", names=header_list)
    df5 = pd.read_csv(path+"/Y5.csv", names=header_list)
    df_concat = pd.concat([df1, df3, df5], ignore_index=False)
    # Transform
    result = pd.DataFrame()
    result['Date'] = pd.to_datetime([f'{y}-{m}-{d}' for m, d, y in zip(df_concat['Date'].str[4:7], df_concat['Date'].str[8:10], df_concat['Date'].str[11:15])])
    result['Price'] = (df_concat['Price']).str[1:].values.astype('float')
    result.sort_values(by=['Date'], axis=0, ascending=True, inplace=True)
    result.drop_duplicates(inplace=True)
    return result

datasets_path = "Datasets/Price_chart"

dir_list = os.listdir(datasets_path)

for dir in dir_list:
    combine = transform(datasets_path+'/'+dir)
    combine.to_csv(datasets_path+'/'+dir+'/'+"Combine_all.csv")
