# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:39:22 2020

@author: Karthik
"""

import pandas as pd


tripadvisor_files = [
    r"C:\Users\Karthik\Desktop\SMA_Assignment_Files\main_df_with_word_count.csv",
    r"C:\Users\Karthik\Desktop\SMA_Assignment_Files\multigram_df.csv",
    r"C:\Users\Karthik\Desktop\SMA_Assignment_Files\summarized_noun_df.csv",
    r"C:\Users\Karthik\Desktop\SMA_Assignment_Files\noun_feature_polairty.csv"
                    ]

google_files = [
    r"C:\Users\Karthik\Desktop\SMA_Assignment_Files\google_main_df_with_word_count.csv",
    r"C:\Users\Karthik\Desktop\SMA_Assignment_Files\google_multigram_df.csv",
    r"C:\Users\Karthik\Desktop\SMA_Assignment_Files\google_summarized_noun_df.csv",
    r"C:\Users\Karthik\Desktop\SMA_Assignment_Files\google_noun_feature_polairty.csv"
    ]

file_name_list = [
    r"combined_main_df_with_word_count.csv",
    r"combined_multigram_df.csv",
    r"combined_summarized_noun_df.csv",
    r"combined_noun_feature_polarity.csv"
    ]

file_path_1 = r'C:\\Users\\Karthik\\Desktop\\SMA_Assignment_Files\\'
for i in range(0,4):
    temp_list = []
    temp_df = pd.read_csv(tripadvisor_files[i], sep='|', header=0)
    temp_df['hotel_name'].replace(
                                to_replace='travelodge cardiff m4 hotel',
                                value='travelodge cardiff m4',
                                inplace=True
                                )
    temp_list.append(temp_df)
    temp_df1 = pd.read_csv(google_files[i], sep='|', header=0)
    temp_df1['hotel_name'].replace(
                                to_replace='travelodge cardiff llanedeyrn',
                                value='travelodge cardiff llanederyn',
                                inplace=True
                                )
    temp_list.append(temp_df1)
    temp_combined_df = pd.concat(temp_list, axis=0, ignore_index=True)
    file_name = file_name_list[i]
    temp_combined_df.to_csv(f"{file_path_1}" + f"{file_name}", sep='|')