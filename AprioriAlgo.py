import pandas as pd
import numpy as np


encoded_df = pd.read_csv(r'C:\datasets\Encoded_data.csv')


def apriori_basic(df, min_support, use_colnames=False):
    item_support_dict = {}
    num_records = len(df)
    for index, row in df.iterrows():
        for item in row[row > 0].index:
            if item in item_support_dict:
                item_support_dict[item] += 1
            else:
                item_support_dict[item] = 1

    qualifying_items = {item: support / num_records for item,
                        support in item_support_dict.items() if support / num_records >= min_support}

    if use_colnames:
        frequent_itemsets = pd.DataFrame(
            list(qualifying_items.items()), columns=["itemsets", "support"])
    else:
        frequent_itemsets = pd.DataFrame(
            list(qualifying_items.items()), columns=["itemsets", "support"])
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(
            lambda x: frozenset([df.columns.tolist().index(x)]))

    return frequent_itemsets


one_hot_encoded_df = encoded_df.drop(['Invoice ID', 'Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs',
                                      'gross margin percentage', 'gross income', 'Rating', 'Datetime'], axis=1)

frequent_itemsets = apriori_basic(
    one_hot_encoded_df, min_support=0.01, use_colnames=True)


print(frequent_itemsets.head())
