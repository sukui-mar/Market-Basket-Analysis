# Install it if not installed 
!pip install apyori

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from apyori import apriori

df = pd.read_csv('../input/groceries-dataset/Groceries_dataset.csv', parse_dates=['Date'])
df.shape
df.columns

df.isnull().any()

all_products = df['itemDescription'].unique()
print("Total products: {}".format(len(all_products)))


all_products = df['itemDescription'].unique()
print("Total products: {}".format(len(all_products)))

df['Member_number'].value_counts()

def ditribution_plot(x,y,name=None,xaxis=None,yaxis=None):
    fig = go.Figure([
        go.Bar(x=x, y=y)
    ])

    fig.update_layout(
        title_text=name,
        xaxis_title=xaxis,
        yaxis_title=yaxis
    )
    fig.show()
    
    x = df['itemDescription'].value_counts()
x = x.sort_values(ascending = False) 
x = x[:10]

ditribution_plot(x=x.index, y=x.values, yaxis="Count", xaxis="Products")



one_hot = pd.get_dummies(df['itemDescription'])
df.drop('itemDescription', inplace=True, axis=1)
df = df.join(one_hot)
df.head()

# Note: if a customer bought multiple products on same day, We will consider it one transaction
records = df.groupby(["Member_number","Date"])[all_products[:]].apply(sum)
records = records.reset_index()[all_products]


## Replacing non-zero values with product names
def get_Pnames(x):
    for product in all_products:
        if x[product] > 0:
            x[product] = product
    return x

records = records.apply(get_Pnames, axis=1)
records.head()

print("Total Transactions: {}".format(len(records)))

## Removing zeros
x = records.values
x = [sub[~(sub == 0)].tolist() for sub in x if sub[sub != 0].tolist()]
transactions = x

transactions[0:10]


rules = apriori(transactions,min_support=0.00030,min_confidance=0.05,min_lift=3,min_length=2,target="rules")
association_results = list(rules)

for item in association_results:

    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    print("Support: " + str(item[1]))

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
