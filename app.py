from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gensim
from gensim.models import Word2Vec
import string
import requests
from flask import Flask, render_template, request
from werkzeug import secure_filename
import random
import sys
from itertools import combinations, groupby
from collections import Counter
from IPython.display import display
import os

pd.options.display.max_rows = 20
sns.set(style="whitegrid", palette="colorblind", font_scale=1, rc={'font.family':'NanumGothic'} )

def toReadable(v):
    value = round(v,2) if isinstance(v, float) else v

    if value < 1000:
        return str(value)
    elif value<1000000:
        return str(round(value/1000,1))+'K'
    elif value>=1000000:
        return str(round(value/1000000,1))+'M'
    return value

app = Flask(__name__)
# Importing datasets i.e. products and order_products__prior
product_ds = pd.read_csv(r"C:\Users\Anupama\Desktop\ProductAffinity\uploads\products.csv")
orders = pd.read_csv(r"C:\Users\Anupama\Desktop\ProductAffinity\uploads\order_products__prior.csv")

def random_string(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def get_response(Product_id,file_name):
   
    df = pd.read_csv(file_name,encoding='utf-8')
    #print(Product_id)   

# Clustering similar product by user's order informaiton
# Traing product2vec using word2vec 
# word = product_id
# scentence = user's order = [product_id1, product_id2, ... ]
# clustering by trained product vector
# Use only products ordered more than 100 times   
    order_product_list = df.sort_values(['user_id','order_id','add_to_cart_order'])[['order_id','product_id']].values.tolist()

    product_corpus = []
    sentence = []
    new_order_id = order_product_list[0][0]
    for (order_id, product_id) in order_product_list:
        if new_order_id != order_id:
            product_corpus.append(sentence)
            sentence = []
            new_order_id = order_id
        sentence.append(str(product_id))

    model = Word2Vec(product_corpus, window=6, size=100, workers=4, min_count=60)
    #model.save('./resource/prod2vec.100d.model')
    #model = Word2Vec.load('./resource/prod2vec.100d.model')
    def toProductName(id):
        #print(id)
        return product_ds[product_ds.product_id==id]['product_name'].values.tolist()[0]

    product_name = toProductName(product_id)
    #return product_name  
    

    def most_similar_readable(model, product_id):
        similar_list = [(product_id,1.0)]+model.wv.most_similar(str(product_id))
        
        return [( toProductName(int(id)), similarity ) for (id,similarity) in similar_list]

    def most_similar_readable(model, product_id):
        similar_list = [(product_id,1.0)]+model.wv.most_similar(str(product_id))
    
        return [( toProductName(int(id)), similarity ) for (id,similarity) in similar_list]

    ##What is the most similar?
    ##most similar to banana(24852) is .
    _5 = pd.DataFrame(most_similar_readable(model, 24852), columns=['product','similarity'])
    _7 = pd.DataFrame(most_similar_readable(model, 27845), columns=['product','similarity'])
    _9 = pd.DataFrame(most_similar_readable(model, Product_id), columns=['product','similarity'])
    

    return _5.to_html(classes='panel-control'),_7.to_html(classes='panel-control'),_9.to_html(classes='panel-control')



# Returns frequency counts for items and item pairs
def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else: 
        return pd.Series(Counter(iterable)).rename("freq")

    
# Returns number of unique orders
def order_count(order_item):
    #order_item = pd.read_csv(order_item,encoding='utf-8')
    return len(set(order_item.index))


# Returns generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    order_item = order_item.reset_index().as_matrix()
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]
              
        for item_pair in combinations(item_list, 2):
            yield item_pair
            

# Returns frequency and support associated with item
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))


# Returns name associated with item
def merge_item_name(rules, item_name):
    columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
               'confidenceAtoB','confidenceBtoA','lift']
    rules = (rules
                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
    return rules[columns]

def association_rules(order_item, min_support):
    order_item = pd.read_csv(order_item,encoding='utf-8')
   
    # Convert from DataFrame to a Series, with order_id as index and item_id as value
    order_item = order_item.set_index('order_id')['product_id'].rename('item_id')
    # print('dimensions: {0};   size: {1};   unique_orders: {2};   unique_items: {3}'
    #   .format(order_item.shape, size(order_item), len(order_item.index.unique()), len(order_item.value_counts())))


    # Calculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Filter from order_item items below min support 
    qualifying_items       = item_stats[item_stats['support'] >= min_support].index
    order_item             = order_item[order_item.isin(qualifying_items)]

    # print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    # print("Remaining order_item: {:21d}".format(len(order_item)))


    # Filter from order_item orders with less than 2 items
    order_size             = freq(order_item.index)
    qualifying_orders      = order_size[order_size >= 2].index
    order_item             = order_item[order_item.index.isin(qualifying_orders)]

    # print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    # print("Remaining order_item: {:21d}".format(len(order_item)))


    # Recalculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Get item pairs generator
    item_pair_gen          = get_item_pairs(order_item)


    # Calculate item pair frequency and support
    item_pairs              = freq(item_pair_gen).to_frame("freqAB")
    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

    # print("Item pairs: {:31d}".format(len(item_pairs)))


    # Filter from item_pairs those below min support
    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

    # print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))


    # Create table of association rules and compute relevant metrics
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)
    
    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    # Return association rules sorted by lift in descending order
    rules = item_pairs.sort_values('lift', ascending=False).head(20)
    
    
    
    ## Replace item ID with item name and display association rules
    item_name   = pd.read_csv(r"C:\Users\Anupama\Desktop\ProductAffinity\uploads\products.csv")
    item_name   = item_name.rename(columns={'product_id':'item_id', 'product_name':'item_name'})
    print(item_name)
    rules_final = pd.DataFrame(merge_item_name(rules, item_name).sort_values('lift', ascending=False))
    _11 = rules_final
    print(_11) 
    return _11.to_html(classes='main-output panel-control')

@app.route('/')
def hello_world():
    return render_template("login.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        Product_id = request.form['Product_id']
        file_name = request.files['in_file1']
        orders = request.files['in_file2']
        min_support = 0.01   
        _5,_7, _9  = get_response(Product_id, file_name)
        _11 = association_rules(orders,min_support)
        get_item_pairs(orders)
        return render_template("index.html",_5 = _5,_7 = _7,_9 = _9,_11 = _11, Product_id=Product_id)

if __name__ == '__main__':
   app.run(debug = True)