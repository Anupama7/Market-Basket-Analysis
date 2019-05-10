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
#print(os.listdir(r"C:\Users\Anupama\Desktop\InstaCart dataset"))

pd.options.display.max_rows = 20
#%matplotlib inline
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
product_ds = pd.read_csv(r"C:\Users\Anupama\Desktop\ProductAffinity\uploads\products.csv")

def random_string(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def get_response(Product_id,file_name):
    df = pd.read_csv(file_name,encoding='utf-8')
    print(Product_id)

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

@app.route('/')
def hello_world():
    return render_template("login.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        Product_id = request.form['Product_id']
        file_name = request.files['in_file']
        #data = get_response(Product_id,file_name)
        
        # return render_template("index.html", data = data)
        # return render_template("index.html", tables=[data.to_html(classes='data')], titles=data.columns.values)
        #return render_template("index.html",  tables=[data.to_html(classes='data', header="true")])
        _5,_7, _9 = get_response(Product_id,file_name)
        return render_template("index.html", _5 = _5, _7 = _7, _9 = _9,Product_id=Product_id)

if __name__ == '__main__':
   app.run(debug = True)