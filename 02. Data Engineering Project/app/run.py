import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.ensemble import _forest


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterMsgDb.db')
df = pd.read_sql_table('DisasterMsg', engine)

# load model
model = joblib.load("../models/rfc.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():    
    
    # Generating the genre count and genre names needs to create the plotly graph
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Filtering the disaster cateories, summing its values and sort in descending order
    # to generate categories from the most common to the most uncommon
    category_counts = df[df.columns[2:]].sum().sort_values(ascending=False)
    # Fixing up the columns names so each first letter is captalized and _ is replaced by spaces
    category_counts = category_counts.rename(index=lambda x: x.title().replace('_', ' '))    
    category_names = list(category_counts.index)

    # Generating the top percentile category (0.75-1.00)
    top_25_cat_count = category_counts[category_counts >= category_counts.quantile(.75)]
    top_25_cat_names = list(top_25_cat_count.index)

    # Generating the bottom percentile caegories (0 - 0.25)
    bottom_25_cat_count = category_counts[category_counts <= category_counts.quantile(.25)].sort_values()
    lower_25_cat_names = list(bottom_25_cat_count.index)
    
    # To send all graphs to the webpage I was only required to add each graph to the graph list
    graphs = [
                {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_25_cat_names,
                    y=top_25_cat_count
                )
            ],

            'layout': {
                'title': 'Top 25 percentile - In need of manpower',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"                    
                }
            }
        },
        {
            'data': [
                Bar(
                    x=lower_25_cat_names,
                    y=bottom_25_cat_count
                )
            ],

            'layout': {
                'title': 'Bottom 25 percentile - Teams can be reallocated',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"                    
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()