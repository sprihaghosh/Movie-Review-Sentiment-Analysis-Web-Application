
# -*- coding: utf-8 -*-

import torch
import inference
from torchtext.legacy import data
import matplotlib.pyplot as plt
import time
from imdb import IMDb
from tmdbv3api import TMDb
from tmdbv3api import Movie
from flask import Flask, render_template, request, jsonify
import requests

# Initialize the TMDb API client
tmdb = TMDb()
tmdb.api_key = '44edf568e2edce38e5154e76525bcede'

# Initialize the IMDb library
ia = IMDb()

#Define the TEXT field
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)

app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

model = inference.get_model()

@app.route('/')
def index():
    return render_template('index.html', sentiment_results = None)

@app.route('/analyze', methods=['POST'])
def analyze():
    
    sentiment_results = None  # Initialize the variable here
    
    movie_name = request.form.get('movie_name')
    
    print(f"Movie Name: {movie_name}")  

    # Fetch movie reviews from the TMDb API
    reviews = fetch_reviews(movie_name)
    
    if not reviews:
        # If reviews were not found, suggest movies with similar names
        suggested_movies = suggest_similar_movies(movie_name)
        return render_template('index.html', sentiment_results={"error": "Movie not found", "suggested_movies": suggested_movies})

    # Perform sentiment analysis on the reviews
    sentiment_results = analyze_sentiment(reviews)
    
    # Generate and save the bar chart
    chart_image_path = generate_bar_chart(sentiment_results)

    # Pass sentiment_results to the template and render it
    return render_template('index.html', sentiment_results=sentiment_results, chart_image=chart_image_path, movie_name=movie_name)


def fetch_reviews(movie_name):
   #Search for the movie using the TMDb API
    search_url = 'https://api.themoviedb.org/3/search/movie'
    search_params = {
        'api_key': tmdb.api_key,
        'query': movie_name
    }
    search_response = requests.get(search_url, params=search_params)
    search_data = search_response.json()

    # Check if any movies were found
    if search_data.get('results'):
        # Get the ID of the first movie in the search results
        movie_id = search_data['results'][0]['id']

        #Fetch reviews for the movie using the TMDb API
        reviews_url = f'https://api.themoviedb.org/3/movie/{movie_id}/reviews'
        reviews_params = {
            'api_key': tmdb.api_key
        }
        reviews_response = requests.get(reviews_url, params=reviews_params)
        reviews_data = reviews_response.json()
        # Extract the "content" from each review and store it in a list
        review_contents = [review['content'] for review in reviews_data['results']]
        return review_contents
    else:
        # No movies found
        return []

def analyze_sentiment(reviews):
    positive_reviews = 0
    neutral_reviews = 0
    negative_reviews = 0

    for review in reviews:
        # Preprocess the review text 
        text = inference.preprocess(review)
        len_txt = inference.length(review)
        
        # Use the loaded model to predict sentiment
        prediction = torch.sigmoid(model.predict_sentiment(text, len_txt))
        
        # Use the model's output to determine sentiment
        #sentiment = "positive" if prediction > 0 else "negative"
        if (prediction.item()<0.3):
            sentiment = "positive"
        
        elif (prediction.item()>0.3 and prediction.item()<0.7):
            sentiment = "neutral"
            
        else:
            sentiment = "negative"
            
        if sentiment == "positive":
            positive_reviews += 1
        elif sentiment == "neutral":
            neutral_reviews += 1
        else:
            negative_reviews += 1

    total_reviews = len(reviews)
    positive_percentage = round((positive_reviews / total_reviews) * 100, 2)
    neutral_percentage = round((neutral_reviews / total_reviews) * 100, 2)
    negative_percentage = round((negative_reviews / total_reviews) * 100, 2)

    return {'positive': positive_percentage, 'neutral': neutral_percentage, 'negative': negative_percentage}

def suggest_similar_movies(movie_name):
    # Use the IMDb library to search for movies with similar names
    results = ia.search_movie(movie_name)
    similar_movie_names = [movie['title'] for movie in results]

    # Limit the number of suggested movies 
    num_suggestions = 5
    return similar_movie_names[:num_suggestions]


def generate_bar_chart(sentiment_results):
    labels = ['Positive', 'Neutral', 'Negative']
    percentages = [sentiment_results['positive'],sentiment_results['neutral'] ,sentiment_results['negative']]
    
    # Define custom colors for the bars
    colors = ['#4CAF50', '#808080', '#FF5733']
    
    # Create a figure and axis for the bar chart
    fig, ax = plt.subplots()

    # Customize the bar chart properties
    ax.bar(labels, percentages, color=colors)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Percentage')
    ax.set_title('Sentiment Analysis Results')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add labels to the bars
    for i, v in enumerate(percentages):
        ax.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    # Save the chart as an image with a unique filename based on timestamp
    timestamp = int(time.time())  # Get the current timestamp
    unique_chart_path = f'static/chart_{timestamp}.png'
    
    # Save the new chart as an image
    plt.tight_layout()  # Adjust spacing to prevent labels from being cut off
    plt.savefig(unique_chart_path, bbox_inches='tight')

    return unique_chart_path  # Return the path to the chart


if __name__ == '__main__':
    app.run(debug=True)


