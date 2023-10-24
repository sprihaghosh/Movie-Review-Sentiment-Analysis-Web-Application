# Movie Review Sentiment Analysis

This project is a Movie Review Sentiment Analysis system that allows users to enter the name of a movie and get an analysis of the sentiment of its reviews. The system fetches movie reviews from the TMDb API, performs sentiment analysis on these reviews, and presents the results in a user-friendly web interface.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Features](#features)
- [File Structure](#file-structure)

## Getting Started

Follow the instructions below to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.6+
- PyTorch
- TorchText
- Flask
- Requests
- IMDbPY
- TMDb API Key

You'll need to install these dependencies to run the project successfully. You can install them using `pip` or any other package manager.

### Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-movie-reviews.git
   ```

2. Install the required packages:

   ```bash
   pip install torch torchtext flask requests imdbpy tmdbv3api matplotlib spacy
   ```

3. Set up your TMDb API key by creating an account on [TMDb](https://www.themoviedb.org/) and generating an API key. Replace the placeholder API key in `application.py`:

   ```python
   tmdb.api_key = 'your_api_key_here'
   ```

4. Run the Flask application:

   ```bash
   python application.py
   ```

5. Open a web browser and access the application at `http://127.0.0.1:5000/`.

6. Enter the name of a movie and click "Analyze" to see the sentiment analysis results.

### Features

- Sentiment analysis of movie reviews
- Fetching movie reviews from TMDb API
- User-friendly web interface
- Suggestions for movies with similar names if the movie is not found

### File Structure

The project contains the following files and folders:

- `data/`: Contains training and test data for the sentiment analysis model.
- `templates/`: Contains the `index.html` file for the web interface.
- `static/`: Stores the charts generated by the web app.
- `application.py`: The main Flask application.
- `inference.py`: Inference module for using the sentiment analysis model.
- `Sentiment_Analysis_Model.ipynb`: Jupyter Notebook containing the sentiment analysis model.
- `tut2-model.pt`: Pre-trained sentiment analysis model.

