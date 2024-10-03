import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')


locations = [
    'Albuquerque, New Mexico',
    'Carlsbad, California',
    'Chula Vista, California',
    'Colorado Springs, Colorado',
    'Denver, Colorado',
    'El Cajon, California',
    'El Paso, Texas',
    'Escondido, California',
    'Fresno, California',
    'La Mesa, California',
    'Las Vegas, Nevada',
    'Los Angeles, California',
    'Oceanside, California',
    'Phoenix, Arizona',
    'Sacramento, California',
    'Salt Lake City, Utah',
    'Salt Lake City, Utah',
    'San Diego, California',
    'Tucson, Arizona'
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":

            try:
                # Create the response body from the reviews and convert to a JSON byte string
                reviews = pd.read_csv('data/reviews.csv').to_dict('records')
                
                query_params = parse_qs(environ['QUERY_STRING'])

                loc_start_end_filtered_reviews = reviews.copy()
                
                if 'location' in query_params:
                    location = query_params['location'][0]
                    loc_filtered_reviews = [review for review in reviews if review.get('Location') == location]
                    reviews = loc_filtered_reviews.copy()

                
                if 'start_date' in query_params:
                    start_date = datetime.strptime(query_params['start_date'][0], '%Y-%m-%d')
                    loc_start_filtered_reviews = [review for review in reviews if datetime.strptime(review.get('Timestamp'), '%Y-%m-%d %H:%M:%S') >= start_date]
                    reviews = loc_start_filtered_reviews.copy()

                if 'end_date' in query_params:
                    end_date = datetime.strptime(query_params['end_date'][0], '%Y-%m-%d')
                    loc_start_end_filtered_reviews = [review for review in reviews if datetime.strptime(review.get('Timestamp'), '%Y-%m-%d %H:%M:%S') <= end_date]
                    reviews = loc_start_end_filtered_reviews.copy()

                for review in reviews:
                    # sentiment analysis
                    sentiment_scores = self.analyze_sentiment(review.get('ReviewBody'))
                    review['sentiment'] = sentiment_scores
                
                # reordering according to the coumpound score
                reviews = sorted(reviews, key=lambda x: x['sentiment']['compound'], reverse=True)    

                response_body = json.dumps(reviews, indent=2).encode("utf-8")

                    

                # Set the appropriate response headers
                start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
                ])
                
                return [response_body]
            except Exception as e:
                start_response("500 INTERNAL SERVER ERROR", [("Content-Type", "text/plain")])
                return [f"An error occurred: {str(e)}".encode("utf-8")]

        if environ["REQUEST_METHOD"] == "POST":
            reviews = pd.read_csv('data/reviews.csv').to_dict('records')
            # Write your code here
            try:
                request_body_size = int(environ.get("CONTENT_LENGTH", 0))
                request_body = environ["wsgi.input"].read(request_body_size).decode("utf-8")
                query_params = parse_qs(request_body)

                if 'ReviewBody' not in query_params or 'Location' not in query_params:
                    error_response = json.dumps({"error": "Missing required parameters"}).encode("utf-8")
                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(error_response)))
                    ])
                    return [error_response]
                
                if query_params['Location'][0] not in locations:
                    error_response = json.dumps({"error": "Invalid location"}).encode("utf-8")
                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(error_response)))
                    ])
                    return [error_response]

                # create a new review
                new_review = {
                    'ReviewId': str(uuid.uuid4()),
                    'ReviewBody': query_params['ReviewBody'][0],
                    'Location': query_params['Location'][0],
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # add the new review to the reviews list
                reviews.append(new_review)

                # save the new review to the csv file
                pd.DataFrame(reviews).to_csv('data/reviews.csv', index=False)

                response_body = json.dumps(new_review, indent=2).encode("utf-8")

                # Set the appropriate response headers
                start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
                ])

                return [response_body]
                
            except Exception as e:
                error_response = json.dumps({"error": str(e)}).encode("utf-8")
                print(error_response)
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(error_response)))
                ])
                return [f"An error occurred: {str(e)}".encode("utf-8")]
        

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()