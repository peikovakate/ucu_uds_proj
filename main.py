import pandas as pd
from feature_extractor import FeatureExtractor
transitions = pd.read_csv('shared_data/new york_placenet_transitions.txt', sep=',', names=['A_id', 'B_id', 'A_datetime', 'B_datetime'])
ny_venues = pd.read_csv('shared_data/ny_venues.csv', sep=',')
extractor = FeatureExtractor(ny_venues, transitions)

extractor.calculate_squares()
extractor.calculate_transitions()

extractor.calculate_features(['Sandwiches', 'Fast Food'], 'Subway')
extractor.save_into_file('features.csv')
