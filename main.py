import pandas as pd
from feature_extractor import FeatureExtractor
# Reading files
transitions = pd.read_csv('shared_data/new york_placenet_transitions.txt', sep=',', names=['A_id', 'B_id', 'A_datetime', 'B_datetime'])
ny_venues = pd.read_csv('shared_data/ny_venues.csv', sep=',')
taxi_transitions = pd.read_csv('taxi_processing/taxi_needed.csv')

extractor = FeatureExtractor(ny_venues, transitions, taxi_transitions)

# Define model target
# extractor.business_name = 'Subway'
# extractor.categories = ['Sandwiches', 'Fast Food']
# todo rewrite for taxi
extractor.business_name = 'Blockbuster'
extractor.categories = ["Video Store"]

extractor.is_cells = False
# extractor.calculate_areas()
# extractor.save_areas()
# extractor.calculate_transitions()
# extractor.calculate_features()

extractor.calculate_taxi_features()


# Saving features data into file
extractor.taxi_save_into_file('taxi_features.csv')
