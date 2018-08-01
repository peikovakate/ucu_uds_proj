import pandas as pd
from feature_extractor import FeatureExtractor
# Reading files
transitions = pd.read_csv('shared_data/new york_placenet_transitions.txt', sep=',', names=['A_id', 'B_id', 'A_datetime', 'B_datetime'])
ny_venues = pd.read_csv('shared_data/ny_venues.csv', sep=',')

extractor = FeatureExtractor(ny_venues, transitions)

# Define model target
extractor.business_name = 'Subway'
extractor.categories = ['Sandwiches', 'Fast Food']

# extractor.business_name = 'Blockbuster'
# extractor.categories = ["Video Store"]

extractor.is_cells = False
extractor.load_areas()

# extractor.calculate_areas()
# extractor.save_areas()
extractor.calculate_transitions()
extractor.calculate_features()

# Saving features data into file
extractor.save_into_file('shared_data/features.csv')
