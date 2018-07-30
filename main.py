import pandas as pd
from feature_extractor import FeatureExtractor
transitions = pd.read_csv('shared_data/new york_placenet_transitions.txt', sep=',', names=['A_id', 'B_id','A_datetime', 'B_datetime'])
ny_venues = pd.read_csv('shared_data/ny_venues.csv', sep=',')
extractor = FeatureExtractor(ny_venues, transitions)
print(extractor._get_coords(42890))
extractor.calculate_squares()
extractor.calculate_transitions()
venues_grid = extractor.venues_grid
y,x = 0, 0
print(venues_grid[y][x])
print(extractor.transitions_grid)
# print(extractor._get_density(y, x))
# print(extractor._get_number_of_category(y, x, 'Road'))
# print(extractor._get_number_categories(y, x))
# print(extractor._get_neighb_entropy(y, x))
# print(extractor.get_competitiveness(y, x, 'Road'))
# print(extractor._get_neighb_entropy(10, 10))

