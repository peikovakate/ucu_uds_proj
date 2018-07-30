import numpy as np
import pandas as pd
from math import log2
from geopy import distance

class FeatureExtractor:

    lonMin = -74.05  # minimum longitude
    lonMax = -73.8
    lonStep = 0.01  # defines cell size

    latMin = 40.65  # minimum latitude
    latMax = 40.85
    latStep = 0.0025  # defines cell size

    categories = []
    business_name = ''
    is_cells = True
    radius = 200

    def __init__(self, venues, transitions):
        self.venues = {}
        for index, row in venues.iterrows():
            self.venues[row.venue_id] = row.to_dict()
        # self.venues = venues
        self.categories = venues.category.unique()
        self.transitions = transitions.values

    def _get_coords(self, venue_id):
        return self.venues[venue_id]['latitude'], self.venues[venue_id]['longitude']

    def calculate_transitions(self):
        print('calc transitions')
        if self.is_cells:
            latLen = int((self.latMax - self.latMin) / self.latStep) + 1  # number of cells on the y-axis
            lonLen = int((self.lonMax - self.lonMin) / self.lonStep) + 1  # number of cells on the x-axis
            self.transitions_grid = np.zeros((latLen, lonLen, 4))
        else:
            business_n = len(self.business_ids)
            self.transitions_grid = np.zeros((business_n, 4))

        # [0] - area popularity
        # [1] - transition density
        # [2] - incoming flow
        # [3] - transition quality
        for i in range(0, len(self.transitions)):
            a_id = self.transitions[i][0]  # id of A place
            b_id = self.transitions[i][1]  # id of B place
            if self.is_cells:
                a_lat, a_long = self._get_coords(a_id)  # coordinates of A place
                b_lat, b_long = self._get_coords(b_id)  # coordinates of B place
                a_index = self._get_area_index((a_lat, a_long))
                b_index = self._get_area_index((b_lat, b_long))
            else:
                a_index = self._get_circle_index(a_id)
                b_index = self._get_circle_index(b_id)
            if(i%10000 == 0):
                print(i)
            if a_index != None and b_index != None:
                #  area popularity
                self.transitions_grid[a_index][0] += 1
                self.transitions_grid[b_index][0] += 1
                #  transition density
                if a_index == b_index:
                    self.transitions_grid[a_index][1] += 1
                #  incoming flow
                else:
                    self.transitions_grid[b_index][2] += 1
                #  transition quality

    def _get_ids_of_business(self):
        ids = [venue['venue_id'] for venue in self.venues.values() if venue['title'] == self.business_name]
        return ids

    def _calc_rects(self):
        latLen = int((self.latMax - self.latMin) / self.latStep) + 1  # number of cells on the y-axis
        lonLen = int((self.lonMax - self.lonMin) / self.lonStep) + 1  # number of cells on the x-axis

        self.venues_grid = np.zeros((latLen, lonLen), dtype=np.ndarray)
        for y in range(latLen):
            for x in range(lonLen):
                self.venues_grid[y][x] = np.array([])

        for key in self.venues:
            index = self._get_area_index((self.venues[key]['latitude'], self.venues[key]['longitude']))
            if index != None:
                self.venues_grid[index] = np.append(self.venues_grid[index], key)

    def _calc_circles(self):
        print('calc circles')
        self.business_ids = self._get_ids_of_business()
        self.business_venues = [self.venues[key] for key in self.business_ids]
        n_business = len(self.business_ids)
        self.venues_grid = np.zeros(n_business, dtype=np.ndarray)
        for index in range(n_business):

            b_id = self.business_ids[index]

            business_coords = (self.venues[b_id]['latitude'], self.venues[b_id]['longitude'])
            self.venues_grid[index] = self._get_places_around(business_coords)
            print(index, b_id, len(self.venues_grid[index]))

    def calculate_areas(self):
        print('calc areas')
        if self.is_cells:
            self._calc_rects()
        else:
            self._calc_circles()

    def _get_area_index(self, coord):
        lat = coord[0]
        longit = coord[1]
        if self.is_cells:
            # if outside the grid then ingore point
            if (lat < self.latMin) or (lat > self.latMax) or (longit < self.lonMin) or (longit > self.lonMax):
                return None
            cx = int((longit - self.lonMin) / self.lonStep)
            cy = int((lat - self.latMin) / self.latStep)
            return (cy, cx)
        # DO NOT use this function for circle areas
        else:
            print('_get_area_index used for circle')
            return 0

    # what if venue in more then one circle???
    def _get_circle_index(self, venue_id):
        for circle_index in range(len(self.venues_grid)):
            if venue_id in self.venues_grid[circle_index]:
                return circle_index
        return None

    def _get_cell_coordinates(self, index):
        if self.is_cells:
            cy = index[0]
            cx = index[1]
            # top left corner
            lat = self.latMin+self.latStep*cy
            long = self.lonMin+self.lonStep*cx
        else:
            venue = self.venues[self.business_ids[index]]
            lat = venue['latitude']
            long = venue['longitude']
        return lat, long

    def _get_density(self, index):
        return len(self.venues_grid[index])

    def _get_number_of_category(self, index, categories):
        cell = self.venues_grid[index]
        return len([i for i in cell if self.venues[i]['category'] in categories])
        # return len([i for i in cell if self.venues.category[self.venues.venue_id == i].values[0] in categories])

    # not needed???
    def _get_number_categories(self, index):
        cell = self.venues_grid[index]
        categories = [self.venues[self.venues.venue_id == i]['category'].values[0] for i in cell]
        # returning number of unique categories
        return len(set(categories))

    def _get_neighb_entropy(self, index):
        entropy = 0
        density = self._get_density(index)
        for cat in self.categories:
            n_c = self._get_number_of_category(index, cat)
            # log(0) = undefined
            if n_c != 0 and density!=0:
                entropy -= n_c / density * log2(n_c/density)
        return entropy

    def _get_competitiveness(self, index, category):
        density = self._get_density(index)
        if density == 0:
            return 0
        return - self._get_number_of_category(index, category)/self._get_density(index)

    def _calc_features_for_circles(self):
        for index in range(len(self.venues_grid)):
            print(index)
            self.features['venue_id'] = np.append(self.features['venue_id'], self.business_ids[index])
            self._calculate_features_for_area(index)

    def _calc_features_for_rectangles(self):
        for cy in range(len(self.venues_grid)):
            for cx in range(len(self.venues_grid[cy])):
                print(cy, cx)
                self.features['cy'] = np.append(self.features['cy'], cy)
                self.features['cx'] = np.append(self.features['cx'], cx)
                self._calculate_features_for_area((cy, cx))

    def calculate_features(self):
        print('calc features')
        self.features = {
            'latitude': np.array([]),
            'longitude': np.array([]),
            'density': np.array([]),
            'neighbors_entropy': np.array([]),
            'competitiveness': np.array([]),
            'area_popularity': np.array([]),
            'transition_density': np.array([]),
            'incoming_flow': np.array([]),
            'average_check_ins': np.array([]),
        }
        if self.is_cells:
            self.features['cy'] = np.array([])
            self.features['cx'] = np.array([])
            self._calc_features_for_rectangles()
        else:
            self.features['venue_id'] = np.array([])
            self._calc_features_for_circles()

        self.features_dataframe = pd.DataFrame(data=self.features)

    def _calculate_features_for_area(self, index):
        lat, lon = self._get_cell_coordinates(index)
        self.features['latitude'] = np.append(self.features['latitude'], lat)
        self.features['longitude'] = np.append(self.features['longitude'], lon)
        # GEOGRAPHIC FEATURES
        # density
        den = self._get_density(index)
        self.features['density'] = np.append(self.features['density'], den)
        # neighbor entropy
        neight_ent = self._get_neighb_entropy(index)
        self.features['neighbors_entropy'] = np.append(self.features['neighbors_entropy'], neight_ent)
        # competitiveness
        comp = self._get_competitiveness(index, self.categories)
        self.features['competitiveness'] = np.append(self.features['competitiveness'], comp)

        # MOBILITY FEATURES
        # area popularity
        self.features['area_popularity'] = np.append(self.features['area_popularity'], self.transitions_grid[index][0])
        # transition density
        self.features['transition_density'] = np.append(self.features['transition_density'], self.transitions_grid[index][1])
        # incoming flow
        self.features['incoming_flow'] = np.append(self.features['incoming_flow'], self.transitions_grid[index][2])

        # aver check ins for given business
        aver_check_in = self._get_aver_check_ins_for_business(index, self.business_name)
        self.features['average_check_ins'] = np.append(self.features['average_check_ins'], aver_check_in)


    def save_into_file(self, filename):
        self.features_dataframe.to_csv(filename)

    def _get_aver_check_ins_for_business(self, index, name):
        check_ins = [self.venues[id]['total_check-ins'] for id in self.venues_grid[index] if self.venues[id]['title'] == name]
        if len(check_ins) != 0:
            return sum(check_ins)/len(check_ins)
        else:
            return 0


    def _get_places_around(self, center_coordinates):
        # circle area
        places_ids = []
        for key in self.venues:
            dist = distance.distance(center_coordinates, (self.venues[key]['latitude'], self.venues[key]['longitude'])).m
            if dist <= self.radius:
                places_ids.append(self.venues[key]['venue_id'])
        return places_ids







