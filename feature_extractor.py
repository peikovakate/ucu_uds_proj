import numpy as np
import pandas as pd
from math import log2

class FeatureExtractor:
    lonMin = -74.05  # minimum longitude
    lonMax = -73.8
    lonStep = 0.01  # defines cell size

    latMin = 40.65  # minimum latitude
    latMax = 40.85
    latStep = 0.0025  # defines cell size

    def __init__(self, venues, transitions):
        self.venues = {}
        for index, row in venues.iterrows():
            self.venues[row.venue_id] = row.to_dict()
        # self.venues = venues
        self.categories = venues.category.unique()
        self.transitions = transitions.values

    def _get_coords(self, venue_id):
        # venue = self.venues[self.venues.venue_id == venue_id]
        # return venue.latitude.values[0], venue.longitude.values[0]
        return self.venues[venue_id]['latitude'], self.venues[venue_id]['longitude']

    def calculate_transitions(self):
        latLen = int((self.latMax - self.latMin) / self.latStep) + 1  # number of cells on the y-axis
        lonLen = int((self.lonMax - self.lonMin) / self.lonStep) + 1  # number of cells on the x-axis
        self.transitions_grid = np.zeros((latLen, lonLen, 4))

        # [0] - area popularity
        # [1] - transition density
        # [2] - incoming flow
        # [3] - transition quality
        for i in range(0, len(self.transitions)):
            a_id = self.transitions[i][0]  # id of A place
            b_id = self.transitions[i][1]  # id of B place
            a_lat, a_long = self._get_coords(a_id)  # coordinates of A place
            b_lat, b_long = self._get_coords(b_id)  # coordinates of B place
            a_cy, a_cx = self._get_cell_indexes((a_lat, a_long))
            b_cy, b_cx = self._get_cell_indexes((b_lat, b_long))
            if(i%10000==0):
                print(i)
            if a_cy != None and b_cy != None:
                #  area popularity
                self.transitions_grid[a_cy, a_cx][0] += 1
                self.transitions_grid[b_cy, b_cx][0] += 1
                #  transition density
                if a_cx == b_cx and a_cy == b_cy:
                    self.transitions_grid[a_cy, a_cx][1] += 1
                #  incoming flow
                else:
                    self.transitions_grid[b_cy, b_cx][2] += 1
                #  transition quality


    def calculate_squares(self):
        latLen = int((self.latMax - self.latMin) / self.latStep) + 1  # number of cells on the y-axis
        lonLen = int((self.lonMax - self.lonMin) / self.lonStep) + 1  # number of cells on the x-axis

        self.venues_grid = np.zeros((latLen, lonLen), dtype=np.ndarray)
        for y in range(latLen):
            for x in range(lonLen):
                self.venues_grid[y][x] = np.array([])

        for key in self.venues:
            cy, cx = self._get_cell_indexes((self.venues[key]['latitude'], self.venues[key]['longitude']))
            if cy != None:
                self.venues_grid[cy, cx] = np.append(self.venues_grid[cy, cx], key)

    def _get_cell_indexes(self, coord):
        lat = coord[0]
        longit = coord[1]
        # if outside the grid then ingore point
        if (lat < self.latMin) or (lat > self.latMax) or (longit < self.lonMin) or (longit > self.lonMax):
            return None, None
        cx = int((longit - self.lonMin) / self.lonStep)
        cy = int((lat - self.latMin) / self.latStep)
        return cy, cx

    def _get_cell_coordinates(self, cy, cx):
        # top left corner
        lat = self.latMin+self.latStep*cy
        long = self.lonMin+self.lonStep*cx
        return lat, long

    def _get_density(self, cy, cx):
        return len(self.venues_grid[cy][cx])

    def _get_number_of_category(self, cy, cx, categories):
        cell = self.venues_grid[cy][cx]
        return len([i for i in cell if self.venues[i]['category'] in categories])
        # return len([i for i in cell if self.venues.category[self.venues.venue_id == i].values[0] in categories])

    # not needed???
    def _get_number_categories(self, cy, cx):
        cell = self.venues_grid[cy][cx]
        categories = [self.venues[self.venues.venue_id == i]['category'].values[0] for i in cell]
        # returning number of unique categories
        return len(set(categories))

    def _get_neighb_entropy(self, cy, cx):
        entropy = 0
        density = self._get_density(cy, cx)
        for cat in self.categories:
            n_c = self._get_number_of_category(cy, cx, cat)
            # log(0) = undefined
            if n_c != 0 and density!=0:
                entropy -= n_c / density * log2(n_c/density)
        return entropy

    def _get_competitiveness(self, cy, cx, category):
        density = self._get_density(cy, cx)
        if density == 0:
            return 0
        return - self._get_number_of_category(cy,cx, category)/self._get_density(cy, cx)

    def calculate_features(self, categories, business_name):
        self.features = {
            'cy': np.array([]),
            'cx': np.array([]),
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
        #check if vanues_grid calculated
        if self.venues_grid != None:
            for cy in range(len(self.venues_grid)):
                for cx in range(len(self.venues_grid[cy])):
                    print(cy, cx)
                    self._calculate_features_for_cell(cy, cx, categories, business_name)



        self.features_dataframe = pd.DataFrame(data=self.features)
        # check if transitions processed
        # if self.transitions_grid != None:
        #     pass

    def _calculate_features_for_cell(self, cy, cx, categories, business_name):
        self.features['cy'] = np.append(self.features['cy'], cy)
        self.features['cx'] = np.append(self.features['cx'], cx)
        lat, lon = self._get_cell_coordinates(cy, cx)
        self.features['latitude'] = np.append(self.features['latitude'], lat)
        self.features['longitude'] = np.append(self.features['longitude'], lon)
        # GEOGRAPHIC FEATURES
        # density
        den = self._get_density(cy, cx)
        self.features['density'] = np.append(self.features['density'], den)
        # neighbor entropy
        neight_ent = self._get_neighb_entropy(cy, cx)
        self.features['neighbors_entropy'] = np.append(self.features['neighbors_entropy'], neight_ent)
        # competitiveness
        comp = self._get_competitiveness(cy, cx, categories)
        self.features['competitiveness'] = np.append(self.features['competitiveness'], comp)

        # MOBILITY FEATURES
        # area popularity
        self.features['area_popularity'] = np.append(self.features['area_popularity'], self.transitions_grid[cy][cx][0])
        # transition density
        self.features['transition_density'] = np.append(self.features['transition_density'], self.transitions_grid[cy][cx][1])
        # incoming flow
        self.features['incoming_flow'] = np.append(self.features['incoming_flow'], self.transitions_grid[cy][cx][2])

        # aver check ins for given business
        aver_check_in = self._get_aver_check_ins_for_business(cy, cx, business_name)
        self.features['average_check_ins'] = np.append(self.features['average_check_ins'], aver_check_in)


    def save_into_file(self, filename):
        self.features_dataframe.to_csv(filename)

    def _get_aver_check_ins_for_business(self, cy, cx, name):
        check_ins = [self.venues[id]['total_check-ins'] for id in self.venues_grid[cy][cx] if self.venues[id]['title'] == name]
        if len(check_ins) != 0:
            return sum(check_ins)/len(check_ins)
        else:
            return 0







