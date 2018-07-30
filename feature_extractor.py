import numpy as np
from math import log10
#gg

class FeatureExtractor:
    lonMin = -74.1  # minimum longitude
    lonMax = -73.7
    lonStep = 0.01  # defines cell size

    latMin = 40.6  # minimum latitude
    latMax = 41.0
    latStep = 0.01  # defines cell size

    def __init__(self, venues, transitions):
        self.venues = venues
        self.categories = venues.category.unique()
        self.transitions = transitions

    def _get_coords(self, venue_id):
        venue = self.venues[self.venues.venue_id == venue_id]
        return venue.latitude.values[0], venue.longitude.values[0]

    def calculate_transitions(self):
        latLen = int((self.latMax - self.latMin) / self.latStep) + 1  # number of cells on the y-axis
        lonLen = int((self.lonMax - self.lonMin) / self.lonStep) + 1  # number of cells on the x-axis
        self.transitions_grid = np.zeros((latLen, lonLen, 4))
        # [0] - area popularity
        # [1] - transition density
        # [2] - incoming flow
        # [3] - transition quality
        for index, row in self.transitions.iterrows():
            a_id = row.A_id     #id of A place
            b_id = row.B_id     #id of B place
            a_lat, a_long = self._get_coords(a_id) #coordinates of A place
            b_lat, b_long = self._get_coords(b_id) #coordinates of B place
            a_cy, a_cx = self._get_cell_indexes((a_lat, a_long))
            b_cy, b_cx = self._get_cell_indexes((b_lat, b_long))
            # print(index, a_id, b_id)
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

        for index, row in self.venues.iterrows():
            cy, cx = self._get_cell_indexes((row.latitude, row.longitude))
            if cy != None:
                self.venues_grid[cy, cx] = np.append(self.venues_grid[cy, cx], row.venue_id)

    def _get_cell_indexes(self, coord):
        lat = coord[0]
        longit = coord[1]
        # if outside the grid then ingore point
        if (lat < self.latMin) or (lat > self.latMax) or (longit < self.lonMin) or (longit > self.lonMax):
            return None, None
        cx = int((longit - self.lonMin) / self.lonStep)
        cy = int((lat - self.latMin) / self.latStep)
        return cy, cx

    def _get_density(self, cy, cx):
        return len(self.venues_grid[cy][cx])

    def _get_number_of_category(self, cy, cx, category):
        cell = self.venues_grid[cy][cx]
        return len([i for i in cell if self.venues.category[self.venues.venue_id == i].values[0] == category])

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
            if n_c != 0:
                entropy -= n_c / density * log10(n_c/density)
        return entropy

    def get_competitiveness(self, cy, cx, category):
        return - self._get_number_of_category(cy,cx, category)/self._get_density(cy, cx)



