from enum import Enum
import pytz


class PointsOfInterestValidationConfiguration(Enum):

    # Radius for the nearestneighbors algorithm - 100m

    RADIUS = ("radius", 0.1 / 6371, False, "radius in meters for the nearestneighbors alogrithm")

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_key(self):
        return self.value[0]

    def get_value(self):
        return self.value[1]