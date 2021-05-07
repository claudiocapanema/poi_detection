from enum import Enum
import pytz


class PointsOfInterestConfiguration(Enum):

    # Seconds to convert datetime from UTC to SP timezone

    TZ = ("tz", 'America/Sao_Paulo', False, "timezone")

    # DBSCAN

    METERS = ("meters", 20, False, "radius in meters")
    EPSILON = ("epsilon", 0.02 / 6371.0088, "False", "epsilon")
    MIN_SAMPLES = ("min_samples", 8, "False", "minimum number of samples of a cluster")

    # PoI identification

    MAX_EVENTS_TO_CHANGE_PARAMETERS = ("max_events_to_change_parameters", 20, False, "max events to change parameters")

    # PoI classification

    TOP_N_POIS = ("top_n_pois", 7, False, "quantity of pois selected to be classified into home and work")
    HOURS_BEFORE = ("hours_before", 3, False, "hours before inactive interval")
    HOURS_AFTER = ("hours_after", 1,  False, "hours after inactivate interval")
    ONE_HOUR = ("one_hour", 1,  False, "")
    HOME_HOUR = ("home_hour", {'start': 20, 'end': 8},  False, "home time span")
    WORK_HOUR = ("work_hour", {'start': 10, 'end': 18},  False, "work time span")
    MIN_MAX_INVERTED_ROUTINE = ("min_max_inverted_routine", {'min': 8, 'max': 21}, False, "min max inverted routine")
    MIN_HOME_EVENTS = ("min_home_events", 3, False, "minimum number of poi's events to it be classified as Home")
    MIN_WORK_EVENTS = ("min_work_events", 10, False, "minimum number of poi's events to it be classified as Work")
    MIN_DAYS = ("min_days", 4, False, "minimum number of different days of a cluster to become a PoI")
    RADIUS_CLASSIFICATION = ("radius_classification", 0.02 / 6371.0088, "False", "epsilon")

    # run
    MIN_EVENTS = ("min_events", 10,  False, "minimum quantity of events necessary to a user be processed")
    MAX_EVENTS = ("max_events", 500,  False, "maximum quantity of events processed of a user")


    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_key(self):
        return self.value[0]

    def get_value(self):
        return self.value[1]