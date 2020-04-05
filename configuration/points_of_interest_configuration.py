from enum import Enum
import pytz


class PointsOfInterestConfiguration(Enum):

    # Seconds to convert datetime from UTC to SP timezone

    UTC_TO_SP = ("utc_to_sp", 3*3600, False, "Seconds to convert datetime from UTC to SP timezone, 3 hours of difference")

    TZ = ("tz", pytz.timezone('America/Sao_Paulo'), False, "timezone")

    # DBSCAN

    METERS = ("meters", 10, False, "radius in meters")
    EPSILON = ("epsilon", 0.01 / 6371.0088, "False", "epsilon")

    # PoI identification

    MAX_EVENTS_TO_CHANGE_PARAMETERS = ("max_events_to_change_parameters", 20, False, "max events to change parameters")

    # PoI classification

    TOP_N_POIS = ("top_n_pois", 6, False, "quantity of pois selected to be classified into home and work")
    HOURS_BEFORE = ("hours_before", 3, False, "hours before inactive interval")
    HOURS_AFTER = ("hours_after", 1,  False, "hours after inactivate interval")
    ONE_HOUR = ("one_hour", 1,  False, "")
    HOME_HOUR = ("home_hour", {'start': 20, 'end': 8},  False, "home time span")
    WORK_HOUR = ("work_hour", {'start': 10, 'end': 18},  False, "work time span")
    MIN_MAX_INVERTED_ROUTINE = ("min_max_inverted_routine", {'min': 8, 'max': 21}, False, "min max inverted routine")

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