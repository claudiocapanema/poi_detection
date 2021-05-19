import copy
import datetime as dt
from iteration_utilities import duplicates
from foundation.general_code.dbscan import Dbscan
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import pytz
import geopandas as gp
from foundation.util.datetimes_utils import DatetimesUtils
from model.location_type import LocationType

class PerformancePlotsDomain:

    def __init__(self):
        self.location_type = LocationType()
        self.datetime_utils = DatetimesUtils()

