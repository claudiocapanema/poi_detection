#!/usr/bin/python
# -*- coding: utf-8 -*-

from copy import copy
import numpy as np
from configuration.weekday import Weekday

class Poi:

    def __init__(self, coordinates, times):
        self._centroid = self.find_centroid(coordinates)
        self._coordinates = coordinates
        self._n_events = len(coordinates)
        self._times = times
        self._different_hours = None
        self._n_events_week = None
        self._poi_class = "Other"
        self._home_hour = {}
        self._work_hour = {}
        self._different_days = 0
        self._different_schedules = 0
        self._n_events_work_time = 0
        self._n_events_home_time = 0
        self._n_events_week = 0
        self._n_events_weekend = 0
        self._calculate_different_days()
        self._calculate_different_hours()

    def _calculate_different_hours(self):
        """
           Method to calculate the number of different hours that the events occurred.

        """
        self._times
        events_hour = [0] * 24
        for i in range(len(self.times)):
            events_hour[self.times[i].hour] = events_hour[self.times[i].hour] + 1
        self._different_hours = events_hour
        self._different_schedules = np.count_nonzero(events_hour)

    def _calculate_different_days(self):
        """
               Method to calculate the number of different days that the events occurred.

       """
        times = self._times
        events_day = [0] * 32
        different_days = 0
        for i in range(len(times)):
            events_day[times[i].day] = events_day[times[i].day] + 1
        for i in range(len(events_day)):
            if events_day[i] != 0:
                different_days = different_days + 1
        self._different_days =  different_days

    @property
    def different_hours(self):
        return self._different_hours

    @property
    def n_events_week(self):
        return self._n_events_week

    @property
    def n_events_weekend(self):
        return self._n_events_weekend

    @property
    def different_days(self):
        return self._different_days

    @property
    def n_events(self):
        return self._n_events

    @property
    def different_schedules(self):
        return self._different_schedules

    def __repr__(self):
        return "Centroide:"+str(self._centroid)

    @property
    def centroid(self):
        return self._centroid

    @property
    def duration_home_time(self):
        return self._duration_home_time

    @property
    def duration_work_time(self):
        return self._duration_work_time

    @property
    def n_events_work_time(self):
        return self._n_events_work_time

    @property
    def n_events_home_time(self):
        return self._n_events_home_time

    @property
    def times(self):
        return self._times

    @property
    def poi_class(self):
        return self._poi_class

    @poi_class.setter
    def poi_class(self, poi_class):
        self._poi_class = poi_class

    @property
    def home_hour(self):
        return self._home_hour

    @property
    def work_hour(self):
        return self._work_hour

    @home_hour.setter
    def home_hour(self, home_hour):
        self._home_hour = home_hour

    @work_hour.setter
    def work_hour(self, work_hour):
        self._work_hour = work_hour

    def add_n_events_home_time(self):
        self._n_events_home_time = self._n_events_home_time +1

    def add_n_events_work_time(self):
        self._n_events_work_time = self._n_events_work_time +1

    def add_n_events_week(self):
        self._n_events_week = self._n_events_week +1

    def add_n_events_weekend(self):
        self._n_events_weekend = self._n_events_weekend +1

    def to_dict(self):
        return {'location_type': self._poi_class, 'latitude': str(self.centroid[0]), 'longitude': str(self.centroid[1]), \
                 'home_time_events': str(self.n_events_home_time), 'work_time_events': str(self.n_events_work_time)}

    def calculate_n_events(self):
        times = self._times
        for i in range(len(times)):
            if times[i].weekday() < 5:
                self.add_n_events_week()
            else:
                self.add_n_events_weekend()

    def calculate_home_work_n_events(self):

        for datetime in self.times:
            if self.home_hour['start'] < self.home_hour['end']:
                if datetime.hour >= self.home_hour['start'] and datetime.hour <= self.home_hour['end']:
                    self.add_n_events_home_time()
            else:
                if datetime.hour >= self.home_hour['start'] or datetime.hour <= self.home_hour['end']:
                    self.add_n_events_home_time()
            # steps generated on weekends are not accounted for finding the work POI
            if datetime.weekday() >= Weekday.SATURDAY.value:
                continue
            elif self.work_hour['start'] < self.work_hour['end']:
                if datetime.hour >= self.work_hour['start'] and datetime.hour <= self.work_hour['end']:
                    self.add_n_events_work_time()
            else:
                if datetime.hour >= self.work_hour['start'] or datetime.hour <= self.work_hour['end']:
                    self.add_n_events_work_time()


    def find_centroid(self, vertexes):
        _x_list = [vertex[0] for vertex in vertexes]
        _y_list = [vertex[1] for vertex in vertexes]
        _len = len(vertexes)
        _x = round(sum(_x_list) / _len,8)
        _y = round(sum(_y_list) / _len,8)
        return (_x, _y)