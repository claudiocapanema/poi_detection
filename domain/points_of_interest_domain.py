import datetime as dt
from foundation.general_code.dbscan import Dbscan
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import pytz

from model.poi import Poi
from model.user import User
from configuration.points_of_interest_configuration import PointsOfInterestConfiguration
from configuration.weekday import Weekday
from foundation.util.datetimes_utils import DatetimesUtils
from model.location_type import LocationType
from foundation.general_code.nearest_neighbors import NearestNeighbors

class PointsOfInterestDomain:

    def __init__(self):
        self.location_type = LocationType()

    def inactive_interval(self, events_hours: list) ->tuple:
        """
            This function identifies the inactive time interval o a user
            ------
            Parameters
            ----------
            events_hours: list
                It is a list of int representing event's hour
            Return:
            ------
                tuple
                    Tuple of the biggest inactive interval's start and end hours
            Notes:
            ------
            The inactive interval is the maximum time interval that a user didn't generate any step. The hypothesis is that
            at this time interval, the user is at home, so he is not moving and generating any step.
        """

        try:
            """
            list of range from 0 to 47, representing the number of events in each hour
            this is necessary because the inactive interval may be between 22:00 and 02:00, so we need do a circular search
            """
            events_hours = events_hours + events_hours

            # the start hour of an inactive interval
            start_hour = -1
            # the end hour of an inactive interval
            end_hour = -1
            t = 0  # the size of an inactive interval
            intervals = list()
            for i in range(len(events_hours)):
                if events_hours[i] == 0:
                    if start_hour == -1:
                        # the start and end hour have to be into the range from 0 to 23
                        start_hour = i % 24
                        end_hour = i % 24
                        t = t + 1
                    else:
                        end_hour = i % 24
                        t = t + 1
                else:
                    # if the current hour has some event we save the inactive interval previously calculated
                    if start_hour != -1:
                        intervals.append((start_hour, end_hour, t))
                        start_hour = -1
                        end_hour = -1
                        t = 0

            intervals.append((start_hour, end_hour, t))
            # we select the biggest inactive interval
            max_interval = max(intervals, key=lambda x: x[2])
            max_start_hour = max_interval[0]
            max_end_hour = max_interval[1]
            lenght = max_interval[2]

            """ 
            if the lenght of the inactive interval is too small or too big (it happens when we have a few steps), use fixed 
            intervals
            """
            if lenght <= 0 or lenght >= 9:
                return -1, -1
            if max_start_hour > PointsOfInterestConfiguration.MIN_MAX_INVERTED_ROUTINE.get_value()['min'] and \
                    max_start_hour < PointsOfInterestConfiguration.MIN_MAX_INVERTED_ROUTINE.get_value()['max'] and \
                    max_end_hour > PointsOfInterestConfiguration.MIN_MAX_INVERTED_ROUTINE.get_value()['min'] and \
                    max_end_hour < PointsOfInterestConfiguration.MIN_MAX_INVERTED_ROUTINE.get_value()['max']:
                #print(max_start_hour, max_end_hour)
                pass
            return max_start_hour, max_end_hour

        except Exception as e:
            raise e

    def location_hours(self, hours_list: list) ->tuple:
        """
            This function returns the home and work time intervals
            ------
            Parameters
            ----------
            hours_list: list
                It is a list of int representing event's hour
            Return:
            ------
                tuple
                    Tuple of dicts, in what each one represents the home and work time intervals
            Notes:
            The home and work time intervals are the time intervals used to account events. The home and work time intervals
            are defined based on the inactive interval found for each user. The hypothesis is that next to the  maximum
            inactive interval the user generates steps at the home location, by arriving or departing from home.
            The remaining time interval is considered as the work time interval. The home and work time intervals might be
            preset if a inactive interval was not found.
        """
        try:
            inactive_flag = str(False)
            inverted_rotuine_flag = str(False)
            start, end = self.inactive_interval(hours_list)
            # if a inactive interval was not found, we consider pre-defined time slots to detect home and work PoIs
            if start == -1 and end == -1:
                return PointsOfInterestConfiguration.HOME_HOUR.get_value(), \
                       PointsOfInterestConfiguration.WORK_HOUR.get_value(), -1, -1, inactive_flag, inverted_rotuine_flag
            inactive_flag = str(True)
            home_hour = {}
            # the home hour have to starts and ends into a range between 0 and 23
            if (start - PointsOfInterestConfiguration.HOURS_BEFORE.get_value()) < 0:
                # the hour that it starts is before the inactive interval's start
                home_hour['start'] = 24 + (start - PointsOfInterestConfiguration.HOURS_BEFORE.get_value())
            else:
                home_hour['start'] = start - PointsOfInterestConfiguration.HOURS_BEFORE.get_value()
            if (end + PointsOfInterestConfiguration.HOURS_AFTER.get_value()) > 23:
                # the hour that it ends is after the inactive interval's end
                home_hour['end'] = end + PointsOfInterestConfiguration.HOURS_AFTER.get_value() - 24
            else:
                home_hour['end'] = end + PointsOfInterestConfiguration.HOURS_AFTER.get_value()

            work_hour = {}
            # the reason to use - ONE_HOUR is to let the "work hour end" to be one hour before of the "home time start"
            if (home_hour['start'] - PointsOfInterestConfiguration.ONE_HOUR.get_value()) < 0:
                # the work time ends when the home time starts
                work_hour['end'] = 24 + (home_hour['start'] - PointsOfInterestConfiguration.ONE_HOUR.get_value())
            else:
                work_hour['end'] = home_hour['start'] - PointsOfInterestConfiguration.ONE_HOUR.get_value()
                # the reason to use + ONE_HOUR is to let the "work hour start" to be one hour after of the "home time end"
            if (home_hour['end'] + PointsOfInterestConfiguration.ONE_HOUR.get_value()) > 23:
                # the work time starts when the home time ends
                work_hour['start'] = home_hour['end'] + PointsOfInterestConfiguration.ONE_HOUR.get_value() - 24
            else:
                work_hour['start'] = home_hour['end'] + PointsOfInterestConfiguration.ONE_HOUR.get_value()

            inverted_rotuine_flag = str(False)
            if start > PointsOfInterestConfiguration.MIN_MAX_INVERTED_ROUTINE.get_value()['min'] and \
                    start < PointsOfInterestConfiguration.MIN_MAX_INVERTED_ROUTINE.get_value()['max'] and \
                    end > PointsOfInterestConfiguration.MIN_MAX_INVERTED_ROUTINE.get_value()['min'] and \
                    end < PointsOfInterestConfiguration.MIN_MAX_INVERTED_ROUTINE.get_value()['max']:
                inverted_rotuine_flag = str(True)

            return home_hour, work_hour, start, end, inactive_flag, inverted_rotuine_flag

        except Exception as e:
            raise e

    def classify_points_of_interest(self, user: User) -> DataFrame:
        """
            This function selects the home ,work and leisure PoIs of a user
            ------
            Parameters
            ------
            user: User
                It is User object class
            Return:
            ------
                dict
                    Dictionary containing the user's app_user_id, host_app_id, total_user_events and a list of dictionaries
                    containing each POI's features
            Notes:
            ------
            This function tries to find the home and work POIs into the top 5 most importants POIs (Those ones that
            have more events). To find the home POI, it is necessary to calculate the home time interval in order
            to account events into the specific time span. This is analogous for the work finding process. After this,
            the the algorithm classifies as Home the POIs that has most events at the home time interval comparing to
            the others POIs of the user
        """
        try:
            app_user_id = user.id
            min_samples = user.min_samples
            meters = user.meters
            user_pois = user.pois
            total_user_events = user.n_events

            # minimum number of events to a PoI be classified as home
            min_home_events = user.min_home_events
            # minimum number of events to a PoI be classified as work
            min_work_events = user.min_work_events

            if len(user.pois) == 0:
                user_pois_classified = {'app_user_id': app_user_id, 'total_user_steps': total_user_events,
                                        'eps': meters, 'min_samples': min_samples, 'pois_classified': []}

                return user.user_pois_to_pandas_df()
                #return user_pois_classified

            hours_list = [0] * 24
            pois = [[i, user.pois[i]] for i in range(len(user.pois))]

            # Select the top N pois that have the majority quantity of events.
            pois_indexes = sorted(pois, key=lambda p: p[1].n_events, reverse=True)[:PointsOfInterestConfiguration.TOP_N_POIS.get_value()]
            pois_indexes = [poi_index[0] for poi_index in pois_indexes]

            """
            It collects the steps' datetime of each poi. To find the home and work times, it only considers records 
            generated on weekdays
            """
            for index in pois_indexes:
                for datetime in user.pois[index].times:
                    if datetime.weekday() >= Weekday.SATURDAY.value:
                        continue
                    hours_list[datetime.hour] = hours_list[datetime.hour] + 1
            home_hour, work_hour, start, end, inactive_applied_flag, inverted_routine_flag = self.location_hours(hours_list)

            for index in pois_indexes:
                user.pois[index].home_hour = home_hour
                user.pois[index].work_hour = work_hour
                user.pois[index].calculate_home_work_n_events()

            user.inactive_interval_start = start
            user.inactive_interval_end = end
            user.inactive_applied_flag = inactive_applied_flag
            user.inverted_routine_flag = inverted_routine_flag

                # print(user.pois[index].inactive_interval_start, start, user.pois[index].inactive_interval_end, end,
                #       user.pois[index].inactive_applied_flag, inactive_applied_flag)

            """
                From this point forward the algorithm selects the home and work POIs
            """


            home_index = -1
            work_index = -1
            home = None
            work = None
            home_work_distance = ""

            for i in range(len(user.pois)):
                poi = user.pois[i]
                if poi.n_events_home_time > min_home_events:
                    min_home_events = poi.n_events_home_time
                    home_index = i

            for i in range(len(user.pois)):
                if i != home_index:
                    poi = user.pois[i]
                    if poi.n_events_work_time > min_work_events:
                        min_work_events = poi.n_events_work_time
                        work_index = i

            if home_index > -1:
                user.pois[home_index].poi_class = self.location_type.HOME
                home = user.pois[home_index]
            if work_index > -1 and work_index != home_index:
                user.pois[work_index].poi_class = self.location_type.WORK
                work = user.pois[work_index]
            # pois_classified = list()

            # for i in range(len(user.user_pois)):
            #     pois_classified.append(user_pois[i].to_dict())
            #
            # user_pois_classified = {'app_user_id': app_user_id, 'total_user_steps': total_user_events,
            #                         'eps': meters, 'min_samples': min_samples, 'pois_classified': pois_classified}
            #
            # return user_pois_classified

            return user.user_pois_to_pandas_df()

        except Exception as e:
            raise e

    def identify_points_of_interest(self, user_df: pd.DataFrame, utc_to_sp:str) -> DataFrame:
        """
            This function identifies individual points of interest
            ------
            Parameters
            ---------
            user_id: int
                User's id
            user_steps: list
                This is a list of tuples.
            return:
            ------
                list
                    User object class
            Notes:
            -----
            This function identifies points of interest (POIs) by clustering user's steps and selecting as POIs the clusters
            that contains events generated into a minimum amount of different days. This last method is applied to ensure
            that the most relevant POIs will be generated, discarding those ones that were visited in only few days.
        """
        try:
            user_df = user_df.sort_values('datetime')
            user_id = user_df['id'].tolist()
            latitude = user_df['latitude'].tolist()
            longitude = user_df['longitude'].tolist()
            reference_date = user_df['datetime'].tolist()


            user_id = int(user_id[0])

            size = min([len(latitude), len(longitude)])
            coordinates = np.asarray([(latitude[i], longitude[i]) for i in range(size)], dtype=np.float64)

            # if the timezones are utc, you can adjust them here
            if utc_to_sp == "yes":
                sp_time_zone = PointsOfInterestConfiguration.TZ.get_value()
                times = [DatetimesUtils.convert_tz(reference_date[i], pytz.utc, sp_time_zone) for i in range(len(reference_date))]
            else:
                times = reference_date
            n_events = len(coordinates)

            # Setting the identification parameters
            min_samples = PointsOfInterestConfiguration.MIN_SAMPLES.get_value()
            min_days = PointsOfInterestConfiguration.MIN_DAYS.get_value()
            # Setting the classification parameters
            min_home_events = PointsOfInterestConfiguration.MIN_HOME_EVENTS.get_value()
            min_work_events = PointsOfInterestConfiguration.MIN_WORK_EVENTS.get_value()

            dbscan = Dbscan(coordinates, min_samples, PointsOfInterestConfiguration.EPSILON.get_value())
            dbscan.cluster_geo_data()
            pois_coordinates, pois_times = dbscan.get_clusters_with_points_and_datatime(times)

            pois = list()
            size = min([len(pois_coordinates), len(pois_times)])
            for i in range(size):
                if len(pois_coordinates[i]) == 0:
                    continue
                p = Poi(pois_coordinates[i], pois_times[i])
                if p.different_days < min_days:
                    continue
                if p.different_schedules < 2:
                    continue
                pois.append(p)

            user = User(user_id, pois, n_events, PointsOfInterestConfiguration.METERS.get_value(), min_samples, min_home_events, min_work_events)

            return self.classify_points_of_interest(user)

        except Exception as e:
            raise e

    def classify_pois_from_ground_truth(self, user_steps, ground_truth, utc_to_sp):

        ids = ground_truth['id'].unique().tolist()
        classified_users_pois = []
        for user_id in ids:
            us = user_steps.query("id=="+str(user_id))
            gt = ground_truth.query("id==" + str(user_id))
            us_latitudes = us['latitude'].tolist()
            us_longitudes = us['longitude'].tolist()
            gt_latitudes = gt['latitude'].tolist()
            gt_longitudes = gt['longitude'].tolist()
            us_points = np.radians([(long, lat) for long, lat in zip(us_latitudes, us_longitudes)])
            gt_points = np.radians([(long, lat) for long, lat in zip(gt_latitudes, gt_longitudes)])
            distances, indexes = NearestNeighbors. \
                find_radius_neighbors(gt_points, us_points,
                                      PointsOfInterestConfiguration.RADIUS_CLASSIFICATION.get_value())

            pois = []
            for j in range(len(indexes)):
                poi_coordinates = []
                poi_times = []
                for k in range(len(indexes[j])):
                    latitude = us['latitude'].iloc[indexes[j][k]]
                    longitude = us['longitude'].iloc[indexes[j][k]]
                    poi_coordinates.append((latitude, longitude))
                    datetime = us['datetime'].iloc[indexes[j][k]]
                    if utc_to_sp == "yes":
                        sp_time_zone = PointsOfInterestConfiguration.TZ.get_value()
                        datetime = DatetimesUtils.convert_tz(datetime, pytz.utc, sp_time_zone)
                    poi_times.append(datetime)

                if len(poi_coordinates) == 0:
                    continue
                p = Poi(poi_coordinates, poi_times)
                pois.append(p)

            n_events = []
            """
            this parameter is for the dbscan, but as we don't apply it in this moment the parameter value is
            set to -1
            """
            min_samples = -1
            user = User(user_id, pois, n_events, PointsOfInterestConfiguration.METERS.get_value(), min_samples,
                        PointsOfInterestConfiguration.MIN_HOME_EVENTS.get_value(),
                        PointsOfInterestConfiguration.MIN_WORK_EVENTS.get_value())

            classified_user_pois = self.classify_points_of_interest(user)
            classified_users_pois.append(classified_user_pois)

        return pd.Series(classified_users_pois)

    def concatenate_dataframes(self, processed_users_pois):
        """
        Organazing the results into a single table
        """
        concatenated_processed_users_pois = pd.DataFrame({"id": [], "poi_type": [],
                                                         "latitude": [], "longitude": [],
                                                         "work_time_events": [], "home_time_events": []})

        for i in range(processed_users_pois.shape[0]):
            concatenated_processed_users_pois = concatenated_processed_users_pois. \
                append(processed_users_pois.iloc[i], ignore_index=True)

        concatenated_processed_users_pois['id'] = concatenated_processed_users_pois['id'].astype('int64')

        # media de pontos de interesse encontrados por usuário
        media = concatenated_processed_users_pois.groupby(by='id').apply(lambda e: pd.DataFrame({'total': [len(e)]}))
        print("Quantidade de usuários: ", len(concatenated_processed_users_pois['id'].tolist()))
        print("Média de pontos de interesse encontrados por usuário: ", media['total'].mean())
        print("Quantidade total de pontos de interesse encontrados: ", len(concatenated_processed_users_pois))


        return concatenated_processed_users_pois

    def associate_users_steps_with_pois(self, users_steps, pois):

        print(users_steps)
        print(pois)

        users_steps_with_pois = users_steps.groupby(by='id').apply(lambda e: self.associate_user_steps_with_pois(e, pois)).reset_index(drop=True)
        users_steps_with_pois['id'] = users_steps_with_pois['id'].astype('int')
        users_steps_with_pois['index'] = users_steps_with_pois['index'].astype('int')
        users_steps_with_pois['id_right'] = users_steps_with_pois['id_right'].astype("int")
        users_steps_with_pois['index_assign'] = users_steps_with_pois['index_assign'].astype('int')
        users_steps_with_pois['work_time_events'] = users_steps_with_pois['work_time_events'].astype('int')
        users_steps_with_pois['home_time_events'] = users_steps_with_pois['home_time_events'].astype('int')
        users_steps_with_pois['inactive_applied_flag'] = users_steps_with_pois['inactive_applied_flag'].astype("int")
        users_steps_with_pois['inactive_interval_start'] = users_steps_with_pois['inactive_interval_start'].astype("int")
        users_steps_with_pois['inactive_interval_end'] = users_steps_with_pois['inactive_interval_end'].astype("int")
        users_steps_with_pois['inverted_routine_flag'] = users_steps_with_pois['inverted_routine_flag'].astype("int")
        print("final")
        print("colunas: ", users_steps_with_pois.columns)
        print(users_steps_with_pois)

        self.verify_users_steps_pois_assignment(users_steps_with_pois)
        users_steps_with_pois = users_steps_with_pois[
            ['id', 'datetime', 'latitude', 'longitude', 'poi_type', 'poi_latitude', 'poi_longitude',
             'work_time_events', 'home_time_events', 'inactive_applied_flag',
             'inactive_interval_end', 'inactive_interval_start',
             'inverted_routine_flag', 'poi_osm',
        'distance_osm']]
        return users_steps_with_pois

    def associate_user_steps_with_pois(self, user_steps, pois):

        userid = user_steps['id'].iloc[0]
        user_pois = pois.query("id == " + str(userid)).head(2)
        if len(user_pois) == 0:
            return pd.DataFrame({column: [] for column in ['id_right', 'poi_type', 'poi_latitude', 'poi_longitude', 'work_time_events',
       'home_time_events', 'inactive_applied_flag', 'inactive_interval_end',
       'inactive_interval_start', 'inverted_routine_flag', 'poi_osm',
        'distance_osm']})
        poi_latitudes = user_pois['latitude'].tolist()
        poi_longitudes = user_pois['longitude'].tolist()
        poi_points = np.radians([(long, lat) for long, lat in zip(poi_latitudes, poi_longitudes)])
        dp_latitudes = user_steps['latitude'].tolist()
        dp_longitudes = user_steps['longitude'].tolist()
        user_steps['index'] = np.array([i for i in range(len(user_steps))])
        user_steps_points = np.radians([(long, lat) for long, lat in zip(dp_latitudes, dp_longitudes)])
        user_pois = user_pois[['id', 'poi_type', 'latitude', 'longitude', 'work_time_events',
       'home_time_events', 'inactive_applied_flag', 'inactive_interval_end',
       'inactive_interval_start', 'inverted_routine_flag', 'poi_osm',
       'distance_osm']]
        user_pois.columns = ['id_right', 'poi_type', 'poi_latitude', 'poi_longitude', 'work_time_events',
       'home_time_events', 'inactive_applied_flag', 'inactive_interval_end',
       'inactive_interval_start', 'inverted_routine_flag', 'poi_osm',
        'distance_osm']
        # if len(dp_points) < 1:
        #     continue
        distances, indexes = NearestNeighbors. \
            find_radius_neighbors(poi_points, user_steps_points,
                                  PointsOfInterestConfiguration.EPSILON.get_value())

        columns = user_steps.columns.tolist() + ['index_assign'] + user_pois.columns.tolist()
        new_users_steps = {column: [] for column in columns}
        """
            get user_steps that don't belong to POIs
        """
        flatten_indexes = []
        for index in indexes:
            flatten_indexes = flatten_indexes + index.tolist()
        flatten_indexes = sorted(flatten_indexes)
        non_indexes = []
        for i in range(1, len(flatten_indexes)):
            non_indexes = non_indexes + [i for i in range(flatten_indexes[i-1], flatten_indexes[i])]

        pattern_row = {}
        for column in ['id_right', 'inactive_applied_flag', 'inactive_interval_end',
       'inactive_interval_start', 'inverted_routine_flag']:
            pattern_row[column] = user_pois.iloc[0][column]
        pattern_row['poi_type'] = "Displacement"
        for column in ['poi_latitude', 'poi_longitude', 'work_time_events',
        'home_time_events']:
            pattern_row[column] = -1
        pattern_row['poi_osm'] = 'empty'
        pattern_row['distance_osm'] = 999
        for i in range(len(non_indexes)):
            index = non_indexes[i]

            user_step = user_steps.iloc[index]
            for column in user_steps.columns:
                new_users_steps[column].append(user_step[column])

            for column in pattern_row.keys():
                new_users_steps[column].append(pattern_row[column])

        new_users_steps['index_assign'] = [i for i in non_indexes]
        tamanhos = []
        for column in columns:
            if len(new_users_steps[column]) == 0:
                print(column)
            tamanhos.append(len(new_users_steps[column]))

        """
            Calculating the metrics
        """
        for i in range(len(indexes)):
            poi_indexes = indexes[i]
            poi = user_pois.iloc[i]
            for j in poi_indexes:
                user_steps['index_assign'] = j
                row = pd.concat([user_steps.iloc[j], poi])
                for column in row.index.tolist():
                    new_users_steps[column].append(row[column])

        # print("usuario ----------")
        # for k in new_users_steps:
        #     print("Chave: ", k)
        #     print("Tamanho: ", len(new_users_steps[k]))

        users_steps_with_pois = pd.DataFrame(new_users_steps)
        #users_steps_with_pois.append(lambda e: self.verify_users_steps_pois_assignment(e))

        return users_steps_with_pois


    def verify_users_steps_pois_assignment(self, row):
        print(row)
        print("Verificação")
        if row['id'].tolist() != row['id_right'].tolist() or row['index'].tolist() != row['index_assign'].tolist():
            print("Erro")
            raise