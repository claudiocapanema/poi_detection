from enum import Enum


class DataSources(Enum):
    USERS_STEPS_CSV                               = '/home/claudio/Documentos/users_steps_datasets/194_users_data.csv'
    FILES_DIRECTORY                               = '/home/claudio/Documentos/pycharmprojects/poi_detection_output/'


    def get_name(self):
        return self.name

    def get_value(self):
        return self.value