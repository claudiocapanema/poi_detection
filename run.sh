# !/bin/bash

# points_of_interest_job
# This job is applied to detect users' pois
#CONFIG='{
#          "job": "points_of_interest_job",
#          "users_steps_csv": "/home/claudio/Documentos/users_steps_datasets/194_users_data.csv",
#          "poi_detection_output": "/home/claudio/Documentos/pycharmprojects/poi_detection_output/users_classified_pois.csv",
#          "ground_truth": "/home/claudio/Documentos/pycharmprojects/poi_detection_output/pontosmarcados_corrigido_periodico.csv"
#          }'

# points_of_interest_validation_job
# This job is applied to validate the pois found by the points_of_interest_job
CONFIG='{
          "job": "points_of_interest_validation_job",
          "users_steps_csv": "/home/claudio/Documentos/users_steps_datasets/194_users_data.csv",
          "poi_detection_output": "/home/claudio/Documentos/pycharmprojects/poi_detection_output/users_classified_pois.csv",
          "ground_truth": "/home/claudio/Documentos/pycharmprojects/poi_detection_output/pontosmarcados_corrigido_periodico.csv"
          }'

echo $CONFIG

python main.py "${CONFIG}"