# !/bin/bash
echo "Argument: "$1

# common variables
USERS_STEPS_FILENAME="/home/claudio/Documentos/users_steps_datasets/194_users_data.csv"
POI_DETECTION_FILENAME="/home/claudio/Documentos/pycharmprojects/poi_detection_output/users_detected_pois.csv"
POI_CLASSIFICATION_FILENAME="/home/claudio/Documentos/pycharmprojects/poi_detection_output/users_classified_pois.csv"
GROUND_TRUTH="/home/claudio/Documentos/pycharmprojects/poi_detection_output/pontosmarcados_corrigido_periodico.csv"

# points_of_interest_job
# This job is applied to detect users' pois
POI_CONFIG='{
          "job": "points_of_interest_job",
          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
          "poi_classification_filename": "'$POI_CLASSIFICATION_FILENAME'",
          "ground_truth": "'$GROUND_TRUTH'",
          "utc_to_sp": "yes"
          }'

# points_of_interest_validation_job
# This job is applied to validate the pois found by the points_of_interest_job
VALIDATION_CONFIG='{
          "job": "points_of_interest_validation_job",
          "users_steps_filename": "'$USERS_STEPS_FILENAME'",
          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
          "poi_classification_filename": "'$POI_CLASSIFICATION_FILENAME'",
          "ground_truth": "'$GROUND_TRUTH'"
          }'

echo $CONFIG

case $1 in
  "find_poi")
    python main.py "${POI_CONFIG}"
    ;;
  "validate")
    python main.py "${VALIDATION_CONFIG}"
    ;;
  "find_poi_and_validate")
    python main.py "${POI_CONFIG}"
    python main.py "${VALIDATION_CONFIG}"
    ;;
esac