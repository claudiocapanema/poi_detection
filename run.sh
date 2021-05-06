# !/bin/bash
echo "Argument: "$1

# common variables
USERS_STEPS_BASE_DIR="/media/claudio/Data/backup_linux/Documentos/users_steps_datasets/"
USERS_STEPS_OUTPUT_BASE_DIR="/media/claudio/Data/backup_win_hd/Downloads/doutorado/users_steps_output/"
USERS_194_STEPS_FILENAME="${USERS_STEPS_BASE_DIR}194_users_data.csv"
USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_BASE_DIR}df_mais_de_10_mil_limite_500_pontos.csv"
DETECTED_USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}detected_df_mais_de_10_mil_limite_500_pontos.csv"
CLASSIFIED_USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}classified_df_mais_de_10_mil_limite_500_pontos.csv"
USERS_5_MIL_MAX_500_POINTS="${USERS_STEPS_BASE_DIR}df_mais_de_5_mil_limite_500_pontos.csv"
DETECTED_USERS_5_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}detected_df_mais_de_5_mil_limite_500_pontos.csv"
CLASSIFIED_USERS_5_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}classified_df_mais_de_5_mil_limite_500_pontos.csv"
POI_DETECTION_FILENAME="/home/claudio/Documentos/pycharmprojects/poi_detection_output/users_detected_pois.csv"
POI_CLASSIFICATION_FILENAME="/home/claudio/Documentos/pycharmprojects/poi_detection_output/users_classified_pois.csv"
GROUND_TRUTH="/home/claudio/Documentos/pycharmprojects/poi_detection_output/pontosmarcados_corrigido_periodico.csv"

# points_of_interest_job
# This job is applied to detect users' pois
POI_CONFIG='{
          "job": "points_of_interest_job",
          "users_steps_filename": "'$USERS_5_MIL_MAX_500_POINTS'",
          "poi_detection_filename": "'$DETECTED_USERS_5_MIL_MAX_500_POINTS'",
          "poi_classification_filename": "'$CLASSIFIED_USERS_5_MIL_MAX_500_POINTS'",
          "ground_truth": "'$GROUND_TRUTH'",
          "utc_to_sp": "no"
          }'

# points_of_interest_validation_job
# This job is applied to validate the pois found by the points_of_interest_job
VALIDATION_CONFIG='{
          "job": "points_of_interest_validation_job",
          "users_steps_filename": "'$USERS_194_STEPS_FILENAME'",
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