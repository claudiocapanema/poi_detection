# !/bin/bash
echo "Argument: "$1

# common variables
USERS_STEPS_BASE_DIR="/media/claudio/Data/backup_linux/Documentos/users_steps_datasets/"
USERS_STEPS_OUTPUT_BASE_DIR="/media/claudio/Data/backup_win_hd/Downloads/doutorado/users_steps_output/"
USERS_STEPS_SEQUENCES_OUTPUT_BASE_DIR="${USERS_STEPS_OUTPUT_BASE_DIR}sequences/"
USERS_STEPS_10_CATEGORIES_SEQUENCES_FILENAME="${USERS_STEPS_SEQUENCES_OUTPUT_BASE_DIR}users_steps_10_categories_sequences.csv"
USERS_STEPS_8_CATEGORIES_SEQUENCES_FILENAME="${USERS_STEPS_SEQUENCES_OUTPUT_BASE_DIR}users_steps_8_categories_sequences.csv"
USERS_194_STEPS_FILENAME="${USERS_STEPS_BASE_DIR}194_users_data.csv"
USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_BASE_DIR}df_mais_de_10_mil_limite_500_pontos_local_datetime.csv"
DETECTED_USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}detected_df_mais_de_10_mil_limite_500_pontos_local_datetime.csv"
CLASSIFIED_USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}classified_df_mais_de_10_mil_limite_500_pontos.csv"
USERS_STEPS_10_MIL_MAX_500_POINTS_WITH_POIS_FILENAME="${USERS_STEPS_OUTPUT_BASE_DIR}detected_df_mais_de_10_mil_limite_500_pontos_local_datetime_with_osm_pois_50.csv"
USERS_STEPS_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME="${USERS_STEPS_OUTPUT_BASE_DIR}users_steps_10_mil_limite_500_pontos_local_datetime_with_detected_pois_with_osm_pois_50.csv"
USERS_STEPS_8_CATEGORIES_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME="${USERS_STEPS_OUTPUT_BASE_DIR}users_steps_8_categories_10_mil_limite_500_pontos_local_datetime_with_detected_pois_with_osm_pois_50.csv"
USERS_5_MIL_MAX_500_POINTS="${USERS_STEPS_BASE_DIR}df_mais_de_5_mil_limite_500_pontos.csv"
DETECTED_USERS_5_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}detected_df_mais_de_5_mil_limite_500_pontos.csv"
CLASSIFIED_USERS_5_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}classified_df_mais_de_5_mil_limite_500_pontos.csv"
CLASSIFIED_USERS_10_MIL_MAX_500_POINTS="${USERS_STEPS_OUTPUT_BASE_DIR}classified_df_mais_de_10_mil_limite_500_pontos.csv"
POI_DETECTION_FILENAME="/home/claudio/Documentos/pycharmprojects/poi_detection_output/users_detected_pois.csv"
POI_CLASSIFICATION_FILENAME="/home/claudio/Documentos/pycharmprojects/poi_detection_output/users_classified_pois.csv"
GROUND_TRUTH="/home/claudio/Documentos/pycharmprojects/poi_detection_output/pontosmarcados_corrigido_periodico.csv"

# gowalla
GOWALLA_BASE_DIR="/media/claudio/Data/backup_win_hd/Downloads/doutorado/gowalla/"
GOWALLA_US="${GOWALLA_BASE_DIR}gowalla_checkins_7_categories_local_datetime_columns_reduced_us.csv"
GOWALLA_US_SEQUENCES="${GOWALLA_BASE_DIR}sequences/gowalla_7_categories_sequences.csv"

# points_of_interest_job
# This job is applied to detect users' pois
POI_CONFIG='{
          "job": "points_of_interest_job",
          "users_steps_filename": "'$USERS_10_MIL_MAX_500_POINTS'",
          "poi_detection_filename": "'$DETECTED_USERS_10_MIL_MAX_500_POINTS'",
          "poi_classification_filename":"'$CLASSIFIED_USERS_10_MIL_MAX_500_POINTS'",
          "users_detected_pois_with_osm_pois_filename":"'$USERS_STEPS_10_MIL_MAX_500_POINTS_WITH_POIS_FILENAME'",
          "users_steps_with_detected_pois_with_osm_pois_filename":"'$USERS_STEPS_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME'",
          "ground_truth": "'$GROUND_TRUTH'",
          "utc_to_sp": "no",
          "users_steps_join_detected_pois":"yes"
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

SEQUENCE_GENERATION_FOR_POI_CATEGORIZATION_SEQUENTIAL_CONFIG='{
          "job": "next_poi_category_prediction_sequences_generation_job",
          "users_steps_filename": "'$USERS_STEPS_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME'",
          "base_dir":"'$USERS_STEPS_BASE_DIR'",
          "users_sequences_folder":"'$USERS_STEPS_OUTPUT_BASE_DIR'sequences/",
          "8_categories_filename":"'$USERS_STEPS_8_CATEGORIES_10_MIL_MAX_500_POINTS_WITH_DETECTED_POIS_WITH_OSM_POIS_FILENAME'",
          "categories_type":"8_categories",
          "ground_truth": "'$GROUND_TRUTH'",
          "to_8_categories":"yes",
          "dataset_name":"users_steps"
          }'

#SEQUENCE_GENERATION_FOR_POI_CATEGORIZATION_SEQUENTIAL_CONFIG='{
#          "job": "next_poi_category_prediction_sequences_generation_job",
#          "users_steps_filename": "'$GOWALLA_US'",
#          "base_dir":"'$GOWALLA_BASE_DIR'",
#          "users_sequences_folder":"'$GOWALLA_BASE_DIR'sequences/",
#          "8_categories_filename":"'$GOWALLA_US'",
#          "categories_type":"7_categories",
#          "ground_truth": "'$GROUND_TRUTH'",
#          "to_8_categories":"no",
#          "dataset_name":"gowalla"
#          }'

#NEXT_POI_CATEGORY_PREDICTION_CONFIG='{
#          "job": "next_poi_category_prediction_job",
#          "users_sequences": "'$USERS_STEPS_8_CATEGORIES_SEQUENCES_FILENAME'",
#          "baseline": "serm",
#          "dataset_name":"users_steps",
#          "categories_type":"8_categories"
#          }'

NEXT_POI_CATEGORY_PREDICTION_CONFIG='{
          "job": "next_poi_category_prediction_job",
          "users_sequences": "'$GOWALLA_US_SEQUENCES'",
          "baseline": "map",
          "dataset_name":"gowalla",
          "categories_type":"7_categories"
          }'

PERFORMANCE_PLOTS_CONFIG='{
          "job": "performance_plots_job",
          "categories_type":"7_categories",
          "dataset_name":"gowalla"
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
  "sequences_generation")
    python main.py "${SEQUENCE_GENERATION_FOR_POI_CATEGORIZATION_SEQUENTIAL_CONFIG}"
    ;;
  "next")
    python main.py "${NEXT_POI_CATEGORY_PREDICTION_CONFIG}"
    ;;
  "plots")
    python main.py "${PERFORMANCE_PLOTS_CONFIG}"
    ;;
esac