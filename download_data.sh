SUBDIRECTORY="output"
mkdir -p ${SUBDIRECTORY}
TEMP_DIR=$(mktemp -d)

# download colmap results
mkdir -p "$SUBDIRECTORY/colmap_saved_output"
CM_FILE_ID="181EG4I6sXurPOUU72vab3rNBuEqrybgd"
CM_FILE_NAME="colmap_saved_output.zip"
FULL_PATH="${SUBDIRECTORY}/${CM_FILE_NAME}"
gdown --id ${CM_FILE_ID} -O ${FULL_PATH}
unzip ${FULL_PATH} -d "$TEMP_DIR"
mv "$TEMP_DIR/colmap_saved_output/"* "$SUBDIRECTORY/colmap_saved_output"
rm ${FULL_PATH}

# # download dust3r results
# mkdir -p "dust3r_saved_output"
# D_FILE_ID="155n30aD1k37u7XIX78YlpgOBx9i8znkr"
# D_FILE_NAME="dust3r_saved_output.zip"
# D_FULL_PATH="${SUBDIRECTORY}/${D_FILE_NAME}"
# gdown --id ${D_FILE_ID} -O ${D_FULL_PATH}
# unzip ${D_FULL_PATH} -d "$TEMP_DIR"
# mv "$TEMP_DIR/dust3r_saved_output/"* "$SUBDIRECTORY/dust3r_saved_output"
# rm ${D_FULL_PATH}

# # download dust3r segmented results
# SUB_SUBDIRECTORY="output/dust3r_saved_output"
# DS_FILE_ID="18G0cpXU03METQKsDZmgjxGZ6smN0tu6Y"
# DS_FILE_NAME="dust3r_segmented_output.zip"
# DS_FULL_PATH="${SUB_SUBDIRECTORY}/${DS_FILE_NAME}"
# gdown --id ${DS_FILE_ID} -O ${DS_FULL_PATH}
# unzip ${DS_FULL_PATH} -d ${SUB_SUBDIRECTORY}
# rm ${DS_FULL_PATH}

# download tinysam mannual segmentation results
mkdir -p "$SUBDIRECTORY/tiny_sam_output"
TS_FILE_ID="1TcnNvTVP-JdQDYqIl1LYLvzLF4QF5rTf"
TS_FILE_NAME="tiny_sam_output.zip"
TS_FULL_PATH="${SUBDIRECTORY}/${TS_FILE_NAME}"
gdown --id ${TS_FILE_ID} -O ${TS_FULL_PATH}
unzip ${TS_FULL_PATH} -d "$TEMP_DIR"
mv "$TEMP_DIR/tiny_sam_finished_output/"* "$SUBDIRECTORY/tiny_sam_output"
rm ${TS_FULL_PATH}

sudo rm -r "$TEMP_DIR"
echo "Done"
