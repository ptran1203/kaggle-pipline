
# Script used to download competition dataset

DATA="plant-pathology-2020-fgvc7"

WORK_DIR="/content"

pip uninstall -y kaggle
pip install kaggle --upgrade

echo "
{
    \"username\":\"$1\",
    \"key\":\"$2\"
}
" > kaggle.json

if [ ! -d ~/.kaggle ]; then
    mkdir ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
fi

if [ ! -d $WORK_DIR/dataset ]; then
    kaggle competitions download -c $DATA
    unzip $DATA.zip -d $WORK_DIR/dataset
    rm $DATA.zip
fi

# download_csv() {
#     if [ ! -f $WORK_DIR/$1 ]; then
#         kaggle competitions download -c siim-covid19-detection -f $1
#         if [ -f $1.zip ]; then
#             unzip $1.zip
#         fi

#         if [ ${PWD##*/} != "content" ]; then
#             mv $1 $WORK_DIR/.
#         fi
#     fi
# }

# download_csv train_study_level.csv
# download_csv train_image_level.csv