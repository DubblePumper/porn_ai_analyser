# Video AI Analyser
An AI-powered video content analysis tool.


# Note
### my first language is dutch and not english. Because of this some variables, constants, functions and comentations may be in dutch. feel free to put them in english
### Also a lot is done with chatgpt and github copilot. Not everything. just things i didnt knew.
## TODO
### bra size Recognition see:
- https://github.com/gg46power/Oppai-dataset
- https://github.com/samiurprapon/BraSizePrediction
- data is available in the performers.json file

### Height and weight Recognition see:
- https://github.com/canaltinigne/HeightWeightFinder

### cloth Recognition see:
- https://github.com/kritanjalijain/Clothing_Detection_YOLO?tab=readme-ov-file
- https://github.com/Rabbit1010/Clothes-Recognition-and-Retrieval
- https://github.com/normalclone/fashion-ai-analysis

### Scene Recognition see:
- https://github.com/shreyagu/Scene_Recognition
- https://github.com/flytxtds/scene-recognition
- https://github.com/vpulab/Semantic-Aware-Scene-Recognition


### Porn Human Action Recognition
- https://github.com/ryanjay0/miles-deep
- https://github.com/rlleshi/phar

### Age and gender detection:
- https://github.com/rlleshi/phar
- https://github.com/Ebimsv/Facial_Age_estimation_PyTorch
- https://github.com/Aayush3014/Gender-and-Age-Detection
- data is available in the performers.json file


## Project Structure
### It is a bit of a mess right now. i usally don't work with so mutch files.
- key map is /app
- /app/ai_things are pretrained models (i think)
- /app/datasets/ has everything that connects with datasets.
  - recognize_person and the dataset.csv is old. i now use pornstar_images and performers_details_data.json
- /app/get_data/ is things with data for the datasets, 
  - download_videos.py is to download test porn videos.
  - scrape_pornhub.py is for  /app/get_data/datasets: get list of every performer on pornhub
  - scrape_pornstar.py is to get the images of app\get_data\datasets\all_performers.json
- aitrainingv2.py is the file to train the ai to recognise a pornstart with tensorflow -- currently worked on
- trainaionimageanddescofpeople.py - old file to train ai with pytorch
- recognisePerson.py - idk what this is
