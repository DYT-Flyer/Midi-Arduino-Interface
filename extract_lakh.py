import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pretty_midi
import warnings
import os

def get_genres(path):
    ids = []
    genres = []
    with open(path) as f:
        line = f.readline()
        while line:
            if line[0] != '#':
                [x, y, *_] = line.strip().split("\t")
                ids.append(x)
                genres.append(y)
            line = f.readline()
    genre_df = pd.DataFrame(data={"Genre": genres, "TrackID": ids})
    return genre_df
# Get the Genre DataFrame
genre_path = "C:/Users/remove/Downloads/msd_tagtraum_cd1.cls/msd_tagtraum_cd1.cls"
genre_df = get_genres(genre_path)

def get_matched_midi(midi_folder, genre_df):
    # Get All Midi Files
    track_ids, file_paths = [], []
    for dir_name, subdir_list, file_list in os.walk(midi_folder):
        dir_name_temp = dir_name[26:]
        if len(dir_name_temp) == 36:
            track_id = dir_name_temp[18:]
            file_path_list = ["/".join([dir_name, file]) for file in file_list]
            for file_path in file_path_list:
                track_ids.append(track_id)
                file_paths.append(file_path)
                
    all_midi_df = pd.DataFrame({"TrackID": track_ids, "Path": file_paths})
    
    # Inner Join with Genre Dataframe
    df = pd.merge(all_midi_df, genre_df, on='TrackID', how='inner')
    return df.drop(["TrackID"], axis=1)

# Obtain DataFrame with Matched Genres to File Paths
midi_path = "C:/Users/remove/Downloads/lmd_matched/"
matched_midi_df = get_matched_midi(midi_path, genre_df)

# Print to Check Correctness
print(matched_midi_df.head())


files = os.listdir('C:/Users/remove/Downloads/adl-piano-midi/adl-piano-midi/Classical')
root = 'C:/Users/remove/Downloads/adl-piano-midi/adl-piano-midi/Soul/'
folders = os.listdir('C:/Users/remove/Downloads/adl-piano-midi/adl-piano-midi/Soul/')
for folder in folders:
    artists = os.listdir(root+folder)
    for artist in artists:
        files = os.listdir(root+folder+'/'+artist)
        for file in files:
            os.replace(root+folder+'/'+artist+'/'+file,'C:/Users/remove/Documents/GitHub/Midi-Arduino-Interface/Dataset/Jazz/'+file)















