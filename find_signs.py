
import pandas as pd,numpy as np


import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.preprocessing import normalize



def strict_translate(encoded_list,  known_letters , debug:bool = False):
    my_encoded_list = encoded_list.copy()
    encoded_fragment = [str(element) for element in my_encoded_list[:len(known_letters)].copy()]
    
    for find, replace in zip(encoded_fragment,known_letters):
        for i,element in enumerate(my_encoded_list):
            if not isinstance(element, str) and str(element) == find:
                if debug:
                    print(f"replacing {element} with {replace}")
                    
                my_encoded_list[i] = replace
    
    decoded_text = ''.join([str(element) for element in my_encoded_list])
    return decoded_text


def try_clustering(sample_id: int, data : pd.DataFrame, savethreshold = 80):
        #get the text that's being signed
        phrase_string = data.loc[data["sequence_id"] == sample_id].phrase.values[0]
        
        #where's the "movie" stored?
        filename = data.loc[data["sequence_id"] == sample_id].file_id.values[0]
        
        #get the movie itself
        target_phrase = pd.read_parquet(f"input/train_landmarks/{filename}.parquet").loc[sample_id]
        
        #delete everything but the detected hand, and the frame number
        hand = target_phrase.filter(regex="hand|frame").copy().dropna(axis=1,how="all")
        
        #remove frames where there's very few hand points detected
        hand = hand.dropna(axis=0,thresh=20)
        
        #find the dimensionality of the dataset:
        feat = hand.filter(regex="hand").to_numpy()
        feat = normalize(np.nan_to_num(feat))
        n_components = min(feat.shape)
        # find unique letters (less one for " " space)
        unique_letters = len(set(phrase_string)) -1 
        
        if unique_letters <= n_components:
            pca  = PCA(n_components=n_components)
            pca.fit(feat)

            explained_variance = pca.explained_variance_
            
            #work out a cut-off threshold for PCA  - the last dimension that matters.
            #in this case I've selected the dimension that contributes 1% as much information as the first one - or 50 if that fails.
            n_components = next((i for i, dimension in enumerate(explained_variance) if dimension / explained_variance[0] <= 0.01), 50)

            pca = PCA(n_components=n_components)
            pca.fit(feat)
            x = pca.transform(feat)
            
            
           
            kmeans = KMeans(n_clusters=unique_letters,n_init="auto")
            kmeans.fit(x)
            
            cipher = []
            #let's only keep labels found in consecutive detections - discarding others as noise
            for i, element in enumerate(kmeans.labels_):
                if len(cipher) == 0 or (i+1 < len(kmeans.labels_) and element == kmeans.labels_[i+1] and element != cipher[-1]):
                    cipher.append(element)

            cipherstr = [str(element) for element in cipher]
            
            unique_string = ""
            for value in cipherstr:
                if value not in unique_string:
                    unique_string += value
                else:
                    break
            
            unrepeated_phrase = no_repeats(phrase_string.replace(" ",""))
            
            plain_subtext =  unrepeated_phrase[:(len(unique_string))]
            
            
            decoded_word = strict_translate(cipher,plain_subtext)
            
            print(f"We're supposed to get: {unrepeated_phrase}")
            print(f"We got:                {decoded_word}")
            this_sim = similarity(unrepeated_phrase,decoded_word)
            print(f"This is a { this_sim : .2f}% similarity.")
            
            if this_sim > savethreshold:
                save_gestures(feat,kmeans.labels_,cipher,unrepeated_phrase,decoded_word,sample_id)
            return this_sim
            
        else:
            #there's not enough frames for the number of letters
            return 0
        
def no_repeats(text: str): # removes consecutive repeated letters
    return "".join(dict.fromkeys(text))

def similarity(string1 : str, string2 : str):
    hits = 0
    for i, let in enumerate(string1):
        try:
            if string2[i] == let:
                hits += 1
        except:
            ... # the two strings are not the same length, so there's no "hit" in this context
    
    return 100 * hits / len(string1)


def save_gestures(hand_numpy,groups,cipher, unrepeated_phrase,decoded_word,sequence_id):
    print(f"writing {sequence_id} Pickle")
    gestures = {}
    
    for i, letter in enumerate(unrepeated_phrase):
        if letter == decoded_word[i]:
            cluster = cipher[i]
            #print(f"Letter: {letter}, Cluster: {cluster}")
            for gesture,group in zip(hand_numpy,groups):
                
                if group == cluster:
                    if letter in gestures:
                        glist = gestures[letter]
                        glist.append(gesture)
                        gestures[letter] = glist
                    else:
                        gList = [gesture]
                        gestures[letter] = gList
    
    with open(f'output/{sequence_id}.pickle', 'wb') as f:
        pickle.dump(gestures, f)


def main():
    cumulative_similarity = 0
    
    data = pd.read_csv("input/train.csv", delimiter=',', encoding='UTF-8')
    for sequence in data.sequence_id:
        print(f"Sequence ID: {sequence}")
        cumulative_similarity += try_clustering(sequence, data,75)
    
    print(f"Average Similarity was: {cumulative_similarity/len(data) : .2f}%")

if __name__ == "__main__":
    main()