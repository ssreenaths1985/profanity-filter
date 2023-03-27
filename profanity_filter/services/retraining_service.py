import json
import pandas as pd
from services.text_profanity_service import text_profanity_svc

class retraining:

    def __init__(self):
        pass


    def add_words(self, data: list,  profanity_svc: text_profanity_svc):
        profanity_set = set(json.load(open('data/custom_profanity.json')))
        for word in data:
            word = str(word).lower().strip()
            profanity_set.add(word)
        json.dump(list(profanity_set), open('data/custom_profanity.json','w'))
        profanity_svc.retrain_profanity_filter()
        return profanity_svc

    def add_text(self, data, profanity_svc: text_profanity_svc):
        df  =  pd.read_csv('data/consolidated_data.csv', index_col=[0])
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        for row in data:
            df = df.append({"is_offensive": row['class'], "text" : row['text']}, ignore_index = True)
        df.to_csv('data/consolidated_data.csv')
        return
        
    def train_model(self, profanity_svc: text_profanity_svc):
        profanity_svc.retrain_text_model()
        return
