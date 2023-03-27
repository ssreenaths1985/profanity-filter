from profanity_filter import ProfanityFilter
import spacy
import textacy
import json
import re


class profanity_filter:

    def __init__(self):
        self.nlp = spacy.load('en')
        self.profanity_filter = ProfanityFilter(nlps={'en': self.nlp})
        custom_p = json.load(open('data/custom_profanity.json'))
        self.custom_p_set = set()
        for profanity in custom_p:
            self.custom_p_set.add(profanity)

        self.profanity_filter.extra_profane_word_dictionaries = {'en' : self.custom_p_set}
        self.nlp.add_pipe(self.profanity_filter.spacy_component, last=True)
        


    def predict(self, data, ngram_flag):
        text = self.normalize(data) 
        doc = self.nlp(text)
        profane_words = set()
        frequecy = dict()
        #classifying spacy word tokens
        for token in doc:
                if token._.is_profane:
                    if token._.original_profane_word in profane_words:
                        frequecy[token._.original_profane_word] = frequecy[token._.original_profane_word] + 1
                    else:
                        profane_words.add(token._.original_profane_word)
                        frequecy[token._.original_profane_word] = 1
        #classifying n-gram tokens
        if ngram_flag:
            ngrams = self.ngram(text)
            for token in ngrams:
                token_str = str(token)
                if token_str in self.custom_p_set:
                    if token_str in profane_words:
                        frequecy[token_str] = frequecy[token_str] + 1
                    else:
                        profane_words.add(token_str)
                        frequecy[token_str] = 1

        #changing frequency object structure
        freq_obj = []
        for key, value in frequecy.items():
            freq_obj.append({'no_of_occurrence' : value, 'word' : key})

        result = {'profane_words' : list(profane_words), 'frequecy' : freq_obj}
        return result

    def ngram(self, text):
        ngrams = []
        doc = textacy.make_spacy_doc(text, lang='en')
        ngrams.extend(list(textacy.extract.ngrams(doc, 3)))
        ngrams.extend(list(textacy.extract.ngrams(doc, 2)))
        return ngrams

    def normalize(self, text):
        text = text.strip()
        text = text.lower()
        text = re.sub(r'[^\w\s]','', text)
        return text

    def retrain(self):
        self.nlp = spacy.load('en')
        self.profanity_filter = ProfanityFilter(nlps={'en': self.nlp})
        custom_p = json.load(open('data/custom_profanity.json'))
        self.custom_p_set = set()
        for profanity in custom_p:
            self.custom_p_set.add(profanity)

        self.profanity_filter.extra_profane_word_dictionaries = {'en' : self.custom_p_set}
        self.nlp.add_pipe(self.profanity_filter.spacy_component, last=True)
        return