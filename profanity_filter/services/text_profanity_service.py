from modules.custom_word_model import profanity_filter
from modules.custom_doc_model import svm_classifier
import time

class text_profanity_svc:

    svm_classifier_obj = object()
    profanity_filter_obj = object()

    def __init__(self):
        self.svm_classifier_obj   = svm_classifier()
        self.profanity_filter_obj = profanity_filter()
        pass


    def infer(self, data):
        #check optional analysis attributes
        data = self.option_check(data)
        data = self.curate_text(data)
        word_analysis_result    = []
        content_analysis_result = []
        line_analysis_result    = []

        performance = {}
        
        t = time.time()
        if data['content_analysis'] == True:
            content_analysis_result = self.svm_classifier_obj.predict(data['text'])
        performance['content_analysis'] = time.time()-t
        t = time.time()

        if data['line_analysis'] == True:
            line_analysis_result    = self.svm_classifier_obj.line_analysis(data['text'])
        performance['line_analysis'] = time.time()-t
        t = time.time()

        if data['word_analysis'] == True:
            word_analysis_result    = self.profanity_filter_obj.predict(data['text'], data['phrase_analysis'])
        performance['word_analysis'] = time.time()-t
        t = time.time()
        
        return  {'possible_profanity' : word_analysis_result['profane_words'],
            'possible_profane_word_count' : len(word_analysis_result['profane_words']),
            'possible_profanity_frequency' :  word_analysis_result['frequecy'],
            'line_analysis' : line_analysis_result, 
            'overall_text_classification' : content_analysis_result, 
            'performance' : performance,
            'text_original' : data['text'],
            'code' : 200,
            'message' : 'Success'
            }


    def option_check(self, data):
        #instantiate absent attributes 
        if 'word_analysis' not in data:
            data['word_analysis']     = False
        if 'line_analysis' not in data:
            data['line_analysis']     = False
        if 'content_analysis' not in data:
            data['content_analysis']  = False
        if 'phrase_analysis' not in data:
            data['phrase_analysis']   = False

        #turn all analysis on if non are mentioned
        if not data['word_analysis'] and not data['line_analysis'] and not data['content_analysis'] and not data['phrase_analysis']:
            data['word_analysis']     = True 
            data['line_analysis']     = True 
            data['content_analysis']  = True
            data['phrase_analysis']   = True
        print(data)
        return data

    def retrain_profanity_filter(self):
        self.profanity_filter_obj.retrain()
        return
    
    def retrain_text_model(self):
        self.svm_classifier_obj.retrain_model()
        return
    
    def curate_text(self, data):
        return data