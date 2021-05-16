import os
import logging
import sys
import subprocess

from position_rank import position_rank_mod
from tokenizer import StanfordCoreNlpTokenizer

import pke
from pke.utils import compute_document_frequency

from keybert import KeyBERT

import spacy
import en_core_web_sm
from textacy.ke import sgrank
from textacy.ke import scake
from textacy.ke import textrank, yake

from bert_serving.client import BertClient
path_uke = "unsupervised_keyword_extraction/" #path for the dir from  https://github.com/AnzorGozalishvili/unsupervised_keyword_extraction
sys.path.append(path_uke)
from model.embedrank_transformers import EmbedRankTransformers #uncomment to use EmbedRank

import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords

# import silence_tensorflow.auto
# from tensorflow.python.keras.models import Model, Sequential, model_from_json, load_model
# from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences



class KeyModel:
    """ Main Model class
    """
    def __init__(self):
        pass

    def normalize_weights(self, weights):
        max_value = max(weights)
        min_val = min(weights)
        normalized =[]
        for w in weights:
            new_w = (w-min_val)/(max_value-min_val)
            normalized.append(new_w)
        return  normalized



class PKE (KeyModel):
    """Class for pke models developed by Florian Boudin
    """

    def __init__(self, model_name, models=None, _df_counts=None, frequency_path = None):
        self.df = None

        self.model_name = "PKE_"+model_name
        if model_name.lower() == "topicrank":
            self.model = pke.unsupervised.TopicRank()
        elif model_name.lower() == "textrank":
            self.model = pke.unsupervised.TextRank()
        elif model_name.lower() == "singlerank":
            self.model = pke.unsupervised.SingleRank()
        elif model_name.lower() == "topicalpagerank":
            self.model = pke.unsupervised.TopicalPageRank()
        elif model_name.lower() == "positionrank":
            self.model = pke.unsupervised.PositionRank()
        elif model_name.lower() == "multipartiterank":
            self.model = pke.unsupervised.MultipartiteRank()
        elif model_name.lower() == "tfidf":
            self.model = pke.unsupervised.TfIdf()
            if frequency_path:
                self.df = pke.load_document_frequency_file(input_file=frequency_path)

        elif model_name.lower() == "kpminer":
            self.model = pke.unsupervised.KPMiner()
        elif model_name.lower() == "yake":
            self.model = pke.unsupervised.YAKE()
        elif model_name.lower() == "kea":
            self.model = pke.supervised.Kea()
        elif model_name.lower() == "wingnus":
            self.model = pke.supervised.WINGNUS()
        else:
            raise Exception("{} is not a model name defined in PKE library".format(model_name))
        if models:
            self.model._models = models
        else:
            self.model._models = os.path.join(os.path.dirname(pke.__file__), 'models')

        if _df_counts:
            self.model._df_counts = _df_counts

        else:
            self._df_counts = os.path.join(self.model._models, "df-semeval2010.tsv")

    def get_keywords(self, text, n=10, normalization='None'):
        self.model.load_document(text, language='en', normalization=normalization)

        # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
        # and adjectives (i.e. `(Noun|Adj)*`)
        self.model.candidate_selection()

        # candidate weighting, in the case of TopicRank: using a random walk algorithm
        if not self.df:
            self.model.candidate_weighting()
        else:
            self.model.candidate_weighting(df = self.df)

        # N-best selection, keyphrases contains the 10 highest scored candidates as
        # (keyphrase, score) tuples
        try:
            keyphrases, weights = list(map(list, zip(*self.model.get_n_best(n=n))))
        except:
            raise Warning("model was not able to retrieve keywords")
            keyphrases, weights = None, None
        return keyphrases, weights

    def create_doc_frequency(self, texts, outpath= "output.txt"):
        path = os.path.join(os.path.dirname(pke.__file__), 'models/temp/')
        for i in range(len(texts)):
            fpath = os.path.join(path,"f"+str(i)+".txt")
            open(fpath,"w",encoding="utf-8").write(texts[i])
        opath = os.path.join(path, outpath)
        compute_document_frequency(path, opath,"txt")


class keyBert (KeyModel):
    def __init__(self):
        self.model = KeyBERT ('allenai/scibert_scivocab_uncased') #('distilbert-base-nli-mean-tokens')
        self.model_name = "keyBert"
    def get_keywords(self, text, n=10):
        try:
            keyphrases, weights = list(map(list, zip(*self.model.extract_keywords(text,keyphrase_ngram_range=(1, 3), top_n=n))))

        except:
            logging.warning("model was not able to retrieve keywords")
            keyphrases, weights = None, None

        return keyphrases, weights


class Textacy (KeyModel):
    def __init__(self, model_name):
        self.nlp = en_core_web_sm.load()
        self.model_name = "Textacy_"+model_name

    def get_keywords(self, text, n=10):
        try:
            doc = self.nlp(text)
        except:
            raise Exception("Couldn't load Spacy model")

        try:
            if "sgrank" in self.model_name :
                keyphrases, weights = list(map(list, zip(*sgrank(doc, topn=n))))
            elif "scake" in self.model_name:
                keyphrases, weights = list(map(list, zip(*scake(doc, topn=n))))
            elif "textrank" in self.model_name:
                keyphrases, weights = list(map(list, zip(*textrank(doc, topn=n))))
            elif "yake" in self.model_name:
                keyphrases, weights = list(map(list, zip(*yake(doc, topn=n))))
            else:
                raise Exception("{} is not a model name defined in textacy library".format(self.model_name))
        except:
            logging.warning("model was not able to retrieve keywords")
            keyphrases, weights = None, None
        return keyphrases, weights


class BertEmbedRank (KeyModel):
    def __init__(self, pre_load=True, n=10, bert_path=None):
        """

        :param pre_load:
        :param n:

        bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 (to start the service)
        """
        self.model_name = "BertEmbedRank"
        if not bert_path:
            bert_path = "unsupervised_keyword_extraction/cased_L-12_H-768_A-12/"
        #try:
        #with open("stdout.txt", "wb") as out, open("stderr.txt", "wb") as err:
        #    self.p = subprocess.Popen(["bert-serving-start", "-model_dir", bert_path, "-num_worker=4"], stdout=out, stderr=err)
        self.bc = BertClient(output_fmt='list', timeout=120000)
        #except:
        #    logging.warning("BertClient is not working")

        self.nlp = spacy.load("en_core_web_lg", disable=['ner'])
        if pre_load:
            self.model = EmbedRankTransformers(nlp=self.nlp,
                                          dnn=self.bc,
                                          perturbation='replacement',
                                          emb_method='subtraction',
                                          mmr_beta=0.55,
                                          top_n=n,
                                          alias_threshold=0.8)
        else:
            self.model=None

    def get_keywords(self, text, n=10):
        if not self.model:
            self.model = EmbedRankTransformers(nlp=self.nlp,
                                   dnn=self.bc,
                                   perturbation='replacement',
                                   emb_method='subtraction',
                                   mmr_beta=0.55,
                                   top_n=n,
                                   alias_threshold=0.8)

        #marked_target, keywords, keyword_relevance = self.model.fit(text)

        keyphrases, weights = list(map(list, zip(*self.model.extract_keywords(text))))

        return  keyphrases, weights

    def exit(self):
        self.p.terminate()



class BiLSTM (KeyModel):
    def __init__(self, json_config_f, model_f, tokenizer_f):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.stop_words = set(stopwords.words('english'))
        with open(json_config_f) as json_file:
            json_config = json_file.read()
        self.model = model_from_json(json_config)
        self.model_name = "BiLSTM"
        # Load weights
        self.model.load_weights(model_f)

        tokenizer = Tokenizer()
        with open(tokenizer_f, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def get_keywords(self, text, n=10):
        new_t = Tokenizer()
        new_t.fit_on_texts([text])
        tokens = [i for i in new_t.word_index.keys()]

        actual_tokens = new_t.texts_to_sequences([text])
        inv_map_tokens = {v: k for k, v in new_t.word_index.items()}
        actual_tokens = [inv_map_tokens[i] for i in actual_tokens[0]]
        tokens = actual_tokens
        input_ = self.tokenizer.texts_to_sequences([text])
        input_ = pad_sequences(input_, padding="post", truncating="post", maxlen=25, value=0)
        output = self.model.predict([input_])
        output = np.argmax(output, axis=-1)
        where_ = np.where(output[0] == 1)[0]
        output_keywords = np.take(tokens, where_)
        output_keywords = [i for i in output_keywords if i not in self.stop_words]
        output_keywords = list(set(output_keywords))
        k = output_keywords[:n]
        return k, [0.0]*len(k)



class BibRank(KeyModel):
    """
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
    """
    def __init__(self, weights=None, wiki=False):
        self.tokenizer = StanfordCoreNlpTokenizer("http://localhost", port = 9000)
        self.weights = weights
        self.wiki = wiki
        self.model_name = "PositionRankMod"
    def get_keywords(self, text, n=10):
        keyphrases_p = position_rank_mod(text.split(), self.tokenizer, alpha=0.85, window_size=2, num_keyphrase=n, weights=self.weights,
                                     wiki=self.wiki)
        return list(map(list, zip(*keyphrases_p)))

text = '''
   Over the past few years, there has been a strong and growing interest
   in faster network technologies such as FDDI and ATM. However, the
   perceived throughput at the application level has not always increased
   accordingly. Various performance bottlenecks have been encountered
   each of which has to be analysed and corrected.

   This paper presents a performance evaluation of continuous video data
   streams over ATM networks and compares them with similar experiments
   over Ethernet networks. The ATM LAN testbed for these experiments
   consists of three SG R4000 Indigos workstations connected by ATM Fore
   first generation interface cards and a Synoptics ATM switch. It is
   expected that video applications would run faster on ATM networks than
   Ethernet networks. A preliminary examination suggested to us that data
   movements through the protocol stack and the processing overheads were
   considerably high and that with some parameter settings, experiments
   with Ethernet would perform four to five times better than those with
   ATM.

   To address and explore this issue, a packetisation process has been
   added within the video application that would split each video frame
   into a number of packets. We show that by implementing a packetisation
   process at the application level, the end host is able to deliver
   cells at a faster rate. The sensitivity of these parameters, and also
   the overheads involved are discussed. The results are consistent over
   a range of video frame sizes: CIF(25K), QCIF(101K), and SCIF(405K).


'''

'''
extractor = Textacy ("yake")
k,w = extractor.get_keywords(text)
print(k)
print(w)
'''

