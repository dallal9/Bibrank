from stanfordcorenlp import StanfordCoreNLP
import re


class StanfordCoreNlpTokenizer(object):
    """Tokenizer for English using Stanford CoreNLP for tokenization.

    This class tokenize English sentence.
    As a default, tokenizer returns tokens whhic POS are adjective or noun.
    Simultaneously, tokenizer returns specific pattern phrases.

    """
    def __init__(self, url_or_path, port = 9000):
        """Initialize stanford core nlp tokenier.

        Args:
          url_or_path: Url string of path string of Stanford CoreNLP library.
            Provide url string if you already stand up Stanford CoreNLP server.
            If not, provide path to directory of library i.e. JavaLibraries/stanford-corenlp-full-2017-06-09/.
            When you provide path of librart, Stanford CoreNLP server will be up independent of python process.

        """
        self.tokenizer = StanfordCoreNLP(url_or_path, port = port)

    def tokenize(self, sentence, pos_filter=["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"]):
        """Tokenize sentence.

        Tokenize sentence and return token list and phrase list.
        Phrase is continuous tokens which have specific POS pattern '(adjective)*(noun)+'
        and length are more than 3.
        You can edit filter of token POS, default are limited to adjective and noun.

        Args:
          sentence: English sentence.
          pos_filter: POS filter of token. Default are adjective and noun.

        Returns:
          Token list: Filterd by 'pos_filter' param.
          Phrase list: Specific continuous tokens.

        """
        tokens = self.tokenizer.pos_tag(sentence)
        pos_tags = [self._anonymize_pos(token[1]) for token in tokens]
        pattern = r"J*N+"
        iterator = re.finditer(pattern, "".join(pos_tags))
        phrases = filter(lambda x: len(x) <= 3, [[token[0] for token in tokens[match.start():match.end()]] for match in iterator])
        phrases = ["_".join(phrase) for phrase in phrases]
        return [token[0] for token in tokens if token[1] in pos_filter], phrases

    def _anonymize_pos(self, pos):
        """Anonymize POS tags.
        Adjective tags are replaced to 'J', noun are to 'N', and others are to 'O'.

        """
        if (pos == "JJ") or (pos == "JJR") or (pos == "JJS"):
            return "J"
        elif (pos == "NN") or (pos == "NNS") or (pos == "NNP") or (pos == "NNPS"):
            return "N"
        else:
            return "O"
