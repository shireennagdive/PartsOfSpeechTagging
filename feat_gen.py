#!/bin/python
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from metaphone import doublemetaphone
import inflect

p = inflect.engine()

nltk.download('wordnet')

domain_extensions = ['.com', '.org', '.in', '.ly', '.net', '.us', '.int', '.mil', '.edu', '.gov', '.biz', '.info',
                     '.jobs', '.mobi', 'name', '.tel', '.kitchen', '.email', '.tech', '.estate', '.bid', '.html', '.co',
                     '.me', '.gl', '.mp', '.fm', '.fb']

start_of_website = ('http', 'https', 'www')

abbreviations = {'ab', 'abt', 'afaik', 'ayfkmwts', 'b4', 'bfn', 'bgd', 'bh', 'br', 'btw', 'cd9', 'chk', 'cul8r', 'dp',
                 'f2f', 'ffs', 'fotd', 'ftw', 'idk', 'lmao', 'lol', 'omfg', 'orly', 'rt', 'cc', 'cx', 'dm', 'ht', 'mt',
                 'prt', 'em', 'fb', 'li', 'sm', 'smm', 'smo', 'stfu', 'tldr', 'tl;dr', 'yolo', 'yoyo'}

adverb_suffixes = {'ly', 'wise', 'ward'}

noun_suffixes = {'acy', 'al', 'ance', 'ence', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship',
                 'stion', 'tion', 'ian', 'eer', 'age', 'hood', 'ion'}

verb_suffixes = {'ate', 'en', 'fy', 'ize', 'ise', 'ing'}

adjective_suffixes = {'able', 'ible', 'al', 'ful', 'ic', 'ical', 'ous', 'ious', 'ish', 'ive', 'less', 'ate'}

quantifiers = {'all', 'any', 'both', 'each',
               'enough', 'every', 'few', 'fewer', 'little', 'less',
               'lot', 'lots', 'many', 'much',
               'more', 'no', 'several', 'some', 'plenty'}

prefixes = {'ante', 'anti', 'circum', 'co', 'de', 'dis', 'em', 'en',
            'epi', 'ex', 'extra', 'fore', 'homo', 'hyper', 'il', 'im', 'in',
            'ir', 'im', 'in', 'infra', 'inter', 'intra', 'macro', 'micro', 'mid', 'mis',
            'mono', 'non', 'omni', 'para', 'post', 'pre', 're', 'semi', 'sub', 'sub',
            'super', 'therm', 'trans', 'tri', 'un', 'uni'}


def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """


def token2features(sent, i, add_neighs=True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    ftrs = ["BIAS"]
    # bias
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent) - 1:
        ftrs.append("SENT_END")
    metaphones = doublemetaphone(sent[i])
    for metaphone in metaphones:
        if metaphone != "":
            ftrs.append(metaphone)

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    stemmed_word = porter.stem(word)
    ftrs.append("STEMMED_WORD=" + stemmed_word)
    ftrs.append("LEMMATIZED_WORD=" + lemmatizer.lemmatize(word))

    # some features of the word
    if word.isalpha():
        singular = p.singular_noun(sent[i])
        if singular and singular != sent[i]:
            ftrs.append("SINGULAR=" + singular.lower())
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
        listOfWords = p.number_to_words(word, group=1)
        listOfWords = listOfWords.split(",")
        for i in range(len(listOfWords)):
            ftrs.append("IS_" + listOfWords[i].upper())
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    if word.startswith("#"):
        ftrs.append("IS_HASHTAG")
    if word.startswith("@"):
        ftrs.append("IS_MENTION")
    if word.lower() in abbreviations:
        ftrs.append("IS_ABBREVIATION")
    if word in quantifiers:
        ftrs.append("IS_QUANTIFIER")
    for adverb in adverb_suffixes:
        if word.endswith(adverb):
            ftrs.append("IS_ADVERB")
            break
    for verb_sfx in verb_suffixes:
        if word.endswith(verb_sfx):
            ftrs.append("IS_VERB")
            break
    for adjective_sfx in adjective_suffixes:
        if word.endswith(adjective_sfx):
            ftrs.append("IS_ADJECTIVE")
            break
    for noun_sfx in noun_suffixes:
        if word.endswith(noun_sfx):
            ftrs.append("IS_NOUN")
            break
    for extension in domain_extensions:
        if word.startswith(start_of_website) or word.endswith(extension) or word.__contains__(extension + "/"):
            ftrs.append("IS_URL")
            break
    if word.endswith("!"):
        ftrs.append("IS_EXCLAMATION")
    if word.endswith("."):
        ftrs.append("IS_PERIOD")
    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i - 1, add_neighs=False):
                ftrs.append("PREV_" + pf)
        if i < len(sent) - 1:
            for pf in token2features(sent, i + 1, add_neighs=False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs


if __name__ == "__main__":
    sents = [
        ["There", "are", "so", "many", "people", "and", "lots", "of", "food"]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
