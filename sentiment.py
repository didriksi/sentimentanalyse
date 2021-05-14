import os
import pandas as pd
import spacy
import re
import string
import fasttext
import random
from tqdm import tqdm
from itertools import groupby
import pickle
import numpy as np
from sklearn.metrics import r2_score

def get_dataset(path="data"):
    """Find metadata.json file, and make pd.DataFrame from it.
    
    :param path: Path to the metadata.json and authors.csv files.

    :return: pd.DataFrame, with rows for each document.
    """
    print(f"Loading {path}/metadata.json...")
    dataset = pd.read_json(f"{path}/metadata.json").T
    dataset = dataset[dataset.language == "nb"]
    dataset["rating"] = dataset["rating"].astype(int)
    print("Adding the manually labled genders...")
    authors_df = pd.read_csv(f"{path}/authors.csv", index_col="name")
    dataset["gender"] = dataset.authors.apply(
        lambda authors: set_gender(authors, authors_df)
    )

    return dataset

def get_stop_words(path="data"):
    """Extract stop words in bokmål.

    Uses a file that can be found at
    `http://snowball.tartarus.org/algorithms/norwegian/stop.txt`.
    """
    stop_words = []
    with open(f"{path}/stop_words.txt", "r") as stop_words_file:
        for line in stop_words_file:
            if len(line) >= 2 and line[2] != "|":
                stop_word, explanation, = line.split("|")
                if len(stop_word) > 1 and explanation[-2] != "*":
                    stop_words.append(stop_word.strip())

    return stop_words

def set_gender(authors, authors_df):
    """Gives gender of a list of authors based on labeled names in a df.
    
    :param authors: [Str,] where each element is an author
    :param authors_df: pd.DataFrame with names as index and a column with gender info.
    
    :return: `m` for male, `k` for female, or `u` for unknown or ambigous.
    """
    if len(authors) == 1:
        return authors_df.gender[authors[0]]
    else:
        gender = authors_df.gender[authors[0]]
        for author in authors[1:]:
            if gender != authors_df.gender[author]:
                return "u"
        return gender

def gender_to_desc(gender_letter):
    """Returns a descriptive text for a gender_letter.

    :param gender_letter: Either `m`, `k` or `u`.

    :return: Str
    """
    if gender_letter == "k":
        return "kvinne"
    elif gender_letter == "m":
        return "mann"
    else:
        return "ukjent/flere forfattere av forskjellige kjønn"

def process_documents(doc, nlp=None, remove_newlines=False):
    """Tokenize and lemmatize.
    
    :param document: String.
    :param remove_newlines: Bool, whether to remove newline characters.
    :param nlp: Function Str -> iter(Str,)
    
    :return: [Spacy Doc? object,]
    """
    
    assert isinstance(doc, str), f"doc has to be of type str, {type(doc)} is not supported."
    
    if nlp is None:
        nlp = spacy.load("nb_core_news_sm")
    
    if remove_newlines:
        doc = re.sub("\n", " ", doc)
    
    return nlp(doc)

def clean_document(doc, stop_words, **process_kwargs):
    """Cleans up document, normalising words and removing stop_words.
    
    :param doc: [Str]
    :param stop_words: [Str] to exclude
    :param **process_kwargs: Passed to process_documents
    
    :return: [[Str]], list of tokens for each doc
    """
    full_list = []
    for token in process_documents(doc, **process_kwargs):
        if token.lemma_ not in stop_words:
            if token.lemma_ in string.punctuation:
                full_list.append(token.lemma_)
            else:
                full_list.append(f" {token.lemma_}")
    
    return full_list

def get_documents(dataset=None, path="data", ret=["rating", "authors"]):
    """Get documents of a specific type.
        
    :param path: Str path to folder with test and train folders.
    :param dataset: Determines which type to look for. Either `train` or `test`.
    :param ret: Columns from each document to return alongside the document itself. [Rating and authors]
    
    :return: iter(Str,) of documents
    """
    
    if dataset is None:
        dataset = pd.read_json(f"{path}/metadata.json").T
    
    full_path = f"{path}/%s/%s.txt"
    
    for (_, review) in dataset.iterrows():
        document = open(f"{path}/{review['split']}/{str(review['id']).zfill(6)}.txt", "r").read()
        yield (document, *[review[col] for col in ret])

def make_processed_datasets(dataset, stop_words, nlp=None, debug=False):
    """Make fasttext-style dataset, with each line being a text.
    
    :param stop_words: List of stop words.
    :param debug: Set to true to only process first doc.
    :param dataset: pd.DataFrame of the metadata format
    
    Create files `../data/processed/<kwarg_key>.txt`, with each line
    being on the form `__label__<1-6> <a document, without linebreaks>.
    
    :return: None
    """

    if nlp is None:
        nlp = spacy.load("nb_core_news_sm")

    sentence_set = {gender: [] for gender in ["m", "k", "u"]}
    fasttext_set = {split: [] for split in ["train", "test", "dev"]}

    docs = get_documents(dataset = dataset, ret=["rating", "split", "gender"])
    for doc, rating, split, gender in tqdm(docs, total=len(dataset)):

        # This is a bit messy, because we want to have two very similar datasets
        # One for fasttext, which should be a lemmatized string, one line per review, and be split
        # into one training set, and one testing set
        # Another for word groupings, which should not be lemmatized, and stored as a nested
        # list, with sentences as one dimension, and tokens as the next. This should be split on gender.
        fasttext_list = []
        for sentence in process_documents(doc, nlp=nlp, remove_newlines=True).sents:
            sentence_list = []
            for token in sentence:
                if token.lemma_ in string.punctuation:
                    fasttext_list.append(token.lemma_)
                else:
                    if token.lemma_ not in stop_words:
                        fasttext_list.append(" " + token.lemma_)
                    sentence_list.append(token.text)
            sentence_set[gender].append(sentence_list)
        
        fasttext_set[split].append(f"__label__{rating} {''.join(fasttext_list)}\n")
        
        if debug:
            return fasttext_set, sentence_set
    
    for split, text_list in fasttext_set.items():
        random.shuffle(text_list)
    
        with open(f"data/processed/{split}.txt", "w") as file:
            file.write("\n".join(text_list))

    with open("data/sentence_sets.pkl", "wb") as handle:
        pickle.dump(sentence_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

def predict(doc, model, stop_words, nlp=None):
    """Process text, and use the model to predict a label.
    
    :param doc: Str
    :param model: Model with predict method.
    :param stop_words: List of words to exclude.
    :nlp: Function Str -> iter(Str,)
    """
    if nlp is None:
        nlp = spacy.load("nb_core_news_sm")
    
    clean_doc = "".join(clean_document(doc, stop_words, nlp=nlp, remove_newlines=True))
    prediction = model.predict(clean_doc)
    return int(prediction[0][0][-1]), clean_doc

def sort_by_value(unsorted_dict, descending=True):
    """Sorts dictionary by its value.

    :param unsorted_dict: Dict with numerical values
    :param descending: Bool, whether to sort descending or ascending. [True]
    
    :return: Sorted dict
    """
    return {k: v for k, v in sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=descending)}

def group_words(*lengths, **sentence_sets):
    """Groups words in lists of length == length for all lengths
        
    :param *length: Int. Number of words in each string to return.
    :param **sentence_sets: [Str], where each string is a sentence.
    
    :return: Dict {length: {gender: [word groups]}}
    """
    word_groups = {length: {gender: [] for gender in sentence_sets.keys()} for length in lengths}
    
    with tqdm(total = sum([len(sentence_set) for _, sentence_set in sentence_sets.items()])) as pbar:
        for gender, sentence_set in sentence_sets.items():
            for sentence in sentence_set:
                pbar.update(1)
                for length in lengths:
                    for i in range(len(sentence) - length):
                        word_groups[length][gender].append(" ".join([sentence[i+ii].lower() for ii in range(length)]))
            
            for length in lengths:
                word_groups[length][gender].sort()
                word_groups[length][gender] = {key: len(list(group)) for key, group in groupby(word_groups[length][gender])}
                word_groups[length][gender] = sort_by_value(word_groups[length][gender])
    return word_groups

def sentence_complexity(nlp=None, **sentence_sets):
    """Finds the average word count per sentence, and char count per word.
    
    :param nlp:
    :param **sentence_sets:
    
    :return: Dict with same keys as sentence_sets, and 2-tuples with avg word/sent and char/word as values.
    """
    
    if nlp is None:
        nlp = spacy.load("nb_core_news_sm")
        
    complexity = pd.DataFrame()
    for gender, sentence_set in sentence_sets.items():
        words = 0
        chars = 0
        for sentence in sentence_set:
            words += len(sentence)
            chars += len("".join(sentence))
                
        complexity.at[gender, "ord per setning"] = words / len(sentence_set)
        complexity.at[gender, "bokstaver per ord"] = chars / words
        
    return complexity

def word_ratios(ratio_threshold, absolute_threshold, **word_groups_dicts):
    """Finds the words that are used most in a group
        
    :param ratio_threshold: Float
    :param absolute_threshold: Int
    :param **word_groups: Dictionary with all unique groups, string : int number of occurences.
    
    :return: Dictionary with the 
    """
    split = list(word_groups_dicts.keys())
    split_combos = [(split[i], split[j]) for j in range(len(split)) for i in range(len(split)) if i != j]
    
    ratios = {}
    for split1, split2 in split_combos:
        ratios[split1 + split2] = {}
        for word_group in word_groups_dicts[split1]:
            if word_group in word_groups_dicts[split2] and word_groups_dicts[split2][word_group] >= absolute_threshold:
                ratio = word_groups_dicts[split1][word_group]/word_groups_dicts[split2][word_group]
                if ratio >= ratio_threshold:
                    ratios[split1 + split2][word_group] = ratio
        
        ratios[split1 + split2] = sort_by_value(ratios[split1 + split2])
    
    return ratios

def eec_generator(templates, people, feelings):
    """Generates corpus with all combinations of templates, people and feelings.
    
    :param templates: [Str]
    :param people: [(Str: id, Str: value)] 
    :param feelings: [(Str: id, Str: value)]
    
    :return: 2-tuple og corpus and ids
    """
    
    corpus = []
    ids = []
    for template in templates:
        for person in people:
            for feeling in feelings:
                corpus.append(template.format(person=person[1], feeling=feeling[1]))
                ids.append((template, person[0], feeling[0]))
    
    return corpus, ids

def test_model_gender_bias(model, dataset, stop_words, nlp=None):
    """Test the model on test split of dataset, and find r2, actual avg, and predicted avg.
    
    :param model: Fasttext model object.
    :param dataset: pd.DataFrame metadata-style object.
    :param stop_words: List of words to exclude.
    :nlp: Function Str -> iter(Str,)
    
    :return: pd.DataFrame with genders as index, and r2, actual avg and predicted avg as cols.
    """

    if nlp is None:
        nlp = spacy.load("nb_core_news_sm")

    dataset = dataset[dataset.split == "test"]
    genders = ["m", "k"]
    gender_i = {gender: 0 for gender in genders}
    ratings = {gender: np.zeros((sum(dataset.gender == gender), 2)) for gender in genders}
    docs = get_documents(path="../data", dataset=dataset[dataset.gender != "u"], ret=["rating", "gender"])
    for doc, actual_rating, gender in tqdm(docs, total=sum(dataset.gender != "u")):

        predicted_rating, _ = predict(doc, model, stop_words, nlp=nlp)
        ratings[gender][gender_i[gender]] = np.array([actual_rating, predicted_rating])
        gender_i[gender] += 1

    metrics = pd.DataFrame()
    for gender, ratings_array in ratings.items():
        metrics.at[gender, "r2"] = r2_score(ratings_array[:,0], ratings_array[:,1])
        metrics.at[gender, "actual_avg"] = ratings_array[:,0].mean()
        metrics.at[gender, "pred_avg"] = ratings_array[:,1].mean()
    
    return metrics

def find_words_in_docs(words, dataset, context_chars=200, max_occurences=3):
    """Find examples of a specific string occuring in datasets.
    
    :param words: Str, the words you are interested in seeing.
    :param context_chars: How many charaters to include on either side of the words, in the print. [100]
    :param max_occurences: How many occurences you want to find before stopping. [3]
    :param **datasets: pd.DataFrame metadata-style objects.
    
    Prints out the context as fast as it is found.
    
    :return: Tuple of ids where text was found.
    """
    ids = []
    docs = get_documents(path="../data", dataset = dataset, ret=["id", "excerpt"])
    for doc, doc_id, excerpt in tqdm(docs):
        if doc.find(words) != -1:
            words_idx = doc.find(words)
            print("---")
            print(f"\033[1m\033[95m{excerpt}: \033[0m")
            print("...", doc[max(0, words_idx-context_chars):words_idx], end="")
            print("\033[1m\033[4m", doc[words_idx:words_idx+len(words)], "\033[0m", end="")
            print(doc[words_idx+len(words): min(len(doc), words_idx+len(words)+context_chars)])
            ids.append(doc_id)
            if len(ids) >= max_occurences:
                print("---")
                print("Found max occurences, stopped before searching through entire dataset")
                return ids
    return ids

def test_regendered_docs(metadata_df, model, stop_words, nlp=None):
    """Test the model on manually altered reviews, where the gender is flipped.
    
    :param metadata_df:
    :param model: fasttext model object.
    :param nlp: :param nlp: Function Str -> iter(Str,)
    
    :return:
    """

    if nlp is None:
            nlp = spacy.load("nb_core_news_sm")

    index = ["male_predicted", "male_actual", "female_predicted", "female_actual"]
    regendered_df = pd.DataFrame(index=index)
    for _, _, files in os.walk("../data/gender_eval"):
        for filename in files:
            if filename[-5] in ["k", "m"]:
                doc_id = filename[:-5]
                regendered_doc = open(f"../data/gender_eval/{filename}").read()
                regendered_label = predict(regendered_doc, model, stop_words, nlp=nlp)[0]
                
                unchanged_doc = open(f"../data/gender_eval/{doc_id}.txt").read()
                unchanged_label = predict(unchanged_doc, model, stop_words, nlp=nlp)[0]
                
                if filename[-5] == "k":
                    regendered_df.at["female_predicted", doc_id] = regendered_label
                    regendered_df.at["male_predicted", doc_id] = unchanged_label
                    
                    regendered_df.at["male_actual", doc_id] = metadata_df.at[int(doc_id), "rating"]
                else:
                    regendered_df.at["male_predicted", doc_id] = regendered_label
                    regendered_df.at["female_predicted", doc_id] = unchanged_label
                    
                    regendered_df.at["female_actual", doc_id] = metadata_df.at[int(doc_id), "rating"]
                        
    return regendered_df.T

if __name__ == "__main__":
    dataset = get_dataset()

    print("Processing data...")
    nlp = spacy.load("nb_core_news_sm")
    stop_words = get_stop_words()
    make_processed_datasets(dataset, stop_words, nlp=nlp)
