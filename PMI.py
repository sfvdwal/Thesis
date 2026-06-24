import pandas as pd
import spacy
from collections import defaultdict
from collections import Counter
import math
import sys
import glob
import os

nlp = spacy.load("nl_core_news_md")

def preprocess(text):
    """ 
    preprocesses text to remove stopwords, punctuation, make everything lowercase and lemmatize each token.
    This is done using the Spacy library to check for stopwords, punctuation etc. 

    :param text: string
    :returns: list of strings
    """

    doc = nlp(str(text).lower())

    tokens = []

    for token in doc:

        if token.is_stop:
            continue

        if token.is_punct:
            continue

        if token.is_space:
            continue

        lemma = token.lemma_.lower()

        if len(lemma) > 1:
            tokens.append(lemma)

    return tokens

def read_folder(input_folder):
    """
    Read a folder with the Microportrait pipeline output csv files. 
    Uses glob to iterate over the whole folder.
    
    :param input_folder: path to to folder
    :returns: dataframe that has concatenated all files
    """
    files = glob.glob(os.path.join(input_folder, "*.csv"))
    dfs = []

    for file in files:
        current_df = pd.read_csv(file, sep = ";")
        dfs.append(current_df)
    
    df = pd.concat(dfs, ignore_index = True)

    return df

def create_nanoportraits(df):
    """
    creates nanoportraits from the dataframe and returns a dictionary

    :param df: pandas dataframe of microportraits pipeline output
    :returns: dictionary where each nanoportrait is the key, and has a nested dictionary with relations (key) and descriptions (value) as value
    """   
    
    #create double defaultdict since we are working with nested dictionaries
    nanoportraits = defaultdict(lambda: defaultdict(set))

    #iterate over the rows in the df, ignore the index of the rows
    for _, row in df.iterrows():
        
        pid = row['mp_identifier']
        relation = row["relation"]

        #preprocess the descriptions on stopwords, punct, etc.
        tokens = preprocess(row["description"])

        #add the tokens to the right relation in the right portrait by using update, as we are working with sets
        nanoportraits[pid][relation].update(tokens)
        
    return nanoportraits


def prepare_data(nanoportraits, query):

    """
    Prepares the dictionaries from create_nanoportraits to calculate the pmi scores.
    The query word is the label we are interested in to find its most associated descriptions.
    
    
    :param nanoportraits: dictionaries
    :param query: query word - label of interest
    :returns N: the amount of nanoportraits
    :returns df_term: amount of descriptions
    :returns cooccur: amount of cooccurrences between query and a description
    :returns query_count: amount of occurrences of the query label
    """
    
    #N = the total amount of nanoportraits, not descriptions, since we are interested how often something occurs within a portrait
    #namely: how often does this description occur together with the query word as defined in a label?
    # in other words, a portrait is one unit
    N = len(nanoportraits)

    #initiate counter dicts
    df_term = Counter()
    cooccur = Counter()
    
    #first loop through the nanoportraits to fill up df_term dict
    for portrait in nanoportraits.values():

        description_terms = set()

        #retrieve all relations and descriptions from each single portrait
        for relation, descriptions in portrait.items():

            description_terms.update(descriptions)

        #count for each term how often they occur
        for term in description_terms:
            df_term[term] += 1

    #second loop through nanoportraits to fill up cooccurrence dict
    for portrait in nanoportraits.values():

        description_terms = set()

        for relation, descriptions in portrait.items():

            description_terms.update(descriptions)
        

        for relation, descriptions in portrait.items():

            for description in descriptions:
                if description != query:
                    cooccur[(relation, description)] += 1

    return N, df_term, cooccur


def calculate_pmi(N, df_term, cooccur, query, min_freq=5):
    """
    calculate the pmi scores for each description
    
    :param N: total amount of nanoportraits
    :param df_term: count dict for the descriptions
    :param cooccur: count dict for cooccurrences with the query
    :param query_count: amount of times the query occurs
    :param min_freq: minimum frequency, default to 5
    :returns: dict with relation, descriptions, description counts, joint frequency and pmi scores
    """

    pmi_scores = []

    #retrieve how often the query term occurs in the term counter dict
    query_count = df_term[query]

    #loop through the cooccurrence dict, while holding on to both relation and description information
    for (relation, description), joint_freq in cooccur.items():

        #minimum co-occurrence frequency
        if joint_freq < min_freq:
            continue
        #make sure the description also occurs more than 5 times on its own
        if df_term[description] < min_freq:
            continue
        
        #probability of target word
        pw = query_count / N
        #probability of context word
        pc = df_term[description] / N
        #joint probability
        pwc = joint_freq / N

        #PMI formula
        pmi = math.log2(pwc / (pw * pc))

        #append all information to the list, to create a list of dicts
        pmi_scores.append({"relation": relation,
            "description": description,
            "description_count": df_term[description],
            "joint_freq": joint_freq,
            "pmi": pmi})

    return pmi_scores

def save_results(pmi_scores, output_file=None):

    """
    Saves the pmi score dictionaries into a csv file, returns the first 40 values.

    :param pmi_scores: dictionary
    :param output_file: path to file
    """

    results = pd.DataFrame(pmi_scores)

    #sort descending
    results = results.sort_values(
        by="pmi",
        ascending=False
    )

    if output_file:
        results.to_csv(output_file, index=False)

    #return the top 40 results, so those are immediately visible when running the script
    return results.head(40)

def main(argv=None):
    
    if argv is None:
        argv = sys.argv
            
    query = argv[1]
    input_folder = argv[2]
    output_file = argv[3]
    
    df = read_folder(input_folder)
    
    nanoportraits = create_nanoportraits(df)
    N, df_term, cooccur = prepare_data(nanoportraits, query)
    pmi_scores = calculate_pmi(N, df_term, cooccur, query)
    results = save_results(pmi_scores, output_file)
    print(results)   

if __name__ == '__main__':
    main()