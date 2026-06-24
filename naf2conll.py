import sys
from KafNafParserPy import KafNafParser
import csv
import glob
import os

def naf_to_conll(input_folder, output_folder):
    """
    Read sentences from all naf files in a folder and write them to a new file in a folder in CoNLL-style format.

    :param input_folder: folder containing only naf files
    :param output_folder: folder where the conll formatted files will be stored
    :returns: nothing, only writes to file
    """
    
    #create the output folder
    os.makedirs(output_folder, exist_ok=True)

    #iterate over all files in the folder
    for filename in glob.glob(f'{input_folder}/*'):
        #create nafobj so we can extract the information
        nafobj = KafNafParser(filename)
        #initiate list to store information
        csv_list = []

        #iterate over each term and get the information
        for term in nafobj.get_terms():
            conll = []
            tid = term.get_id()
            tspan = term.get_span().get_span_ids()
            surface = ''

            #iterate over each word, and retrieve token, lemma, pos and surface form
            for wid in tspan:
                token = nafobj.get_token(wid)
                lemma = term.get_lemma()
                pos = term.get_pos()
                surface += token.get_text() + ''
            conll.append(tid)
            conll.append(surface)
            conll.append(lemma)

            #check for predicates, since the SRL model is not trained for predicate identification
            if pos == "verb":
                conll.append('V')
            else:
                conll.append('_')
            
            #append the conll style sentence to the csv list to create a list of lists
            csv_list.append(conll)

            #append an empty line after each period - not bulletproof 
            if surface == '.':
                csv_list.append([])

        # create output filename to save the original file ID number
        original_name = os.path.basename(filename)
        output_name = original_name + ".conll"
        output = os.path.join(output_folder, output_name)

        # write to csv file
        with open(output, 'w', encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            w.writerows(csv_list)

        print(f"Written: {output}")

def main(argv=None):
    
    if argv is None:
        argv = sys.argv
            
    inputfile = argv[1]
    outputfile = argv[2]
    
    naf_to_conll(inputfile, outputfile)
    
if __name__ == '__main__':
    main()