import sys
from KafNafParserPy.srl_data import Csrl, Cpredicate, Crole
from KafNafParserPy.span_data import Cspan
from KafNafParserPy import *
import csv
import glob
import os

def conll2naf(original_folder, predictions, output_folder):
    """
    turns a conll file with srl predictions to a new naf file with an extra srl layer.

    :param original_folder: folder containing the original naf files without srl
    :param predictions: folder containing the conll files with srl predictions
    :param output_folder: folder where the new naf files will be stored
    :returns: nothing, only writes to file
    """
    os.makedirs(output_folder, exist_ok=True)

    for original_file in glob.glob(f"{original_folder}/*"):
        nafobj = KafNafParser(original_file)

        #match naf file that needs srl layer with the conll file that has the srl information
        base_og, ext_og = os.path.splitext(original_file)
        number_name_og = os.path.basename(base_og)

        #initiate counters to later add to the srl layer
        pred_counter = 0
        role_counter = 0

        #initiate matched_file
        matched_file = None

        #loop through the folder that contains the files with predictions, and save the ID_number of the file base_pred
        for pred_file in glob.glob(f"{predictions}/*"):
            base_pred, ext_pred = os.path.splitext(pred_file)
            number_name_pred = os.path.basename(base_pred)

            #check if files match
            if number_name_og == number_name_pred:
                matched_file = pred_file
                break

        #check if there is a prediction file, as it happens that no SRL predictions took place
        if matched_file is None:
            print(f"No prediction file found for original file: {number_name_og}")
            continue
        
        #initiate empty sentence to run through the sentences
        current_sentence = []

        # create the classes using KafNafParser srl_data and span_data.
        with open(matched_file, "r", encoding="utf-8") as infile:
            for line in infile:
                #check if we are at the end of a sentence
                if line.strip() == "":
                    if current_sentence:
                        #initiate predicate class and predicate span using the KafNafParser module
                        predicate = Cpredicate()
                        pred_span = Cspan()

                        # Find predicate and set its information
                        for columns in current_sentence:
                            if columns[3] == "V":
                                pred_counter += 1
                                predicate.set_id("p" + str(pred_counter))
                                pred_span.create_from_ids([columns[0]])
                                predicate.set_span(pred_span)

                        # Find semantic roles in the last column of the pred file, as the predictions are there
                        for columns in current_sentence:
                            semrole = columns[-1]

                            #count roles to set role_ids, check to filter that it is a semantic role prediction
                            if semrole != "_" and columns[3] != "V":
                                role_counter += 1

                                #initiate role class from KafNafParser module
                                role = Crole()
                                role.set_id("r" + str(role_counter))
                                role.set_semRole(semrole)

                                #initiate role span from KafNafParser module
                                role_span = Cspan()
                                role_span.create_from_ids([columns[0]])
                                role.set_span(role_span)

                                #add roles to the predicate class
                                predicate.add_role(role)

                        #add to the parsed nafobj
                        nafobj.add_predicate(predicate)

                    #empty sentence list to move on to the next
                    current_sentence = []

                else:
                    columns = line.strip().split("\t")
                    current_sentence.append(columns)
        
        # Add new linguistic processor for information at start of NAF file
        nafobj.create_linguistic_processor(layer="srl", name="XLM_RoBERTa SRL predictions", version="1.0")

        #save the nafobj with srl information to a new file
        output_file = os.path.join(output_folder, number_name_og + "_srl.naf")
        nafobj.dump(output_file)

        print(f"The SRL layer has been successfully created for file: {number_name_og}")

def main(argv=None):
    
    if argv is None:
        argv = sys.argv
            
    original_folder = argv[1]
    predictions = argv[2]
    output_folder = argv[3]
    
    conll2naf(original_folder, predictions, output_folder)
    
if __name__ == '__main__':
    main()
