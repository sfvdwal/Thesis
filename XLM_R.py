from transformers import AutoTokenizer
model_checkpoint = 'FacebookAI/xlm-roberta-base'
label_all_tokens = True
import numpy as np

from transformers import AutoModelForTokenClassification
xlmr = AutoModelForTokenClassification.from_pretrained("../SRL/XLMR_SRL_endu")


def duplicate_predicates(input_file, output_file):
    """
    Find sentences in a CoNLL formatted file that contain one or more predicates
    and duplicate the sentence once for each predicate.

    :param input_file: path to the CoNLL file
    :param output_file: path to output file
    :return: CoNLL type file with duplicated sentences
    """
    result = []
    current_sentence = []

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            # Check for sentence boundary
            if line.strip() == "":
                if current_sentence:
                    # Find all predicates in the current sentence
                    predicate_indices = []
                    for i, columns in enumerate(current_sentence):
                        if columns[-1] == "V":
                            predicate_indices.append(i)

                    # Duplicate sentence once for every predicate
                    if len(predicate_indices) >= 1:
                        for pred_i in predicate_indices:
                            predicate_token_id = current_sentence[pred_i][0]

                            for columns in current_sentence:
                                new_columns = columns.copy()

                                # Keep only the selected predicate
                                if columns[0] == predicate_token_id:
                                    new_columns[-1] = 'V'
                                else:
                                    new_columns[-1] = '_'

                                result.append("\t".join(new_columns) + "\n")

                            result.append("\n")

                # Clear for next sentence
                current_sentence = []

            else:
                # Add this line to current_sentence
                columns = line.strip().split("\t")
                current_sentence.append(columns)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(result)

    # print("The file has been successfully preprocessed")
    # print(f"Output written to: {output_file}")

def extract_sentences(input_file):
  """
  Extract sentences from a CoNLL file.

  :param input_file: The path to the CoNLL file
  :return: List of lists with sentences
  """
  sentences = []
  with open(input_file, "r", encoding="utf-8") as file:
    # Initiate the first sentence as an empty list
    current_sentence = []
    for line in file:
      # Check for each line if they are empty, indicating a sentence boundary
      if line.strip() == "":
        # Add the current sentence to the list of sentences
        sentences.append(current_sentence)
        # Empty the current sentence
        current_sentence = []
      else:
        # If our sentence is not yet complete,
        # add the token to the current sentence
        current_sentence.append((line.split()[1], line.split()[-1]))
  return sentences

def gold_labels(input_file):
  """
  Extract gold labels from a CoNLL file. Tokens that are not an argument, are represented with an underscore, others are represented with their respective label

  :param input_file: path to the CoNLL file
  :return: list of lists with gold labels per sentence
  """
  gold_labels = []
  current_sentence = []

  with open(input_file, 'r', encoding='utf-8') as file:
    for line in file:
      # Check the sentence boundary
      if line.strip() == '':
        gold_labels.append(current_sentence)
        current_sentence = []
      # Token is no predicate
      elif line.split('\t')[-1] == '_':
        current_sentence.append('_')
      # Token is a predicate
      else:
        columns = line.strip().split('\t')
        # print(columns)
        if columns[-1] == 'V':
          current_sentence.append('_')
        else:
          current_sentence.append(columns[-1])
        # print(current_sentence)

  return gold_labels

def augment_data(sentences):
  """
  Augment each word by adding PRED or NOPRED to the string.
  The two augments are:
  NOPRED: not a predicate
  PRED: identified predicate

  This augmentation approach was based on the research of Khandelwal and Sawant (2020).

  :param sentences: list of lists with sentences
  :return: list of lists with augmented sentences
  """
  sents = []
  for items in sentences:
      current_sent = []

      for item in items:
        word, pred = item

        if pred != 'V':
            new_word = '[NOP]' + word
        
        else:
            new_word = '[PRED]' + word

        current_sent.append((new_word))

      sents.append(current_sent)

  return sents


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

special_pred_tokens = {"additional_special_tokens": ["[NOP]", "[PRED]"]}
num_added = tokenizer.add_special_tokens(special_pred_tokens)

label_list = ['A0', 'A1', 'A2', 'A3', 'A4', 'AM-ADJ', 'AM-ADV', 'AM-CAU', 'AM-COM', 'AM-DIR', 'AM-DIS', 'AM-EXT', 'AM-LOC', 'AM-LVB', 'AM-MNR', 'AM-MOD', 'AM-NEG', 'AM-PRD', 'AM-PRP', 'AM-REC', 'AM-TMP', 'ARG1-DSP', 'ARG5', 'ARGA', 'ARGM-CXN', 'ARGM-GOL', 'ARGM-PRR', 'C-A1', 'C-A1-DSP', 'C-A2', 'C-ARG0', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADV', 'C-ARGM-COM', 'C-ARGM-CXN', 'C-ARGM-DIR', 'C-ARGM-EXT', 'C-ARGM-GOL', 'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-PRP', 'C-ARGM-PRR', 'C-ARGM-TMP', 'C-V', 'R-A0', 'R-A1', 'R-A2', 'R-AM-CAU', 'R-AM-LOC', 'R-AM-MNR', 'R-AM-TMP', 'R-ARG3', 'R-ARG4', 'R-ARGM-ADV', 'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-GOL', '_']

label2id = {"A0": 0, "A1": 1, "A2": 2, "A3": 3, "A4": 4, "AM-ADJ": 5, "AM-ADV": 6, "AM-CAU": 7, "AM-COM": 8, "AM-DIR": 9, "AM-DIS": 10, "AM-EXT": 11, "AM-LOC": 12, "AM-LVB": 13, "AM-MNR": 14, "AM-MOD": 15, "AM-NEG": 16, "AM-PRD": 17, "AM-PRP": 18, "AM-REC": 19, "AM-TMP": 20, "ARG1-DSP": 21, "ARG5": 22, "ARGM-CXN": 23, "ARGM-GOL": 24, "ARGM-PRR": 25, "C-A1": 26, "C-A1-DSP": 27, "C-A2": 28, "C-ARG0": 29, "C-ARG3": 30, "C-ARG4": 31, "C-ARGM-ADV": 32, "C-ARGM-COM": 33, "C-ARGM-CXN": 34, "C-ARGM-DIR": 35, "C-ARGM-EXT": 36, "C-ARGM-LOC": 37, "C-ARGM-MNR": 38, "C-ARGM-PRP": 39, "C-ARGM-TMP": 40, "C-V": 41, "R-A0": 42, "R-A1": 43, "R-A2": 44, "R-AM-CAU": 45, "R-AM-LOC": 46, "R-AM-MNR": 47, "R-AM-TMP": 48, "R-ARG3": 49, "R-ARG4": 50, "R-ARGM-ADV": 51, "R-ARGM-COM": 52, "R-ARGM-DIR": 53, "R-ARGM-GOL": 54, "_": 55}

def tokenize_and_align_labels(words, gold_labels):
    """
    Tokenise the input words and align the labels with the token ids.

    :param words: list of lists of words
    :param gold_labels: list of lists of gold labels
    :return: dictionary
    """
    tokenized_inputs = tokenizer(words, truncation=False, is_split_into_words=True)

    dictionary = {
        'input_ids' : tokenized_inputs["input_ids"],
        'attention_mask' : tokenized_inputs["attention_mask"],
    }
    labels = []
    word_ids_per_sentence =[]
    for i, (token, label) in enumerate(zip(words, gold_labels)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_ids_per_sentence.append(word_ids)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
            if word_id is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_id != previous_word_id:
                label_ids.append(label2id[label[word_id]]) # get the label of the original word index (word_id) that this subtoken belongs to. label == list, so we have to index it to get the right label
            # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
            else:
                 label_ids.append(label2id[label[word_id]] if label_all_tokens else -100)
            previous_word_id = word_id

        labels.append(label_ids)

    dictionary["labels"] = labels
    dictionary["word_ids"] = word_ids_per_sentence

    return dictionary

def dict2dataset(dictionary):
  """
  Convert a dictionary to a list of dictionaries per sentence, instead of per token.

  :param dictionary: dictionary
  :return: list of dictionaries
  """
  input = []
  for sentence, attention_mask, labels in zip (dictionary['input_ids'], dictionary['attention_mask'], dictionary['labels']):
    input.append({'input_ids':sentence, 'attention_mask': attention_mask, 'labels':labels})
  return input

def majority(lst):
  """
  Get the majority label from a list of labels.

  :param lst: List of labels
  :return: Majority label
  """

  count_dict = {}
  for label in lst:
    if label not in count_dict:
      count_dict[label] = 1
    else:
      count_dict[label] += 1

  m = max(count_dict, key=count_dict.get)

  return m

def subtoken_to_token_predictions(predictions, word_ids, subtoken_ids):
  """
  Map subtoken predictions to token predictions.

  :param predictions: List of lists of subtoken predictions
  :param word_ids: List of lists of word ids
  :param subtoken_ids: List of lists of subtoken ids
  :return: List of lists of token predictions
  """
  # Remove ignored index (special tokens)
  special_pred_tokens = ["[PRED]", "[NOP]"]
  special_pred_ids = tokenizer.convert_tokens_to_ids(special_pred_tokens)
  normalized_predictions = [np.argmax(np.stack(sentence), axis=1) for sentence in predictions]
  
  token_predictions = []

  # Loop through the sentences
  for prediction, words, subtokens in zip(normalized_predictions, word_ids, subtoken_ids):
    previous_word_id = None
    sentence_prediction_ids = []
    current_word_preds = []

    #loop through the subtokens of each sentence
    for pred,word_id,subtoken_id in zip(prediction,words,subtokens):
        
        # exclude special tokens from the majority vote.
        if word_id is None: 
           continue
        
        if subtoken_id in special_pred_ids:
           continue
        
        # if word_id != previous_word_id, it means we have moved on to a different token in the sentence, and as a result we have collected all subtokens of a single token
        if word_id != previous_word_id:
          # if current_word_preds holds information
          if current_word_preds:
            # We set the label by majority voting
            sentence_prediction_ids.append(majority(current_word_preds))

            # initiate the collecting of predictions for the next pred and word_id loop
          current_word_preds = [pred]
        else:
        # if word_id == previous_word_id, we are still looking at a subtoken of the same token, so collect the predictions in a list
            current_word_preds.append(pred)

        #move to the next token
        previous_word_id = word_id

    #because we only append the majority vote of predictions when a new word starts, it means we would structurally miss out the last token of each sentence
    #by executing this line of code at the end of a sentence loop, we make sure to include the predictions for the subtokens of the last token of each sentence
    if current_word_preds:
       sentence_prediction_ids.append(majority(current_word_preds))

    #collect all majority vote sentence predictions into the token_predictions list that will be returned, meaning we will return a list of lists with numerized predictions
    token_predictions.append(sentence_prediction_ids)

  return token_predictions

id2label = {0: 'A0', 1: 'A1', 2: 'A2', 3: 'A3', 4: 'A4', 5: 'AM-ADJ', 6: 'AM-ADV', 7: 'AM-CAU', 8: 'AM-COM', 9: 'AM-DIR', 10: 'AM-DIS', 11: 'AM-EXT', 12: 'AM-LOC', 13: 'AM-LVB', 14: 'AM-MNR', 15: 'AM-MOD', 16: 'AM-NEG', 17: 'AM-PRD', 18: 'AM-PRP', 19: 'AM-REC', 20: 'AM-TMP', 21: 'ARG1-DSP', 22: 'ARG5', 23: 'ARGM-CXN', 24: 'ARGM-GOL', 25: 'ARGM-PRR', 26: 'C-A1', 27: 'C-A1-DSP', 28: 'C-A2', 29: 'C-ARG0', 30: 'C-ARG3', 31: 'C-ARG4', 32: 'C-ARGM-ADV', 33: 'C-ARGM-COM', 34: 'C-ARGM-CXN', 35: 'C-ARGM-DIR', 36: 'C-ARGM-EXT', 37: 'C-ARGM-LOC', 38: 'C-ARGM-MNR', 39: 'C-ARGM-PRP', 40: 'C-ARGM-TMP', 41: 'C-V', 42: 'R-A0', 43: 'R-A1', 44: 'R-A2', 45: 'R-AM-CAU', 46: 'R-AM-LOC', 47: 'R-AM-MNR', 48: 'R-AM-TMP', 49: 'R-ARG3', 50: 'R-ARG4', 51: 'R-ARGM-ADV', 52: 'R-ARGM-COM', 53: 'R-ARGM-DIR', 54: 'R-ARGM-GOL', 55: '_'}

def final_predictions(new_predictions):
  """
  Change all labels to the original SRL string labels, rather than the integers.

  :param labels: List of lists of id predictions of type integer
  :return: List of lists of final predictions of type string
  """
  final_preds = []
  current_sentence = []
  for sentence in new_predictions:
    for id in sentence:
      current_sentence.append(id2label[id])
    final_preds.append(current_sentence)
    current_sentence = []
  return final_preds

def predict_srl(file, trainer, outputfile):
  """
  Predict the semantic role labels for a given sentence using a trained BERT model.

  :param tokens: List of tokens in the sentence
  :param predicates: List of numerical values indicating the predicate
  :param model: Trained model
  :return: List of predicted labels
  """

  # Extract sentences and predicates from the file
  sentences = extract_sentences(file)
  gold = gold_labels(file)

  # it can happen that after duplicating the sentences, no sentences remain as none of them have a predicate. 
  # return nothing when this happens and continue in the outer loop of the notebook.
  if len(sentences) == 0:
      print(f"No predicates found in file: {file}")
      return

  # Augment the tokens by adding the PRED information
  aug_tokens = augment_data(sentences)

  # Tokenise the (sub)words
  tokenised = tokenize_and_align_labels(aug_tokens, gold)
  subtoken_ids = tokenised['input_ids']
  word_ids = tokenised['word_ids']

  # Transform the tokenised (sub)words into a dataset
  dataset = dict2dataset(tokenised)

  # Use the trained model to predict the labels
  predictions, labels, _ = trainer.predict(dataset)

  true_predictions = [[p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

  filtered_word_ids = [[w for (w, l) in zip(words, label) if l != -100] for words, label in zip(word_ids, labels)]

  filtered_subtoken_ids = [[s for (s, l) in zip(subtokens, label) if l != -100] for subtokens, label in zip(subtoken_ids, labels)]


  # Transform the subtokens back to tokens
  new_predictions = subtoken_to_token_predictions(true_predictions, filtered_word_ids, filtered_subtoken_ids)

  finals = final_predictions(new_predictions)

  # Flatten sentence-level predictions into one token-level list
  flat_predictions = [label for sentence in finals for label in sentence]

  # Write predictions to file, preserving sentence boundaries
  counter = 0
  with open(file, "r", encoding="utf-8") as infile, open(outputfile, "w", encoding="utf-8") as outfile:
        for line in infile:
            if line.strip() == "":
                outfile.write("\n")
            else:
                outfile.write(line.rstrip("\n") + "\t" + flat_predictions[counter] + "\n")
                counter += 1
  
