# encoding = "utf-8"
import torch.nn.functional as F
from nltk import word_tokenize
import re

def postprocess_function_by_prob(model_outputs, tokenizer, targets):
    '''
    output_token_probs: (batch_size, vocab_size)
    '''
    output_token_probs = F.softmax(model_outputs.scores[0], dim=-1)

    A_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("A"))[0]
    B_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("B"))[0]
    C_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("C"))[0]
    D_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("D"))[0]

    acc = 0
    predictions = []

    for output_token_prob, target in zip(output_token_probs, targets): # for each sample
        
        max_prob = 0
        max_id = None
        for (idx, choice) in [(A_id, "A"), (B_id, "B"), (C_id, "C"), (D_id, "D")]:
            prob = output_token_prob[idx]
            if prob>max_prob:
                max_prob = prob
                max_id = choice
        predictions.append(max_id)
        if max_id==target:
            acc += 1

    return acc, predictions

def postprocess_function_by_txt(predictions, targets, choices):

    acc = 0
    for pred, target, cur_choices in zip(predictions, targets, choices):

        cur_choices = eval(cur_choices)
        letter = None

        try:
            pred_words = word_tokenize(pred)

            for word in pred_words:
                if word in ["A", "B", "C", "D"]:
                    letter = word
                    break
            assert letter!=None

        except AssertionError:
            for choice_idx, choice in enumerate(cur_choices):
                if choice in pred:
                    letter = ["A", "B", "C", "D"][choice_idx]
                    break
            
        if letter and letter==target:
            acc += 1

    return acc
