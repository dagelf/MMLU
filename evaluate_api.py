import argparse
import os
import numpy as np
import pandas as pd
from categories import subcategories, categories
import time
import requests
import json
import sys
import torch

from crop import crop

choices = ["A", "B", "C", "D"]

def queryLLM(prompt):
    # Define the URL to send the request to
    url = "http://llm:5000/api/v1/generate"

    # Define the JSON payload to send in the request
    payload = {
        "prompt": prompt,
        "temperature": .9,
        "max_new_tokens": 1,
    }

    # Set the timeout for the request (in seconds)
    timeout = 60

    # Send the POST request with the JSON payload and timeout
    response = requests.post(url, json=payload, timeout=timeout)

    text_value=''

    # Check if the request was successful (i.e. the response status code is 200)
    if response.status_code == 200:
        # If the request was successful, print the response content

        # Convert the response data from bytes to a string
        response_str = response.content.decode('utf-8')

        # Parse the JSON data into a Python data structure
        response_dict = json.loads(response_str)

        # Extract the value of the "text" key from the data structure
        text_value = response_dict['results'][0]['text']
        answer = text_value.strip()

        # print("Query : >" + prompt + "<\n" )
        # print("Answer: >" + answer + "<\n"+ "\n" )

        return answer

    else:
        # If the request was not successful, print an error message
        print("Error: Request failed with status code", response.status_code)

    return response.status_code, text_value



def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def get_softmax_input(answer):
    """
    Returns the one-hot encoded tensor for a given answer as input for softmax function.
    :param answer: the selected answer as string ('A', 'B', 'C', or 'D')
    :return: torch array representing the one-hot encoded tensor
    """

    # One-hot encode the answer
    tensor = np.zeros(len(choices))

    if answer in choices:
        index = choices.index(answer)
        tensor[index] = 1

    return tensor


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(args, subject, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        c = queryLLM(prompt)

        lprobs = []

        pred = c
        probs = softmax(get_softmax_input(c))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args):

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, dev_df, test_df)
        subcats = subcategories[subject]

        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="google/flan-t5-small",
    )
    args = parser.parse_args()
    main(args)

