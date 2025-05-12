import numpy as np
import pandas as pd
import math


def bayesian_classifier(outcome, collected_data, given_data):
    #variables:
    #outcome[i]; i represents class(i=0,1 => class 1, class 2)
    #collected_data[i][j][k]; 'i' represents the feature(i=0,1 => glucose, bmi) and,
    #                         'j' represents the class(j=0,1 => class 1, class 2)
    #                         'k' represents the data value of 'i'th feature in 'j'th class.
    #given_data[i]; i represents the feature value(i=0,1 => given_glucose, given_bmi)

    prior = [outcome.count(0) / len(outcome), outcome.count(1) / len(outcome)]  # prior probabilities

    # Calculating likelihoods by multiplying the individual conditional probabilities of both the classes
    likelihoods = [gaussian_prob(given_data[0], np.mean(collected_data[0][0]), np.std(collected_data[0][0])) *
                   gaussian_prob(given_data[1], np.mean(collected_data[1][0]), np.std(collected_data[1][0])),
                   gaussian_prob(given_data[0], np.mean(collected_data[0][1]), np.std(collected_data[0][1])) *
                   gaussian_prob(given_data[1], np.mean(collected_data[1][1]), np.std(collected_data[1][1]))]

    evidence = prior[0] * likelihoods[0] + prior[1] * likelihoods[1]

    result = [(likelihoods[0] * prior[0]) / evidence, (likelihoods[1] * prior[1]) / evidence]

    print("Prior: {}\nLikelihoods: {}\nEvidence: {}\nConditional Probabilities: \n\tClass 1: {}\n\tClass 2: {}"
          .format(prior, likelihoods, evidence, result[0], result[1]))
    return result


def gaussian_prob(x, mu, std):
    return (1 / (2 * math.pi * std)) * math.exp((-(x - mu) ** 2) / (2 * (std ** 2)))


if __name__ == '__main__':
    path = r"D:\Study\IITJ\IIT Delhi\Bayesian\diabetes.csv"
    file = pd.read_csv(path)
    outcome = list(file["Outcome"])
    glucose_data = list(file["Glucose"])
    bmi_data = list(file["BMI"])

    bmi = [[], []]
    glucose = [[], []]

    #For managing the dataset for both the classes
    for i in range(len(outcome)):
        if outcome[i] == 0:
            glucose[0].append(glucose_data[i])
            bmi[0].append(bmi_data[i])
        else:
            glucose[1].append(glucose_data[i])
            bmi[1].append(bmi_data[i])

    #Past data
    collected_data = [glucose, bmi]


    #Data for prediction
    given_glucose = 150
    given_bmi = 40
    given_data = [given_glucose, given_bmi]
    print("Given Data:\nGlucose:{}\nBMI:{}\n".format(given_data[0], given_data[1]))
    result = bayesian_classifier(outcome, collected_data, given_data)
    if result[0] > result[1]:
        print("No Diabetes")
    else:
        print("Diabetes")
