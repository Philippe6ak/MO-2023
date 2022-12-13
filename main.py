import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_csv(file):
    reading = pd.read_csv(file)
    return reading


def survivor_by_gender(dataset):
    survived_people = dataset.groupby(['Pclass', 'Sex']).Survived.agg('sum').reset_index()
    survived_men = np.array(survived_people[survived_people['Sex'] == 'male'].Survived)
    survived_women = np.array(survived_people[survived_people['Sex'] == 'female'].Survived)

    labels = ["1", "2", "3"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, survived_men, width, label='Men')
    ax.bar(x + width / 2, survived_women, width, label='Women')

    ax.set_ylabel('Survived')
    ax.set_xlabel('Class')
    ax.set_title('Survivors by gender')
    plt.xticks(x, labels)

    ax.legend()
    plt.show()
    print(survived_men)


def survivor_per_class(dataset):
    passenger_by_class = np.array(dataset.groupby('Pclass').Name.count())
    fig, ax = plt.subplots()

    def func(pct, allvals):
        absolute = int(np.round(pct / 100. * np.sum(allvals)))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    ax.pie(passenger_by_class, autopct=lambda pct: func(pct, passenger_by_class))
    ax.legend(['first class', 'second class', 'third class'],
              title="Classes",
              loc="upper left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.show()


def age_dist(dataset):
    fig, ax = plt.subplots()
    all_ages = dataset['Age'].dropna().apply(np.floor)
    ax.hist(np.array(all_ages), 70)

    plt.show()


data = read_csv("tested.csv")
survivor_by_gender(data)
survivor_per_class(data)
age_dist(data)



# 1. Distribution of survivors among men and women by ticket class.
# 2. Build a pie chart for the "passenger class" attribute
#   (the number of people in each class)
# 3. Build a distribution of the ages of all passengers.
