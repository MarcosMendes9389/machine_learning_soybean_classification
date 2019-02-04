import pickle
import SoybeanClassification
import numpy as np
import pandas



def user_interaction():
    while True:
        print("##########################################################\n")
        print("-> Do you want to enter with attributes to classify ")
        print("or run again the algorithm selection and save model?\n")
        print("##########################################################\n")
        print("1 - Attributes")
        print("2 - Selection Algorithm")
        print("3 - Exit")
        option = input()

        if option == 1:
            classify_data_entry()
        if option == 2:
            run_selection_algorithm()
        if option == 3:
            break


def classify_data_entry():
    print("################################\n\n")
    print("Enter the sort attributes")
    print("in the order you requested\n")
    print("################################\n\n")

    model = pickle.load(open('../classification_model/classification_model.sav', 'rb'))
    new_entries = load_data_set().values
    x = new_entries[:, 0:34]

    predictions = model.predict(x)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    print(">>>> New Entries Classification: >>>>>>\n")
    print(predictions)
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


def load_data_set():
    url = "../data/new_entries.data"
    names = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged',
             'severity', 'seed-tmt', 'germination', 'plant-growth',
             'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf',
             'leaf-mild', 'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies',
             'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots',
             'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots']
    dataset = pandas.read_csv(url, names=names)
    return dataset


def run_selection_algorithm():
    SoybeanClassification.run()


if __name__ == '__main__':

    user_interaction()
