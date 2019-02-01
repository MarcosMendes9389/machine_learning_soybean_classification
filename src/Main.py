import pickle
import TourismClassification
import numpy as np


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


    #att = [6,0,2,1,0,1,1,1,0,0,1,1,0,2,2,0,0,0,1,1,3,1,1,1,0,0,0,0,4,0,0,0,0,0,0]
    model = pickle.load(open('../classification_model/classification_model.sav', 'rb'))
    #predictions = model.predict(att)
    #print(predictions)


def run_selection_algorithm():
    TourismClassification.run()


if __name__ == '__main__':

    user_interaction()
