import pickle
import traceback
import random
from sklearn.model_selection import StratifiedKFold
from fold_msb3_tcc import *
from folder_utils import *

ROOT_DIR = get_root_dirname()


def generate_cross_datasets(final_string, indexes, input, label):
    x_out = []
    y_out = []
    for index in indexes:
        x_out.append(input[index])
        y_out.append(label[index])
        final_string = final_string + ",".join(map(str, input[index])) + "," + label[index] + "\n"

    return x_out, y_out


def get_random_labels(actual_label, size, uniqueList):
    new_label = str(random.choice(uniqueList))

    if actual_label == new_label:
        get_random_labels(actual_label, size, uniqueList)
    return new_label


def generate_random_labels(yTrain0, uniqueActivitiesList):
    yTrain = []
    size = len(yTrain0)
    yTrain.append(yTrain0)
    yTrainCopy = yTrain0
    numChanges = int(abs(size / 10))

    for times in range(0, 5):
        yTrainCopy = yTrainCopy.copy()
        for change in range(0, numChanges):
            index = random.randint(0, size - 1)
            actual_label = yTrainCopy[index]
            yTrainCopy[index] = get_random_labels(actual_label, size, uniqueActivitiesList)
        yTrain.append(yTrainCopy)
    print('fold created!')
    return yTrain


def create_or_get_existing_folds(input, file):
    folds = []
    # getExisting
    folds_dir = os.path.dirname(join_paths(ROOT_DIR, "folds/"))
    path_fold = os.path.dirname(join_paths(folds_dir, input + "/"))

    input_dir = os.path.dirname(join_paths(ROOT_DIR, "input_data/"))
    path_input = os.path.dirname(join_paths(input_dir, input + "/"))

    if (os.path.exists(path_fold + "/" + file)):
        with open(path_fold + "/" + file, 'rb') as f:
            folds = pickle.load(f)
            uniqueActivitiesList = pickle.load(f)
            f.close()
    else:  # Create
        with open(path_input + "/" + file, 'rb') as f:
            x = pickle.load(f)  # Get the vector sensor activation
            y = pickle.load(f)  # Get the activities for each activation
            uniqueActivitiesList = pickle.load(f)  # Get unique activities

            if (x != [] and y != []):
                kFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)

                for foldNumber, (trainIndexes, testIndexes) in enumerate(kFold.split(x, y)):
                    trainingString, testString = "", ""

                    # save the training datasets
                    xTrain, yTrain = generate_cross_datasets(trainingString, trainIndexes, x, y)
                    # save the validation datasets
                    xTest, yTest = generate_cross_datasets(testString, testIndexes, x, y)

                    yTrains = generate_random_labels(yTrain, uniqueActivitiesList)

                    fold = fold_msb3_tcc(xTrain, yTrains, xTest, yTest)
                    folds.append(fold)

                with open(path_fold + "/" + file, 'wb') as fp:
                    pickle.dump(folds, fp)
                    pickle.dump(uniqueActivitiesList, fp)
                    fp.close()

            f.close()
    return [folds, uniqueActivitiesList]


if __name__ == '__main__':
    inputs = ["Dynamic", "Static"]

    dir_name = os.path.dirname(join_paths(ROOT_DIR, "input_data/"))

    for input in inputs:
        path = os.path.dirname(join_paths(dir_name, input + "/"))
        files = list_directory(path)
        for file in files:
            if file not in [".DS_Store", "csv"]:
                try:
                    print("~~~~~~~~~~~ Database : " + file + " ~~~~~~~~~~~\n")
                    folds, uniqueActivitiesList = create_or_get_existing_folds(input, file)
                except:
                    print('Exception : ' + file)
                    traceback.print_exc()
                    pass
