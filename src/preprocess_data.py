import pickle
import numpy as np
import traceback
from folder_utils import *
from data_utils import *


class DataProcessing:
    def __init__(self, rows=None, window_size=0):
        self.file_rows = rows
        self.window = window_size
        self.bag_of_sensors = get_empty_bag_of_sensors(rows)

    def process_data(self):
        i = 0
        feature_vector = []
        label_array = []
        # Sensor events from raw data
        rows = self.file_rows
        # Represents activities that began but didn't end
        activities_not_finished = []

        # Fixed Windows size
        if self.window > 0:
            total_windows = len(rows) - self.window + 1
            for i in range(0, total_windows):
                window_rows = rows[i:self.window + i]

                if window_rows:
                    features = generate_feature_vector(window_rows, activities_not_finished, self.bag_of_sensors)

                    if features[1] != "Other":
                        feature_vector.append(features[0])
                        label_array.append(features[1])
                else:
                    feature_vector = []

        # Dynamic Windows size
        else:
            raw_filtered, windows_size = get_dynamic_window(rows)
            bag_of_sensors = get_empty_bag_of_sensors(raw_filtered)

            for window in windows_size:
                window_rows = raw_filtered[i:window + i]

                if window_rows:
                    features = generate_feature_vector(window_rows, activities_not_finished, bag_of_sensors)

                    if features[1] != "Other":
                        feature_vector.append(features[0])
                        label_array.append(features[1])

                    i = window + 1
                else:
                    feature_vector = []

        feature_vector_array = normalize_data(feature_vector)
        unique_activities_list = np.unique(np.array(label_array))
        return [feature_vector_array, label_array, unique_activities_list]


if __name__ == '__main__':
    # Get path from files
    ROOT_DIR = get_root_dirname()
    FILES_DIR = join_paths(ROOT_DIR, "datasets")

    # Get datasets on directory
    dataFiles = list_directory(FILES_DIR)

    # Window size
    windowSize = [0, 30]

    for window in windowSize:
        print("Window size: ", window)
        # Filename to save
        filename = "__window_" + str(window)
        for file in dataFiles:
            if not file.startswith('.'):
                print("File to be processed: ", file)
                try:
                    file_rows = open_dataset_files(join_paths(FILES_DIR, file + "/ann.txt"))
                    dataProcessed = DataProcessing(file_rows, window)
                    data_array = dataProcessed.process_data()

                    inputData, labelData, uniqueActivitiesList = data_array

                    dir_name = os.path.dirname(join_paths(ROOT_DIR, "input_data/"))
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)

                    if window == 0:
                        path = join_paths(dir_name, "Dynamic") + "/" + file + filename
                    else:
                        path = join_paths(dir_name, "Static") + "/" + file + filename

                    with open(path, 'wb') as fp:
                        pickle.dump(inputData, fp)
                        pickle.dump(labelData, fp)
                        pickle.dump(uniqueActivitiesList, fp)
                        fp.close()

                except:
                    print('Exception : ' + file)
                    traceback.print_exc()
                    pass
