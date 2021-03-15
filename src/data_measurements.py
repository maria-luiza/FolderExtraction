import numpy as np
import pandas as pd
import pickle
import traceback
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from folder_utils import *


def total_windows(windows):
    return len(windows)


def total_activities(activities):
    counter_activities = OrderedDict(Counter(activities).most_common())
    df = pd.DataFrame.from_dict(counter_activities, orient='index')
    df.columns = ['count']

    return df


def plot_activities(df, title):
    title = title.split("__")
    dataset_name = title[0].upper()

    df.plot(kind='bar')
    plt.title("Dataset: {0}".format(dataset_name))
    plt.xlabel("Activities")
    plt.ylabel("#Windows")
    plt.xticks(rotation=15)
    plt.savefig("activities__" + dataset_name + title[1] + ".jpeg", quality=95, format="jpeg")


def total_activities_transitions(activities):
    df_activities = pd.DataFrame(index=set(activities), columns=set(activities)).fillna(0)
    total_labels = len(activities) - 1

    for i in range(0, total_labels):
        if activities[i] != activities[i + 1]:
            df_activities[activities[i]][activities[i + 1]] += 1

    return df_activities


def heatmap(activities_transitions, title):
    ax = plt.axes()

    title = title.split("__")
    dataset_name = title[0].upper()
    # Calculate correlation between each pair of variable
    corr_matrix = activities_transitions.corr()
    # Can be great to plot only a half matrix
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    # plot a heatmap with annotation
    # Draw the heatmap with the mask
    sns.heatmap(activities_transitions,
                ax=ax,
                mask=mask,
                square=True,
                annot=True,
                annot_kws={"size": 7},
                cmap="Blues",
                cbar_kws={'label': 'Transitions'})
    ax.set_title("Dataset: {0}".format(dataset_name))
    plt.xlabel('Next', fontsize=10, fontweight='bold')  # x-axis label with fontsize 15
    plt.ylabel('Actual', fontsize=10, fontweight='bold')  # y-axis label with fontsize 15
    plt.show()


def sensor_profile(input, labels):
    dict_activities = dict.fromkeys(set(labels), [])
    total_sensors = len(input[0]) - 3

    for row, label in zip(input, labels):
        dict_activities[label].append(row[3:])

    for activity in set(labels):
        dict_activities[activity] = np.mean(dict_activities[activity], axis=0)

    return total_sensors, dict_activities


if __name__ == '__main__':
    # Get path from files
    ROOT_DIR = get_root_dirname()
    FILES_DIR = join_paths(ROOT_DIR, "input_data")

    # Get datasets on directory
    paths = list_directory(FILES_DIR)

    for file in paths:
        if not file.startswith('.'):
            try:
                datasets = list_directory(join_paths(FILES_DIR, file))
                for dataset in datasets:
                    if dataset not in [".DS_Store", "csv"]:
                        path = join_paths(FILES_DIR, file) + "/" + dataset
                        with open(path, 'rb') as fp:
                            windows = pickle.load(fp)
                            labels = pickle.load(fp)
                            unique_activities = pickle.load(fp)
                            fp.close()

                        # len_windows = total_windows(windows)
                        # total_sensors, sensor_activation = sensor_profile(windows, labels)
                        # heatmap(total_activities_transitions(labels), dataset)
                        plot_activities(total_activities(labels), dataset)
            except:
                print('Exception : ' + file)
                traceback.print_exc()
                pass
