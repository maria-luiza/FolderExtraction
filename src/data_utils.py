from datetime import datetime


def get_milliseconds_from_date(date, time):
    if '.' not in time:
        time = time + ".00"
    # strptime parses a string representing a time according to a format
    date_and_time = datetime.strptime(date+" "+time, '%Y-%m-%d %H:%M:%S.%f')
    # dateAndTime.timestamp() Return POSIX timestamp corresponding to the datetime instance
    return date_and_time.timestamp() * 1000


def get_empty_bag_of_sensors(file_rows):
    sensors = {}
    for row in file_rows:
        if row:
            sensor = row[2]
            if sensor not in sensors:
                sensors.update({sensor: 0})
    return sensors


def get_dynamic_window(file_rows):
    feature_vector = []
    size_vectors = []
    activities_size = 0
    has_label = False

    for row in file_rows:
        # If the row has label information
        if len(row) > 4:
            # Get the label
            label = row[4]
            # It's the first label appearance
            if '"begin"' in label:
                feature_vector.append(row)
                activities_size = 1
                has_label = True
            # It's the ending of the activity interval
            if '"end"' in label:
                activities_size += 1
                feature_vector.append(row)
                size_vectors.append(activities_size)

                activities_size = 0
                has_label = False

        else:
            if has_label:
                feature_vector.append(row)
                activities_size += 1

    return feature_vector, size_vectors


def get_activity_label(row, label, was_activity_end, activities_not_finished):
    if was_activity_end:
        # was ending an activity previously... it's not anymore. (That class scope is over)
        label = ""
        was_activity_end = False
    if len(row) > 4:
        # it has an activity
        label = row[4]
        if '"begin"' in label:
            label = label.split("=")[0]
            activities_not_finished.append(label)

        elif '"end"' in label:
            label = label.split("=")[0]
            # Remove last occurrence of the activity at activities_not_finished
            activities_not_finished.reverse()
            was_activity_end = True
            try:
                activities_not_finished.remove(label)
            except:
                pass
            activities_not_finished.reverse()

        else:
            if len(row) == 6:
                if row[5] == "begin":
                    activities_not_finished.append(label)
                elif row[5] == "end":
                    # Remove last occurrence of the activity at activitiesNotFinished
                    activities_not_finished.reverse()
                    was_activity_end = True
                    try:
                        activities_not_finished.remove(label)
                    except:
                        pass
                    activities_not_finished.reverse()
            else:
                was_activity_end = True
    return label, was_activity_end


def generate_feature_vector(windows, activities_not_finished, empty_bag):
    """
    Return the feature vector for each n windows
    :param windows:
    :param activities_not_finished:
    :param empty_bag:
    :return:
    """

    activity_label = ""
    was_activity_end = False

    # Get the first activation captured in that window
    first_row = windows[0]
    first_sensor_time = get_milliseconds_from_date(first_row[0], first_row[1])
    # Get the last activation captured in that window
    last_row = windows[len(windows)-1]
    last_sensor_time = get_milliseconds_from_date(last_row[0], last_row[1])
    # Get the duration of the window
    window_temporal_span = (last_sensor_time - first_sensor_time)

    bag_of_sensors = empty_bag.copy()

    for window in windows:
        sensor = window[2]
        bag_of_sensors[sensor] = bag_of_sensors[sensor] + 1
        activity_label, was_activity_end = get_activity_label(window,
                                                              activity_label,
                                                              was_activity_end,
                                                              activities_not_finished)

    # What was the label computed for the last activity in the window?
    if not activity_label:
        activities_not_finished_length = len(activities_not_finished)
        # If there is an incomplete activity or an activity just finished (Gets its class)
        # otherwise label it 'Other' class
        if activities_not_finished_length != 0 or was_activity_end:
            activity_label = activities_not_finished[activities_not_finished_length-1]
        else:
            activity_label = "Other"
    bag_of_sensors = [value for(key, value) in sorted(bag_of_sensors.items())]
    return [[first_sensor_time, last_sensor_time, window_temporal_span, *bag_of_sensors], activity_label]


def normalize_data(feature_vector):
    return [[(x-min(l))/(max(l)-min(l)) for x in l] for l in feature_vector]
