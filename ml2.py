import coremltools
import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.stats import skew
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib


def majority_voting(labels):
    return labels.value_counts().idxmax()


def bowley_skewness(arr):
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    if q3 == q1:
        return 0.0
    return (q1 + q3 - 2 * q2) / (q3 - q1)


def feature_extract(filename):
    data = pd.read_csv(filename)
    activity = data['Labels']
    window_size = 5
    data_xyz = data[['X', 'Y', 'Z']].values  # Convert to NumPy array
    rolling_data = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size) / window_size, mode='valid'),
                                       axis=0, arr=data_xyz)

    features_list = []
    activity_windowed = []
    m = 0
    # Slide the window through the data and calculate features for each window
    for i in range(len(rolling_data) - window_size + 1):

        window_data = rolling_data[i:i + window_size]

        mean_x = sum(window_data[:, 0]) / 5
        if m < 1 :
            print(window_data)
            m=m+1
        mean_y = window_data[:, 1].mean()
        mean_z = window_data[:, 2].mean()
        max_x = np.max(window_data[:, 0])
        max_y = np.max(window_data[:, 1])
        max_z = np.max(window_data[:, 2])
        # Calculate other features (min, std, skew) separately for each direction
        min_x = window_data[:, 0].min()
        min_y = window_data[:, 1].min()
        min_z = window_data[:, 2].min()

        std_x = window_data[:, 0].std()
        std_y = window_data[:, 1].std()
        std_z = window_data[:, 2].std()

        skew_x = bowley_skewness(window_data[:, 0])
        skew_y = bowley_skewness(window_data[:, 1])
        skew_z = bowley_skewness(window_data[:, 2])

        # ... (rest of the feature calculations)

        # Calculate FFT separately for each direction
        fft_x = np.abs(fft(window_data[:, 0]))[:window_size]
        fft_y = np.abs(fft(window_data[:, 1]))[:window_size]
        fft_z = np.abs(fft(window_data[:, 2]))[:window_size]

        fft_mean_x = fft_x.mean()
        fft_mean_y = fft_y.mean()
        fft_mean_z = fft_z.mean()

        fft_max_x = fft_x.max()
        fft_max_y = fft_y.max()
        fft_max_z = fft_z.max()

        fft_min_x = fft_x.min()
        fft_min_y = fft_y.min()
        fft_min_z = fft_z.min()

        fft_std_x = fft_x.std()
        fft_std_y = fft_y.std()
        fft_std_z = fft_z.std()

        fft_2max_x = np.partition(fft_x, -2)[-2]
        fft_2max_y = np.partition(fft_y, -2)[-2]
        fft_2max_z = np.partition(fft_z, -2)[-2]

        # Calculate the PSD separately for each axis
        psd_x = np.square(np.abs(fft_x)) / window_size
        psd_y = np.square(np.abs(fft_y)) / window_size
        psd_z = np.square(np.abs(fft_z)) / window_size

        # Calculate the mean of each axis separately for the PSD features
        psd_mean_x = psd_x.mean()
        psd_mean_y = psd_y.mean()
        psd_mean_z = psd_z.mean()

        # Handle division by zero using NumPy's divide function
        mean_x_div_mean_z = np.divide(mean_x, mean_z, out=np.zeros_like(mean_x), where=mean_z != 0)
        mean_y_div_mean_z = np.divide(mean_y, mean_z, out=np.zeros_like(mean_y), where=mean_z != 0)
        mean_x_div_mean_y = np.divide(mean_x, mean_y, out=np.zeros_like(mean_x), where=mean_y != 0)
        majority_label = majority_voting(activity[i:i + window_size])
        activity_windowed.append(majority_label)

        # Create a DataFrame with all the features
        features_list.append({
            'mean_x': mean_x,
            'mean_y': mean_y,
            'mean_z': mean_z,
            'max_x': max_x,
            'max_y': max_y,
            'max_z': max_z,
            'min_x': min_x,
            'min_y': min_y,
            'min_z': min_z,
            'std_x': std_x,
            'std_y': std_y,
            'std_z': std_z,
            'skew_x': skew_x,
            'skew_y': skew_y,
            'skew_z': skew_z,
            'fft_mean_x': fft_mean_x,
            'fft_mean_y': fft_mean_y,
            'fft_mean_z': fft_mean_z,
            'fft_2max_x': fft_2max_x,
            'fft_2max_y': fft_2max_y,
            'fft_2max_z': fft_2max_z,
            'fft_max_x': fft_max_x,
            'fft_max_y': fft_max_y,
            'fft_max_z': fft_max_z,
            'fft_min_x': fft_min_x,
            'fft_min_y': fft_min_y,
            'fft_min_z': fft_min_z,
            'fft_std_x': fft_std_x,
            'fft_std_y': fft_std_y,
            'fft_std_z': fft_std_z,
            'psd_mean_x': psd_mean_x,
            'psd_mean_y': psd_mean_y,
            'psd_mean_z': psd_mean_z,
            'mean_x_div_mean_z': mean_x_div_mean_z,
            'mean_y_div_mean_z': mean_y_div_mean_z,
            'mean_x_div_mean_y': mean_x_div_mean_y
        })

    activity_windowed = np.array(activity_windowed)

    features = pd.DataFrame(features_list)

    output_file = "features.csv"
    features['activity_label'] = activity_windowed

    # Save the combined features (including activity labels) to a CSV file
    features.to_csv(output_file, index=False)
    return features, np.array(activity_windowed)


def train_and_test(filename):
    pd.set_option("display.max_columns", None)
    features, activity = feature_extract(filename)
    imputer = SimpleImputer(strategy='mean')
    X = features.values
    print(features.iloc[:10, :])
    print(activity[:10])
    print("X shape:", X.shape)
    print("Activity shape:", activity.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, activity, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=13)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Set: {accuracy}")

    coreml_model = coremltools.converters.sklearn.convert(clf)
    coreml_model.save('activity_classifier_knn.mlmodel')

    report = classification_report(y_test, y_pred)
    print(report)
    print(cv_scores)
