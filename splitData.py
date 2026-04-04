import pandas as pd

data = pd.read_csv("dataset.csv")

categorical_features = [
    'explicit',
    'genre'
]

numerical_features = [
    'popularity',
    'duration_ms',
    'energy',
    'key',
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentallness',
    'liveness',
    'tempo',
    'valence',
    'time_signature'
]


def split():
    df = data.sample(frac=1).reset_index(drop=True)

    split1 = int(0.7 * len(df))
    split2 = int(0.2 * len(df)) + split1
    train = df[:split1]
    validation = df[split1:split2]
    test = df[split2:]


    dropped_features = ['danceability', 'track_name', 'track_id', 'artists', 'album_name']

    y_train = train['danceability']
    x_train = train.drop(dropped_features, axis=1)
    x_train = pd.get_dummies(x_train, columns=['track_genre'], drop_first=True)

    y_validation = validation['danceability']
    x_validation = validation.drop(dropped_features, axis=1)
    x_validation = pd.get_dummies(x_validation, columns=['track_genre'], drop_first=True)

    y_test = test['danceability']
    x_test = test.drop(dropped_features, axis=1)

    return (y_train, x_train, y_validation, x_validation, y_test, x_test)