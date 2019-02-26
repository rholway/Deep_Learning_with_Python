from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation ='relu',
                            input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

if __name__ == '__main__':
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    # look at the data
    print(f'shape of training data: {train_data.shape}')
    print(f'shape of test data: {test_data.shape}')
    # look at the targets
    print(f'training targets are in thousands of dollars: {train_targets}')
    # must standardize/normalize the data
    print('due to numeric features on different scales, must standardize the data. Must standardize training and test data from training mean/std ')
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    # must use mean and std from training data to apply to test data, cannot
    # calculate m/std of train data and apply that, its cheating!
    test_data -= mean
    test_data /= std

    # using k-fold cross val
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []

    for i in range(k):
        print(f'processing fold #: {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]

        partial_train_data = np.concatenate(
                    [train_data[:i * num_val_samples],
                    train_data[(i+1) * num_val_samples:]],
                    axis=0)
        partial_train_targets = np.concatenate(
                    [train_targets[:i * num_val_samples],
                    train_targets[(i+1) * num_val_samples:]],
                    axis=0)

        model = build_model()
        model.fit(partial_train_data, partial_train_targets,
                    epochs=num_epochs,
                    batch_size=1,
                    verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)

    print(f'all scores: {all_scores}')
    print(f'mean of all scores: {np.mean(all_scores)}')

    new_num_epochs = 500
    all_mae_histories = []
    for i in range(k):
        print(f'processing fold #: {i}')
        val_data_2 = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets_2 = train_targets[i * num_val_samples: (i+1) * num_val_samples]

        partial_train_data_2 = np.concatenate(
                    [train_data[:i * num_val_samples],
                    train_data[(i+1) * num_val_samples:]],
                    axis=0)
        partial_train_targets_2 = np.concatenate(
                    [train_targets[:i * num_val_samples],
                    train_targets[(i+1) * num_val_samples:]],
                    axis=0)

        model2 = build_model()
        model2.fit(partial_train_data_2, partial_train_targets_2,
                    validation_data=(val_data_2, val_targets_2),
                    epochs=new_num_epochs,
                    batch_size=1,
                    verbose=0)
        mae_history = model2.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)

    # history of successive mean k-fold validation scores
    average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    print(f'average MAE is: {average_mae_history}')

    # plot validation scores
    plt.plot(range(1,len(average_mae_history) + 1), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    # plt.savefig('validaiton score')
    plt.show()

    # plot validation scores, excluding the first 10 data points
    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt.plot(range(1,len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()
    plt.savefig('val_scores_not_first_10')

    # look at where we start to overfit (80 epochs?)

    # build final model
    model3 = build_model()
    model3.fit(train_data, train_targets,
                epochs=80, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model3.evaluate(test_data, test_targets)

    print(f'here is our final results! (in thousands of $$): {test_mae_score}')
