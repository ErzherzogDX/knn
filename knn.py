import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class KernelKNN:
    DATASET_TEST = ""
    DATASET_TRAIN = ""
    data = []

    train_raw = None
    test_raw = None
    weights = None


    def __init__(self, train_file:str, test_file:str):
        self.DATASET_TEST = test_file
        self.DATASET_TRAIN = train_file
        self.weights = np.full(8000, 1.0)

    def count_lowess_weights(self):
        pass

    def get_f1_score(self, y_true, y_pred):
        unique_classes = set(y_true) | set(y_pred)
        f1_scores = []

        for cls in unique_classes:
            TP = sum((y_pred[i] == cls and y_true[i] == cls) for i in range(len(y_true)))
            FP = sum((y_pred[i] == cls and y_true[i] != cls) for i in range(len(y_true)))
            FN = sum((y_pred[i] != cls and y_true[i] == cls) for i in range(len(y_true)))
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            f1_scores.append(f1)
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0

    def uniform_kernel(self, distance):
        return np.full_like(distance, 0.5)

    def gaussian_kernel(self, distance):
        return (1/np.sqrt(2 * np.pi)) * np.exp(-distance ** 2 / 2)

    def triangular_kernel(self, distance):
        return 1 - np.abs(distance)

    def epachnikov_kernel(self, distance):
        return 3/4 * (1 - distance ** 2)

    def logistic_kernel(self, distance):
        return 1/(np.exp(distance) + 2 + np.exp(-distance))

    def cosine_kernel(self, distance):
        return np.pi/4 * np.cos(np.pi * distance / 2)

    def real_load_dataset(self):
        self.train_raw = pd.read_csv(self.DATASET_TRAIN, header=None)
        self.test_raw = pd.read_csv(self.DATASET_TEST, header=None)

    def load_dataset(self, is_fixed:bool, window_param, kernel_function, metric, weightening:bool):
        train_results = self.train_raw.iloc[:, 0]
        train_data = self.train_raw.iloc[:, 1:]

        test_results = self.test_raw.iloc[:, 0]
        test_data = self.test_raw.iloc[:, 1:]

        if is_fixed:
            neighbors_model = NearestNeighbors(radius=window_param, algorithm='auto', metric=metric)
        else:
            neighbors_model = NearestNeighbors(n_neighbors=window_param+1, algorithm='auto',metric=metric)

        neighbors_model.fit(train_data)
        correct_pred = 0

        y_true = []
        y_pred = []

        if weightening:
            target = train_data
        else:
            target = test_data

        for index, row in target.iterrows():
            if is_fixed:
                distances, indices = neighbors_model.radius_neighbors([row])
                filtered_distances = distances[0]
                filtered_indices = indices[0]
            else:
                distances, indices = neighbors_model.kneighbors([row])
                filtered_distances = distances[0][1:]
                filtered_indices = indices[0][1:]

            if len(filtered_distances) == 0:
                distances, indices = neighbors_model.kneighbors([row])
                filtered_distances = distances[0][1:]
                filtered_indices = indices[0][1:]

            if is_fixed and len(filtered_distances) > 0:
                window = window_param
            else:
                window = filtered_distances[-1]

            normed_distances = filtered_distances/window
            if kernel_function == "gaussian":
                kernel_distances = self.gaussian_kernel(normed_distances)
            elif kernel_function == "triangular":
                kernel_distances = self.triangular_kernel(normed_distances)
            elif kernel_function == "epachnikov":
                kernel_distances = self.epachnikov_kernel(normed_distances)
            elif kernel_function == "logistic":
                kernel_distances = self.logistic_kernel(normed_distances)
            elif kernel_function == "cosine":
                kernel_distances = self.cosine_kernel(normed_distances)
            else:
                kernel_distances = self.uniform_kernel(normed_distances)

            class_set = {}
            for ix in range(0, len(filtered_indices)):
                number = train_results[filtered_indices[ix]]
                if number in class_set:
                    class_set[number] += kernel_distances[ix] * self.weights[filtered_indices[ix]]
                else:
                    class_set[number] = kernel_distances[ix] * self.weights[filtered_indices[ix]]

            sorted_class_set = sorted(class_set.items(), key=lambda item: item[1], reverse=True)
            total_sum = sum(class_set.values())
            most_common_class, max_value = sorted_class_set[0]

            if total_sum != 0:
                normed_max_value = max_value / total_sum
            else:
                normed_max_value = 1
            if weightening:
                weight = self.uniform_kernel(1 - normed_max_value)
                self.weights[index] = weight

            if not weightening:
                true_value = test_results[index]
                isEq = (most_common_class == true_value)

                y_true.append(true_value)
                y_pred.append(most_common_class)

                #print(most_common_class, true_value, " |", isEq)
                if isEq:
                    correct_pred+=1

        if not weightening:
            f1 = self.get_f1_score(y_true, y_pred)
            print("-------------------------")

            print("Method:", kernel_function, "| isFixed:", is_fixed, "| Window/K:", window_param)
            print("Correct:", correct_pred, "/", 2000)
            print("F1 Percent:", f1 * 100, "%")
            #print(near_min, near_avg/2000)
        else:
            print("Finish weightening!")



if __name__ == '__main__':
    knn = KernelKNN("mnist_train.csv", "mnist_test.csv")
    kernels = ["uniform", "gaussian", "triangular", "epachnikov", "logistic", "cosine"]
    dists = [750, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500, 3000]
    dists_cos = [ 0.6, 0.7, 0.8]
    k = [75, 100, 250, 500, 750, 1000, 2000]
    metrics = ["cosine"]
    radiuses = []

    knn.real_load_dataset()

    knn.load_dataset(False, 10, "epachnikov", "cosine", True)

    knn.load_dataset(False, 10, "uniform", "cosine", False)
    knn.load_dataset(False, 10, "epachnikov", "cosine", False)
    knn.load_dataset(True, 0.2, "guassian", "cosine", False)
    knn.load_dataset(True, 0.2, "logistic", "cosine", False)


    # for metric in metrics:
    #     for ks in dists_cos:
    #         for kernel in kernels:
    #             knn.load_dataset(True, ks, kernel, metric, False)
    #         print("=========================")

    # for ks in dists:
    #     for kernel in kernels:
    #         knn.load_dataset(True, ks, kernel, metric)
    #     print("=========================")



