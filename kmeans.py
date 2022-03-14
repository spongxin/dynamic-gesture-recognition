import numpy as np
import torch


class KMeans:
    def __init__(self, use_gpu=False):
        self.centers = None
        self.device = torch.device('cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu')

    def fit(self, X, num_clusters, distance="euclidean", tol=1e-4):
        """
        perform kmeans
        :param X: (torch.tensor) matrix
        :param num_clusters: (int) number of clusters
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :param tol: (float) threshold [default: 0.0001]
        :return: (torch.Tensor, torch.Tensor) cluster ids, cluster centers
        """
        if distance == 'euclidean':
            pairwise_distance_function = self.pairwise_distance
        elif distance == 'cosine':
            pairwise_distance_function = self.pairwise_cosine
        else:
            raise NotImplementedError

        X = X.float()
        X = X.to(self.device)
        self.centers = self.initialize(X, num_clusters)

        iteration = 0
        while True:
            dis = pairwise_distance_function(X, self.centers)
            clusters = torch.argmin(dis, dim=1)
            initial_state_pre = self.centers.clone()
            for index in range(num_clusters):
                selected = torch.nonzero(clusters == index).squeeze().to(self.device)
                selected = torch.index_select(X, 0, selected)
                self.centers[index] = selected.mean(dim=0)
            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((self.centers - initial_state_pre) ** 2, dim=1)
                ))
            iteration = iteration + 1
            if center_shift ** 2 < tol:
                break
        return clusters.cpu(), self.centers.cpu()

    @staticmethod
    def initialize(X, num_clusters):
        """
        初始化中心对象
        :param X: 待聚类队列 (torch.tensor)
        :param num_clusters: 聚类数 (int)
        :return: 初始化中心
        """
        indices = np.random.choice(len(X), num_clusters, replace=False)
        return X[indices]

    def predict(self, X, distance='euclidean'):
        """
        predict using cluster centers
        :param X: (torch.tensor) matrix
        :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
        :return: (torch.tensor) cluster ids
        """
        if distance == 'euclidean':
            pairwise_distance_function = self.pairwise_distance
        elif distance == 'cosine':
            pairwise_distance_function = self.pairwise_cosine
        else:
            raise NotImplementedError

        X = X.float()
        X = X.to(self.device)
        dis = pairwise_distance_function(X, self.centers)
        choice_cluster = torch.argmin(dis, dim=1)
        return choice_cluster.cpu()

    def pairwise_distance(self, data1, data2):
        data1, data2 = data1.to(self.device), data2.to(self.device)
        # N*1*M
        A = data1.unsqueeze(dim=1)
        # 1*N*M
        B = data2.unsqueeze(dim=0)
        dis = (A - B) ** 2.0
        dis = dis.sum(dim=-1).squeeze()
        return dis

    def pairwise_cosine(self, data1, data2):
        data1, data2 = data1.to(self.device), data2.to(self.device)
        A = data1.unsqueeze(dim=1)
        B = data2.unsqueeze(dim=0)
        # normalize the points  | [0.3, 0.4] ->
        # [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
        A_normalized = A / A.norm(dim=-1, keepdim=True)
        B_normalized = B / B.norm(dim=-1, keepdim=True)
        cosine = A_normalized * B_normalized
        cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
        return cosine_dis


if __name__ == '__main__':
    import time

    data_size, dims, clu = 1000000, 2, 4
    x = np.random.randn(data_size, dims) / 6
    x = torch.from_numpy(x)

    mean = KMeans()
    mean1 = KMeans(use_gpu=False)
    start = time.perf_counter()
    mean.fit(X=x, num_clusters=5)
    end = time.perf_counter()
    print(f"gpu: {(end-start)*1000} ms.")
    start = end
    mean1.fit(X=x, num_clusters=5)
    end = time.perf_counter()
    print(f"cpu: {(end - start) * 1000} ms.")

    print(mean.predict(x, distance="cosine"))
