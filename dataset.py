from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

np.random.seed(15)


class cifar10Nosiy(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, nosiy_rate=0.0):
        super(cifar10Nosiy, self).__init__(root, transform=transform, target_transform=target_transform)
        if nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = self.other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

    def other_class(self, n_classes, current_class):
        """
        Returns a list of class indices excluding the class indexed by class_ind
        :param nb_classes: number of classes in the task
        :param class_ind: the class index to be omitted
        :return: one random class that != class_ind
        """
        if current_class < 0 or current_class >= n_classes:
            error_str = "class_ind must be within the range (0, nb_classes - 1)"
            raise ValueError(error_str)

        other_class_list = list(range(n_classes))
        other_class_list.remove(current_class)
        other_class = np.random.choice(other_class_list)
        return other_class


class cifar100Nosiy(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, nosiy_rate=0.0):
        super(cifar100Nosiy, self).__init__(root, download=download, transform=transform, target_transform=target_transform)
        if nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
            class_noisy = int(n_noisy / 100)
            noisy_idx = []
            for d in range(100):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = self.other_class(n_classes=100, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(100):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

    def other_class(self, n_classes, current_class):
        """
        Returns a list of class indices excluding the class indexed by class_ind
        :param nb_classes: number of classes in the task
        :param class_ind: the class index to be omitted
        :return: one random class that != class_ind
        """
        if current_class < 0 or current_class >= n_classes:
            error_str = "class_ind must be within the range (0, nb_classes - 1)"
            raise ValueError(error_str)

        other_class_list = list(range(n_classes))
        other_class_list.remove(current_class)
        other_class = np.random.choice(other_class_list)
        return other_class


class cifarDataset():
    def __init__(self,
                 batchSize=128,
                 dataPath='data/',
                 numOfWorkers=4,
                 is_cifar100=False,
                 cutout_length=16,
                 noise_rate=0.4):
        self.batchSize = batchSize
        self.dataPath = dataPath
        self.numOfWorkers = numOfWorkers
        self.cutout_length = cutout_length
        self.noise_rate = noise_rate
        self.is_cifar100 = is_cifar100
        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

        if self.is_cifar100:
            train_dataset = cifar100Nosiy(root=self.dataPath,
                                          train=True,
                                          transform=train_transform,
                                          download=True,
                                          nosiy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR100(root=self.dataPath,
                                             train=False,
                                             transform=test_transform,
                                             download=True)

        else:
            train_dataset = cifar10Nosiy(root=self.dataPath,
                                         train=True,
                                         transform=train_transform,
                                         download=True,
                                         nosiy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR10(root=self.dataPath,
                                            train=False,
                                            transform=test_transform,
                                            download=True)

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.batchSize,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.numOfWorkers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.batchSize,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.numOfWorkers)

        print("Num of train %d" % (len(train_dataset)))
        print("Num of test %d" % (len(test_dataset)))

        return data_loaders
