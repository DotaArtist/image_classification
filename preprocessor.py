"""图片数据读取"""
import numpy as np
import cv2
import os

label_map = {
    "0": 0,  # 其他
    "1": 1,  # 出院小结首页
    "2": 1,  # 出院小结其他页
    "3": 1,  # 入院记录首页
    "4": 1,  # 入院记录其他页
    "5": 2,  # 病案首页
    "6": 3,  # 专项检查（如心电图，脑电图、超声、CT、病理等）
}


class BatchPreprocessor(object):

    def __init__(self, dataset_file_path, num_classes, output_size=[224, 224], horizontal_flip=False, shuffle=False,
                 batch_size=32,
                 mean_color=[132.2766, 139.6506, 146.9702], multi_scale=None, is_load_img=True, alpha_label_smooth=0):
        self.num_classes = num_classes
        self.output_size = output_size
        self.horizontal_flip = horizontal_flip
        self.shuffle = shuffle
        self.mean_color = mean_color
        self.multi_scale = multi_scale
        self.alpha_label_smooth = alpha_label_smooth  # true label = 1 - alpha_label_smooth
        self.batch_size = batch_size

        self.pointer = 0
        self.images = []
        self.labels = []

        if is_load_img:
            # Read the dataset file
            dataset_file = open(dataset_file_path, encoding='utf-8')
            lines = dataset_file.readlines()

            # NoneType = type(None)
            counter = 0

            for line in lines:
                items = line.strip().split('\t')
                # tmp = cv2.imread(items[0])
                self.images.append(items[0])
                self.labels.append(label_map[items[1]])
                counter += 1
                # if os.path.isfile(items[0]) and os.path.getsize(items[0]) > 1024:
                #     self.images.append(items[0])
                #     self.labels.append(int(items[1]))
                #     counter += 1
                # else:
                #     pass

                if counter % 10000 == 0:
                    pass
                    # print(counter)

            # Shuffle the data
            if self.shuffle:
                self.shuffle_data()

    def shuffle_data(self):
        images = self.images[:]
        labels = self.labels[:]
        self.images = []
        self.labels = []

        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        paths = self.images[self.pointer:(self.pointer + batch_size)]
        labels = self.labels[self.pointer:(self.pointer + batch_size)]

        self.pointer += batch_size

        images = np.ndarray([batch_size, self.output_size[0], self.output_size[1], 3])
        for i in range(len(paths)):
            # img = cv2.imread(paths[i])
            img = cv2.imdecode(np.fromfile(paths[i], dtype=np.uint8), -1)
            # img = paths[i]

            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            if self.multi_scale is None:
                img = cv2.resize(img, (self.output_size[0], self.output_size[0]))
                img = img.astype(np.float32)

            elif isinstance(self.multi_scale, list):
                new_size = np.random.randint(self.multi_scale[0], self.multi_scale[1], 1)[0]
                img = cv2.resize(img, (new_size, new_size))
                img = img.astype(np.float32)

                diff_size = new_size - self.output_size[0]
                random_offset_x = np.random.randint(0, diff_size, 1)[0]
                random_offset_y = np.random.randint(0, diff_size, 1)[0]
                img = img[random_offset_x:(random_offset_x + self.output_size[0]),
                      random_offset_y:(random_offset_y + self.output_size[0])]

            img = np.array(img, dtype=np.float64)

            if img.shape[2] == 4:
                img = img[:, :, :3]
            try:
                img -= np.array(self.mean_color)
            except ValueError:
                print(paths[i])

            images[i] = img

        one_hot_labels = np.zeros((batch_size, self.num_classes))
        for i in range(len(labels)):
            # one_hot_labels[i][labels[i]] = 1
            # label smoothing
            one_hot_labels[i] += self.alpha_label_smooth / (self.num_classes - 1)
            one_hot_labels[i][labels[i]] = 1 - self.alpha_label_smooth

        return images, one_hot_labels, labels

    def get_all_data(self):
        x_train = np.empty((len(self.labels), self.output_size[0], self.output_size[1], 3), dtype='uint8')
        y_train = np.empty((len(self.labels),), dtype='uint8')
        y_train_one_hot = np.empty((len(self.labels), self.num_classes), dtype='uint8')
        batch_number = np.floor(len(self.labels) / self.batch_size).astype(np.int16)
        for i in range(batch_number):
            print(i)
            x_train[(i) * self.batch_size: (i + 1) * self.batch_size, :, :, :], \
            y_train_one_hot[(i) * self.batch_size: (i + 1) * self.batch_size], \
            y_train[(i) * self.batch_size: (i + 1) * self.batch_size] = self.next_batch(self.batch_size)
        return x_train, y_train_one_hot, y_train

    @staticmethod
    def process_single_img(img_path):
        try:
            images = np.ndarray([1, 224, 224, 3])
            img = cv2.imread(img_path).astype(np.float32)
            img = cv2.resize(img, (224, 224))
            img -= np.array([132.2766, 139.6506, 146.9702])
            images[0] = img
            return images
        except AttributeError:
            return None

    @staticmethod
    def process_batch_img(img_path, counter=0):
        images = np.ndarray([64, 224, 224, 3])
        name_list = []

        _index = 0
        for i in range(counter, len(img_path)):
            counter += 1
            _images = np.ndarray([64, 224, 224, 3])
            _name_list = []
            try:
                img = cv2.imread(img_path[i]).astype(np.float32)
                img = cv2.resize(img, (224, 224))
                img -= np.array([132.2766, 139.6506, 146.9702])
                if (not isinstance(img, type(None))) and img.shape == (224, 224, 3):
                    images[_index] = img
                    _index += 1
                    name_list.append(img_path[i])
                else:
                    pass
            except AttributeError:
                pass

            if len(name_list) >= 64:
                _images = images
                _name_list = name_list

                images = np.ndarray([64, 224, 224, 3])
                name_list = []
                _index = 0

                yield _images, _name_list

        if len(name_list) != 0:
            yield images, name_list


if __name__ == '__main__':
    train_preprocessor = BatchPreprocessor(dataset_file_path='../data/train_100.txt',
                                           num_classes=4,
                                           output_size=[224, 224],
                                           horizontal_flip=True,
                                           shuffle=True, multi_scale=None)
    x_train, y_train_one_hot, y_train = train_preprocessor.get_all_data()
    print(x_train.shape, y_train_one_hot.shape, y_train.shape)
    # train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / 32).astype(np.int16)
    # for _ in range(train_batches_per_epoch):
    #     batch_tx, batch_ty, batch_ty_label = train_preprocessor.next_batch(32)
    #     print(batch_tx.shape, batch_ty.shape)
