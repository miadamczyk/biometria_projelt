import cv2
import numpy as np
import torch

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class FaceDataset:
    def __init__(self, min_faces_per_person=8, num_system_users=100, mode="facenet", img_size=160):
        self.min_faces_per_person = min_faces_per_person
        self.num_system_users = num_system_users
        self.mode = mode
        self.img_size = img_size

        self.data = fetch_lfw_people(
            min_faces_per_person=self.min_faces_per_person,
            resize=1.0,
            color=(mode == "facenet")
        )
        self.images = self.data.images
        self.labels = self.data.target

    def preprocess_and_normalize(self, img):
        if img.max() <= 1.0:
            img = img * 255.0
        img_res = cv2.resize(img, (self.img_size, self.img_size)).astype(np.float32)

        if self.mode == "facenet":
            img_res = (img_res - 127.5) / 128.0
        else:
            img_res /= 255.0

        if len(img_res.shape) == 3:
            img_res = np.transpose(img_res, (2, 0, 1))
        else:
            img_res = np.expand_dims(img_res, axis=0)
        return img_res

    def build(self):
        unique_ids = np.unique(self.labels)
        np.random.RandomState(42).shuffle(unique_ids)

        actual_num_users = min(self.num_system_users, len(unique_ids) // 2)
        system_ids = set(unique_ids[:actual_num_users])
        outsider_ids = set(unique_ids[actual_num_users: actual_num_users * 2])

        X_system, y_system, X_out, y_out = [], [], [], []

        for img, label in zip(self.images, self.labels):
            processed_img = self.preprocess_and_normalize(img)
            if label in system_ids:
                X_system.append(processed_img)
                y_system.append(label)
            elif label in outsider_ids:
                X_out.append(processed_img)
                y_out.append(-1)

        le = LabelEncoder()
        y_system = torch.tensor(le.fit_transform(np.array(y_system)), dtype=torch.long)
        y_out = torch.tensor(np.array(y_out), dtype=torch.long)
        X_system, X_out = torch.tensor(np.array(X_system)), torch.tensor(np.array(X_out))

        idx_train, idx_test = train_test_split(
            np.arange(len(y_system)), test_size=0.3, random_state=42, stratify=y_system.numpy()
        )

        return {
            "train": (X_system[idx_train], y_system[idx_train]),
            "test": (X_system[idx_test], y_system[idx_test]),
            "out": (X_out, y_out),
            "encoder": le
        }