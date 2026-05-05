import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from facenet_pytorch import InceptionResnetV1
from dataset import FaceDataset


class FaceEmbeddingExtractor:
    def __init__(self, output_dir="features", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.output_dir = os.path.join(os.getcwd(), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def extract(self, images_tensor, labels_tensor, filename_x, filename_y, batch_size=32):
        print(f"Generating embeddings for {filename_x}...")
        loader = DataLoader(TensorDataset(images_tensor, labels_tensor), batch_size=batch_size)

        embeddings, labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                emb = self.model(batch_x)
                embeddings.append(emb.cpu().numpy())
                labels.append(batch_y.numpy())

        np.save(os.path.join(self.output_dir, filename_x), np.vstack(embeddings))
        np.save(os.path.join(self.output_dir, filename_y), np.concatenate(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--min_faces", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    data = FaceDataset(num_system_users=args.users, min_faces_per_person=args.min_faces).build()

    extractor = FaceEmbeddingExtractor()
    extractor.extract(data["train"][0], data["train"][1], "X_train_feat.npy", "y_train.npy", args.batch_size)
    extractor.extract(data["test"][0], data["test"][1], "X_test_feat.npy", "y_test.npy", args.batch_size)
    extractor.extract(data["out"][0], data["out"][1], "X_out_feat.npy", "y_out.npy", args.batch_size)

    print("\nExtraction complete. Files saved in 'features/' folder.")