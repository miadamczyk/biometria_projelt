import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import Normalizer
from dataset import FaceDataset


def calculate_biometric_metrics(s_known, s_out):
    all_scores = np.concatenate([s_known, s_out])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), 1000)
    eer = 1.0
    best_thr = 0.0

    for t in thresholds:
        far = np.mean(s_out >= t)
        frr = np.mean(s_known < t)
        if abs(far - frr) < eer:
            eer = (far + frr) / 2
            best_thr = t

    return {"eer": eer, "threshold": best_thr}


def tensor_to_numpy(tensor_img):
    img = tensor_img.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return img


def visualize_results_grid(model_name, imgs_te, s_known, imgs_out, s_out, threshold):
    categories = [
        ("True Positives (OK)", np.where(s_known >= threshold)[0], imgs_te, s_known),
        ("False Negatives (Błąd)", np.where(s_known < threshold)[0], imgs_te, s_known),
        ("True Negatives (OK)", np.where(s_out < threshold)[0], imgs_out, s_out),
        ("False Positives (Błąd)", np.where(s_out >= threshold)[0], imgs_out, s_out)
    ]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f"Przykłady klasyfikacji (4 per kategoria) - Model: {model_name.upper()}", fontsize=16)

    for row_idx, (title, indices, source_imgs, scores) in enumerate(categories):
        selected_indices = np.random.choice(indices, size=min(4, len(indices)), replace=False) if len(
            indices) > 0 else []

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            if col_idx < len(selected_indices):
                idx = selected_indices[col_idx]
                img = tensor_to_numpy(source_imgs[idx])
                ax.imshow(img)
                ax.set_title(f"Score: {scores[idx]:.3f}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "Brak danych", ha='center', va='center')

            if col_idx == 0:
                ax.set_ylabel(title, fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"faces_grid_{model_name}.png")


def run_evaluation(model_name, xtr, ytr, xte, xout, imgs_te, imgs_out):
    models = {
        "svm": SVC(probability=True, kernel='rbf', C=10),
        "knn": KNeighborsClassifier(n_neighbors=3, metric='cosine'),
        "rf": RandomForestClassifier(n_estimators=100)
    }

    clf = models[model_name].fit(xtr, ytr)

    s_known = np.max(clf.predict_proba(xte), axis=1)
    s_out = np.max(clf.predict_proba(xout), axis=1)

    m = calculate_biometric_metrics(s_known, s_out)
    thr = m['threshold']

    y_true = np.concatenate([np.ones(len(s_known)), np.zeros(len(s_out))])
    y_pred = np.concatenate([s_known >= thr, s_out >= thr])
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- RESULTS: {model_name.upper()} ---")
    print(f"  EER: {m['eer']:.4f} (Thr: {thr:.3f})")
    print(f"  Confusion Matrix (Samples: {len(y_true)}):\n{cm}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Outsider', 'User'])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix {model_name.upper()} (100 vs 100)")
    plt.savefig(f"cm_{model_name}.png")

    visualize_results_grid(model_name, imgs_te, s_known, imgs_out, s_out, thr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--min_faces", type=int, default=8)
    parser.add_argument("--features_path", type=str, default="features")
    args = parser.parse_args()

    xtr = np.load(f"{args.features_path}/X_train_feat.npy")
    ytr = np.load(f"{args.features_path}/y_train.npy")
    xte_full = np.load(f"{args.features_path}/X_test_feat.npy")
    xout_full = np.load(f"{args.features_path}/X_out_feat.npy")
    yte_full = np.load(f"{args.features_path}/y_test.npy")

    dataset_build = FaceDataset(num_system_users=args.users, min_faces_per_person=args.min_faces).build()
    imgs_te_full = dataset_build["test"][0]
    imgs_out_full = dataset_build["out"][0]

    idx_te = np.random.choice(len(xte_full), size=min(args.users, len(xte_full)), replace=False)
    xte = xte_full[idx_te]
    yte = yte_full[idx_te]
    imgs_te = imgs_te_full[idx_te]

    idx_out = np.random.choice(len(xout_full), size=min(args.users, len(xout_full)), replace=False)
    xout = xout_full[idx_out]
    imgs_out = imgs_out_full[idx_out]

    scaler = Normalizer(norm='l2')
    xtr = scaler.fit_transform(xtr)
    xte = scaler.transform(xte)
    xout = scaler.transform(xout)

    for m in ["svm", "knn", "rf"]:
        run_evaluation(m, xtr, ytr, xte, xout, imgs_te, imgs_out)