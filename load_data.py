import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 100)

DATA_PATH = r"C:\Users\S.Dhathri\Desktop\ML\placement_predict_50k Dataset.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at:\n{path}"
        )

    df = pd.read_csv(path)

    return df


def get_data_summary(path: str = DATA_PATH) -> dict:
    df = load_data(path)

    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "missing_counts": {
            col: int(df[col].isna().sum())
            for col in df.columns
        },
        "preview": df.head(10).to_dict(orient="records"),
    }

    return summary


if __name__ == "__main__":

    print("Loading dataset...")

    data = load_data()

    print("\nDataset path:")
    print(DATA_PATH)

    print("\nDataset shape:")
    print(data.shape)

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDataset summary:")
    summary = get_data_summary()

    print("Number of rows:", summary["n_rows"])
    print("Number of columns:", summary["n_cols"])

    print("\nProgram completed successfully.")