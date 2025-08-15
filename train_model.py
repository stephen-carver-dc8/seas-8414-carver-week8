import pandas as pd
import numpy as np
from pycaret.classification import *
from pycaret.clustering import (
    setup as cluster_setup,
    create_model as cluster_create_model,
    save_model as cluster_save_model,
)
import os
import matplotlib.pyplot as plt


def generate_synthetic_data(num_samples=600):
    """Generates a synthetic dataset with patterns for different threat actors."""
    print("Generating synthetic dataset...")
    np.random.seed(42)

    features = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service',
        'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
        'having_Sub_Domain', 'SSLfinal_State', 'URL_of_Anchor', 'Links_in_tags',
        'SFH', 'Abnormal_URL', 'has_political_keyword'
    ]

    num_phishing = num_samples // 2
    num_benign = num_samples - num_phishing
    per_profile = num_phishing // 3

    # --- State-Sponsored ---
    state_data = {
        'having_IP_Address': np.random.choice([1, -1], per_profile, p=[0.1, 0.9]),
        'URL_Length': np.random.choice([1, 0, -1], per_profile, p=[0.4, 0.5, 0.1]),
        'Shortining_Service': np.random.choice([1, -1], per_profile, p=[0.1, 0.9]),
        'having_At_Symbol': np.random.choice([1, -1], per_profile, p=[0.2, 0.8]),
        'double_slash_redirecting': np.random.choice([1, -1], per_profile, p=[0.2, 0.8]),
        'Prefix_Suffix': np.random.choice([1, -1], per_profile, p=[0.8, 0.2]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], per_profile, p=[0.6, 0.3, 0.1]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], per_profile, p=[0.05, 0.05, 0.9]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], per_profile, p=[0.3, 0.4, 0.3]),
        'Links_in_tags': np.random.choice([-1, 0, 1], per_profile, p=[0.2, 0.5, 0.3]),
        'SFH': np.random.choice([-1, 0, 1], per_profile, p=[0.4, 0.4, 0.2]),
        'Abnormal_URL': np.random.choice([1, -1], per_profile, p=[0.2, 0.8]),
        'has_political_keyword': np.full(per_profile, -1)
    }

    # --- Organized Cybercrime ---
    org_data = {
        'having_IP_Address': np.random.choice([1, -1], per_profile, p=[0.7, 0.3]),
        'URL_Length': np.random.choice([1, 0, -1], per_profile, p=[0.7, 0.2, 0.1]),
        'Shortining_Service': np.random.choice([1, -1], per_profile, p=[0.8, 0.2]),
        'having_At_Symbol': np.random.choice([1, -1], per_profile, p=[0.5, 0.5]),
        'double_slash_redirecting': np.random.choice([1, -1], per_profile, p=[0.6, 0.4]),
        'Prefix_Suffix': np.random.choice([1, -1], per_profile, p=[0.6, 0.4]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], per_profile, p=[0.7, 0.2, 0.1]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], per_profile, p=[0.8, 0.1, 0.1]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], per_profile, p=[0.6, 0.2, 0.2]),
        'Links_in_tags': np.random.choice([-1, 0, 1], per_profile, p=[0.6, 0.2, 0.2]),
        'SFH': np.random.choice([-1, 0, 1], per_profile, p=[0.6, 0.2, 0.2]),
        'Abnormal_URL': np.random.choice([1, -1], per_profile, p=[0.8, 0.2]),
        'has_political_keyword': np.full(per_profile, -1)
    }

    # --- Hacktivist ---
    hack_data = {
        'having_IP_Address': np.random.choice([1, -1], per_profile, p=[0.4, 0.6]),
        'URL_Length': np.random.choice([1, 0, -1], per_profile, p=[0.5, 0.3, 0.2]),
        'Shortining_Service': np.random.choice([1, -1], per_profile, p=[0.5, 0.5]),
        'having_At_Symbol': np.random.choice([1, -1], per_profile, p=[0.4, 0.6]),
        'double_slash_redirecting': np.random.choice([1, -1], per_profile, p=[0.4, 0.6]),
        'Prefix_Suffix': np.random.choice([1, -1], per_profile, p=[0.5, 0.5]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], per_profile, p=[0.5, 0.3, 0.2]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], per_profile, p=[0.4, 0.3, 0.3]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], per_profile, p=[0.4, 0.3, 0.3]),
        'Links_in_tags': np.random.choice([-1, 0, 1], per_profile, p=[0.4, 0.3, 0.3]),
        'SFH': np.random.choice([-1, 0, 1], per_profile, p=[0.5, 0.3, 0.2]),
        'Abnormal_URL': np.random.choice([1, -1], per_profile, p=[0.6, 0.4]),
        'has_political_keyword': np.full(per_profile, 1)
    }

    df_phishing = pd.concat([
        pd.DataFrame(state_data),
        pd.DataFrame(org_data),
        pd.DataFrame(hack_data)
    ], ignore_index=True)

    # Benign traffic
    benign_data = {
        'having_IP_Address': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'URL_Length': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.6, 0.3]),
        'Shortining_Service': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_At_Symbol': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'double_slash_redirecting': np.random.choice([1, -1], num_benign, p=[0.05, 0.95]),
        'Prefix_Suffix': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'having_Sub_Domain': np.random.choice([1, 0, -1], num_benign, p=[0.1, 0.4, 0.5]),
        'SSLfinal_State': np.random.choice([-1, 0, 1], num_benign, p=[0.05, 0.15, 0.8]),
        'URL_of_Anchor': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'Links_in_tags': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.2, 0.7]),
        'SFH': np.random.choice([-1, 0, 1], num_benign, p=[0.1, 0.1, 0.8]),
        'Abnormal_URL': np.random.choice([1, -1], num_benign, p=[0.1, 0.9]),
        'has_political_keyword': np.full(num_benign, -1)
    }

    df_benign = pd.DataFrame(benign_data)

    df_phishing['label'] = 1
    df_benign['label'] = 0

    final_df = pd.concat([df_phishing, df_benign], ignore_index=True)
    return final_df.sample(frac=1, random_state=42).reset_index(drop=True)


def train():
    model_path = 'models/phishing_url_detector'
    cluster_path = 'models/threat_actor_profiler'
    plot_path = 'models/feature_importance.png'

    if os.path.exists(model_path + '.pkl') and os.path.exists(cluster_path + '.pkl'):
        print("Models already exist. Skipping training.")
        return

    data = generate_synthetic_data()
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/phishing_synthetic.csv', index=False)

    print("Initializing PyCaret Setup for classification...")
    s = setup(data, target='label', session_id=42, verbose=False)

    print("Comparing models...")
    best_model = compare_models(n_select=1, include=['rf', 'et', 'lightgbm'])

    print("Finalizing classification model...")
    final_model = finalize_model(best_model)

    print("Saving feature importance plot...")
    os.makedirs('models', exist_ok=True)
    plot_model(final_model, plot='feature', save=True)
    os.rename('Feature Importance.png', plot_path)

    print("Saving classification model...")
    save_model(final_model, model_path)

    # --- Clustering workflow ---
    print("Training clustering model for threat attribution...")
    cluster_data = data[data['label'] == 1].drop(columns=['label'])
    csetup = cluster_setup(data=cluster_data, session_id=42, verbose=False)
    kmeans = cluster_create_model('kmeans', num_clusters=3)

    # Determine mapping from cluster id to threat profile
    centers = kmeans.cluster_centers_
    feature_list = cluster_data.columns.tolist()
    mapping = {}
    for idx, center in enumerate(centers):
        if center[feature_list.index('has_political_keyword')] > 0:
            mapping[idx] = "Hacktivist"
        elif center[feature_list.index('Shortining_Service')] > 0.5 and center[feature_list.index('having_IP_Address')] > 0.5:
            mapping[idx] = "Organized Cybercrime"
        else:
            mapping[idx] = "State-Sponsored"

    # Attach mapping to model before saving so inference can access it directly
    kmeans.cluster_mapping = mapping
    cluster_save_model(kmeans, cluster_path)

    print("Models and artifacts saved successfully.")


if __name__ == "__main__":
    train()