from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def prepare(X_Data, Y_Data, test_size, sampling_mode, task_type):

    # chosing strtify for classification or regression accordingly
    stratify = Y_Data if task_type == 'classification' else None

    # Split once
    X_train, X_test, y_train, y_test = train_test_split(
            X_Data, Y_Data, stratify=stratify, random_state=42, test_size=test_size
    )

    if task_type == 'classification':
        if sampling_mode == "Over":
            sampler = SMOTE(random_state=42)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        elif sampling_mode == "Under":
            sampler = RandomUnderSampler(random_state=42)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        elif sampling_mode != "None":
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    elif sampling_mode != "None":
        print("[INFO] Sampling is ignored for regression.")

    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def prepareKfold(X_train, y_train, sampling_mode, task_type):
    if task_type == 'classification':
        if sampling_mode == "Over":
            sampler = SMOTE(random_state=42)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        elif sampling_mode == "Under":
            sampler = RandomUnderSampler(random_state=42)
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        elif sampling_mode == "None":
            pass  # no resampling
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    
    elif sampling_mode != "None":
        print("[INFO] Sampling skipped for regression.")

    return X_train, y_train
