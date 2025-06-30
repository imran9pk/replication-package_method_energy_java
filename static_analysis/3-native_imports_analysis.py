# %%
import pandas as pd
from pathlib import Path

# %%
# Paths and Config
root_dir = Path.cwd().parent
data_dir = root_dir / "data"
all_class_metrics_csv = data_dir / "calss_metrics.classes.csv"
feature_packages_csv = "library_call_features-unique.csv"
extracted_features_classes_csv = "feature_classes.csv"

# %%
def extract_class_name(import_statement):

    class_names = []
    parts = import_statement.split(".")
    
    # Using basic heuristic to determine the class name and
    # consider any part starting with upper case letter as a class name
    for part in parts:
        #if the part starts with an upper case letter and not all the letters are upper case
        if part[0].isupper() and not part.isupper():
            class_names.append(part)

    return class_names

# %%
def get_all_native_imports(all_class_metrics_csv):
    
    total_native_imports = 0
    all_native_imports_set = set()
    
    classes_metrics_df = pd.read_csv(all_class_metrics_csv)
    native_imports_list = classes_metrics_df["cNativeImportsList"]
    total_native_imports += classes_metrics_df["#cNativeImports"].sum()
        

    for native_import in native_imports_list:
        import_statements = native_import.split(",") if type(native_import) == str else []

        for import_statement in import_statements:
            all_native_imports_set.add(import_statement)
    
    return all_native_imports_set

# %%
def get_all_feature_classes(native_imports_set, feature_package):
        
        all_feature_classes_set = set()

        for import_statement in native_imports_set:
            #if the import statement contains the feature package
            if feature_package in import_statement:     
                #then extract the class names from the import statement
                class_names = extract_class_name(import_statement)
                #and add these to the set of all feature classes
                for class_name in class_names:
                    all_feature_classes_set.add(class_name)
        
        #this is the set of classes imported in our projects from the given feature package
        return all_feature_classes_set

# %%


# %%
native_imports_set = get_all_native_imports(all_class_metrics_csv)
features_df = pd.read_csv(feature_packages_csv)

feature_classes_count = []
feature_classes = []
for feature_package in features_df["Packages"]:
    classes_set = get_all_feature_classes(native_imports_set, feature_package)
    print (classes_set)
    feature_classes.append(",".join(list(classes_set)))
    feature_classes_count.append(len(classes_set))

features_df["ClassesCount"] = feature_classes_count
features_df["Classes"] = feature_classes
    

# %%
features_df


# %%
# Post processing to remove the overlapping in the resulting features and usage identification
# then Update the DataFrame with the updated classes

# Loading all the feature classes into sets
set_java_util = set((features_df[features_df["Packages"] == "java.util"]["Classes"]).values[0].split(","))
set_java_util_concurrent = set((features_df[features_df["Packages"] == "java.util.concurrent"]["Classes"]).values[0].split(","))
set_java_util_regex = set((features_df[features_df["Packages"] == "java.util.regex"]["Classes"]).values[0].split(","))

set_java_lang = set((features_df[features_df["Packages"] == "java.lang"]["Classes"]).values[0].split(","))
set_java_lang_management = set((features_df[features_df["Packages"] == "java.lang.management"]["Classes"]).values[0].split(","))
set_java_lang_Thread = set((features_df[features_df["Packages"] == "java.lang.Thread"]["Classes"]).values[0].split(","))

set_java_io = set((features_df[features_df["Packages"] == "java.io"]["Classes"]).values[0].split(","))

set_java_nio = set((features_df[features_df["Packages"] == "java.nio"]["Classes"]).values[0].split(","))
set_java_nio_channels = set((features_df[features_df["Packages"] == "java.nio.channels"]["Classes"]).values[0].split(","))
set_java_nio_file = set((features_df[features_df["Packages"] == "java.nio.file"]["Classes"]).values[0].split(","))
set_java_nio_charset = set((features_df[features_df["Packages"] == "java.nio.charset"]["Classes"]).values[0].split(","))

set_java_net = set((features_df[features_df["Packages"] == "java.net"]["Classes"]).values[0].split(","))
set_javax_net_ssl = set((features_df[features_df["Packages"] == "javax.net.ssl"]["Classes"]).values[0].split(","))

set_java_text = set((features_df[features_df["Packages"] == "java.text"]["Classes"]).values[0].split(","))

set_java_math = set((features_df[features_df["Packages"] == "java.math"]["Classes"]).values[0].split(","))

# Subtracting the Subsets from SuperSets to remove the overlapping in the resulting features and usage identification
set_java_util = set_java_util - set_java_util_concurrent
set_java_util = set_java_util - set_java_util_regex

set_java_lang = set_java_lang - set_java_lang_management
set_java_lang = set_java_lang - set_java_lang_Thread

set_java_nio = set_java_nio - set_java_nio_channels
set_java_nio = set_java_nio - set_java_nio_file
set_java_nio = set_java_nio - set_java_nio_charset

# Updating the DataFrame with the updated classes
features_df.loc[features_df["Packages"] == "java.util", "Classes"] = ",".join(list(set_java_util))
features_df.loc[features_df["Packages"] == "java.lang", "Classes"] = ",".join(list(set_java_lang))
features_df.loc[features_df["Packages"] == "java.nio", "Classes"] = ",".join(list(set_java_nio))

#updating the classes count
features_df.loc[features_df["Packages"] == "java.util", "ClassesCount"] = len(set_java_util)
features_df.loc[features_df["Packages"] == "java.lang", "ClassesCount"] = len(set_java_lang)
features_df.loc[features_df["Packages"] == "java.nio", "ClassesCount"] = len(set_java_nio)


# %%
features_df

# %%
#Finally saving the updated DataFrame to a CSV file
features_df.to_csv(extracted_features_classes_csv, index=False)


