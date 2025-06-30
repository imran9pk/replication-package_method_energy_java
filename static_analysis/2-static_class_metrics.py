# %%
import os, glob
import shutil
from lxml import etree as let
import pandas as pd
import stack
import xpath_queries
import helpers as hlp
import json
from pathlib import Path

# %%
def get_full_class_name(aclass):
    full_class_name = ""
    
    package_name = ""
    package_name = aclass.xpath(xpath_queries.class_package_name_query, namespaces=xpath_queries.srcml_namespace)
    package_name = "".join(package_name)

    #If the class is a nested class, the sibling package will not exist, the name will include parent class + package name
    if(len(package_name)==0):
        package_name = aclass[0].xpath(xpath_queries.package_name_query, namespaces=xpath_queries.srcml_namespace)
        if(len(package_name)==0):
            package_name = aclass[0].xpath(xpath_queries.package_name_query_from_ancestor_interface, namespaces=xpath_queries.srcml_namespace)
        package_name = "".join(package_name)
        
    ancestor_class_name = aclass[0].xpath(xpath_queries.ancestor_class_name_query, namespaces=xpath_queries.srcml_namespace)
    if(len(ancestor_class_name)==0):
        ancestor_class_name = aclass[0].xpath(xpath_queries.ancestor_class_inner_name_query, namespaces=xpath_queries.srcml_namespace)
    if(len(ancestor_class_name)==0):
        ancestor_class_name = aclass[0].xpath(xpath_queries.ancestor_interface_name_query, namespaces=xpath_queries.srcml_namespace)
    ancestor_class_name = ".".join(ancestor_class_name)
    
    package_name = package_name+"."+ancestor_class_name   
        
    current_class_name = aclass.xpath(xpath_queries.current_name_query, namespaces=xpath_queries.srcml_namespace)
    if len(current_class_name)==0:
                    current_class_name=aclass.xpath(xpath_queries.current_full_name_query,namespaces=xpath_queries.srcml_namespace)
    current_class_name = "".join(current_class_name)
    
    seperator = "." if len(package_name)>0 else ""
    full_class_name = package_name+seperator+current_class_name
    return full_class_name

# %%
def get_all_class_names(classes):
    class_names = []
    for aclass in classes:
        full_name = get_full_class_name(aclass)
        class_names.append(full_name)

    return class_names

# %%
def get_nesting_levels(classes):
    nesting_levels = []
    for aclass in classes:
        ancestors_class_len = (len(aclass.xpath("ancestor::srcml:class", namespaces=xpath_queries.srcml_namespace)))
        ancestors_interface_len = (len(aclass.xpath("ancestor::srcml:interface", namespaces=xpath_queries.srcml_namespace)))
        nesting_levels.append(ancestors_class_len+ancestors_interface_len)
        
    return nesting_levels

# %%
def get_class_specifiers(classes):
    class_scopes=[]
    abstract_flags = []

    for aclass in classes:
        #Getting the specifiers of curret class
        specifier = aclass.xpath("./srcml:specifier//text()", namespaces=xpath_queries.srcml_namespace)
        specifier = "".join(specifier)

        scope = "default"
        #check if specifier contains any of the access specifiers (public, private, protected) set the value of scope to either public, private or protected
        if("public" in specifier):
            scope = "public"
        elif("private" in specifier):
            scope = "private"
        elif("protected" in specifier):
            scope = "protected"
        
        abstract_flag = 0
        #check if specifier contains abstract keyword set abstract flag to true
        if("abstract" in specifier):
            abstract_flag = 1

        class_scopes.append(scope)
        abstract_flags.append(abstract_flag)

    return class_scopes, abstract_flags

# %%
def get_class_in_pkg_count(classes):
    class_pkgs=[]
    count_PkgClasses = []
    for aclass in classes:
        pkg_name = aclass.xpath(xpath_queries.class_package_name_query, namespaces=xpath_queries.srcml_namespace)
        pkg_name = "".join(pkg_name)
        
        if(len(pkg_name)==0):
            pkg_name = aclass[0].xpath(xpath_queries.package_name_query, namespaces=xpath_queries.srcml_namespace)
            pkg_name = "".join(pkg_name)
        
        class_pkgs.append(pkg_name)
    
    #create a dictionary of package name and its occurence count in the class_pkgs
    pkg_count = {i:class_pkgs.count(i) for i in class_pkgs}

    for package in class_pkgs:
        count_PkgClasses.append(pkg_count[package])

    return count_PkgClasses

# %%
def get_number_imports(classes):
    number_imports = []
    for aclass in classes:
        imports = aclass.xpath(xpath_queries.all_imports_query, namespaces=xpath_queries.srcml_namespace)
        number_imports.append(len(imports))
    return number_imports

# %%
def get_extends_count (classes):
    number_extends = []
    for aclass in classes:
        extends = aclass.xpath("./srcml:super_list/srcml:extends", namespaces=xpath_queries.srcml_namespace)
        number_extends.append(len(extends))
    return number_extends

# %%
def get_implements_count(classes):
    number_implements = []
    for aclass in classes:
        implements = aclass.xpath("./srcml:super_list/srcml:implements/srcml:super", namespaces=xpath_queries.srcml_namespace)
        number_implements.append(len(implements))
    return number_implements

# %%
def get_native_imports_count(classes):
    number_native_imports = []
    for aclass in classes:
        imports = aclass.xpath(xpath_queries.java_imports_query, namespaces=xpath_queries.srcml_namespace)
        number_native_imports.append(len(imports))
    return number_native_imports

# %%
def get_native_imports_list(classes):
    list_native_imports = []
    
    for aclass in classes:
        this_class_imports = []
        imports = aclass.xpath(xpath_queries.java_imports_query, namespaces=xpath_queries.srcml_namespace)
        
        for aimport in imports:
            import_name = aimport.xpath(".//text()", namespaces=xpath_queries.srcml_namespace)
            import_name = "".join(import_name)
            this_class_imports.append(import_name)
    
        list_native_imports.append(",".join(this_class_imports))   
    return list_native_imports

# %%
def collect_class_metrics(classes):
    
    #Collect all the metrics from Soruce File in DataFrame
    df = pd.DataFrame()

    #Extract Class Names
    # df['Class_Names'] = get_all_class_names(classes)
    df['Class_Names'] = hlp.extract_class_names(classes)
    df['#cPkgClasses'] = get_class_in_pkg_count(classes)
    df['classScope'], df["cIsAbstract"] = get_class_specifiers(classes)
    df['cNestingLevel'] = get_nesting_levels(classes)
    df['#cImports'] = get_number_imports(classes)
    df['#cNativeImports'] = get_native_imports_count(classes)
    df['cNativeImportsList'] = get_native_imports_list(classes)
    df['#cInherits'] = get_extends_count (classes)
    df['#cImplements'] = get_implements_count(classes)
    

    return df

# %%
def process_srcml_for_class_metrics(srcml_file, task):

    srcml_file = Path(srcml_file)

    # Parse the XML file into LXML Tree
    ltree = let.parse(str(srcml_file))
    lroot = ltree.getroot()

    # Query to get all class nodes 
    classes_query = "//srcml:class"
    class_nodes = lroot.xpath(classes_query, namespaces=xpath_queries.srcml_namespace)

    # Filter class nodes excluding those without a name element
    filtered_classes = [
        node
        for node in class_nodes
        if node.xpath("srcml:name", namespaces=xpath_queries.srcml_namespace)
    ]

    df = collect_class_metrics(filtered_classes)
    df.insert(0,"task",task)
    
    return df

# %%
# Main execution
if __name__ == "__main__":

    # Paths and Config
    root_dir = Path.cwd().parent
    data_dir = root_dir / "data"

    srcml_rosetta_dir = data_dir / "srcml_rosetta"
    srcml_clbg_dir = data_dir / "srcml_clbg"
    srcml_dirs = [srcml_rosetta_dir,srcml_clbg_dir]

    OUTPUT_CSV_PATH = data_dir / "calss_metrics.classes.csv"

    all_dfs = []
    for srcml_dir in srcml_dirs:
        dir_dfs = []
        for srcml_file in srcml_dir.glob("*.xml"):
            task = srcml_file.stem        
            print(f"Processing: {task}")
            df = process_srcml_for_class_metrics(srcml_file, task)
            dir_dfs.append(df)

        if dir_dfs:
            combined_dir_df = pd.concat(dir_dfs, ignore_index=True)
            all_dfs.append(combined_dir_df)

    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Saved combined class metrics to: {OUTPUT_CSV_PATH}")
    else:
        print("No class metrics were collected.")


