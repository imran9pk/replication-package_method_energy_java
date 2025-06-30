# %%
import os
import re
import pandas as pd
import lxml.etree as let
import helpers as hlp
import xpath_queries
from pathlib import Path

# %%
from datetime import datetime
import subprocess

def convert_function_node_to_java(function_node) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"temp_method_{timestamp}"
    
    temp_xml_path = os.path.join(os.getcwd(), f"{base_name}.xml")
    temp_java_path = os.path.join(os.getcwd(), f"{base_name}.java")

    try:
        with open(temp_xml_path, 'w', encoding='utf-8') as temp_xml:
            temp_xml.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            temp_xml.write('<unit xmlns="http://www.srcML.org/srcML/src">\n')
            temp_xml.write(let.tostring(function_node, encoding='unicode'))
            temp_xml.write('\n</unit>\n')

        subprocess.run(['srcml', temp_xml_path, '-o', temp_java_path], check=True)

        with open(temp_java_path, 'r', encoding='utf-8') as f:
            java_code = f.read()

    finally:
        if os.path.exists(temp_xml_path):
            os.remove(temp_xml_path)
        if os.path.exists(temp_java_path):
            os.remove(temp_java_path)

    return java_code.strip()


# %%
def get_function_type_scope(functions):
    method_scopes=[]
    method_types=[]

    for function in functions:
        #Getting the specifiers of curret function
        specifier = function.xpath("./srcml:type/srcml:specifier/text()", namespaces=xpath_queries.srcml_namespace)
        specifier = "".join(specifier)

        method_type = function.xpath("./srcml:type/srcml:name/text()", namespaces=xpath_queries.srcml_namespace)
        method_type = "".join(method_type)

        scope = "default"
        #check if specifier contains any of the access specifiers (public, private, protected) set the value of scope to either public, private or protected
        if("public" in specifier):
            scope = "public"
        elif("private" in specifier):
            scope = "private"
        elif("protected" in specifier):
            scope = "protected"
        
        method_scopes.append(scope)
        method_types.append(method_type)

    return method_scopes, method_types

# %%
def get_function_name_length(functions):
    name_lengths = []
    for function in functions:
        # Getting the name of the current function
        function_name = function.xpath("srcml:name/text()", namespaces=xpath_queries.srcml_namespace)
        if function_name:
            name_lengths.append(len(function_name[0]))
        else:
            name_lengths.append(0)
    return name_lengths


# %%
def get_number_fors(functions):
    number_fors = []
    # Getting number of for loops
    for function in functions:
        fors = function.xpath(".//srcml:for", namespaces=xpath_queries.srcml_namespace)
        number_fors.append(len(fors))
    return number_fors

# %%
def get_number_whiles(functions):
    number_whiles = []
    for function in functions:
        # Getting number of while loops
        whiles = function.xpath(".//srcml:while", namespaces=xpath_queries.srcml_namespace)
        number_whiles.append(len(whiles))
    return number_whiles

# %%
def get_number_dowhiles(functions):
    number_dowhiles = []

    for function in functions:  
        # Getting number of do while loops
        dos = function.xpath(".//srcml:do", namespaces=xpath_queries.srcml_namespace)
        number_dowhiles.append(len(dos))
    return number_dowhiles

# %%
def get_number_ifs(functions):
    number_ifs = []
    for function in functions:
        # Getting number of if statements
        ifs = function.xpath(".//srcml:if_stmt", namespaces=xpath_queries.srcml_namespace)
        number_ifs.append(len(ifs))
    return number_ifs

# %%
def get_number_switches(functions):
    number_switches = []
    for function in functions:
        # Getting number of switch statements
        switches = function.xpath(".//srcml:switch", namespaces=xpath_queries.srcml_namespace)
        number_switches.append(len(switches))
    return number_switches

# %%
def get_switch_cases_count(functions):
    switch_cases_count = []
    for function in functions:
        # Getting number of cases in all switch statements
        switch_cases = function.xpath(".//srcml:switch/srcml:case", namespaces=xpath_queries.srcml_namespace)
        switch_cases_count.append(len(switch_cases))
    return switch_cases_count


# %%
def get_loc(functions):
    srcml_statement_tags=["assert","break","case","continue","default","do","empty_stmt","expr_stmt","decl_stmt","for","if","else","label","return","switch","while","throw","function","lambda","enum","annotation"]
    loc = []
    for function in functions:
        count = 0
        for tag in srcml_statement_tags:
            count += len(function.xpath(".//srcml:{0}".format(tag), namespaces=xpath_queries.srcml_namespace))
        loc.append(count)
    return loc

# %%
def get_calls_count(functions):
    calls_count = []
    
    for function in functions:
        simple_calls_names = function.xpath(".//srcml:call[not(preceding-sibling::srcml:operator[1]/text()='new') and count(.//srcml:name/child::srcml:name)=0]/srcml:name/text()", namespaces=xpath_queries.srcml_namespace)
        chained_calls_names = function.xpath(".//srcml:call[not(preceding-sibling::srcml:operator[1]/text()='new') and count(.//srcml:name/child::srcml:name)>0]/srcml:name/child::srcml:name[last()]/text()", namespaces=xpath_queries.srcml_namespace)
        calls_names = simple_calls_names + chained_calls_names
        
        calls_count.append(len(calls_names))
    
    return calls_count

# %%
def get_calls_names(functions):
    calls_names = []
    
    for function in functions:
        names = ""
        simple_calls_names = function.xpath(".//srcml:call[not(preceding-sibling::srcml:operator[1]/text()='new') and count(.//srcml:name/child::srcml:name)=0]/srcml:name/text()", namespaces=xpath_queries.srcml_namespace)
        chained_calls_names = function.xpath(".//srcml:call[not(preceding-sibling::srcml:operator[1]/text()='new') and count(.//srcml:name/child::srcml:name)>0]/srcml:name/child::srcml:name[last()]/text()", namespaces=xpath_queries.srcml_namespace)
        
        if(len(simple_calls_names + chained_calls_names)>0):
            names = ",".join(simple_calls_names + chained_calls_names) 
        
        calls_names.append(names)
    
    return calls_names

# %%
def get_number_returns(functions):
    number_returns = []
    for function in functions:
        # Getting number of return statements
        returns = function.xpath(".//srcml:return", namespaces=xpath_queries.srcml_namespace)
        number_returns.append(len(returns))
    return number_returns

# %%
def get_number_throws(functions):
    number_throws = []
    for function in functions:
        # Getting number of throw statements
        throws = function.xpath(".//srcml:throw", namespaces=xpath_queries.srcml_namespace)
        number_throws.append(len(throws))
    return number_throws


# %%
def get_number_catches(functions):
    number_catches = []
    for function in functions:
        # Getting number of catch statements
        catches = function.xpath(".//srcml:catch", namespaces=xpath_queries.srcml_namespace)
        number_catches.append(len(catches))
    return number_catches


# %%
def get_cyclo(functions): #Cyclomatic Complexity is V(G) = P + 1, where Where P = Number of predicate nodes (node that contains condition)

    cyclos = []
    cyclo = 1 #if there is only 1 path in the method, the Cyclo will be 1
    
    for function in functions:
        # Getting number of Conditions
        condition_count = function.xpath("count(.//srcml:condition)", namespaces=xpath_queries.srcml_namespace)
        count = function.xpath("count(.//srcml:condition | .//srcml:catch | .//srcml:case | .//srcml:default)", namespaces=xpath_queries.srcml_namespace)

        cyclo = cyclo + count
        
        cyclos.append(int(cyclo))
        cyclo = 1
    
    return cyclos

# %%
def get_number_variable_declarations(functions):
    number_variable_declarations = []
    for function in functions:
        # Getting number of variable declarations
        variable_declarations = function.xpath(".//srcml:decl_stmt", namespaces=xpath_queries.srcml_namespace)
        number_variable_declarations.append(len(variable_declarations))
    return number_variable_declarations


# %%
def mark_overloaded(method_names):
    is_overloaded_list = []
    for name in method_names:
        is_overloaded_list.append(1 if method_names.count(name)>1 else 0)
    
    return is_overloaded_list

# %%
def get_nested_loops_count(functions):
    count_list = []
    
    for function in functions:
        query_for = "count(.//srcml:for[(descendant::srcml:for or descendant::srcml:while or descendant::srcml:do) and (not(ancestor::srcml:for) and not(ancestor::srcml:while) and not(ancestor::srcml:do))])"
        for_nesting_count = function.xpath(query_for, namespaces=xpath_queries.srcml_namespace)
        
        query_while = "count(.//srcml:while[(descendant::srcml:for or descendant::srcml:while or descendant::srcml:do) and (not(ancestor::srcml:for) and not(ancestor::srcml:while) and not(ancestor::srcml:do))])"
        while_nesting_count = function.xpath(query_for, namespaces=xpath_queries.srcml_namespace)
        
        query_do = "count(.//srcml:do[(descendant::srcml:for or descendant::srcml:while or descendant::srcml:do) and (not(ancestor::srcml:for) and not(ancestor::srcml:while) and not(ancestor::srcml:do))])"
        do_nesting_count = function.xpath(query_for, namespaces=xpath_queries.srcml_namespace)
        
        #convert the query result to int
        total = for_nesting_count+while_nesting_count+do_nesting_count
        count_list.append(int(total))
    return count_list

# %%
def get_calls_details(call_names_list, method_names_list):
    #Reading each defined method to get internal and external method call from inside its body
    internal_calls = []
    external_calls = []

    internal_calls_count = []
    external_calls_count = []

    for names_list in call_names_list:
        
        current_internal_calls = ""
        current_internal_calls_count = 0
        current_external_calls = ""
        current_external_calls_count = 0
        
        if(isinstance(names_list,str) and len(names_list)>0): #there are any method calls from this method
            for name in names_list.split(","):
                if((method_names_list.str.split(".").str[-1]==name).any()):
                    current_internal_calls += name + ","
                    current_internal_calls_count += 1
                else:
                    current_external_calls += name + ","
                    current_external_calls_count += 1
                            
        internal_calls.append(current_internal_calls)
        external_calls.append(current_external_calls)
        
        internal_calls_count.append(current_internal_calls_count)
        external_calls_count.append(current_external_calls_count)
        
    return internal_calls, external_calls, internal_calls_count, external_calls_count

# %%
def get_classes_by_package(package_name):
    
    features_csv = "feature_classes.csv"
    features_df = pd.read_csv(features_csv)
    
    row = features_df[features_df["Packages"] == package_name]

    # Extract and safely split the Classes field
    if not row.empty:
        classes_raw = row["Classes"].values[0]
        if pd.isna(classes_raw):
            classes_list = []
        else:
            classes_list = classes_raw.split(",")
    else:
        classes_list = []

    return classes_list

# %%
def get_library_usage_features(functions):

    list_java_util = []
    list_java_lang_Thread = []
    list_java_util_concurrent = []
    list_java_io = []
    list_java_nio = []
    list_java_nio_channels = []
    list_java_nio_file = []
    list_java_nio_charset = []
    list_java_net = []
    list_javax_net_ssl = []
    list_java_lang = []
    list_java_lang_management = []
    list_java_util_regex = []
    list_java_text = []
    list_java_math = []

    for function in functions:
        classes = get_classes_by_package("java.util")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_util.append(using_Classes)

        classes = get_classes_by_package("java.lang.Thread")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_lang_Thread.append(using_Classes)

        classes = get_classes_by_package("java.util.concurrent")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_util_concurrent.append(using_Classes)
        
        classes = get_classes_by_package("java.io")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_io.append(using_Classes)

        classes = get_classes_by_package("java.nio")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_nio.append(using_Classes)

        classes = get_classes_by_package("java.nio.channels")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_nio_channels.append(using_Classes)

        classes = get_classes_by_package("java.nio.file")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_nio_file.append(using_Classes)

        classes = get_classes_by_package("java.nio.charset")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_nio_charset.append(using_Classes)

        classes = get_classes_by_package("java.net")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_net.append(using_Classes)

        classes = get_classes_by_package("javax.net.ssl")
        using_Classes = hlp.is_using_classes(function,classes)
        list_javax_net_ssl.append(using_Classes)

        classes = get_classes_by_package("java.lang")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_lang.append(using_Classes)

        classes = get_classes_by_package("java.lang.management")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_lang_management.append(using_Classes)

        classes = get_classes_by_package("java.util.regex")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_util_regex.append(using_Classes)

        classes = get_classes_by_package("java.text")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_text.append(using_Classes)

        classes = get_classes_by_package("java.math")
        using_Classes = hlp.is_using_classes(function,classes)
        list_java_math.append(using_Classes)

    return list_java_util, list_java_lang_Thread, list_java_util_concurrent, list_java_io, list_java_nio, list_java_nio_channels, list_java_nio_file, list_java_nio_charset, list_java_net, list_javax_net_ssl, list_java_lang, list_java_lang_management, list_java_util_regex, list_java_text, list_java_math

# %%
def get_concurrency_and_collection_usage(class_names_list, functions, classes_dict):
    list_uses_concurrency = []
    list_uses_collections = []

    #iterate on class names list and functions list using index
    #for each funciton check if the class uses concurrency and collections
    #if the class uses concurrency, then check if the current method uses concurrency
    #if the class uses collection, then check if the current method uses collection
    for i in range(len(class_names_list)):
        class_name = class_names_list[i]
        function = functions[i]
        
        uses_concurrency = 0
        uses_collections = 0
        
        #Getting the values of uses_concurrency and uses_collections from the dictionary for current class
        if(class_name in classes_dict.keys()):
            uses_concurrency = 1 if classes_dict[class_name][0] else 0
            uses_collections = 1 if classes_dict[class_name][1] else 0
        
        #If the class uses concurrency, then check if the current method uses concurrency
        if(uses_concurrency):
            #pass the current function to is_using_concurrency function to check if it uses concurrency
            uses_concurrency = hlp.is_using_concurrency(function)

        #If the class uses collection, then check if the current method uses collection
        if(uses_collections):
            #pass the current function to is_using_collection function to check if it uses collection
            uses_collections = hlp.is_using_collection(function)
            
        list_uses_concurrency.append(uses_concurrency)
        list_uses_collections.append(uses_collections)

    return list_uses_concurrency, list_uses_collections

# %%
def collect_method_metrics(functions, task):
    df = pd.DataFrame()
    df['Method_name'] = hlp.extract_method_names(functions)

    class_names = df['Method_name'].str.split('.').str[0:-1]
    df.insert(0,"Class_Names",class_names)
    df["Class_Names"] = df["Class_Names"].str.join(".")
    df.loc[df["Class_Names"].str[-1] == ".", "Class_Names"] = "MISSING"
    
    df.insert(0,"task",task)

    df["methodScope"], df["methodType"] = get_function_type_scope(functions)
    df["nameLen"] = get_function_name_length(functions)
    df["isOverloaded"] = mark_overloaded(df['Method_name'].values.tolist())
    df["methodLoc"] = get_loc(functions)
    df["#for"] = get_number_fors(functions)
    df["#while"] = get_number_whiles(functions)
    df["#do"] = get_number_dowhiles(functions)
    df["#nestedLoops"] = get_nested_loops_count(functions)
    df["#if"] = get_number_ifs(functions)
    df["#switch"] = get_number_switches(functions)
    df["#case"] = get_switch_cases_count(functions)
    df["#return"] = get_number_returns(functions)
    df["#throw"] = get_number_throws(functions)
    df["#catch"] = get_number_catches(functions)
    df["cyclo"] = get_cyclo(functions)
    df["#vars"] = get_number_variable_declarations(functions)
    df["#methodCalls"] = get_calls_count(functions)
    df["methodCallNames"] = get_calls_names(functions)
    df["internalCallsList"], df["externalCallsList"],df["#internalCalls"],df["#externalCalls"] = get_calls_details(df["methodCallNames"], df["Method_name"])
    df["usesJavaUtil"], df["usesJavaLangThread"], df["usesJavaUtilConcurrent"], df["usesJavaIo"], df["usesJavaNio"], df["usesJavaNioChannels"], df["usesJavaNioFile"], df["usesJavaNioCharset"], df["usesJavaNet"], df["usesJavaxNetSsl"], df["usesJavaLang"], df["usesJavaLangManagement"], df["usesJavaUtilRegex"], df["usesJavaText"], df["usesJavaMath"] = get_library_usage_features(functions)

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

    OUTPUT_CSV_PATH = data_dir / "method_static_metrics_temp.csv"


    all_dfs = []
    for srcml_dir in srcml_dirs:
        dir_dfs = []
        for srcml_file in srcml_dir.glob("*.xml"):
            task = srcml_file.stem
            print(f"Processing: {task}")
            
            functions = hlp.get_filtered_methods(srcml_file, False, xpath_queries.srcml_namespace)
            if not functions:
                continue

            df_metrics = collect_method_metrics(functions, task)
            dir_dfs.append(df_metrics)
        
        if dir_dfs:
            combined_dir_df = pd.concat(dir_dfs, ignore_index=True)
            all_dfs.append(combined_dir_df)
    
    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Saved combined method metrics to: {OUTPUT_CSV_PATH}")
    else:
        print("No method metrics were collected.")



