#Defining the namespace to be used with the xpath queries
srcml_namespace = {"srcml": "http://www.srcML.org/srcML/src"}


#Defining xpath queries to be used across the script

# xPath query for counting all the File Units in a SRCML file
file_units_query = "//srcml:unit"


# xPath query for all the functions that have annotation @Test 
test_methods_query = "//srcml:function[srcml:annotation/srcml:name='Test' or (srcml:annotation/srcml:name='ParameterizedTest')]"

# xPath query for all the functions EXCEPT @benchmarked
all_methods_except_bench_query = "//srcml:function[not(srcml:annotation/srcml:name='Benchmark')]"

# xPath query for all the functions that have annotation @Benchmark
benched_methods_query = "//srcml:function[srcml:annotation/srcml:name='Benchmark']"

# xPath query for all the functions that do not have annotation @Test, Setup, Benchmark, ParameterizedTest in the Java file of Project only
project_methods_query = "//srcml:function[not(srcml:annotation/srcml:name='Test') and not(srcml:annotation/srcml:name='Benchmark') and not(srcml:annotation/srcml:name='Setup') and not(srcml:annotation/srcml:name='ParameterizedTest')]"
#project_methods_query = "//srcml:function"

#xPath query to get the package name from a given function node, by looking for the preceding siblings
package_name_query = ".//ancestor::srcml:class/preceding-sibling::srcml:package/srcml:name//text()"
package_name_query_from_ancestor_interface = ".//ancestor::srcml:interface/preceding-sibling::srcml:package/srcml:name//text()"

#xPath query to get the package name from a given class node
class_package_name_query = ".//preceding-sibling::srcml:package/srcml:name//text()"

#xPath query to get the ancestor class name for an INNER class
ancestor_class_name_query = "../ancestor::srcml:class/srcml:name/text()"
ancestor_class_inner_name_query = "../ancestor::srcml:class/srcml:name/srcml:name/text()" 
ancestor_interface_name_query = "../ancestor::srcml:interface/srcml:name/text()"

#xPath query to get the Text Name of the current node
current_full_name_query = "srcml:name/srcml:name/text()"

#xPath query to get the Text Name of the current node
current_name_query = "srcml:name/text()"

#xPath query to get all the imports for a class
#all_imports_query = ".//preceding-sibling::srcml:import"
all_imports_query = "ancestor::srcml:unit/srcml:import"

#xPath query to get only Java imports for a class
#java_imports_query = ".//preceding-sibling::srcml:import/srcml:name[(srcml:name='java') or (srcml:name='javax') or (srcml:name='com' and srcml:name[2] = 'sun')]"
java_imports_query = "ancestor::srcml:unit/srcml:import/srcml:name[(srcml:name='java') or (srcml:name='javax') or (srcml:name='com' and srcml:name[2] = 'sun')]"
                   



