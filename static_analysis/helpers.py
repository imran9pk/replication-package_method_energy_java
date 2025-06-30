import stack
import xpath_queries
import lxml.etree as let
import re

concurrency_classes_in_dataset = [
    "AbstractExecutorService", "AbstractQueuedSynchronizer", "ArrayBlockingQueue", "AtomicBoolean", "AtomicInteger",
    "AtomicIntegerArray", "AtomicIntegerFieldUpdater", "AtomicLong", "AtomicLongArray", "AtomicLongFieldUpdater",
    "AtomicMarkableReference", "AtomicReference", "AtomicReferenceArray", "AtomicReferenceFieldUpdater", "BlockingDeque",
    "BlockingQueue", "BrokenBarrierException", "Callable", "CancellationException", "CompletableFuture", "CompletionException",
    "CompletionService", "CompletionStage", "ConcurrentHashMap", "ConcurrentLinkedDeque", "ConcurrentLinkedQueue",
    "ConcurrentMap", "ConcurrentNavigableMap", "ConcurrentSkipListMap", "ConcurrentSkipListSet", "Condition",
    "CopyOnWriteArrayList", "CopyOnWriteArraySet", "CountDownLatch", "CyclicBarrier", "Delayed", "DelayQueue",
    "DoubleAdder", "Exchanger", "ExecutionException", "Executor", "ExecutorCompletionService", "Executors",
    "ExecutorService", "Flow", "ForkJoinPool", "ForkJoinTask", "ForkJoinWorkerThread", "Future", "FutureTask",
    "LinkedBlockingDeque", "LinkedBlockingQueue", "LinkedTransferQueue", "Lock", "LockSupport", "LongAccumulator",
    "LongAdder", "Phaser", "PriorityBlockingQueue", "ReadWriteLock", "ReentrantLock", "ReentrantReadWriteLock",
    "RejectedExecutionException", "RejectedExecutionHandler", "RunnableFuture", "RunnableScheduledFuture",
    "ScheduledExecutorService", "ScheduledFuture", "ScheduledThreadPoolExecutor", "Semaphore", "StampedLock",
    "SynchronousQueue", "ThreadFactory", "ThreadLocalRandom", "ThreadPoolExecutor", "TimeoutException", "TimeUnit"
]

collection_classes_in_dataset = [
    "AbstractCollection", "AbstractList", "AbstractMap", "AbstractQueue", "AbstractSequentialList", "AbstractSet",
    "ArrayDeque", "ArrayList", "Arrays", "Collection", "Collections", "Collector", "Collectors", "Comparator",
    "Deque", "Dictionary", "DoubleStream", "DoubleSummaryStatistics", "EmptyStackException", "Enumeration", "EnumMap",
    "EnumSet", "HashMap", "HashSet", "Hashtable", "IdentityHashMap", "Iterator", "LinkedHashMap", "LinkedHashSet",
    "LinkedList", "List", "ListIterator", "LongStream", "Map", "NavigableMap", "NavigableSet", "PrimitiveIterator",
    "PriorityQueue", "Queue", "Set", "SortedMap", "SortedSet", "Spliterator", "Spliterators", "Stack", "Stream",
    "TreeMap", "TreeSet", "Vector", "WeakHashMap"
]

def is_using_classes(function,classes):

    #get all the types used in the method return type, parameters and local variables
    types = function.xpath(".//srcml:type//srcml:name//text()", namespaces=xpath_queries.srcml_namespace)
    #check if any value in the list of method_parameter_types is in the list of concurrency_classes_in_dataset and retun true if yes
    uses_classes = 1 if any(aClass in types for aClass in classes) else 0

    return uses_classes


def is_using_concurrency(function):

    #get all the types used in the method return type, parameters and local variables
    types = function.xpath(".//srcml:type//srcml:name//text()", namespaces=xpath_queries.srcml_namespace)
    #check if any value in the list of method_parameter_types is in the list of concurrency_classes_in_dataset and retun true if yes
    uses_concurrency = 1 if any(concurrency_class in types for concurrency_class in concurrency_classes_in_dataset) else 0

    return uses_concurrency

def is_using_collection(function):

    #get all the types used in the method return type, parameters and local variables
    types = function.xpath(".//srcml:type//srcml:name//text()", namespaces=xpath_queries.srcml_namespace)
    #check if any value in the list of method_parameter_types is in the list of collection_classes_in_dataset and retun true if yes
    uses_collection = 1 if any(collection_class in types for collection_class in collection_classes_in_dataset) else 0    

    return uses_collection


# Get the filtered methods based on the input flag and given function nodes
def get_filtered_methods(srcml_file, isBenchTagged, srcml_namespace):
    #Parse the XML file into LXML Tree
    ltree = let.parse(srcml_file)
    lroot = ltree.getroot()

    # First query to get all function nodes
    functions_query = "//srcml:function"
    function_nodes = lroot.xpath(functions_query, namespaces=srcml_namespace)

    functions=[]
    match isBenchTagged:
        case True:
            # Filter functions based on 'Benchmark' annotation
            functions = [
                node
                for node in function_nodes
                if any(
                    annotation.xpath("srcml:name/text()", namespaces=srcml_namespace)
                    and annotation.xpath("srcml:name/text()", namespaces=srcml_namespace)[0] == 'Benchmark'
                    for annotation in node.xpath("srcml:annotation", namespaces=srcml_namespace)
                )
            ]

        case False:
            # Filter functions exclude specific annotated functions
            functions = [
                node
                for node in function_nodes
                if all(
                    annotation.xpath("srcml:name/text()", namespaces=srcml_namespace)
                    and annotation.xpath("srcml:name/text()", namespaces=srcml_namespace)[0] not in {'Test', 'Benchmark', 'Setup', 'ParameterizedTest'}
                    for annotation in node.xpath("srcml:annotation", namespaces=srcml_namespace)
                )
            ]
    
    return functions


# check if the name of the method is complete
def is_complete_name_RE(method):
    pattern = r"^\w+(\.\w+)*\.\w+\.\w+$"
    return bool(re.match(pattern, method))

def remove_duplicate_methods_incomplete_names(df):
    #calling the method is_complete_name_RE on each value of df["Method_name"]

    # Apply the is_complete_name_RE function to each value in the "Method_name" column
    df['is_complete_name'] = df['Method_name'].apply(is_complete_name_RE)

    # Filter the DataFrame to keep only the rows where 'is_complete_name' is True
    df = df[df['is_complete_name']]

    # Drop the 'is_complete_name' column as it's no longer needed
    df = df.drop('is_complete_name', axis=1)

    #Drop the duplicate rows based on Method_name column
    df = df.drop_duplicates(subset="Method_name", keep="first")

    return df


def extract_method_names(functions):

    function_names = []

    #Loops starts here to iterate on each function of the file and get the 
    #fully qualified name by traversing to all parent classes until we reach the Unit node

    for function in functions:
        
        name_stack = stack.Stack()
        
        package_name = ""
        package_name = function.xpath(xpath_queries.package_name_query, namespaces=xpath_queries.srcml_namespace)
        package_name = "".join(package_name).strip()
        
        #get name of the current function and push it on stack
        current_name = function.xpath(xpath_queries.current_name_query,namespaces=xpath_queries.srcml_namespace)
        name_stack.push("".join(current_name))
        name_stack.push(".")

        #Now get parent of the current function and keep on getting parents of parent until we reach the top
        parent = function.getparent()
        while not(parent is None):
            #if the current parent node is a class, get the name of this node (i.e. name of the class)
            if(parent.tag=="{http://www.srcML.org/srcML/src}class" or parent.tag=="{http://www.srcML.org/srcML/src}interface"):
                
                current_class_name=""
                current_class_name=parent.xpath(xpath_queries.current_name_query,namespaces=xpath_queries.srcml_namespace)
               
                if len(current_class_name)==0:
                    current_class_name=parent.xpath(xpath_queries.current_full_name_query,namespaces=xpath_queries.srcml_namespace)

                name_stack.push("".join(current_class_name))
                name_stack.push(".")

            parent = parent.getparent()

        #finally pushing the package name to stack to complete the fully qualified name of function
        # Only push package name if it's not empty
        if package_name:
            name_stack.push(package_name)

        #now pop the stack to form the complete name string and store the name with all function names
        name = ""
        while name_stack.is_empty() == False:
            name = name + str(name_stack.pop())
        name = name.strip(".")  # clean up
        function_names.append(name)

    return function_names



def extract_class_names(classes):

    class_names = []

    #Loops starts here to iterate on each class
    #fully qualified name by traversing to all parent classes and packages until we reach the Unit node

    for aClass in classes:
        
        name_stack = stack.Stack()
        
        package_name = "MISSING"
        package_name = aClass.xpath("ancestor::srcml:unit/srcml:package/srcml:name//text()", namespaces=xpath_queries.srcml_namespace)
        package_name = "".join(package_name)
    
        #get name of the current class and push it on stack
        current_class_name = "MISSING"
        current_class_name = aClass.xpath(xpath_queries.current_name_query, namespaces=xpath_queries.srcml_namespace)
        if len(current_class_name)==0:
            current_class_name=aClass.xpath(xpath_queries.current_full_name_query,namespaces=xpath_queries.srcml_namespace)
        current_class_name = "".join(current_class_name)
        name_stack.push(current_class_name)
        name_stack.push(".")


        #Now get parent of the current class and keep on getting parents of parent until we reach the top
        parent = aClass.getparent()
        while not(parent is None):
            #if the current parent node is a class, get the name of this node (i.e. name of the class)
            if(parent.tag=="{http://www.srcML.org/srcML/src}class" or parent.tag=="{http://www.srcML.org/srcML/src}interface"):
                
                current_class_name="MISSING"
                current_class_name=parent.xpath(xpath_queries.current_name_query,namespaces=xpath_queries.srcml_namespace)
                if len(current_class_name)==0:
                    current_class_name=aClass.xpath(xpath_queries.current_full_name_query,namespaces=xpath_queries.srcml_namespace)
                current_class_name = "".join(current_class_name)

                name_stack.push(current_class_name)
                name_stack.push(".")

            parent = parent.getparent()

        #finally pushing the package name to stack to complete the fully qualified name of function
        name_stack.push(package_name)

        #now pop the stack to form the complete name string and store the name with all function names
        name = ""
        while name_stack.is_empty() == False:
            name = name + str(name_stack.pop())
        class_names.append(name)

    return class_names