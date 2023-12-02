from collections import defaultdict
from collections import Counter
import math
import random
import os



# 0 -> business
# 1 -> entertainment
# 2 -> politics
# 3 -> sport
# 4 -> tech

def fetch_data():
    class_path = os.path.join(os.path.dirname(__file__), 'bbc_data','bbc.classes')
    mtx_path = os.path.join(os.path.dirname(__file__), 'bbc_data','bbc.mtx')

    docId_to_classId = {}
    docId_to_termIds = defaultdict(Counter)
    termIds = set()

    with open(class_path) as class_file:
        doc_class_pairs = class_file.readlines()[4:]
        
        for i in range(len(doc_class_pairs)):
            docId, classId = doc_class_pairs[i].split()
            docId = int(docId)
            classId = int(classId)
            docId_to_classId[docId] = classId
        class_file.close()
    
    
    with open(mtx_path) as matrix_file:
        rows = matrix_file.readlines()[2:]

        for row in rows:
            termId, docId, count = row.split()

            termId = int(termId)-1
            docId = int(docId)-1
            term_count = int(float(count))

            termIds.add(termId)
            term_counter = docId_to_termIds[docId]
            term_counter[termId] += term_count

        matrix_file.close()
    return (docId_to_classId, docId_to_termIds, list(termIds))


def copy_X_dict(docIds, master_X_dict):
    X_dict = {}
    for docId in docIds:
        term_counter = master_X_dict[docId]
        X_dict[docId] = term_counter
    return X_dict

def get_X_and_y(X_dict, y_dict):
    X = []
    y = []

    for docId in X_dict.keys():
        X.append(X_dict[docId])
        y.append(y_dict[docId])

    return X, y
    
def split_list(lengths, original_list):
    list_of_lists = []
    for length in lengths:
        segment = []
        for _ in range(length):
            segment.append(original_list.pop())
        list_of_lists.append(segment)
    return list_of_lists


def split_X_dict(docId_to_termIds):
    docIds = list(docId_to_termIds.keys())
    random.shuffle(docIds)

    num_heldout = int(0.1 * len(docIds))
    num_test = int(0.2 * len(docIds))
    num_train = len(docIds) - num_test - num_heldout

    heldout_docIds, test_docIds, train_docIds = split_list(
        [num_heldout, num_test, num_train],
        docIds
    )

    heldout_X_dict = copy_X_dict(heldout_docIds, docId_to_termIds) 
    test_X_dict = copy_X_dict(test_docIds, docId_to_termIds) 
    train_X_dict = copy_X_dict(train_docIds, docId_to_termIds) 

    return heldout_X_dict, test_X_dict, train_X_dict


docId_to_classId, docId_to_termIds, termIds = fetch_data()
heldout_X_dict, test_X_dict, train_X_dict = split_X_dict(docId_to_termIds)



#X is [ {termId:count} ] #bag of words by default
#Y is [ classId ]
heldout_X, heldout_y = get_X_and_y(heldout_X_dict, docId_to_classId)
test_X, test_y = get_X_and_y(test_X_dict, docId_to_classId)
train_X, train_y = get_X_and_y(train_X_dict, docId_to_classId)



def one_hot_encoding(X):
    #to

    # [
    #   {termId: 1 or 0} #present or absent
    # ]

    one_hot_X = []
    for counter in X:
        presence_dict = defaultdict(int) #0 if termId doesnt exist
        for termId in counter.keys():
            if counter[termId] > 0:
                presence_dict[termId] = 1
        one_hot_X.append(presence_dict)

    return one_hot_X


def count_term_span(X):
    term_spans = defaultdict(int)

    for counter in X:
        for termId in counter.keys():
            if counter[termId] > 0:
                term_spans[termId] += 1

    return term_spans

def tf_idf(X):
    #[
    #   {termId:weight} 
    #   #big weight if very frequent in document AND very unique to document
    #]


    term_spans = count_term_span(X)
    tf_idf_dicts = []

    for counter in X:        
        term_weights = {}
        num_docs = len(X)
        total_terms = sum(list(counter.values()))

        for termId in counter.keys():
            term_frequency = counter[termId]/total_terms
            inverse_dense_frequency = math.log(num_docs/term_spans[termId])

            weight = term_frequency * inverse_dense_frequency
            term_weights[termId] = weight

        tf_idf_dicts.append(term_weights)
    
    return tf_idf_dicts

'''
#test 
print(len(heldout_X))
print(len(test_X))
print(len(train_X))

print()
print("should vary", test_y[0])

print()

print(len(train_X))
print(len(one_hot_encoding(train_X)))
print(len(tf_idf(train_X)))

print()

print("BAG OF WORDS")
print(train_X[0])

print("\nONE HOT ENCODING")
print(one_hot_encoding(train_X)[0])

print("\nTF IDF")
print(tf_idf(train_X)[0])

'''