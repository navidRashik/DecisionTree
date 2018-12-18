
# coding: utf-8

# # Assignment Week 3
# ## Python
# 
# 1. Implement a function to compute Entropy of a set of examples with respect
# to a target attribute.
# 2. Implement a function to compute Information gain if we split a set of
# examples based on a given attribute.
# 3. Implement decision tree algorithm ID3 using the above functions.
# 4. Implement support for continuous valued attributes and missing data while
# training a decision tree.
# 5. Bonus: Implement Reduced Error Pruning.
# You will be given a dataset and you will need to report the accuracy of your
# decision tree model on that dataset.
# Bonus: You can also report other performance measures such as precision, and
# recall.
"""
x is examples in training set
y is set of attributes
labels is labeled data
Node is a class which has properties values, childs, and next
root is top node in the decision tree

Declare:
x =  Multi dimensional arrays
y =  Column names of x
labels =  Classification values, for example {0, 1, 0, 1}
          correspond that row 1 is false, row 2 is true, and so on
root = ID3(x, y, label, root)
Define:
ID3(x, y, label, node)
  initialize node as a new node instance
  if all rows in x only have single classification c, then:
    insert label c into node
    return node
  if x is empty, then:
    insert dominant label in x into node
    return node
  bestAttr is an attribute with maximum information gain in x
  insert attribute bestAttr into node
  for vi in values of bestAttr:
    // For example, Outlook has three values: Sunny, Overcast, and Rain
    insert value vi as branch of node
    create viRows with rows that only contains value vi
    if viRows is empty, then:
      this node branch ended by a leaf with value is dominant label in x
    else:
      newY = list of attributes y with bestAttr removed
      nextNode = next node connected by this branch
      nextNode = ID3(viRows, newY, label, nextNode)
  return node
"""
# In[288]:


import pandas as pd
import numpy as np
from math import log
from sklearn.model_selection import train_test_split
import pptree

# In[289]:


def fileRead():
    reader = pd.read_csv('car.data', sep = ',' )
#    reader = reader.sample(100)
    
#    reader = pd.read_csv('txt.data', sep = ' ' )
#    reader = reader.drop(columns='Day')
    
    data, test_data = train_test_split(reader, test_size=0.2, shuffle=True)
    return data, test_data



# In[291]:


def solutionCol(reader):
    return reader[reader.columns[-1]]


# In[292]:


def entropy_s(reader, full_col):

    value_s_series = full_col #last coloumn
    total_element = len(reader)
    frequency_dict  = value_s_series.value_counts() #returns dict with unique elements and its frequency of occerance
    # print(frequency_dict)
    entropy_total = 0
    for value in frequency_dict:
        p_i = value/total_element
        entropy_total += -p_i*log(p_i,2)
    return entropy_total


# In[293]:


def entropy_col(reader, col_name_str):
    dict_col = reader.groupby([col_name_str, solutionCol(reader)]).groups
    total_element = len(reader)
    
    unique_elements = reader[col_name_str].unique()
    unique_elements_dic = dict.fromkeys(unique_elements, 0)
    
    frequency_dict = reader[col_name_str].value_counts()
#     freq_sol = solutionCol(reader).nunique()
    for item in dict_col:
#         flag = freq_sol
        count_for_solution = len(dict_col[item])
#         unique_elements_dic[item[0]]
        p_i = count_for_solution/frequency_dict[item[0]]
        unique_elements_dic[item[0]] += -p_i*log(p_i,2)
    
      
    
    total_entropy= 0 
    for value in unique_elements_dic.keys():
        ratio = frequency_dict[value]/total_element
        total_entropy += ratio*unique_elements_dic[value]
    
    return total_entropy


# In[294]:


def gain(ent_s , ent_col, column , reader):
    gain = ent_s-ent_col
    
#     full_col = reader[column]

    split_info = entropy_col(reader,column)
#    print('split info',split_info)
    gain_ratio = gain*split_info
#     gain = gain_ratio*gain
#    print("----------------")1
#    print('gain' , gain , 'gain ratio  ' , gain_ratio)
    return gain


# In[]:
def decision_node(node):

#    if node is none:
#        
    
    s_col = solutionCol(node.data)
    ent_s = entropy_s(node.data, s_col)
    if ent_s == 0:
        new_node = Node(max(s_col), None)
        new_node.expected_result = Node(max(s_col), None)
        node.add_child(new_node)
        return node
    total_col = node.data.columns[:-1]
    gains = {}
    for column in total_col:
        ent_col = entropy_col(node.data,column)
        gains[column] = gain(ent_s,ent_col,column,node.data)
    
                    
    max_gain = max(gains.keys(), key = (lambda k : gains[k]))
    if gains[max_gain] < node.gain:
        new_node = Node(max(s_col), None)
        new_node.expected_result = Node(max(s_col), None)
        node.add_child(new_node)
        return node
        

    else:

        new_node = Node(max_gain, node.data)
        new_node.expected_result = Node(max(s_col), None)
        new_node.gain = gains[max_gain]
        node.add_child(new_node)
        children_names = list(node.data[max_gain].unique())
     
    
    new_node_parent = node.children[0]
    for index , name in enumerate(children_names):
        sub_data = new_node_parent.data.loc[new_node_parent.data[new_node_parent.name] == name]
        sub_data = sub_data.drop(columns = [new_node_parent.name])
        new_node = Node(name,sub_data )
        new_node.expected_result = Node(max(s_col), None)
        new_node = decision_node(new_node)
        new_node_parent.add_child(new_node)
    return node


# In[]:

    
# In[]:
class Node(object):
    def __init__(self,name, data):
        self.name = name
        self.data = data
        self.children = []
        self.gain = 0
        self.expected_result = None
    def name(self):
        return self.name

    def add_child(self, Node):
        self.children.append(Node)
# In[]:
        
        
        
def printTree(tree):
    pptree.print_tree(tree) 
    
# In[]:


    
def returnMatchedSubtree(root , name):
    new_root = root
    found = False
    if not bool(new_root.children):
        found = True
        return new_root, found
    
    for child in new_root.children:
        if child.name == name :
            if not bool(child.children[0].children):
                found = True
            return child.children[0], found
    else:
        # if the list does not have the child which was asked then it will provide a node(result ) which is most likely to occer 
        found = True
        return child.expected_result, found
#    return new_root, found
# In[]:
def test_validation(test_data, tree_root):
    match = 0
    unmatch = 0
    for index, row in test_data.iterrows():
#    row[0] means index no row[1] is actual row row[1]['col_1'] means for that perticular row the value of col_1
        sub_tree = tree_root
        found = False
        while not found:
            value = row[sub_tree.name]
            sub_tree, found = returnMatchedSubtree(sub_tree, value)
            if found:
                if sub_tree.name == row[-1]:
                    match += 1
                else:
                    unmatch += 1
                break
    return match , unmatch

def accuracy(match, unmatch , total):
    acc = float(match/total)*100
    print ('accuracy is : '+ str(acc) + "%")
# In[]:
    
data, test_data = fileRead()
root = Node('root',data)
root = decision_node(root)
root = root.children[0] #removing 1st child
printTree(root)
match , unmatch = test_validation(test_data , root)
accuracy(match,  unmatch , len(test_data))
# In[297]:



