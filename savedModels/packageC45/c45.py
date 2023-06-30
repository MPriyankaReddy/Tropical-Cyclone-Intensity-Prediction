import numpy as np
import pandas as pd
import math
import random
import itertools

# Split and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Datasets
from sklearn.datasets import load_wine, load_iris

#Detect string/numeric
from numbers import Number
from pandas.api.types import is_string_dtype, is_numeric_dtype
class TreeNode:
    #Inicialization of the class, with some default parameters
    def __init__(self, min_samples_split=2, max_depth=None, seed=2, 
                 verbose=False):
        # Sub nodes -- recursive, those elements of the same type (TreeNode)
        self.children = {} 
        self.decision = None # Undecided
        self.split_feat_name = None # Splitting feature
        self.threshold = None #Where to split the feature
        #Minimun number of samples to do a split
        self.min_samples_split=min_samples_split
        #Maximun number of nodes/end-nodes: 0 => root_node
        self.max_depth=max_depth
        self.seed=seed #Seed for random numbers
        self.verbose=verbose #True to print the splits
    
    #This function validate the datatype and equal len of data and target
    def validate_data(self, data, target):
        #Validating data type for sample (X)
        if not isinstance(data,(list,pd.core.series.Series,np.ndarray, 
                                pd.DataFrame)):
            return False
        #Validating data type for target (y)
        if not isinstance(target,(list,pd.core.series.Series,np.ndarray)):
            return False
        #Validating same len of X and y
        if len(data) != len(target):
            return False
        return True

    #This function initialises the training of the algorithm
    def fit(self, data, target):
        #Validate data type and length
        if not self.validate_data(data,target):
            print('Error: The data is not in the correct format')
        #If all is correct, then the first node is created
        else:
            current_depth = 0 #When it starts the deep of the nodes is 0
            self.recursiveGenerateTree(data, target, current_depth)

        #This is a recursive function, which selects the decision if possible. 
        #If not, create children nodes
    def recursiveGenerateTree(self, sample_data, sample_target, current_depth):
        #If there is only one possible outcome, select that as the decision
        if len(sample_target.unique())==1:
            self.decision = sample_target.unique()[0]
        #If the sample has less than min_samples_split select the majority class
        elif len(sample_target)<self.min_samples_split:
            self.decision = self.getMajClass(sample_target)
        #If the deep of the current branch is equal to max_depth \
        #select the majority class
        elif current_depth == self.max_depth:
            self.decision = self.getMajClass(sample_target)
        #If there is not possible to make a decision, \
        #create a new node based on the best split
        else:
            #Call the function to select best_attribute to split
            best_attribute,best_threshold,splitter = self.splitAttribute(sample_data,
                                                                         sample_target)
            self.children = {} #Initializing a dictionary with the children nodes
            self.split_feat_name = best_attribute  #Name of the feature
            self.threshold = best_threshold #Threshold for continuous variable
            current_depth += 1 #Increase the deep by 1
            #Create a new node for each class of the best feature
            for v in splitter.unique():
                index = splitter == v #Select the indexes of each class
                #If there is data in the node, create a new tree node with that partition
                if len(sample_data[index])>0:
                    self.children[v] = TreeNode(min_samples_split = self.min_samples_split,
                                                max_depth=self.max_depth,
                                                seed=self.seed,
                                                verbose=self.verbose)
                    self.children[v].recursiveGenerateTree(sample_data[index],
                                                           sample_target[index],
                                                           current_depth)
                #If there is no data in the node, use the previous node data (this one) \
                #and make a decision based on the majority class
                else:
                    #Force to make a decision based on majority class \
                    #simulating that it reached max_depth
                    self.children[v] = TreeNode(min_samples_split = self.min_samples_split,
                                                max_depth=1,
                                                seed=self.seed,
                                                verbose=self.verbose)
                    self.children[v].recursiveGenerateTree(sample_data,                                          
                                                           sample_target,
                                                           current_depth=1)

    #This function define which is the best attribute to split (string \
    #or continious)
    def splitAttribute(self, sample_data,sample_target):
        info_gain_max = -1*float("inf") #Info gain set to a minimun
        #Creating a blank serie to store variable in which the split is based
        splitter = pd.Series(dtype='str') 
        best_attribute = None #Setting attribute to split to None
        best_threshold = None #Setting the threshold to None
        #Iterate for every attribute in the sample_data
        for attribute in sample_data.keys():
            #If the attribute is a string
            if is_string_dtype(sample_data[attribute]):
                #Compute information gain using that attribute to split the target
                aig = self.compute_info_gain(sample_data[attribute], sample_target)
                #If the information gain is more than the previous one, store
                if aig > info_gain_max:
                    splitter = sample_data[attribute] #Store the variable
                    info_gain_max = aig #Store the information gain
                    best_attribute = attribute #Store the name of the attribute
                    #In this case there is no threshold (string)
                    best_threshold = None 
            #If the attribute is a continuous
            else:
                #Sort the continuous variable in an asc order. Change the target order \
                # based on that
                sorted_index = sample_data[attribute].sort_values(ascending=True).index
                sorted_sample_data = sample_data[attribute][sorted_index]
                sorted_sample_target = sample_target[sorted_index]
                #Iterate between each sample, except the last one
                for j in range(0, len(sorted_sample_data) - 1):
                    #Create a blank serie to store the classification (less or greater)
                    classification = pd.Series(dtype='str')
                    #If two consecutives samples are not the same, use its mean as \
                    #a threshold
                    if sorted_sample_data.iloc[j] != sorted_sample_data.iloc[j+1]:
                        threshold = (sorted_sample_data.iloc[j] + 
                                     sorted_sample_data.iloc[j+1]) / 2
                        #Assign greater or less acording to the threshold
                        classification = sample_data[attribute] > threshold
                        classification[classification] = 'greater'
                        classification[classification == False] = 'less'
                        #Calculate the information gain using previous variable \
                        # (now categorical)
                        aig = self.compute_info_gain(classification, sample_target)
                        #If the information gain is more than the previous one, store
                        if aig >= info_gain_max:
                            splitter = classification #Store the variable
                            info_gain_max = aig #Store the information gain
                            best_attribute = attribute #Store the name of the attribute
                            best_threshold = threshold #Store the threshold
        #If verbose is true print the result of the split
        if self.verbose:
            if is_string_dtype(sample_data[best_attribute]):
                print(f"Split by {best_attribute}, IG: {info_gain_max:.2f}")
            else:
                print(f"Split by {best_attribute}, at {threshold}, IG: {info_gain_max:.2f}")
        return (best_attribute,best_threshold,splitter)

    #This function selects the majority class of the target to make a decision
    def getMajClass(self, sample_target):
        #Compute the number of records per class and order it (desc)
        freq = sample_target.value_counts().sort_values(ascending=False)
        #Select the name of the class (classes) that has the max number of records
        MajClass = freq.keys()[freq==freq.max()]
        #If there are two classes with equal number of records, select one randomly
        if len(MajClass) > 1:
            decision = MajClass[random.Random(self.seed).randint(0,len(MajClass)-1)]
        #If there is only onle select that
        else:
            decision = MajClass[0]
        return decision

    #This function calculates the entropy based on the distribution of \
    #the target split
    def compute_entropy(self, sample_target_split):
        #If there is only only one class, the entropy is 0
        if len(sample_target_split) < 2:
            return 0
        #If not calculate the entropy
        else:
            freq = np.array(sample_target_split.value_counts(normalize=True))
            return -(freq * np.log2(freq + 1e-6)).sum()
    
    #This function computes the information gain using a specific \
    #attribute to split the target
    def compute_info_gain(self, sample_attribute, sample_target):
        #Compute the proportion of each class in the attribute
        values = sample_attribute.value_counts(normalize=True)
        split_ent = 0 #Set the entropy to 0
        #Iterate for each class of the sample attribute
        for v, fr in values.iteritems():
            #Calculate the entropy for sample target corresponding to the class
            index = sample_attribute==v
            sub_ent = self.compute_entropy(sample_target[index])
            #Weighted sum of the entropies
            split_ent += fr * sub_ent
        #Compute the entropy without any split
        ent = self.compute_entropy(sample_target)
        #Return the information gain of the split
        return ent - split_ent

    #This function returns the class or prediction given an X
    def predict(self, sample):
        #If there is a decision in the node, return it
        if self.decision is not None:
            #Print when verbose is true
            if self.verbose:
                print("Decision:", self.decision)
            return self.decision #Return decision
        #If not, it means that it is an internal node
        else:
            #Select the value of the split attribute in the given data
            attr_val = sample[self.split_feat_name]
            #Print if verbose is true
            if self.verbose:
                print('attr_val')
            #If the value for the feature is not numeric just go to the\
            # corresponding child node and print
            if not isinstance(attr_val, Number):
                child = self.children[attr_val]
                if self.verbose:
                    print("Testing ", self.split_feat_name, "->", attr_val)
            #If the value is numeric see if it is greater or less than the \
            #threshold
            else:
                if attr_val > self.threshold:
                    child = self.children['greater']
                    if self.verbose:
                        print("Testing ", self.split_feat_name, "->",
                              'greater than ', self.threshold)
                else:
                    child = self.children['less']
                    if self.verbose:
                        print("Testing ", self.split_feat_name, "->",
                              'less than or equal', self.threshold)
            #Do it again with the child until finding the terminal node
            return child.predict(sample)
    
    #This function prints the structure of the three and its nodes/leaves
    def pretty_print(self, prefix=''):
        if self.split_feat_name is not None:
            for k, v in self.children.items():
                if self.threshold is not None:
                    v.pretty_print(f"{prefix}:When {self.split_feat_name} is {k} than {self.threshold}")
                else:
                    v.pretty_print(f"{prefix}:When {self.split_feat_name} is {k}")
        else:
            print(f"{prefix}:{self.decision}")

    #This functions returns the hyperparameters defined for TreeNode 
    def get_params(self):
        return {'min_samples_split':self.min_samples_split,
                'max_depth':self.max_depth,
                'seed':self.seed}

    #This function changes the hyperparameters of TreeNode given a dictionary with new parameters
    def set_params(self, params):
        if 'min_samples_split' in params:
            self.min_samples_split=params['min_samples_split']
        if 'max_depth' in params:
            self.max_depth=params['max_depth']
        if 'seed' in params:
            self.seed=params['seed']

