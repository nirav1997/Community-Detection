#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 9 15:32:14 2020

@author: nirav
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from igraph import Graph, clustering
import sys

# Function for mapping and inverse mapping of clusters
def Mapp_Invmapp(clusters):
    i=0
    mapp = dict()
    for v in clusters:
        if len(v) == 0:
            continue
        mapp[i] = v
        i += 1
    invmapp = dict()
    for k, vs in mapp.items():
        for v in vs:
            invmapp[v] = k
    return mapp, invmapp

#Calculate Delta Q Newman using the original membership and new membership by changing the membership of current vertex
def Delta_Q_Newman(cluster_membership_old, cluster_membership_new):
    old_modularity = graph.modularity(cluster_membership_old)    
    new_modularity = graph.modularity(cluster_membership_new)
    return new_modularity - old_modularity

#Calculate Delta Q Attr
def Delta_Q_Attr(clusters, c_id, v):
    #Similarity List for all similarities for current vertex in the cluster
    similarities = []
    #print(clusters)
    #print(c_id)
    for i in clusters[c_id]:
        similarities.append(similarity_mat[v][i])
        
    #Return mean of the similarities
    return np.mean(similarities)

#Phase 1 Begins
def Phase1(alpha):
    #Generate Initial Clusters
    clusters = clustering.VertexClustering(graph, [v for v in range(graph.vcount())])
    #Converge Initialized
    converge = False
    
    #Iterations Initialized
    iterations = 0
    while not(converge) and iterations < 15:
        #Converge Reinitialized to true
        converge = True
        
        for i in graph.vs.indices:
            max_gain = 0
            max_vertex = -1
            
            clusters_membership = clusters.membership
            
            #Iterate through all vertices except the current 1 and the others in the same cluster
            for j in graph.vs.indices:
                if clusters_membership[i] == clusters_membership[j]:
                    continue
                
                if i == j:
                    continue
                
                cluster_membership_old = clusters_membership.copy()
                cluster_membership_new = clusters_membership.copy()
                cluster_membership_new[i] = cluster_membership_new[j]
                
                delta_Q_Newman = Delta_Q_Newman(cluster_membership_old, cluster_membership_new)
                delta_Q_Attr = Delta_Q_Attr(clusters, clusters_membership[j], i)
                delta_Q = alpha * delta_Q_Newman + (1 - alpha) * delta_Q_Attr
                
                #If changing the cluster for current vertex to jth vertex increases delta_Q update values of max_gain and max_vertex
                #Also as value is changed we will not converge so update it to False
                if delta_Q > 0 and delta_Q > max_gain:
                    converge = False
                    max_gain = delta_Q
                    max_vertex = j
            
            #Update the clusters from max_gain and max_vertex values
            if not(max_vertex == -1):
                clusters_membership[i] = clusters_membership[max_vertex]
                clusters = clustering.VertexClustering(graph, clusters_membership) 
           
        iterations+=1
        
    return clusters

#Phase 2 Begins
def Phase2(clusters,attributes,alpha):
    #Get the global graph and similarity matrix
    global graph,similarity_mat
    
    #Get inverse mapping of the clusters
    mapp, inv_mapp = Mapp_Invmapp(clusters)
    
    #Contract the vertices
    graph.contract_vertices([inv_mapp[mem] for mem in clusters.membership], combine_attrs="first")
    
    #Simplify the Graph
    graph = graph.simplify()
    
    #Get new attributes by summing up all the attributes in the cluster or the new updated vertex
    attributes_new = np.array([list(attributes[vertices].sum(0)) for vertices in mapp.values()])
    
    #Generate updated Similarity Matrix using contracted Graph
    similarity_mat = cosine_similarity(attributes_new, attributes_new)
    
    #Run Phase1 on contracted Graph
    clusters_p2 = Phase1(alpha)
    
    return clusters_p2
    
    
if __name__ == '__main__':
    #Input Path initialized
    attribute_path = 'data/fb_caltech_small_attrlist.csv'
    edges_path = 'data/fb_caltech_small_edgelist.txt'
    alpha = float(sys.argv[1])
    #alpha = 1
    
    #Read Edges of the graph
    with open(edges_path, 'r') as f:
        edges = [tuple([int(v) for v in line.strip().split(' ')]) for line in f.readlines()]
    
    #Read the attributes for the graph
    attributes = pd.read_csv(attribute_path)
    attributes = np.array(attributes)
    
    #Initialize Graph as Global
    global graph
    graph = Graph()
    
    #Add vertices to the graph
    graph.add_vertices(len(attributes))
    #Add edges to the graph
    graph.add_edges(edges)
    
    
    #Intialize Similarity Matrix to global
    global similarity_mat
    
    #Update Similarity Matrix based ob cosine similarity of the attriubutes
    similarity_mat = cosine_similarity(attributes, attributes)
    
    #Run Phase 1 and Get the clusters
    clusters_p1 = Phase1(alpha)
    
    #print(len(set(clusters_p1.membership)))
    print("Phase 1 generated "+str(len(set(clusters_p1.membership)))+" Clusters")
    
    #Run Phase 2 And get the clusters
    clusters_p2 = Phase2(clusters_p1,attributes,alpha)
    
    #print(len(set(clusters_p2.membership)))
    print("Phase 2 generated "+str(len(set(clusters_p2.membership)))+" Clusters")
    
    #Get inverse mappings of both the phases and concat them and output them to a file named clusters+'alpha'+.txt
    mapp_phase1, _ = Mapp_Invmapp(clusters_p1)
    mapp_phase2, _ = Mapp_Invmapp(clusters_p2)
    clusters = []
    for vs in mapp_phase2.values():
        cluster = []
        for v in vs:
            cluster.extend(mapp_phase1[v])
        clusters.append(cluster)
    with open('communities_'+str(alpha)+'.txt', 'w') as fp:
        for cluster in clusters:
            fp.write(','.join([str(x) for x in cluster]) + '\n')
