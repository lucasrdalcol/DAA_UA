#!/usr/bin/env python3

#########################
##    IMPORT MODULES   ##
#########################
import pprint
import argparse
import time
from itertools import combinations, groupby
import random
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


#########################
##      VARIABLES      ##
#########################


#########################
##      FUNCTIONS      ##
#########################
def tic():
    """
    Functions used to return the numbers of seconds passed since epoch to use with function toc() afterwards.
    Like tic toc matlab functions.
    :return start_time: a float.
    """

    # Get the number os seconds passed since epoch
    start_time = time.time()

    return start_time


def toc(start_time):
    """
    Function used to return the elapsed time since function tic() was used. tic() and toc() works together.
    :param start_time: number of seconds passed since epoch given by tic() function. Datatype: float
    :return elapsed_time: a float.
    """

    # Get the number of seconds passed since epoch and subtract from tic(). This is the elapsed time from tic to toc.
    end_time = time.time()
    elapsed_time = end_time - start_time

    return elapsed_time


def generateRandomGraph(number_nodes, probability_edges, seed):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx
    """
    edges = combinations(range(number_nodes), 2)
    G = nx.Graph()
    G.add_nodes_from(range(number_nodes))
    random.seed(seed)
    if probability_edges <= 0:
        return G
    if probability_edges >= 1:
        return nx.complete_graph(number_nodes, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < probability_edges:
                G.add_edge(*e)
    return G


def allCliquesGraph(G):
    """
    find and return the list of all cliques of a graph.
    :return: list of all cliques of G
    :param G: the networkx graph
    """
    # Create all the combinations of nodes possible
    nodes = list(G.nodes)
    all_nodes_combinations = list()
    for n in range(2, len(nodes) + 1):
        all_nodes_combinations += list(combinations(nodes, n))
    all_nodes_combinations.reverse()

    # Create subgraphs and verify if they are a clique
    all_cliques = list()
    counter = 0
    for nodes_subgraph in all_nodes_combinations:
        counter += 1
        SG = G.subgraph(nodes_subgraph)
        # nx.draw(SG, with_labels=True)
        # plt.show()
        if isCompleteGraph(SG):
            # print('the subgraph ' + str(list(SG.nodes)) + ' is a clique of G.')
            all_cliques.append(SG)
    # print('the number of cliques of G is: ' + str(len(all_connected_subgraphs)))

    return all_cliques, counter


def maximumCliquesGraph(G):
    """
    find a return the list of all maximum cliques of a graph.
    :return: list of all maximum cliques of G
    :param G: the graph
    """
    # Create all the combinations of nodes possible
    nodes = list(G.nodes)
    all_nodes_combinations = list()
    for n in range(2, len(nodes) + 1):
        all_nodes_combinations += list(combinations(nodes, n))
    all_nodes_combinations.reverse()  # reverse the list to iterate from up to down

    # Create subgraphs and verify if they are a clique
    all_maximum_cliques = list()
    maximum_number_nodes = 0
    counter = 0
    for nodes_subgraph in all_nodes_combinations:
        counter += 1
        SG = G.subgraph(nodes_subgraph)
        # nx.draw(SG, with_labels=True)
        # plt.show()
        if isCompleteGraph(SG):
            if len(list(SG.nodes)) >= maximum_number_nodes:
                # print('the subgraph ' + str(list(SG.nodes)) + ' is a maximum clique of G.')
                maximum_number_nodes = len(list(SG.nodes))
                all_maximum_cliques.append(SG)
    # print('the number of maximum cliques of G is: ' + str(len(all_maximum_cliques)))

    return all_maximum_cliques, counter


def isCompleteGraph(G):
    """ Check if a graph is complete.
        If any edge is missing, the graph is not complete.
        If the graphs contains all edges, the graph is complete
    """
    for (u, v) in combinations(list(G.nodes), 2):  # check each possible pair
        if not G.has_edge(u, v):
            return False  # if any edge is missing, the graph is not complete
    return True  # if there are all possible edges, the graph is complete


#########################
##     MAIN SCRIPT     ##
#########################
def main():
    # ---------------------------------------------------
    # Initialization of the argparse arguments
    # ---------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-nn', '--number_nodes', required=True, type=int, help="Define number of nodes of the graph")
    ap.add_argument('-pe', '--probability_edges', required=True, type=float,
                    help="Define probability to connect two nodes")
    ap.add_argument('-s', '--seed', type=int, default=None,
                    help="Define probability to connect two nodes")

    args = vars(ap.parse_args())

    print('The inputted arguments are: ' + str(args))

    # Create a random graph with the student number as seed
    print('Creating a random undirected graph with ' + str(args['number_nodes']) + ' nodes, and edge probability of ' +
          str(args['probability_edges']) + ' .')
    G = generateRandomGraph(number_nodes=args['number_nodes'],
                            probability_edges=args['probability_edges'],
                            seed=args['seed'])
    nx.draw(G, with_labels=True)
    plt.show()

    # find all cliques of the graph
    print('Finding all cliques from graph G...')
    timer_all_cliques = tic()  # Start the timer
    all_cliques, counter_all_cliques = allCliquesGraph(G)
    execution_time_all_cliques = toc(timer_all_cliques)  # Stop the timer
    print(execution_time_all_cliques)
    print('All the clicks of G are: \n')
    for clique_subgraph in all_cliques:
        print(str(list(clique_subgraph.nodes)))
    print('The number of cliques of G is: ' + str(len(all_cliques)))

    # find the maximum clique of the graph
    print('Finding all maximum cliques from graph G...')
    timer_all_maximum_cliques = tic()  # Start the timer
    all_maximum_cliques, counter_all_maximum_cliques = maximumCliquesGraph(G)
    execution_time_all_maximum_cliques = toc(timer_all_maximum_cliques)  # Stop the timer
    print(execution_time_all_maximum_cliques)
    print('All the maximum clicks of G are: \n')
    for maximum_clique_subgraph in all_maximum_cliques:
        print(str(list(maximum_clique_subgraph.nodes)))
    print('The number of maximum cliques of G is: ' + str(len(all_maximum_cliques)))


if __name__ == "__main__":
    main()
