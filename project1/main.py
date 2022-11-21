#!/usr/bin/env python3
#########################
##    IMPORT MODULES   ##
#########################
import pprint
import argparse
import csv
import time
from itertools import combinations, groupby
import random
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from colorama import Back, Fore, Style

#########################
##      VARIABLES      ##
#########################
global counter_solutions_bb_algorithm
counter_solutions_bb_algorithm = 0


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


def generateRandomGraph(number_nodes, probability_edges):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi
    graph, but enforcing that the resulting graph is connected
    https://stackoverflow.com/questions/61958360/how-to-create-random-graph-where-each-node-has-at-least-1-edge-using-networkx
    """
    edges = combinations(range(number_nodes), 2)
    G = nx.Graph()
    G.add_nodes_from(range(number_nodes))
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


def allCliquesGraphExhaustiveSearch(G):
    """
    find and return the list of all cliques of a graph.
    :return: list of all cliques of G
    :param G: networkx graph
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


def maximumCliquesGraphExhaustiveSearch(G):
    """
    find a return the list of all maximum cliques of a graph.
    :return: list of all maximum cliques of G
    :param G: networkx graph
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
    counter_solutions = 0
    counter_basic_operations = 0
    for nodes_subgraph in all_nodes_combinations:
        counter_solutions += 1
        SG = G.subgraph(nodes_subgraph)
        complete_graph, counter_basic_operations = isCompleteGraph(SG, counter_basic_operations)
        if complete_graph:
            if len(list(SG.nodes)) >= maximum_number_nodes:
                # print('the subgraph ' + str(list(SG.nodes)) + ' is a maximum clique of G.')
                maximum_number_nodes = len(list(SG.nodes))
                all_maximum_cliques.append(SG)
    # print('the number of maximum cliques of G is: ' + str(len(all_maximum_cliques)))

    return all_maximum_cliques, counter_solutions, counter_basic_operations


def isCompleteGraph(G, counter_basic_operations):
    """ Check if a graph is complete.
        If any edge is missing, the graph is not complete.
        If the graphs contains all edges, the graph is complete
        :param counter_basic_operations:
        :param G: networkx graph
        :return: True or False, depending if the graph is complete or not
    """
    for (u, v) in combinations(list(G.nodes), 2):  # check each possible pair
        counter_basic_operations += 1
        if not G.has_edge(u, v):
            return False, counter_basic_operations  # if any edge is missing, the graph is not complete
    return True, counter_basic_operations  # if there are all possible edges, the graph is complete


def findSingleMaximalClique(G):
    """
    Greedy heuristic do find a single maximal clique.
    :param G: networkx graph
    :return: the networkx subgraph maximal clique
    """
    maximal_clique = []  # Initialize the maximal clique list
    nodes = list(G.nodes)
    random_node = random.randrange(0, len(nodes), 1)  # Get a random node
    maximal_clique.append(nodes[random_node])  # put this first node in the list
    # iterate through each node of the graph
    counter_solutions = 0
    counter_basic_operations = 0
    for node in nodes:
        counter_solutions += 1
        if node in maximal_clique:
            continue
        next_node = True
        for node_maximal_clique in maximal_clique:
            if G.has_edge(node, node_maximal_clique):
                counter_basic_operations += 1
                continue
            else:
                next_node = False
                break
        if next_node is True:
            maximal_clique.append(node)

    maximal_clique_subgraph = G.subgraph(maximal_clique)

    return maximal_clique_subgraph, counter_solutions, counter_basic_operations


def branchBoundAlgorithmMaximumClique(G):
    """
    Branch and Bound algorithm for solving Maximum clique problem using greedy coloring heuristic to estimate upper
    bound and greedy clique heuristic for lower bound on each step. https://github.com/donfaq/max_clique
    :param G: networkx graph
    :return: the maximum clique
    """
    global counter_solutions_bb_algorithm
    maximum_clique = greedyCliqueHeuristic(G)
    chromatic_number = greedyColoringHeuristic(G)
    counter_solutions_bb_algorithm += 1
    if len(maximum_clique) == chromatic_number:
        maximum_clique_subgraph = G.subgraph(maximum_clique)
        return maximum_clique_subgraph
    else:
        g1, g2 = branching(G)
        return max(branchBoundAlgorithmMaximumClique(g1), branchBoundAlgorithmMaximumClique(g2), key=lambda x: len(x))


def greedyCliqueHeuristic(G):
    """
    Greedy search for clique iterating by nodes
    with the highest degree and filter only neighbors
    :param G:
    :return:
    """
    K = set()
    nodes = [node[0] for node in sorted(nx.degree(G), key=lambda x: x[1], reverse=True)]
    while len(nodes) != 0:
        neigh = list(G.neighbors(nodes[0]))
        K.add(nodes[0])
        nodes.remove(nodes[0])
        nodes = list(filter(lambda x: x in neigh, nodes))

    return K


def greedyColoringHeuristic(G):
    """
    Greedy graph coloring heuristic with degree order rule
    :param G:
    :return:
    """
    color_num = iter(range(0, len(G)))
    color_map = {}
    # used_colors = set()
    nodes = [node[0] for node in sorted(nx.degree(G), key=lambda x: x[1], reverse=True)]
    color_map[nodes.pop(0)] = next(color_num)  # color node with color code
    used_colors = {i for i in color_map.values()}
    while len(nodes) != 0:
        node = nodes.pop(0)
        neighbors_colors = {color_map[neighbor] for neighbor in list(filter(lambda x: x in color_map,
                                                                            G.neighbors(node)))}
        if len(neighbors_colors) == len(used_colors):
            color = next(color_num)
            used_colors.add(color)
            color_map[node] = color
        else:
            color_map[node] = next(iter(used_colors - neighbors_colors))

    return len(used_colors)


def branching(G):
    """
    Branching procedure
    :param G:
    :return:
    """
    g1, g2 = G.copy(), G.copy()
    max_node_degree = len(G) - 1
    nodes_by_degree = [node for node in sorted(nx.degree(G),  # All graph nodes sorted by degree (node, degree)
                                               key=lambda x: x[1], reverse=True)]
    # Nodes with (current clique size < degree < max possible degree)
    partial_connected_nodes = list(filter(lambda x: x[1] != max_node_degree and x[1] <= max_node_degree,
                                          nodes_by_degree))
    # graph without partial connected node with the highest degree
    g1.remove_node(partial_connected_nodes[0][0])
    # graph without nodes which is not connected with partial connected node with the highest degree
    g2.remove_nodes_from(G.nodes() - G.neighbors(partial_connected_nodes[0][0]) - {partial_connected_nodes[0][0]})
    return g1, g2


def drawGraph(G, SG, title):
    """
    Draws the undirected graphs, highlighting the clique
    @param G: the graph created with networkx library. Datatype: networkx object
    @param title: the title of the graph. Datatype: string
    @return: a matplotlib figure. Datatype: matplotlib object
    """
    plt.figure()
    plt.title(title)
    plt.axis('off')
    pos = nx.spring_layout(G, seed=91352)

    color_map_nodes = []
    for node in G:
        if node in SG.nodes:
            color_map_nodes.append('red')
        else:
            color_map_nodes.append('#1f78b4')

    color_map_edges = []
    for edge in G.edges:
        if edge in SG.edges:
            color_map_edges.append('red')
        else:
            color_map_edges.append('black')

    nx.draw(G, pos, node_color=color_map_nodes, edge_color=color_map_edges, with_labels=True)

    return plt


#########################
##     MAIN SCRIPT     ##
#########################
def main():
    ###################################################################
    # Initialization of the argparse arguments
    ###################################################################
    ap = argparse.ArgumentParser()
    ap.add_argument('-nmn', '--number_maximum_nodes', required=True, type=int, help="Define maximum number of nodes.")
    ap.add_argument('-pe', '--probabilities_edges', type=list, default=[0.125, 0.25, 0.5, 0.75],
                    help="Define probability to connect two nodes")
    ap.add_argument('-s', '--seed', type=int, default=91352,
                    help="Define probability to connect two nodes")

    args = vars(ap.parse_args())

    print('The inputted arguments are: ' + str(args))
    results_folder_path = './results/'
    figure_results_folder_path = results_folder_path + 'figures/'
    random.seed(args['seed'])

    header = ['Number of nodes', 'Number of edges', 'Edge probability', 'Algorithm', 'Results',
              'Number of basic operations', 'Execution time', 'Number of solutions tested']
    file = open(results_folder_path + 'results.csv', 'w')
    writer = csv.writer(file)
    writer.writerow(header)

    list_nodes = list(range(4, args['number_maximum_nodes'] + 1))
    for number_nodes in list_nodes:
        for edge_probability in args['probabilities_edges']:
            ###################################################################
            # Create a random graph with the student number as seed
            ###################################################################
            print(Back.RED + 'Creating a random undirected graph with ' + str(number_nodes) +
                  ' nodes, and edge probability of ' + str(edge_probability) + Back.RESET)
            G = generateRandomGraph(number_nodes=number_nodes,
                                    probability_edges=edge_probability)
            standard_filename = 'fig_n' + str(number_nodes) + '_p' + str(edge_probability) + '_'

            ###################################################################
            # # find all cliques of the graph
            ###################################################################
            # print('Finding all cliques from graph G...')
            # timer_all_cliques = tic()  # Start the timer
            # all_cliques, counter_all_cliques = allCliquesGraphExhaustiveSearch(G)
            # execution_time_all_cliques = toc(timer_all_cliques)  # Stop the timer
            # print(execution_time_all_cliques)
            # print('All the clicks of G are: \n')
            # for clique_subgraph in all_cliques:
            #     print(str(list(clique_subgraph.nodes)))
            # print('The number of cliques of G is: ' + str(len(all_cliques)))

            ###################################################################
            # find the maximum clique of the graph
            ###################################################################
            timer_all_maximum_cliques = tic()  # Start the timer
            all_maximum_cliques, counter_solutions_all_maximum_cliques, counter_basic_operation_all_maximum_cliques = maximumCliquesGraphExhaustiveSearch(
                G)
            execution_time_all_maximum_cliques = toc(timer_all_maximum_cliques)  # Stop the timer
            # Save results in the csv file
            maximum_cliques_list = [sorted(list(maximum_clique_subgraph.nodes)) for maximum_clique_subgraph in
                                    all_maximum_cliques]
            row = [number_nodes, G.number_of_edges(), edge_probability, 'Exhaustive Search', str(maximum_cliques_list),
                   counter_basic_operation_all_maximum_cliques, execution_time_all_maximum_cliques,
                   counter_solutions_all_maximum_cliques]
            writer.writerow(row)
            # Draw graph and save in disk
            for index, maximum_clique_subgraph in enumerate(all_maximum_cliques):
                title = str(number_nodes) + ' nodes - ' + str(edge_probability) + ' edge probability - ' + \
                        'Exhaustive Search Solution ' + str(index + 1)
                graph_plt = drawGraph(G=G, SG=maximum_clique_subgraph, title=title)
                filename = figure_results_folder_path + standard_filename + 'exhaustive_' + str(index + 1) + '.png'
                graph_plt.savefig(filename)
                graph_plt.close()

            ###################################################################
            # find a single maximal clique from graph G (not always an maximum clique), using a simple greedy heuristic
            ###################################################################
            timer_single_maximal_clique = tic()  # Start the timer
            single_maximal_clique, counter_solutions_single_maximal_clique, counter_basic_operations_single_maximal_clique = findSingleMaximalClique(
                G)
            execution_time_single_maximal_clique = toc(timer_single_maximal_clique)  # Stop the timer
            # Save results in the csv file
            row = [number_nodes, G.number_of_edges(), edge_probability, 'Maximal Clique - Simple Greedy',
                   str(sorted(list(single_maximal_clique.nodes))), counter_basic_operations_single_maximal_clique,
                   execution_time_single_maximal_clique, counter_solutions_single_maximal_clique]
            writer.writerow(row)
            # Draw graph and save in disk
            title = str(number_nodes) + ' nodes - ' + str(edge_probability) + ' edge probability - ' + 'Simple Greedy Heuristic'
            graph_plt = drawGraph(G=G, SG=single_maximal_clique, title=title)
            filename = figure_results_folder_path + standard_filename + 'simple-greedy.png'
            graph_plt.savefig(filename)
            graph_plt.close()

            ###################################################################
            # find the maximum clique from graph G, using the branch bound greedy heuristic
            ###################################################################
            global counter_solutions_bb_algorithm
            timer_maximum_clique = tic()  # Start the timer
            maximum_clique_bb_subgraph = branchBoundAlgorithmMaximumClique(G)
            execution_time_maximum_clique = toc(timer_maximum_clique)  # Stop the timer
            # Save results in the csv file
            row = [number_nodes, G.number_of_edges(), edge_probability, 'Maximum Clique - Branch Bound Greedy',
                   str(sorted(list(maximum_clique_bb_subgraph.nodes))), 'Not used', execution_time_maximum_clique,
                   counter_solutions_bb_algorithm]
            writer.writerow(row)
            # Draw graph and save in disk
            title = str(number_nodes) + ' nodes - ' + str(edge_probability) + ' edge probability - ' + 'Branch Bound Greedy Heuristic'
            graph_plt = drawGraph(G=G, SG=maximum_clique_bb_subgraph, title=title)
            filename = figure_results_folder_path + standard_filename + 'branch-bound-greedy.png'
            graph_plt.savefig(filename)
            graph_plt.close()

    file.close()
    print('Results csv and graph figures saved successfully!')


if __name__ == "__main__":
    main()
