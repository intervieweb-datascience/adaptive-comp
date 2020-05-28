####
# imports

import math
import numpy as np
import random
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import pandas as pd
import copy
import time



# parameters - vote collection
N_items = 990
MM = 20
alpha = 0.5
n_b = 7
items = list(range(N_items))


# parameters - stochastic model
n_simulations = 50
n_voters = 100
sigma_min = 0.01
sigma_max = 0.1
epsilon_min = 0.005
epsilon_max = 0.05


# parameters - ranking correlation
n_0 = 2


# random seed
np.random.seed(0)
random.seed(0)


####
# underlying similarity distributions
underlying_sim_exp = [ 2 * math.exp(- i / N_items) - 1 for i in range(N_items) ]
underlying_sim_powerlaw = [ 2 / (1 + i / N_items) - 1 for i in range(N_items)]
underlying_sim_samearea = list(pd.read_csv("underlying_sim_samearea.csv", sep=",")["similarity"])


#####
# create random voters

def create_voters(zz):
    sigma_list = list(np.random.uniform(sigma_min, sigma_max, n_voters))
    epsilon_list =  list(np.random.uniform(epsilon_min, epsilon_max, n_voters))
    n_scores = len(zz)

    voters = []
    for voter in range(n_voters):
        eta = list(np.random.normal(0, 1, n_scores))
        oo = [abs(max(-1, min(1, zz[i] + sigma_list[voter] * zz[i] * (1- zz[i]) * eta[i])))  for i in range(n_scores)]
        voters.append((epsilon_list[voter], oo))

    return voters

def choose_winner(pair_of_items, voter):
    epsilon = voter[0]
    oo = voter[1]
    z1 = oo[pair_of_items[0]]
    z2 = oo[pair_of_items[1]]

    if z1 < z2:
        winner = pair_of_items[1]
        if random.uniform(0, 1) < epsilon:
            winner = pair_of_items[0]
    else:
        winner = pair_of_items[0]
        if random.uniform(0, 1) < epsilon:
            winner = pair_of_items[1]
    return winner


####
#  definition of rank correlation coefficients

def weighted_spearman(aa, bb):
    nn = len(aa)
    aa_w = [1 / (x + n_0) ** 2  for x in aa]
    bb_w = [1 / (x + n_0) ** 2  for x in bb]
    w = [aa_w[i] + bb_w[i]  for i in range(nn)]
    wsum = sum(w)
    aam = sum([w[i] * aa[i] for i in range(nn)]) / wsum
    bbm = sum([w[i] * bb[i] for i in range(nn)]) / wsum
    ssa = math.sqrt(sum([w[i] * (aa[i] ** 2 - aam ** 2) for i in range(nn)]) / wsum)
    ssb = math.sqrt(sum([w[i] * (bb[i] ** 2 - bbm ** 2) for i in range(nn)]) / wsum)
    coeff = sum([w[i] * (aa[i] - aam) * (bb[i] - bbm) for i in range(nn)]) / (wsum * ssa * ssb)
    return coeff

def weighted_kendall(aa, bb):
    nn = len(aa)
    aa_w = [1 / (x + n_0) ** 2  for x in aa]
    bb_w = [1 / (x + n_0) ** 2  for x in bb]
    w = [aa_w[i] + bb_w[i]  for i in range(nn)]

    coeff = 0
    for i in range(0, nn-1):
        for j in range(i + 1, nn):
            coeff += w[i] * w[j] * np.sign(aa[j] - aa[i]) * np.sign(bb[j] - bb[i])
    coeff = 2 * coeff / (sum(w) ** 2 - sum([x ** 2 for x in w]))
    return coeff


#####
# create random pairs of items

def create_pairs_of_items(items_now, MM_now, voters_unique):
    # create table_items -- 1 row for each item -- priority = number of times the item has still to be choosen
    table_items = pd.DataFrame()
    table_items["item"] = items_now
    nrows = table_items.shape[0]
    table_items["priority"] = [MM_now] * nrows
    table_items["random"] = pd.Series([int(round(x)) for x in np.random.uniform(0, 100000, nrows)])
    table_items = table_items.sort_values(by=["priority", "random"], ascending=False, ignore_index=True)


    # number of pairs of items to be created (each of them contains 2 items --> divide by 2)
    items_to_choose = round(nrows * MM_now / 2)

    # create list of voters
    nn = items_to_choose // n_voters
    voters = []
    for i in range(n_voters):
        voters = voters + [voters_unique[i]] * nn
    nn2 = items_to_choose - (nn * n_voters)
    if nn2 > 0:
        pos = list(np.random.choice(range(n_voters), nn2))
        voters = voters + [voters_unique[i] for i in pos]
    np.random.shuffle(voters)

    # create of pairs of items (from table_items) and associate a voter to each one
    pairs_of_items = []
    for count in range(0, (items_to_choose + 1)):
        if items_to_choose < 1:
            break
        pairs_of_items.append([list(table_items["item"])[0:2], voters[count]])
        table_items["priority"][0] -= 1
        table_items["priority"][1] -= 1
        table_items["random"][0] = int(round(np.random.uniform(0, 100000, 1)[0]))
        table_items["random"][1] = int(round(np.random.uniform(0, 100000, 1)[0]))

        table_items = table_items.sort_values(by=["priority", "random"], ascending=False, ignore_index=True)
        items_to_choose -= 1

    return pairs_of_items



def voting(distribution_type):

    if distribution_type == "exponential":
        underlying_scores = underlying_sim_exp
    elif distribution_type == "power_law":
        underlying_scores = underlying_sim_powerlaw
    elif distribution_type == "same_area":
        underlying_scores = underlying_sim_samearea
    else:
        print("Unknown distribution type, using exponential")
        distribution_type = "exponential"
        underlying_scores = underlying_sim_exp


    dic_coeff_tot = {}
    runtimes_uniform = []
    runtimes_adaptive = []

    theoretical_ranks = sorted([[abs(underlying_scores[i]), i] for i  in range(N_items)], reverse=True)
    theoretical_ranks = sorted([[theoretical_ranks[i][1], i] for i  in range(N_items)])
    theoretical_ranks = [x[1] for x in theoretical_ranks]


    for votation in range(n_simulations):
        print("\rUniform simulation {}/{}                  ".format(votation + 1, n_simulations), end="")

        #####
        # create voters
        voters_unique = create_voters(underlying_scores)

        #####
        # case 1 - UNIFORM APPROACH - all votes together

        t0 = time.time()
        MM1 = round(MM * (1 - alpha ** n_b) / (1 - alpha))
        pairs_of_items = create_pairs_of_items(items, MM1, voters_unique)
        pairs_of_items_voted = [[x, choose_winner(x, voter)] for (x, voter) in pairs_of_items]


        dic_wins = {}
        for x in pairs_of_items_voted:
            k = x[1]
            dic_wins[k] = dic_wins.get(k, 0) + 1

        dic_votes = {}
        for x in pairs_of_items_voted:
            k = x[0][0]
            dic_votes[k] = dic_votes.get(k, 0) + 1
            k = x[0][1]
            dic_votes[k] = dic_votes.get(k, 0) + 1

        dic_scores = {}

        for k in dic_votes.keys():
            dic_scores[k] = dic_wins.get(k, 0) / dic_votes[k]

        uniform_ranks = [x[0] for x in sorted(list(dic_scores.items()), key=lambda x: x[1], reverse=True)]


        #####
        # case 2 - ADAPTIVE APPROACH - n_b iterations

        t1 = time.time()
        items_cicle = items
        list_scores_tot = []

        dic_scores_final = {}


        for kk in range(0, n_b):

            print("\rAdaptive simulation {}/{}  (k = {})".format(votation + 1, n_simulations, kk + 1), end="")
            pairs_of_items = create_pairs_of_items(items_cicle, MM, voters_unique)
            pairs_of_items_voted = [[x, choose_winner(x, voter)] for (x, voter) in pairs_of_items]

            dic_wins = {}
            for x in pairs_of_items_voted:
                k = x[1]
                dic_wins[k] = dic_wins.get(k, 0) + 1

            dic_votes = {}
            for x in pairs_of_items_voted:
                k = x[0][0]
                dic_votes[k] = dic_votes.get(k, 0) + 1
                k = x[0][1]
                dic_votes[k] = dic_votes.get(k, 0) + 1

            dic_x_kk = {}

            for k in dic_votes.keys():
                dic_x_kk[k] = dic_wins.get(k, 0) / dic_votes[k]

            if kk > 0:
                bb = sum([(1 - dic_x_kk[k]) * (1 - dic_scores_final[k]) for k in dic_x_kk.keys()]) / sum([(1 - dic_x_kk[k]) ** 2 for k in dic_x_kk.keys()])
                dic_y_kk = {}
                for k in dic_x_kk.keys():
                    dic_y_kk[k] = 1 - bb + bb * dic_x_kk[k]
                    dic_scores_final[k] = ((kk - 1) * dic_scores_final[k] + dic_y_kk[k]) / kk
            else:
                dic_scores_final = copy.deepcopy(dic_x_kk)


            list_scores_tot.append(dic_x_kk)

            n_items_selected = round(len(items_cicle) * alpha)
            items_selected = sorted([(k, v) for (k, v) in dic_x_kk.items()], key=lambda x:x[1], reverse=True)[:n_items_selected]
            items_cicle = [x[0] for x in items_selected]

        print("\rFinal ranks calculation {}/{}       ".format(votation + 1, n_simulations), end="")
        final_ranks = [x[0] for x in sorted(list(dic_scores_final.items()), key=lambda x: x[1], reverse=True)]

        t2 = time.time()
        runtimes_uniform.append(t1 - t0)
        runtimes_adaptive.append(t2 - t1)


        #####
        # coefficients
        bb = [ x + 1 for x in theoretical_ranks]
        aa = [ x + 1 for x in uniform_ranks]
        coeff = weighted_spearman(aa, bb)
        dic_coeff_tot["rho_w_uniform"] = dic_coeff_tot.get("rho_w_uniform", []) + [coeff]
        aa = [ x + 1 for x in final_ranks]
        coeff = weighted_spearman(aa, bb)
        dic_coeff_tot["rho_w_adaptive"] = dic_coeff_tot.get("rho_w_adaptive", []) + [coeff]

        bb = [ x + 1 for x in theoretical_ranks]
        aa = [ x + 1 for x in uniform_ranks]
        coeff = weighted_kendall(aa, bb)
        dic_coeff_tot["tau_w_uniform"] = dic_coeff_tot.get("tau_w_uniform", []) + [coeff]
        aa = [ x + 1 for x in final_ranks]
        coeff = weighted_kendall(aa,bb)
        dic_coeff_tot["tau_w_adaptive"] = dic_coeff_tot.get("tau_w_adaptive", []) + [coeff]


        dic_coeff_tot["rho_uniform"] = dic_coeff_tot.get("rho_uniform", []) + [spearmanr(theoretical_ranks, uniform_ranks)[0]]
        dic_coeff_tot["rho_adaptive"] = dic_coeff_tot.get("rho_adaptive", []) + [spearmanr(theoretical_ranks, final_ranks)[0]]

        dic_coeff_tot["tau_uniform"] = dic_coeff_tot.get("tau_uniform", []) + [kendalltau(theoretical_ranks, uniform_ranks)[0]]
        dic_coeff_tot["tau_adaptive"] = dic_coeff_tot.get("tau_adaptive", []) + [kendalltau(theoretical_ranks, final_ranks)[0]]

    print("\n\nUnderlying similarity distribution: {}".format(distribution_type))
    for k in dic_coeff_tot.keys():
        mm = round(np.mean(dic_coeff_tot[k]), 4)
        sd = round(math.sqrt(np.var(dic_coeff_tot[k]) * n_simulations / (n_simulations - 1)), 4)
        print("\t{}: {} ± {}".format(k, mm, sd))

    print("\nAverage runtime uniform: {}".format(np.mean(runtimes_uniform)))
    print("Average runtime adaptive: {}".format(np.mean(runtimes_adaptive)))


    return None


distribution_type = input("Insert distribution type (allowed values: \"exponential\", \"power_law\", \"same_area\"): ")
voting(distribution_type)
