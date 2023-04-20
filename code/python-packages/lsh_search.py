import numpy as np
import pandas as pd
import itertools
from similarities import *
from deep_hashing_models import *
from fnvhash import fnv1a_32, fnv1a_64

def convert_to_hamming(embeddings):
    s = pd.Series(embeddings.tolist())
    s = s.apply(lambda x : np.array(list(map(transform, x))))
    embeddings_hamming = np.vstack(s.tolist())
    return embeddings_hamming

def lsh_hyperparams(m) :
    i = 0
    params = []
    while i <= math.log(m, 2) :
        params.append((2**i, int(m / 2**i)))
        i += 1
    return params  

def create_hash_tables(L, K, b, embeddings_hamming) :
    n_bits = K * b
    hash_tables = {}
    i = 0
    while i < L :
        hash_tables['entry_'+str(i)] = {}
        rows, counts = np.unique(embeddings_hamming[:,i*n_bits:(i+1)*n_bits], axis=0, return_counts = True)
        for row, count in zip(rows, counts):
            if count > 1 :
                indexes = np.where((embeddings_hamming[:,i*n_bits:(i+1)*n_bits] == row).all(axis = 1))[0]
                hash_tables['entry_'+str(i)][row.tostring()] = indexes  
        i += 1
    
    return hash_tables

def near_duplicates_for_runtime(L, K, b, index, embeddings_hamming, hash_tables):
    n_bits = K * b
    l_indexes = []
    for i in range(L) :
        if embeddings_hamming[index][i*n_bits:(i+1)*n_bits].tostring() in hash_tables['entry_'+str(i)] :
            l_indexes.append(list(hash_tables['entry_'+str(i)][embeddings_hamming[index][i*n_bits:(i+1)*n_bits].tostring()]))
    return l_indexes

def near_duplicates(L, K, b, index, embeddings_hamming, hash_tables):
    n_bits = K * b
    l_indexes = []
    for i in range(L) :
        if embeddings_hamming[index][i*n_bits:(i+1)*n_bits].tostring() in hash_tables['entry_'+str(i)] :
            l_indexes.append(list(hash_tables['entry_'+str(i)][embeddings_hamming[index][i*n_bits:(i+1)*n_bits].tostring()]))
    set_indexes = set([item for l in l_indexes for item in l])
    return set_indexes

def near_duplicate_pairs(index, set_indexes, n_stacks):
    l_indexes_sim = []
    for i in set_indexes :
        if i < index :    
            l_indexes_sim.append(get_index_sim(n_stacks, i, index))
        elif i > index :
            l_indexes_sim.append(get_index_sim(n_stacks, index, i))
    return l_indexes_sim

def real_nns(index, df_measures, measure, n_stacks, n_duplicate_pairs):
    set_real_nns = set()
    s1 = pd.Series()
    for i in range (index) :
        s1 = pd.concat([s1, df_measures[measure][[get_index_sim(n_stacks, i, index)]]])
    s2 = df_measures[measure][get_index_sim(n_stacks, index, index+1):get_index_sim(n_stacks, index, n_stacks-1)+1]
    s = pd.concat([s1,s2]).sort_values(ascending = False)
    for ind in s[:n_duplicate_pairs].index :
        a, b = get_indices_sim(n_stacks, ind)
        if a == index :
            set_real_nns.add(b)
        else : 
            set_real_nns.add(a)
    return (set_real_nns, s)
    
    
def generalized_mrr(approximate_nns_sim, real_nns_sim):
    approximate_nns_sim = approximate_nns_sim.values
    real_nns_sim = real_nns_sim.values
    precision = 0
    for i in range(approximate_nns_sim.size) :
        if (i+1) > (np.where(real_nns_sim == approximate_nns_sim[i])[0][0] + 1) :
            precision += 1 
        else :
            #precision += math.log((i+1)+1,2) / math.log((np.where(real_nns_sim == approximate_nns_sim[i])[0][0] + 1) + 1,2)
            precision += ((i+1)+1) / ((np.where(real_nns_sim == approximate_nns_sim[i])[0][0] + 1) + 1)
    return precision / approximate_nns_sim.size
    

def prob_hashing_smallest_elt(approximate_nns_sim, K, L):
    return 1 - (1 - (approximate_nns_sim.values[-1])**K)**L

def mean_prob_hashing_smallest_elt(n_stacks, params, embeddings_hamming, b, df_measures ,measure, trace = True):
    df_probLSH = pd.DataFrame()
    list_indexes = list(np.arange(n_stacks))
    for L, K in params :
        if trace :
            print((L,K))
        l_probas = []
        hash_tables = create_hash_tables(L, K, b, embeddings_hamming)

        for index in list_indexes:
            if index % 100 == 0 and trace :
                print (index)
            l_indexes = near_duplicates(L, K, b, index, embeddings_hamming, hash_tables)
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks)
            if l_indexes_sim :   
                'Find the approximate k-nns'
                smallest_app_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values[-1]
                l_probas.append(1 - (1 - (smallest_app_nns)**K)**L)

            else :
                l_probas.append(None)

        df_probLSH[str((L,K))] = l_probas
        if trace :
            print('-----------------------------')
    return df_probLSH


def recal_rate (n_stacks, params, embeddings_hamming, b, df_measures, k_first_positions, measure) :
    ratio_ann = 0
    list_indexes = list(np.arange(n_stacks))
    for L, K in params :
        old_list_indexes = []
        new_list_indexes = []
        hash_tables = create_hash_tables(L, K, b, embeddings_hamming)

        for index in list_indexes:
            l_indexes = near_duplicates(L, K, b, index, embeddings_hamming, hash_tables)
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks)
            if len(l_indexes_sim) >= k_first_positions :
                'Find the approximate k-nns'
                approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values[:k_first_positions]
                'Find the real nn'
                s1 = pd.Series()
                for i in range (index) :
                    s1 = pd.concat([s1, df_measures[measure][[get_index_sim(n_stacks, i, index)]]])
                s2 = df_measures[measure][get_index_sim(n_stacks, index, index+1):get_index_sim(n_stacks, index, n_stacks-1)+1]
                s = pd.concat([s1,s2])
                real_nns = s.sort_values(ascending = False)[:len(l_indexes_sim)].values[:k_first_positions]
                score = 0
                for nn in real_nns:
                    if nn in approximate_nns :
                        score += 1
                        approximate_nns = np.delete(approximate_nns, np.where(approximate_nns == nn)[0][0])
                ratio_ann += score / k_first_positions 
                old_list_indexes.append(index)
            else :
                new_list_indexes.append(index)
        list_indexes = new_list_indexes[:]
        if not list_indexes : 
            break
    return ratio_ann / n_stacks


def recal_rate_one_param (n_stacks, param, embeddings_hamming, b, df_measures, k_first_positions, measure) :
    ratio_ann = 0
    list_indexes = list(np.arange(n_stacks))
    L = param[0]
    K = param[1]
    cpt = 0
    
    hash_tables = create_hash_tables(L, K, b, embeddings_hamming)

    for index in list_indexes:
        l_indexes = near_duplicates(L, K, b, index, embeddings_hamming, hash_tables)
        l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks)
        if len(l_indexes_sim) >= k_first_positions :
            cpt += 1
            'Find the approximate k-nns'
            approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values[:k_first_positions]

            'Find the real nn'
            s1 = pd.Series()
            for i in range (index) :
                s1 = pd.concat([s1, df_measures[measure][[get_index_sim(n_stacks, i, index)]]])
            s2 = df_measures[measure][get_index_sim(n_stacks, index, index+1):get_index_sim(n_stacks, index, n_stacks-1)+1]
            s = pd.concat([s1,s2])
            real_nns = s.sort_values(ascending = False)[:len(l_indexes_sim)].values[:k_first_positions]
            score = 0
            for nn in real_nns:
                if nn in approximate_nns :
                    score += 1
                    approximate_nns = np.delete(approximate_nns, np.where(approximate_nns == nn)[0][0])
            ratio_ann += score / k_first_positions 

    return ratio_ann / cpt


def mean_generalized_mrr(n_stacks, params, embeddings_hamming, b, df_measures ,measure, trace = True) :
    df_knns = pd.DataFrame()
    list_indexes = list(np.arange(n_stacks))
    for L, K in params :
        if trace :
            print((L,K))
        l_precisions = []
        hash_tables = create_hash_tables(L, K, b, embeddings_hamming)

        for index in list_indexes:
            if index % 100 == 0 and trace:
                print (index)
            l_indexes = near_duplicates(L, K, b, index, embeddings_hamming, hash_tables)
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks)
            if l_indexes_sim :   
                'Find the approximate k-nns'
                approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values

                'Find the real k-nns'
                s1 = pd.Series()
                for i in range(index) :
                    s1 = pd.concat([s1, df_measures[measure][[get_index_sim(n_stacks, i, index)]]])
                s2 = df_measures[measure][get_index_sim(n_stacks, index, index+1):get_index_sim(n_stacks, index, n_stacks-1)+1]
                s = pd.concat([s1,s2])
                real_nns = s.sort_values(ascending = False).values

                precision = 0
                for i in range(approximate_nns.size) :
                    if (i+1) > (np.where(real_nns == approximate_nns[i])[0][0] + 1) :
                        precision += 1 
                    else :
                        #precision += math.log((i+1)+1,2) / math.log((np.where(real_nns == approximate_nns[i])[0][0] + 1)+1,2)
                        precision += ((i+1)+1) / ((np.where(real_nns == approximate_nns[i])[0][0] + 1)+1)
                l_precisions.append(precision / approximate_nns.size) 
            else :
                l_precisions.append(None)

        df_knns[str((L,K))] = l_precisions
        if trace :
            print('-----------------------------')
    return df_knns


def precision_all (n_stacks, params, embeddings_hamming, b, df_measures ,measure, trace = True) :
    df_precision = pd.DataFrame()
    list_indexes = list(np.arange(n_stacks))
    for L, K in params :
        if trace :
            print((L,K))
        l_precisions = []
        hash_tables = create_hash_tables(L, K, b, embeddings_hamming)
        
        for index in list_indexes:
            if index % 100 == 0 and trace:
                print (index)
            l_indexes = near_duplicates(L, K, b, index, embeddings_hamming, hash_tables)
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks)
            if l_indexes_sim :   
                'Find the approximate k-nns'
                approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values
                precision = 0
                for i in range(approximate_nns.size) :
                    prob = (1 - (1 - (approximate_nns[i])**K)**L)
                    if prob >= 0.5 :
                        precision += 1 
                l_precisions.append(precision / approximate_nns.size) 
            else :
                l_precisions.append(None)

        df_precision[str((L,K))] = l_precisions
        if trace :
            print('-----------------------------')
    return df_precision


def recall_all (n_stacks, params, embeddings_hamming, b, df_measures ,measure, trace = True) :
    df_recall = pd.DataFrame()
    list_indexes = list(np.arange(n_stacks))
    for L, K in params :
        if trace :
            print((L,K))
        l_recall = []
        hash_tables = create_hash_tables(L, K, b, embeddings_hamming)
        
        for index in list_indexes:
            if index % 100 == 0 and trace:
                print (index)
            l_indexes = near_duplicates(L, K, b, index, embeddings_hamming, hash_tables)
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks)
            
            'Find the real nn'
            s1 = pd.Series()
            for i in range (index) :
                s1 = pd.concat([s1, df_measures[measure][[get_index_sim(n_stacks, i, index)]]])
            s2 = df_measures[measure][get_index_sim(n_stacks, index, index+1):get_index_sim(n_stacks, index, n_stacks-1)+1]
            s = pd.concat([s1,s2])
            real_nns = s.sort_values(ascending = False)
            prob_real_nns = real_nns.apply(lambda x : (1 - (1 - (x)**K)**L))
            real_nns = real_nns[prob_real_nns >= 0.5].values
            
            if len(real_nns) > 0 :
            
                'Find the approximate k-nns'
                approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values
                score = 0
                for a_nn in approximate_nns:
                    prob = (1 - (1 - (a_nn)**K)**L)
                    if prob >= 0.5 :
                        score += 1

                l_recall.append(score / len(real_nns))
            else :
                l_recall.append(None)

        df_recall[str((L,K))] = l_recall
        if trace :
            print('-----------------------------')
    return df_recall

def fscore_all(df_precision, df_recall):
    precision_values = df_precision.values
    recall_values = df_recall.values
    f_score_values = np.zeros((precision_values.shape[0], precision_values.shape[1]))
    
    for i in range(precision_values.shape[0]):
        for j in range(precision_values.shape[1]):
            precision = precision_values[i][j]
            recall = recall_values[i][j]
            if precision == 0 or recall == 0:
                f_score_values[i][j] = 0
            else :
                f_score_values[i][j] = 2 * (precision * recall) / (precision + recall) 
    
    return pd.DataFrame(f_score_values, columns = df_precision.columns)

def min_hash (nb_permutations, size_vector, df_bag_of_frames) :
    embeddings = np.zeros((df_bag_of_frames.shape[0], nb_permutations * size_vector))
    for cpt in range(0, nb_permutations):
        permutation = np.random.permutation(np.arange(df_bag_of_frames.shape[1]))
        df_bag_of_frames_permuted = df_bag_of_frames.iloc[:,permutation]
        mask = (df_bag_of_frames_permuted != 0).idxmax(1)
        min_index = mask.apply(lambda x : list(df_bag_of_frames_permuted.columns).index(x))
        min_index_binary = min_index.apply(lambda x : list(map(int, [*'{0:016b}'.format(x)])))
        embeddings[:, cpt * size_vector: (cpt+1) * size_vector] = np.array(min_index_binary.values.tolist())
    return embeddings

def get_hash_code_simhash(row, df_hash_codes_frames) :
    tmp = df_hash_codes_frames.iloc[:, np.r_[np.where(row > 0)[0]]].mul(list(row.iloc[np.where(row > 0)[0]]), axis = 1)
    tmp = tmp.replace(0, -1)
    hash_code = tmp.sum(axis = 1)
    hash_code[hash_code < 0] = 0
    hash_code[hash_code > 0] = 1
    return hash_code

def simhash(df_bag_of_frames):
    frames_hash_codes = pd.Series(df_bag_of_frames.columns).apply(lambda x : list(map(int, [*'{0:064b}'.format(fnv1a_64(bytes(x, encoding='utf-8')))])))
    df_hash_codes_frames = pd.DataFrame.from_dict(dict(zip(frames_hash_codes.index, frames_hash_codes.values)))
    df_hash_codes = df_bag_of_frames.apply(lambda x : get_hash_code_simhash (x, df_hash_codes_frames), axis = 1)
    embeddings = df_hash_codes.values
    return embeddings


def create_hash_tables_simhash(combinations, embeddings, size_vector):
    hash_tables = {}
    i = 0
    while i < len(combinations) :
        hash_tables['combination_'+str(i)] = {}

        elt = combinations[i]
        list_slots = []
        for slot in elt : 
            list_slots.append(np.arange(size_vector * slot,size_vector * (slot + 1)))
        embeddings_slice = embeddings[:, np.concatenate(list_slots)]

        rows, counts = np.unique(embeddings_slice, axis=0, return_counts = True)
        for row, count in zip(rows, counts):
            if count > 1 :
                indexes = np.where((embeddings_slice == row).all(axis = 1))[0]
                hash_tables['combination_'+str(i)][row.tostring()] = indexes  
        i += 1
    return hash_tables

def near_duplicates_simhash(index, combinations, embeddings, size_vector, hash_tables):
    l_indexes = []
    for i in range(len(combinations)) :
        
        elt = combinations[i]
        list_slots = []
        for slot in elt : 
            list_slots.append(np.arange(size_vector * slot,size_vector * (slot + 1)))
        embdeddings_slice = embeddings[:, np.concatenate(list_slots)]

        if embdeddings_slice[index].tostring() in hash_tables['combination_'+str(i)] :
            l_indexes.append(list(hash_tables['combination_'+str(i)][embdeddings_slice[index].tostring()]))
    set_indexes = set([item for l in l_indexes for item in l])
    return set_indexes

def recal_rate_simhash (n_stacks, max_permutation, embeddings, size_vector, df_measures, measure, k_first_positions) :
    
    ratio_ann = 0
    list_indexes = list(np.arange(n_stacks))
    for cpt in range(max_permutation + 1)[::-1][:-1] :
        old_list_indexes = []
        new_list_indexes = []
        l = list(np.arange(int(embeddings.shape[1] / size_vector)))
        combinations = list(itertools.combinations(l, cpt))
        hash_tables = create_hash_tables_simhash(combinations, embeddings, size_vector)
        
        for index in list_indexes:
            
            l_indexes = near_duplicates_simhash(index, combinations, embeddings, size_vector, hash_tables)       
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks) 
            
            if len(l_indexes_sim) >= k_first_positions :
                'Find the approximate k-nns'
                approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values[:k_first_positions]
                'Find the real nn'
                s1 = pd.Series()
                for i in range (index) :
                    s1 = pd.concat([s1, df_measures[measure][[get_index_sim(n_stacks, i, index)]]])
                s2 = df_measures[measure][get_index_sim(n_stacks, index, index+1):get_index_sim(n_stacks, index, n_stacks-1)+1]
                s = pd.concat([s1,s2])
                real_nns = s.sort_values(ascending = False)[:len(l_indexes_sim)].values[:k_first_positions]
                score = 0
                for nn in real_nns:
                    if nn in approximate_nns :
                        score += 1
                        approximate_nns = np.delete(approximate_nns, np.where(approximate_nns == nn)[0][0])
                ratio_ann += score / k_first_positions 
                old_list_indexes.append(index)
            else :
                new_list_indexes.append(index)
        list_indexes = new_list_indexes[:]
        if not list_indexes : 
            break
    return ratio_ann / n_stacks


def recal_rate_one_param_simhash (n_stacks, permutation, embeddings, size_vector, df_measures, measure, k_first_positions) :
    
    ratio_ann = 0
    list_indexes = list(np.arange(n_stacks))
    cpt = 0
    
    l = list(np.arange(int(embeddings.shape[1] / size_vector)))
    combinations = list(itertools.combinations(l, permutation))
    hash_tables = create_hash_tables_simhash(combinations, embeddings, size_vector)

    for index in list_indexes:
        
        l_indexes = near_duplicates_simhash(index, combinations, embeddings, size_vector, hash_tables)       
        l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks) 
        
        if len(l_indexes_sim) >= k_first_positions :
            cpt += 1
            'Find the approximate k-nns'
            approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values[:k_first_positions]

            'Find the real nn'
            s1 = pd.Series()
            for i in range (index) :
                s1 = pd.concat([s1, df_measures[measure][[get_index_sim(n_stacks, i, index)]]])
            s2 = df_measures[measure][get_index_sim(n_stacks, index, index+1):get_index_sim(n_stacks, index, n_stacks-1)+1]
            s = pd.concat([s1,s2])
            real_nns = s.sort_values(ascending = False)[:len(l_indexes_sim)].values[:k_first_positions]
            score = 0
            for nn in real_nns:
                if nn in approximate_nns :
                    score += 1
                    approximate_nns = np.delete(approximate_nns, np.where(approximate_nns == nn)[0][0])
            ratio_ann += score / k_first_positions 

    return ratio_ann / cpt


def mean_generalized_mrr_simhash (n_stacks, max_permutation, embeddings, size_vector, df_measures, measure, trace = True) :
    df_knns = pd.DataFrame()
    list_indexes = list(np.arange(n_stacks))
    for cpt in range(max_permutation + 1)[::-1][:-1] :
        if trace :
            print(cpt)
        l_precisions = []
        l = list(np.arange(int(embeddings.shape[1] / size_vector)))
        combinations = list(itertools.combinations(l, cpt))
        hash_tables = create_hash_tables_simhash(combinations, embeddings, size_vector)

        for index in list_indexes:
            if index % 100 == 0 and trace:
                print (index)
            l_indexes = near_duplicates_simhash(index, combinations, embeddings, size_vector, hash_tables)       
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks) 
            if l_indexes_sim :   
                'Find the approximate k-nns'
                approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values

                'Find the real k-nns'
                s1 = pd.Series()
                for i in range(index) :
                    s1 = pd.concat([s1, df_measures[measure][[get_index_sim(n_stacks, i, index)]]])
                s2 = df_measures[measure][get_index_sim(n_stacks, index, index+1):get_index_sim(n_stacks, index, n_stacks-1)+1]
                s = pd.concat([s1,s2])
                real_nns = s.sort_values(ascending = False).values

                precision = 0
                for i in range(approximate_nns.size) :
                    if (i+1) > (np.where(real_nns == approximate_nns[i])[0][0] + 1) :
                        precision += 1 
                    else :
                        precision += ((i+1)+1) / ((np.where(real_nns == approximate_nns[i])[0][0] + 1)+1)
                l_precisions.append(precision / approximate_nns.size) 
            else :
                l_precisions.append(None)

        df_knns[str(cpt)] = l_precisions
        if trace :
            print('-----------------------------')
    return df_knns


def mean_prob_hashing_smallest_elt_simhash (n_stacks, max_permutation, embeddings, size_vector, df_measures ,measure, trace = True):
    df_probLSH = pd.DataFrame()
    list_indexes = list(np.arange(n_stacks))
    for cpt in range(max_permutation + 1)[::-1][:-1] :
        if trace :
            print(cpt)
        l_probas = []
        l = list(np.arange(int(embeddings.shape[1] / size_vector)))
        combinations = list(itertools.combinations(l, cpt))
        hash_tables = create_hash_tables_simhash(combinations, embeddings, size_vector)

        for index in list_indexes:
            if index % 100 == 0 and trace:
                print (index)
            l_indexes = near_duplicates_simhash(index, combinations, embeddings, size_vector, hash_tables)       
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks) 
            if l_indexes_sim :   
                'Find the approximate k-nns'
                smallest_app_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values[-1]
                l_probas.append(1 - (1 - (smallest_app_nns)**(size_vector*cpt))**len(combinations))

            else :
                l_probas.append(None)

        df_probLSH[str(cpt)] = l_probas
        if trace :
            print('-----------------------------')
    return df_probLSH

def precision_all_simhash (n_stacks, max_permutation, embeddings, size_vector, df_measures ,measure, trace = True) :
    df_precision = pd.DataFrame()
    list_indexes = list(np.arange(n_stacks))
    for cpt in range(max_permutation + 1)[::-1][:-1] :
        if trace :
            print(cpt)
        l_precisions = []
        l = list(np.arange(int(embeddings.shape[1] / size_vector)))
        combinations = list(itertools.combinations(l, cpt))
        hash_tables = create_hash_tables_simhash(combinations, embeddings, size_vector)
        
        for index in list_indexes:
            if index % 100 == 0 and trace:
                print (index)
            l_indexes = near_duplicates_simhash(index, combinations, embeddings, size_vector, hash_tables)       
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks) 
            if l_indexes_sim :   
                'Find the approximate k-nns'
                approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values
                precision = 0
                for i in range(approximate_nns.size) :
                    prob = 1 - (1 - (approximate_nns[i])**(size_vector*cpt))**len(combinations)
                    if prob >= 0.5 :
                        precision += 1 
                l_precisions.append(precision / approximate_nns.size) 
            else :
                l_precisions.append(None)

        df_precision[str(cpt)] = l_precisions
        if trace :
            print('-----------------------------')
    return df_precision


def recall_all_simhash (n_stacks, max_permutation, embeddings, size_vector, df_measures ,measure, trace = True) :
    df_recall = pd.DataFrame()
    list_indexes = list(np.arange(n_stacks))
    for cpt in range(max_permutation + 1)[::-1][:-1] :
        if trace :
            print(cpt)
        l_recall = []
        l = list(np.arange(int(embeddings.shape[1] / size_vector)))
        combinations = list(itertools.combinations(l, cpt))
        hash_tables = create_hash_tables_simhash(combinations, embeddings, size_vector)
        
        for index in list_indexes:
            if index % 100 == 0 and trace:
                print (index)
            l_indexes = near_duplicates_simhash(index, combinations, embeddings, size_vector, hash_tables)       
            l_indexes_sim = near_duplicate_pairs(index, l_indexes, n_stacks) 
            
            'Find the real nn'
            s1 = pd.Series()
            for i in range (index) :
                s1 = pd.concat([s1, df_measures[measure][[get_index_sim(n_stacks, i, index)]]])
            s2 = df_measures[measure][get_index_sim(n_stacks, index, index+1):get_index_sim(n_stacks, index, n_stacks-1)+1]
            s = pd.concat([s1,s2])
            real_nns = s.sort_values(ascending = False)
            prob_real_nns = real_nns.apply(lambda x : 1 - (1 - (x)**(size_vector*cpt))**len(combinations))
            real_nns = real_nns[prob_real_nns >= 0.5].values
            
            if len(real_nns) > 0 :
            
                'Find the approximate k-nns'
                approximate_nns = df_measures[measure][l_indexes_sim].sort_values(ascending = False).values
                score = 0
                for a_nn in approximate_nns:
                    prob = 1 - (1 - (a_nn)**(size_vector*cpt))**len(combinations)
                    if prob >= 0.5 :
                        score += 1

                l_recall.append(score / len(real_nns))
            else :
                l_recall.append(None)

        df_recall[str(cpt)] = l_recall
        if trace :
            print('-----------------------------')
    return df_recall
