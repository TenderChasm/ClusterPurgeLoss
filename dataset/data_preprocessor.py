import pandas as pd
from dataclasses import dataclass, field
from unionfind import UnionFind
from itertools import combinations
import numpy as np
import random

@dataclass
class RelatedMutantGroup():
    ancestor : int = None
    eqivalents : set = field(default_factory=set)
    non_eqivalents : set = field(default_factory=set)

    def copy(self):
        return RelatedMutantGroup(self.ancestor, self.eqivalents.copy(), self.non_eqivalents.copy())


class MutantStorage():

    def __init__(self, mutants : pd.DataFrame, mutant_pairs : pd.DataFrame):
        self.mutants : pd.DataFrame = mutants
        self.mutant_pairs : pd.DataFrame = mutant_pairs
        self.mutant_collection : list[RelatedMutantGroup] = MutantStorage._create_mutant_collection(mutant_pairs)
        
        self._sanity_check()
        
    @classmethod    
    def _create_mutant_collection(cls, mutant_pairs : pd.DataFrame):
        eq_mutants_sets_dsu = UnionFind()
        mutant_pairs.apply(
            lambda x: MutantStorage._insert_mutants_into_dsu(x, eq_mutants_sets_dsu),
            axis = 1)
        eq_mutants_sets : list[set] = eq_mutants_sets_dsu.components()

        mutant_collection = [RelatedMutantGroup(None,eq_set) for eq_set in eq_mutants_sets]
        mutant_pairs.apply(
            lambda x: MutantStorage._find_relative_group_for_neqmutants(x, mutant_collection),
            axis = 1)
        
        ancestral_fertility : dict[int,int] = {}
        mutant_pairs.apply(
            lambda x: MutantStorage._count_fertility(x, ancestral_fertility),
            axis = 1)
        
        max_fertility_for_each_group = [0] * len(mutant_collection)
        for ancestor,offspring_number in ancestral_fertility.items():
            for i, group in enumerate(mutant_collection):
                if ancestor in group.eqivalents and offspring_number > max_fertility_for_each_group[i]:
                    group.ancestor = ancestor
                    max_fertility_for_each_group[i] = offspring_number
        
        [group.eqivalents.remove(group.ancestor) for group in mutant_collection]        

        return mutant_collection

        
    @classmethod
    def _insert_mutants_into_dsu(cls, mutant_pair_row : pd.Series, dsu : UnionFind):
        if mutant_pair_row['label'] == 1:
            dsu.union(mutant_pair_row['code_id_1'], mutant_pair_row['code_id_2'])

    
    @classmethod
    def _find_relative_group_for_neqmutants(cls, mutant_pair_row : pd.Series, mutant_collection : list[RelatedMutantGroup]):
        if mutant_pair_row['label'] == 0:
            mutant_1 = mutant_pair_row['code_id_1']
            mutant_2 = mutant_pair_row['code_id_2']

            group_found = False

            for group in mutant_collection:
                if mutant_1 in group.eqivalents:
                    group.non_eqivalents.add(mutant_2)
                    group_found = True
            
            for group in mutant_collection:
                if mutant_2 in group.eqivalents:
                    group.non_eqivalents.add(mutant_1)
                    group_found = True
            
            if not group_found:
                mutant_collection.append(RelatedMutantGroup(None, set([mutant_1]), set([mutant_2])))

    @classmethod            
    def _count_fertility(cls, mutant_pair_row : pd.Series, ancestral_fertility : dict[int,int]):
        if mutant_pair_row['code_id_1'] in ancestral_fertility:
            ancestral_fertility[mutant_pair_row['code_id_1']] += 1
        else:
            ancestral_fertility[mutant_pair_row['code_id_1']] = 1

    def _sanity_check(self):
        pairs_of_sets = [(group.eqivalents, group.non_eqivalents) for group in self.mutant_collection]
        all_sets = [mut_set for group in pairs_of_sets for mut_set in group]
        intersection_of_all_sets = set.intersection(*all_sets)
        if len(intersection_of_all_sets) != 0:
            raise SystemError('disjoint sets are not disjoint, critical error')

    
    def generate_triplets(self, remove_empty = True):
        triplets : list[list] = []
        for group in self.mutant_collection:
            group_clone_ancestor_to_equivalents = group.copy()
            group_clone_ancestor_to_equivalents.eqivalents.add(group.ancestor)
            group_triplets = [(m1, m2, m3) 
                              for (m1, m2) in combinations(group_clone_ancestor_to_equivalents.eqivalents, 2) 
                               for m3 in group_clone_ancestor_to_equivalents.non_eqivalents]
            triplets.append(group_triplets)
        
        if remove_empty:
            triplets = list(filter( lambda x : len(x) != 0, triplets))
        return triplets
    
    def generate_clusters(self):
        clusters : list[list] = []
        for group_id, group in enumerate(self.mutant_collection):
            cluster_pairs_positive = [(group_id, group.ancestor,positive, 1) for positive in group.eqivalents]
            cluster_pairs_negative = [(group_id, group.ancestor,negative, 0) for negative in group.non_eqivalents]
            clusters.append(cluster_pairs_positive+cluster_pairs_negative)
        return clusters

    
    
    #def generate_triplets_dss(self, fr)


class TripletProcessor():
    @staticmethod
    def count_cv(triplets : list[list]):
        sizes = TripletProcessor.count_stratum_sizes(triplets)
        return np.std(sizes) / np.mean(sizes) 
    
    @staticmethod
    def count_stratum_sizes(triplets : list[list]):
        return [len(stratum) for stratum in triplets]

    @staticmethod
    def count_total_size(triplets : list[list]):
        return sum(TripletProcessor.count_stratum_sizes(triplets))
    
    @staticmethod
    def undersample_by_nth_largest(triplets : list[list], n : int):
        nth_largest = sorted(TripletProcessor.count_stratum_sizes(triplets), reverse = True)[n]
        return TripletProcessor.undersample(triplets, nth_largest)
    
    @staticmethod
    def undersample(triplets : list[list], value):
        shuffled_triplets = [random.sample(group, len(group)) for group in triplets]
        undersampled = [triplet[:value] for triplet in shuffled_triplets]
        return undersampled
    
    @staticmethod
    def export_to_csv(triplets : list[list], name : str, shuffle = True):
        data = []
        for i, group in enumerate(triplets):
            for triplet in group:
                data.append([i, *triplet])

        if shuffle:
            random.shuffle(data)
        df = pd.DataFrame(data = data, columns=['group_id', 'anchor', 'positive', 'negative'])
        df.to_csv(name, index = False)


class ClusterProcessor(TripletProcessor):
    
    @staticmethod
    def export_to_csv(clusters : list[list], name : str, shuffle = True):
        data = [cluster_shard for cluster in clusters for cluster_shard in cluster]
        if shuffle:
            random.shuffle(data)
        df = pd.DataFrame(data = data, columns=['group_id', 'centroid', 'mutant', 'sign'])
        df.to_csv(name, index = False)


    

mutants : pd.DataFrame = pd.read_csv('dataset/MutantBench_code_db_java.csv')
mutants = mutants.set_index('id')

mutant_pairs_train : pd.DataFrame = pd.read_csv('dataset/train_pairs.csv')
mutant_pairs_test : pd.DataFrame = pd.read_csv('dataset/test_pairs.csv')


mutant_pairs_train = mutant_pairs_train[['code_id_1','code_id_2','label']]
storage = MutantStorage(mutants, mutant_pairs_train)


triplets = storage.generate_triplets()
triplets_1 = TripletProcessor.undersample_by_nth_largest(triplets,1)
triplets_2 = TripletProcessor.undersample_by_nth_largest(triplets,13)
print(TripletProcessor.count_total_size(triplets), TripletProcessor.count_cv(triplets))
print(TripletProcessor.count_total_size(triplets_1), TripletProcessor.count_cv(triplets_1))
print(TripletProcessor.count_total_size(triplets_2), TripletProcessor.count_cv(triplets_2))
TripletProcessor.export_to_csv(triplets, 'dataset/train_triplets_all.csv')
TripletProcessor.export_to_csv(triplets_1, 'dataset/train_triplets_to_first.csv')
TripletProcessor.export_to_csv(triplets_2, 'dataset/train_triplets_to_thirteenth.csv')

mutant_pairs_test = mutant_pairs_test[['code_id_1','code_id_2','label']]
storage = MutantStorage(mutants, mutant_pairs_test)
triplets = storage.generate_triplets()
triplets_1 = TripletProcessor.undersample_by_nth_largest(triplets,1)
triplets_2 = TripletProcessor.undersample_by_nth_largest(triplets,13)
print(TripletProcessor.count_total_size(triplets), TripletProcessor.count_cv(triplets))
print(TripletProcessor.count_total_size(triplets_1), TripletProcessor.count_cv(triplets_1))
print(TripletProcessor.count_total_size(triplets_2), TripletProcessor.count_cv(triplets_2))
TripletProcessor.export_to_csv(triplets, 'dataset/test_triplets_all.csv')
TripletProcessor.export_to_csv(triplets_1, 'dataset/test_triplets_to_first.csv')
TripletProcessor.export_to_csv(triplets_2, 'dataset/test_triplets_to_thirteenth.csv')


storage = MutantStorage(mutants, mutant_pairs_train)
clusters = storage.generate_clusters()
print(ClusterProcessor.count_total_size(clusters), ClusterProcessor.count_cv(clusters))
ClusterProcessor.export_to_csv(clusters, 'dataset/train_clusters.csv')

storage = MutantStorage(mutants, mutant_pairs_test)
clusters = storage.generate_clusters()
print(ClusterProcessor.count_total_size(clusters), ClusterProcessor.count_cv(clusters))
ClusterProcessor.export_to_csv(clusters, 'dataset/test_clusters.csv')


'''a = [i for i in groups if len(i.neqs) != 0]
lamb = lambda group : len(mutants.query(f'id == {next(iter(group.eq))}').iloc[0]['code'])
sum = 0
for i in a:
    sum += lamb(i) * len(i.eq) * (len(i.eq) - 1) * len(i.neqs[0])'''

a = 1
#all_mutants_ids = 
