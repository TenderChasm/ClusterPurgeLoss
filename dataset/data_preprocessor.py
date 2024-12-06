import pandas as pd
from dataclasses import dataclass, field
from unionfind import UnionFind
from itertools import combinations
import numpy as np
import random

@dataclass
class RelatedMutantGroup():
    eqivalents : set = field(default_factory=set)
    non_eqivalents : set = field(default_factory=set)


class MutantStorage():

    def __init__(self, mutants : pd.DataFrame, mutant_pairs : pd.DataFrame):
        self.mutants : pd.DataFrame = mutants
        self.mutant_pairs : pd.DataFrame = mutant_pairs
        self.mutant_collection : list[RelatedMutantGroup] = MutantStorage._create_mutant_collection(mutant_pairs)

    @classmethod    
    def _create_mutant_collection(cls, mutant_pairs : pd.DataFrame):
        eq_mutants_sets_dsu = UnionFind()
        mutant_pairs.apply(
            lambda x: MutantStorage._insert_mutants_into_dsu(x, eq_mutants_sets_dsu),
            axis = 1)
        eq_mutants_sets : list[set] = eq_mutants_sets_dsu.components()

        mutant_collection = [RelatedMutantGroup(eq_set) for eq_set in eq_mutants_sets]
        mutant_pairs.apply(
            lambda x: MutantStorage._find_relative_group_for_neqmutants(x, mutant_collection),
            axis = 1)
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

            for group in mutant_collection:
                if mutant_1 in group.eqivalents:
                    group.non_eqivalents.add(mutant_2)
            
            for group in mutant_collection:
                if mutant_2 in group.eqivalents:
                    group.non_eqivalents.add(mutant_1)
    
    def generate_triplets(self, remove_empty = True):
        triplets : list[list] = []
        for group in self.mutant_collection:
            group_triplets = [(m1, m2, m3) 
                              for (m1, m2) in combinations(group.eqivalents, 2) 
                               for m3 in group.non_eqivalents]
            triplets.append(group_triplets)
        
        if remove_empty:
            triplets = list(filter( lambda x : len(x) != 0, triplets))
        return triplets
    
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


'''a = [i for i in groups if len(i.neqs) != 0]
lamb = lambda group : len(mutants.query(f'id == {next(iter(group.eq))}').iloc[0]['code'])
sum = 0
for i in a:
    sum += lamb(i) * len(i.eq) * (len(i.eq) - 1) * len(i.neqs[0])'''

a = 1
#all_mutants_ids = 