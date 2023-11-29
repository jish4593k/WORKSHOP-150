import argparse
import numpy as np
from itertools import combinations
from sklearn.metrics import precision_recall_curve, auc

class Apriori:
    def __init__(self, database_file, min_supp, output_file):
        self.database_file = database_file
        self.min_supp = min_supp
        self.output_file = output_file
        self.freq_set = []  # List to store frequent itemsets at each level of iteration

    def read_database(self):
        # Read the transactional database from the specified file
        with open(self.database_file, 'r') as fp:
            database = [list(map(int, line.split())) for line in fp.readlines()]
        return database

    def generate_F1(self, database):
        # Generate frequent 1-itemsets (F1)
        min_count = int(database[0][0] * self.min_supp)
        candidates = database[0][1]
        F1 = [x for x in range(candidates) if sum(1 for row in database[1:] if x in row) >= min_count]
        return [[x] for x in F1]

    def generate_candidate(self, last_freq_set, k):
        # Generate candidate itemsets of size (k + 1) from the frequent itemsets of size k
        unique_elem = list(set(item for sublist in last_freq_set for item in sublist))
        return list(combinations(unique_elem, k + 1))

    def prune_candidate(self, last_freq_set, candidates, k):
        # Prune candidates that have infrequent subsets
        return [c for c in candidates if all(set(subset) in last_freq_set for subset in combinations(c, k))]

    def count_support(self, candidates, database):
        # Count the support of each candidate in the database
        return [sum(1 for row in database if set(candidate).issubset(row)) for candidate in candidates]

    def eliminate_candidate(self, support, candidates):
        # Eliminate candidates with support below the minimum threshold
        min_count = int(database[0][0] * self.min_supp)
        return [c for c, s in zip(candidates, support) if s >= min_count]

    def output_freq_itemsets(self):
        # Write frequent itemsets to the specified output file
        with open(self.output_file, "w") as f:
            items = len(self.freq_set[0])
            transactions = sum(len(trans) for trans in self.freq_set)
            f.write(f"{transactions} {items}\n")

            for x in self.freq_set[:-1]:
                for p in x:
                    f.write(' '.join(map(str, p)) + "\n")
                f.write("\n")

    def apriori(self, database):
        # Apriori algorithm to find frequent itemsets
        k = 1
        self.freq_set.append(self.generate_F1(database))  # Initialize with frequent 1-itemsets

        while len(self.freq_set[k - 1]) > 0:
            candidates = self.generate_candidate(self.freq_set[k - 1], k)

            if k != 1:
                candidates = self.prune_candidate(self.freq_set[k - 1], candidates, k)

            support = self.count_support(candidates, database)
            candidates = self.eliminate_candidate(support, candidates)
            self.freq_set.append(candidates)
            k += 1

        self.output_freq_itemsets()

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-database_file')
    parser.add_argument('-minsupp')
    parser.add_argument('-output_file')
    args = parser.parse_args()

    # Create Apriori instance and execute the algorithm
    apriori = Apriori(args.database_file, float(args.minsupp), args.output_file)
    database = apriori.read_database()
    apriori.apriori(database)

if __name__ == "__main__":
    main()
