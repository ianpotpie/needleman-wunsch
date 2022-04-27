import sys
import argparse
import numpy as np
from scoring_scheme import ScoringScheme


def needleman_wunsch(sequence1, sequence2, scoring_scheme):
    """
    The Needleman-Wunsch algorithm performs global alignment according to the provided scoring scheme.
    The algorithm does not implement affine gap alignment, each gap has a fixed alignment penalty.

    :param sequence1: the first sequence to align
    :param sequence2: the second sequence to align
    :param scoring_scheme: the scoring system used in the alignment
    :return: a max alignment score and a list of alignments that produce that score
    """
    # initialize constants
    gap_penalty = scoring_scheme.gap_extend_penalty
    max_i = len(sequence1) + 1
    max_j = len(sequence2) + 1

    # initialize the scoring matrix
    A = np.empty((max_i, max_j), dtype=float)
    A[0, :] = gap_penalty * np.arange(max_j)
    A[:, 0] = gap_penalty * np.arange(max_i)

    # populate the dynamic programming matrix
    for i in range(1, max_i):
        for j in range(1, max_j):
            A[i, j] = max(A[i - 1, j - 1] + scoring_scheme(sequence1[i - 1], sequence2[j - 1]),
                          A[i - 1, j] + gap_penalty,
                          A[i, j - 1] + gap_penalty)
    max_score = A[max_i - 1, max_j - 1]

    # backtrace to find the alignments that created the max alignment score
    alignments = []
    partials = [("", "", max_i - 1, max_j - 1)]  # each partial alignment has the form (alignment1, alignment2, i, j)
    while len(partials) > 0:
        next_partials = []
        for alignment1, alignment2, i, j in partials:
            if i == 0:
                alignments.append((alignment1, (j * "-") + alignment2))
            elif j == 0:
                alignments.append(((i * "-") + alignment1, alignment2))
            else:
                if A[i, j] == A[i - 1, j - 1] + scoring_scheme(sequence1[i - 1], sequence2[j - 1]):
                    next_partials.append((sequence1[i - 1] + alignment1, sequence2[j - 1] + alignment2, i - 1, j - 1))
                if A[i, j] == A[i - 1, j] + gap_penalty:
                    next_partials.append((sequence1[i - 1] + alignment1, "-" + alignment2, i - 1, j))
                if A[i, j] == A[i, j - 1] + gap_penalty:
                    next_partials.append(("-" + alignment1, sequence2[j - 1] + alignment2, i, j - 1))
        partials = next_partials

    return max_score, alignments


def main():
    parser = argparse.ArgumentParser(description="Align two sequences.")
    parser.add_argument("sequence1", type=str)
    parser.add_argument("sequence2", type=str)
    parser.add_argument("--gap-penalty", type=float)
    parser.add_argument("--match-score", type=float)
    parser.add_argument("--mismatch-penalty", type=float)
    parser.add_argument("--matrix-file", type=str)
    args = parser.parse_args(sys.argv[1:])

    # the first sequence to align
    sequence1 = args.sequence1

    # the second sequence to align
    sequence2 = args.sequence2

    # create a scoring scheme
    scoring_scheme = ScoringScheme()
    if args.gap_penalty is not None:
        scoring_scheme.gap_extend_penalty = -abs(args.gap_penalty)
    if args.match_score is not None:
        scoring_scheme.match_score = args.match_score
    if args.mismatch_penalty is not None:
        scoring_scheme.mismatch_penalty = -abs(args.mismatch_penalty)
    if args.matrix_file is not None:
        scoring_scheme.load_matrix(args.matrix_file)

    max_score, alignments = needleman_wunsch(sequence1, sequence2, scoring_scheme)

    print(f"Sequence 1: {sequence1}")
    print(f"Sequence 2: {sequence2}")
    print(f"Alignment Score: {max_score}")
    print(30 * "-")
    for i, alignment in enumerate(alignments):
        print(alignment[0])
        print(alignment[1])
        print(30 * "-")


if __name__ == "__main__":
    main()
