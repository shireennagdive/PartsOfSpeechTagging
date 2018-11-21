#Reference : https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode

def get_best_score(best_scores, transitions, emission_score, current_token, current_tag, L):
    max_score = float("-Inf")
    final_tag_for_token = -1
    for temp_tag in range(L):
        temp_score = best_scores[current_token - 1][temp_tag] + transitions[temp_tag][current_tag]
        if temp_score > max_score:
            max_score = temp_score
            final_tag_for_token = temp_tag

    return (max_score + emission_score), final_tag_for_token


def get_best_path(path, end_tag_index):
    final_path = [end_tag_index]
    token = len(path) - 1
    tag_index = end_tag_index
    while path[token][tag_index] != -1 and token > -1:
        final_path.append(path[token][tag_index])
        tag_index = path[token][tag_index]
        token -= 1
    return final_path


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    path = [[-1 for x in range(L)] for y in range(N)]

    best_scores = [[0 for x in range(L)] for y in range(N)]

    for i in range(L):
        best_scores[0][i] = emission_scores[0][i] + start_scores[i]

    for token in range(1, N):
        for tag in range(L):
            best_scores[token][tag], path[token][tag] = get_best_score(best_scores, trans_scores,
                                                                       emission_scores[token][tag], token, tag,
                                                                       L)
    max_final_score = float("-Inf")
    final_column_tag = -1

    for tag in range(L):
        best_scores[N - 1][tag] = best_scores[N - 1][tag] + end_scores[tag]
        if best_scores[N - 1][tag] > max_final_score:
            max_final_score = best_scores[N - 1][tag]
            final_column_tag = tag

    final_path = get_best_path(path, final_column_tag)

    final_path = final_path[::-1]

    assert len(final_path) == N
    assert max_final_score != float("-Inf")

    return max_final_score, final_path
