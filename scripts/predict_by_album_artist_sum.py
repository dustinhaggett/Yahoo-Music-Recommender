from __future__ import print_function
from operator import itemgetter
import time
import os

RESULT_FILE = "data/prediction_simple_sum.csv"
TEST_SCORE_FILE = "data/test_hie_score.txt"
none_value = 50  # Replace 'None' with this value

def sort_list(input_list):
    sorted_list = [[x[0], x[1] + x[2]] for x in input_list]
    sorted_list = sorted(sorted_list, key=itemgetter(1))
    pred_dic = {}
    for i, item in enumerate(sorted_list):
        pred_dic[item[0]] = 0 if i < 3 else 1
    return [pred_dic[item[0]] for item in input_list]

def read_lines(file, num):
    lines = []
    line = file.readline()
    if not line:
        return []
    lines.append(line)
    for _ in range(1, num):
        l = file.readline()
        if l:
            lines.append(l)
    return lines

start_time = time.time()
with open(RESULT_FILE, "w") as predictionFile:
    predictionFile.write("TrackID,Predictor\n")
    with open(TEST_SCORE_FILE) as testHierarchy:
        test_list = read_lines(testHierarchy, 6)
        while test_list:
            # Columns: user|track|track_rating|album_rating|artist_rating|...
            test_list_raw = [item.strip("\n").split("|") for item in test_list]
            test_list = [row[1:4] for row in test_list_raw]
            user_id = test_list_raw[0][0]
            for i in range(6):
                test_list[i] = [test_list[i][0]] + [
                    int(x) if x != "None" else none_value for x in test_list[i][1:3]
                ]
            prediction_result = sort_list(test_list)
            for i, item in enumerate(prediction_result):
                track_id = test_list[i][0]
                predictionFile.write(f"{user_id}_{track_id},{item}\n")
            test_list = read_lines(testHierarchy, 6)
print("Finished, spend %.2f s" % (time.time() - start_time)) 