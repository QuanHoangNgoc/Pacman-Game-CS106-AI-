import pandas as pd
from collections import defaultdict

def remove_char_in_list(line):
    """
    Remove unwanted characters from a string.

    Args:
    line (str): Input string.

    Returns:
    str: String with unwanted characters removed.
    """
    for rm in ["[", "]", "'", ",", "(", ")"]:
        line = line.replace(rm, "")
    return line

file = open("defFn.txt", "r")
column_name = "def"
score_dict = defaultdict(list)

def post():
    """
    Process data from files and populate score dictionary.

    Returns:
    defaultdict: Score dictionary containing processed data.
    """
    global file, column_name, score_dict

    name_map_list = []
    score_dict["map"] = []
    NUM_TURN = 6 * 3

    for turn in range(NUM_TURN):
        line = file.readline()
        #####
        line = remove_char_in_list(file.readline())
        words = line.split()
        name_map = words[1].replace("Classic", "")
        name_algo = words[3][:3]
        #####
        scores = remove_char_in_list(file.readline()).split()
        wins = remove_char_in_list(file.readline()).split()
        scores = [int(float(x)) for x in scores]
        _wins = []
        for x in wins:
            if x == 'True':
                _wins.append(1)
            else:
                _wins.append(0)
        wins = _wins
        com = [x for x in zip(scores, wins)]
        #####
        if turn % 3 == 0:
            score_dict["map"] += [name_map] * 5
        score_dict[column_name + name_algo] += com
        name_map_list += [name_map]
        #####
        line = file.readline()
        line = file.readline()
    print(name_map_list)
    return score_dict

post()
file = open("scFn.txt", "r")
column_name = "score"
post()
file = open("myFn.txt", "r")
column_name = "my"
post()

# Print the length of each key in the score dictionary
for key in score_dict.keys():
    print(len(score_dict[key]))

# Create a DataFrame from the score dictionary
df = defaultdict(list)
for column_name in ['map', 'scoreExp', 'defExp', 'myExp', 'scoreMin', 'defMin', 'myMin', 'scoreAlp', 'defAlp', 'myAlp']:
    _column_name = column_name.replace("score", "sc")
    df[_column_name] = score_dict[column_name]

df = pd.DataFrame(df)

# Write the DataFrame to a LaTeX file
writer = open('long_table.txt', 'w')
writer.write(df.to_latex(longtable=True))
writer.close()
print(df)

###$----------------------------------------------
new_dict = defaultdict(list)
new_dict['mertric'] = ['meanAvgScore', 'meanAvgWinRate']

for col in df.columns:
    if col == 'map':
        continue

    cnt = -1
    s, w = [], []
    for i in range(6):
        scores, wins = [], []
        for j in range(5):
            cnt += 1
            print(df[col][cnt], col, cnt)
            scores.append(df[col][cnt][0])
            wins.append(df[col][cnt][1])
        s.append(sum(scores) / len(scores))
        w.append(sum(wins) / len(wins))
    new_dict[col] += [round(sum(s) / len(s), 2), round(sum(w) / len(w), 2)]

df2 = pd.DataFrame(new_dict)

# Write the new DataFrame to a LaTeX file
writer = open('long_table2.txt', 'w')
writer.write(df2.to_latex(longtable=True, float_format="%.2f"))
writer.close()
print(df2)
