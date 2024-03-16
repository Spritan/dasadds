def cal_score(dif_angle):
    score = 0
    for key, values in dif_angle.items():
        if abs(values) <= 25:
            score +=1
    return score

def DROP_score(dif_angle):
    for key, value in list(dif_angle.items()):
        if abs(value) <= 25:
            dif_angle.pop(key)
    return dif_angle