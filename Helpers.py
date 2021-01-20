from tqdm import tqdm
import pandas as pd
from datetime import timedelta
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def read():
    """
    :return: first dataset and GDU dataset

    df.columns = ['population', 'site', 'original_planting_date', 'early_planting_date',
       'late_planting_date', 'required_gdus', 'scenario_1_harvest_quantity',
       'scenario_2_harvest_quanitity']

    dfGDU.columns = ['index', 'date', 'site_0', 'site_1']
    """
    df = pd.read_csv('Dataset_1.csv')
    df['original_planting_date'] = pd.to_datetime(df['original_planting_date'])
    df['early_planting_date'] = pd.to_datetime(df['early_planting_date'])
    df['late_planting_date'] = pd.to_datetime(df['late_planting_date'])

    gdu_df = pd.read_csv('finalGDUs.csv')
    gdu_df['date'] = pd.to_datetime(gdu_df['date'])

    return df, gdu_df

def day_calculator(planting_date, gdu_df, req_gdu, site):
    """
    :param planting_date:
    :param gdu_df:
    :param req_gdu:
    :param site:
    :return: the exact date of harvest for the given planting date, based on required GDU
    """
    gdu_counter = 0

    while gdu_counter < req_gdu:

        gdu_of_day = gdu_df[site][gdu_df['date'] == planting_date]
        try:
            gdu_counter += float(gdu_of_day)
        except:
            pass
        planting_date += timedelta(days=1)

    return planting_date - timedelta(days=1)

def week_calculator(day):
    """
    :param day: a pandas datetime type of a date
    :return: the number of week past from starting_day
    """
    starting_day = pd.to_datetime('2020-01-05')
    temp = day - starting_day
    week = np.ceil(temp.days/7)
    if temp.days%7==0:
        return week +2
    return week+1

def harvest_changer_init(pop, week_condidates, df, weekly_harvest, scenario):

    # get the only row that is belong to the chosen population
    tempdf = df[df['population'] == pop]

    minimum = np.inf
    place_holder = []

    for weeks in week_condidates: # weeks : ( week number, planting date, harvest date)
        if weekly_harvest[weeks[0]] < minimum:
            minimum = weekly_harvest[weeks[0]]
            place_holder = weeks #keep the week that have less harvest quantity on it

    idx = tempdf.index

    if minimum == np.inf:
        raise Exception('the population {} is not in this site'.format(pop))

    if scenario == 1:
        weekly_harvest[place_holder[0]] += int(tempdf['scenario_1_harvest_quantity'])
    else:
        weekly_harvest[place_holder[0]] += int(tempdf['scenario_2_harvest_quanitity'])

    df['original_planting_date'].loc[idx] = place_holder[1]
    df['harvest_time'].loc[idx] = place_holder[2]

    return df, weekly_harvest


def loss(weekly_harvest, location_capacity):

    weekly_harvest = list(weekly_harvest.values())
    weekly_harvest = np.array(list(filter(lambda a: a != 0, weekly_harvest)))

    location_capacity = np.ones(len(weekly_harvest)) * location_capacity

    su = np.abs(location_capacity - weekly_harvest)

    return np.sum(su)

def loss2(weekly_harvest):
    weekly_harvest = list(weekly_harvest.values())
    weekly_harvest = np.array(list(filter(lambda a: a != 0, weekly_harvest)))
    mean = np.mean(weekly_harvest)

    print('number of weeks: {}\nthe mean: {}\nthe max: {}\nthe median: {}\n######'.format(
        len(weekly_harvest),mean,np.max(weekly_harvest),np.median(weekly_harvest)))

    mean = np.ones(len(weekly_harvest)) * mean

    objective = np.abs(mean - weekly_harvest)

    return np.sum(objective)


def refiner(df, weekly_harvest, objective, populations, lop, location_capacity, scenario):
    # we want to move from right to left and left to right simultaneously
    left2right = np.array(list(weekly_harvest.keys()))
    right2left = left2right[-1::-1]

    # the weeks that could be delete and not hurting the objective function saves hare
    forbidden_weeks = []

    weekly_harvest = {i: 0 for i in weekly_harvest.keys()}

    for left, right in tqdm(zip(left2right,right2left)):
        ok = True
        temp_weekly_harvest = weekly_harvest.copy()
        temp_df = df.copy()
        for pop in lop:
            week_candidates = populations[pop]

            temp_df, temp_weekly_harvest, ok = harvest_changer_refine(
                pop, week_candidates, temp_df, temp_weekly_harvest.copy(), forbidden_weeks.copy(), left, scenario)
            if not ok:
                break

        if ok:
            if loss(temp_weekly_harvest, location_capacity) < objective:
                forbidden_weeks.append(left)
                objective = loss(temp_weekly_harvest, location_capacity)
                df = temp_df.copy()
                plotter(df, objective)
            else:
                print('objective before adding {} : {}\nbut after: {}'.format(
                    left, objective, loss(temp_weekly_harvest, location_capacity)
                ))
        else:
            print("couldn't add", left)

        temp_weekly_harvest = weekly_harvest.copy()
        ok = True
        temp_df = df.copy()
        for pop in lop:
            week_candidates = populations[pop]

            temp_df, temp_weekly_harvest, ok = harvest_changer_refine(
                pop, week_candidates, temp_df, temp_weekly_harvest.copy(), forbidden_weeks.copy(), right, scenario)

            if not ok:
                break

        if ok:
            if loss(temp_weekly_harvest, location_capacity) < objective:
                objective = loss(temp_weekly_harvest, location_capacity)
                forbidden_weeks.append(right)
                df = temp_df.copy()
                plotter(df, objective)
            else:
                print('objective before adding {} : {}\nbut after: {}'.format(
                    right, objective, loss(temp_weekly_harvest, location_capacity)
                ))
        else:
            print("couldn't add", right)

        print("the forbidden weeks are : ", forbidden_weeks)

    return df, forbidden_weeks, objective


def harvest_changer_refine(pop, week_candidates, df, weekly_harvest, forbidden_weeks, test, scenario):

    if test in forbidden_weeks:
        return False, False, False

    tempdf = df[df['population'] == pop]
    forbidden_weeks += [test]

    minimum = np.inf
    place_holder = []

    for weeks in week_candidates:
        if weeks[0] in forbidden_weeks:
            continue
        if weekly_harvest[weeks[0]] < minimum:
            minimum = weekly_harvest[weeks[0]]
            place_holder = weeks

    if minimum == np.inf:
        return False, False, False

    idx = tempdf.index

    if scenario == 1:
        weekly_harvest[place_holder[0]] += int(tempdf['scenario_1_harvest_quantity'])
    else:
        weekly_harvest[place_holder[0]] += int(tempdf['scenario_2_harvest_quanitity'])

    df['original_planting_date'].loc[idx] = place_holder[1]
    df['harvest_time'].loc[idx] = place_holder[2]

    return df, weekly_harvest, True


def tunner(df, weekly_harvest, objective, populations, lop, location_capacity, forbidden_weeks, scenario):
    results = {}
    # see which weeks could be replaced to get a better objective
    other_weeks = [i for i in weekly_harvest.keys() if i not in forbidden_weeks]

    weekly_harvest = {i: 0 for i in weekly_harvest.keys()}
    for week in tqdm(other_weeks):
        for i in range(len(forbidden_weeks)):

            temp_weekly_harvest = weekly_harvest.copy()
            temp_df = df.copy()
            ok = True
            for pop in lop:
                week_candidates = populations[pop]
                temp_df, temp_weekly_harvest, ok = harvest_changer_tuning(
                    pop, week_candidates, temp_df, temp_weekly_harvest.copy(), forbidden_weeks.copy(), week, i, scenario)

                if not ok:
                    break
            if ok:
                if loss(temp_weekly_harvest, location_capacity) < objective:
                    objective = loss(temp_weekly_harvest, location_capacity)
                    print('instead of week {} we put week {} and got objective {}'.format(
                        forbidden_weeks[i], week, objective
                    ))
                    temp_forbidden_weeks = [k for j, k in enumerate(forbidden_weeks) if j != i]
                    results[objective] = temp_forbidden_weeks + [week]
    return results


def harvest_changer_tuning(pop, week_candidates, df, weekly_harvest, forbidden_weeks, test, i, scenario):
    tempdf = df[df['population'] == pop]

    del forbidden_weeks[i]
    forbidden_weeks += [test]

    minimum = np.inf
    place_holder = []

    for weeks in week_candidates:
        if weeks[0] in forbidden_weeks:
            continue
        if weekly_harvest[weeks[0]] < minimum:
            minimum = weekly_harvest[weeks[0]]
            place_holder = weeks

    if minimum == np.inf:
        return False, False, False

    idx = tempdf.index

    if scenario == 1:
        weekly_harvest[place_holder[0]] += int(tempdf['scenario_1_harvest_quantity'])
    else:
        weekly_harvest[place_holder[0]] += int(tempdf['scenario_2_harvest_quanitity'])

    df['original_planting_date'].loc[idx] = place_holder[1]
    df['harvest_time'].loc[idx] = place_holder[2]

    return df, weekly_harvest, True

def final(df, weekly_harvest, populations, lop, location_capacity, forbidden_weeks, scenario):

    for pop in tqdm(lop):
        week_candidates = populations[pop]
        df, weekly_harvest, ok = harvest_changer_final(
            pop, week_candidates, df, weekly_harvest, forbidden_weeks, scenario)

        if not ok:
            break
    if ok:
        objective = loss(weekly_harvest, location_capacity)

    return df, objective


def harvest_changer_final(pop, week_candidates, df, weekly_harvest, forbidden_weeks, scenario):
    tempdf = df[df['population'] == pop]

    minimum = np.inf
    place_holder = []

    for weeks in week_candidates:
        if weeks[0] in forbidden_weeks:
            continue
        if weekly_harvest[weeks[0]] < minimum:
            minimum = weekly_harvest[weeks[0]]
            place_holder = weeks

    if minimum == np.inf:
        return False, False, False

    idx = tempdf.index

    if scenario == 1:
        weekly_harvest[place_holder[0]] += int(tempdf['scenario_1_harvest_quantity'])
    else:
        weekly_harvest[place_holder[0]] += int(tempdf['scenario_2_harvest_quanitity'])

    df['original_planting_date'].loc[idx] = place_holder[1]
    df['harvest_time'].loc[idx] = place_holder[2]

    return df, weekly_harvest, True

def plotter(site0, objective):
    site0_sc1_weekly = {}

    # the start time to count the weeks
    date = pd.to_datetime("2020-01-05")
    date_max = site0['harvest_time'].max()

    pl = False
    week_counter = 2
    while True:
        temp = site0[(site0['harvest_time'] >= date) & (site0['harvest_time'] < date + timedelta(days=7))]

        if len(temp) > 0:
            pl = True

        if pl:
            site0_sc1_weekly[week_counter] = temp['scenario_1_harvest_quantity'].sum()

        if date > date_max:
            break

        date += timedelta(days=7)
        week_counter += 1

    title = 'site 0 scenario 1'

    names = [str(i) for i in site0_sc1_weekly.keys()]
    values = site0_sc1_weekly.values()

    fig, ax = plt.subplots(figsize=(20, 10), constrained_layout=True)
    ax.bar(names, values)
    ax.set(xlabel='weeks', ylabel='Quantity', title=title)
    ax.set_title('the objective {}'.format(objective))

    plt.xticks(rotation=90)
    plt.show()

def refiner1(df, weekly_harvest, objective, populations, lop, location_capacity, scenario):
    # we want to move from right to left and left to right simultaneously

    # left2right = {k: v for k, v in sorted(weekly_harvest.items(), key=lambda item: item[1], reverse=True)}
    # left2right = np.array(list(weekly_harvest.keys()))
    # left2right = list(left2right.keys())

    # objective = abs(max(list(weekly_harvest.values())) - np.median(list(weekly_harvest.values())))
    objective = loss2(weekly_harvest)
    # the weeks that could be delete and not hurting the objective function saves hare
    forbidden_weeks = []

    print('before refining the objective is ',objective)

    sequence = {k: v for k, v in sorted(weekly_harvest.items(), key=lambda item: item[1])}
    sequence = list(sequence.keys())

    for left in sequence:
        ok = True
        weekly_harvest = {i: 0 for i in weekly_harvest.keys()}
        temp_weekly_harvest = weekly_harvest.copy()
        temp_df = df.copy()
        for pop in lop:
            week_candidates = populations[pop]

            temp_df, temp_weekly_harvest, ok = harvest_changer_refine(
                pop, week_candidates, temp_df, temp_weekly_harvest.copy(), forbidden_weeks.copy(), left, scenario)
            if not ok:
                break

        if ok:
            temp_objective = loss2(temp_weekly_harvest)
            if temp_objective < objective:
                forbidden_weeks.append(left)
                objective = temp_objective
                print(objective)
                df = temp_df.copy()
                weekly_harvest = temp_weekly_harvest
                # plotter(df, objective)
        else:
            print("couldn't add", left)

        print("the forbidden weeks are : ", forbidden_weeks)

    return df, forbidden_weeks, objective


def next_step(weekly_harvest):

    original_weekly = weekly_harvest.copy()

    weekly_harvest = list(weekly_harvest.values())
    weekly_harvest = np.array(list(filter(lambda a: a != 0, weekly_harvest)))

    mean = np.mean(weekly_harvest)
    mean = np.ones(len(weekly_harvest)) * mean

    nex = np.abs(weekly_harvest - mean)

    nex = np.argmax(nex)

    step = [i for i in original_weekly.keys() if original_weekly[i] == weekly_harvest[nex]]

    return step[0]