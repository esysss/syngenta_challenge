from tqdm import tqdm
import pandas as pd
from datetime import timedelta
import pickle
import numpy as np
import matplotlib.pyplot as plt
import Helpers as H
plt.style.use('ggplot')

def data_preporation(site):
    """
    :param site: site 0 or 1
    :return: a pickle file including two dictionaries including all possibilities that a population
    can be planted and its harvest week. second dictionary have all the possibilities of every week in
    populations
    """
    df, gdu_df = H.read()
    df = df[df['site'] == site]

    weeks = {i: [] for i in range(100)}
    populations = {i: [] for i in df['population']}

    for i in tqdm(range(len(df))):

        start = df['early_planting_date'].iloc[i]
        end = df['late_planting_date'].iloc[i]
        req_gdu = df['required_gdus'].iloc[i]
        site_ = "site_{}".format(str(site))
        population = df['population'].iloc[i]

        lastweek = ""
        while start <= end:
            harvest_day = H.day_calculator(start, gdu_df, req_gdu, site_)
            week = H.week_calculator(harvest_day)

            if week != lastweek:
                weeks[week].append((population, start, harvest_day))
                populations[population].append((week, start, harvest_day))
                lastweek = week
            start += timedelta(days=1)

    theFile = open("network{}.p".format(str(site)), "wb")  # it says to write in bite
    pickle.dump((weeks, populations), theFile)
    theFile.close()


def read_data(site):
    """
    Read the pickle data made by data preporation function
    :param site:
    :return:
    """
    pickleIN = open("saved/network{}.p".format(str(site)), "rb")  # says read it to bite
    (weeks, populations) = pickle.load(pickleIN)
    pickleIN.close()

    #delete the weeks that don't have any harvest on them (harvest = 0)
    weeks = {i: weeks[i] for i in weeks.keys() if len(weeks[i]) > 0}

    return weeks, populations

def optimizer(populations, weeks, site, location_capacity, scenario):
    df, _ = H.read()
    df['original_planting_date'] = 10 #just for the heck of it!
    df['harvest_time'] = 10 #just for the heck of it!
    df = df[df['site'] == site]

    # we sort the population from their posibility to be harvest to how many weeks
    # that means if a population only can be harvest in one week, it goes first
    # so the priority is the populations with less harvest week choice.
    lens_of_populations = {i: len(populations[i]) for i in populations.keys()}
    lop = sorted(lens_of_populations, key=lambda k: lens_of_populations[k], reverse=False)

    #Reset the harvests in the weeks to count how much harvest do we have in each week
    weekly_harvest = {i: 0 for i in weeks.keys()}

    # initial scheduling
    print('initial scheduling')
    for pop in tqdm(lop):
        week_candidates = populations[pop]

        df, weekly_harvest = H.harvest_changer_init(pop, week_candidates, df, weekly_harvest, scenario)

    if scenario == 2:
        location_capacity = int(location_capacity/len(weekly_harvest))
    objective = H.loss(weekly_harvest, location_capacity)

    # refine the scheduling
    print('refining the scheduling (this might take a few minutes ...)')

    df, forbidden_weeks, objective = H.refiner1(df, weekly_harvest, objective, populations, lop, location_capacity, scenario)

    return df, forbidden_weeks, objective

################################### parameter tuning ################################
def tuning(populations, weeks, location_capacity, forbidden_weeks, df, objective, scenario):
    # sore these as well
    lens_of_populations = {i: len(populations[i]) for i in populations.keys()}
    lop = sorted(lens_of_populations, key=lambda k: lens_of_populations[k], reverse=False)

    #reset weekly harvest
    weekly_harvest = {i: 0 for i in weeks.keys()}

    print('initial tuning')
    results = H.tunner(df, weekly_harvest, objective, populations, lop, location_capacity, forbidden_weeks, scenario)

    return results

################################### final results and testing #######################

def final(populations, weeks, location_capacity, forbidden_weeks, site, scenario):
    df, _ = H.read()
    df['original_planting_date'] = 10  # just for the heck of it!
    df['harvest_time'] = 10  # just for the heck of it!
    df = df[df['site'] == site]

    # sore these as well
    lens_of_populations = {i: len(populations[i]) for i in populations.keys()}
    lop = sorted(lens_of_populations, key=lambda k: lens_of_populations[k], reverse=False)

    # reset weekly harvest
    weekly_harvest = {i: 0 for i in weeks.keys()}

    print('initial tuning')
    df, objective = H.final(df, weekly_harvest, populations, lop, location_capacity, forbidden_weeks, scenario)

    return df, objective