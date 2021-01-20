import Funcs as f
import pandas as pd

############# Controller ###############
SITE = 1
# LOCATIONCAPACITY = [7000, 6000]  #scenario 1
LOCATIONCAPACITY = [496836, 426576] #scenario 2
HAVE_DATA = True
SCENARIO = 2

print("we are running the program on site {} with scenario {}".format(str(SITE), str(SCENARIO)))
################## Date preparation #######################
if not HAVE_DATA:
    f.data_preporation(site=SITE)

weeks, populations = f.read_data(site = SITE)


################## Initial optimization (step 1 & 2) ######################
df, forbidden_weeks, objective = f.optimizer(
    populations, weeks, site= SITE, location_capacity = LOCATIONCAPACITY[SITE], scenario = SCENARIO)

df.to_csv('saved/initial_results_for_site_{}_scenario_{}.csv'.format(str(SITE), str(SCENARIO)))
print('the optimized data frame has been saved')


######################### Parameter tuning ###########################
results = f.tuning(populations, weeks, LOCATIONCAPACITY[SITE], forbidden_weeks, df, objective, scenario = SCENARIO)

print('the objectives and the final forbidden dictionary are ')
for i in results.keys():
    print(i,':',results[i])


######################## final data results ############################

forbidden_weeks = results[min(list(results.keys()))]

df, objective = f.final(populations, weeks, LOCATIONCAPACITY[SITE], forbidden_weeks, SITE, SCENARIO)
df.to_csv('saved/final_results_for_site_{}_scenario_{}.csv'.format(str(SITE), str(SCENARIO)))

print('the objective is : ', objective)