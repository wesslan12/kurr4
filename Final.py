##############################  IMPORT LIBRARIES ###############################
import random
import pandas as pd
import numpy as np
from functions import dsearch,get_swipe_prob,create_agents
from recipe_db import filtered_df

################################################################################
##############################  DATA WRANGLING #################################

# Loading database of recipes from Kurr. Database is filtered to only have the
#columns of interest (see recipe_db for more info)
data = filtered_df
#Removing recipes which are not complete
df = data.loc[(data['Title'] != "Up and running")
              & (data['Title'] != "hej")
              & (data['Title'] != "Test")
              & (data['Title'] != "test")
              & (data['Title'] != "test2")]

#Function that creates a dictionary from the recipe database.The dictionary contains the keys: "title","cuisine",
# "type", and "time".
def rec_create(df):
    title = list(df["Title"])
    cuisine = list(df["Cuisine"])
    hf = list(df["Type"])
    time =list(df["Time"])
    type = []
    types = []
#For each dictionary of filters, we select the filters vegan and vegetarian (could be extended to contain other filters).
    for i in range(len(hf)):
        keys = ['vegan', 'vegetarian']
        values = list(map(hf[i].get, keys))
        type.append(values)

#Then for each combination of vegan/vegetarian we append either "vegan","vegetarian", or "omnivore" to the final dictionary
    for j in type:
        if j == [True,True]: # If vegan = true and "vegetarian" = True append "vegan
            types.append("vegan")
        elif j == [False,True]: # If vegan = true and "vegetarian" = False append "vegan (should be 0 but found 1 recipe)
            types.append("vegetarian")
        elif j == [True,False]:# If vegan = False and "vegetarian" = True append "vegetarian"
            types.append("vegan")
        else: # The rest are omnivore recipes
            types.append("omnivore")
    recipes = []

    for i in range(len(df)):
        dict = {"title":title[i],"cuisine":cuisine[i],"type":types[i],"time":time[i]}
        recipes.append(dict)
    return recipes

recept = rec_create(df) #Final recipe object
vegan_rec = list(dsearch(recept,type="vegan")) #Only vegan recipes
vegetarian_rec = list(dsearch(recept,type="vegetarian"))#Only vegetarian recipes

######################################################################################
##############################  GENERATE AGENTS ######################################
#Initialize the agents (see function.py for details of how they were defined)

agents = create_agents(10,10,10)

######################################################################################
##############################  SIMULATION ###########################################

def simulation(agents,n):
    data = []
    for i in range(n):
        for agent in agents:
                if agent.food_preference == "vegan":
                    veg_rec = random.choice(vegan_rec)
                    if veg_rec["cuisine"] in agent.cuisine:
                        agent_swipe = get_swipe_prob(agent.trans_mat[1])
                        if agent_swipe == 0:
                            agent.patience = agent.patience - 1
                        data.append((agent.name,str(i + 1),
                                    agent.food_preference,
                                    agent.cuisine,
                                    veg_rec["title"],
                                    veg_rec["cuisine"],
                                    veg_rec["type"],
                                    veg_rec["time"],
                                    agent_swipe))
                    else:
                        agent_swipe = get_swipe_prob(agent.trans_mat[0])

                        data.append((agent.name,str(i + 1),
                                    agent.food_preference,
                                    agent.cuisine,
                                    veg_rec["title"],
                                    veg_rec["cuisine"],
                                    veg_rec["type"],
                                    veg_rec["time"],
                                    agent_swipe))
                elif agent.food_preference == "vegetarian":
                    vege_rec = random.choice(vegan_rec + vegetarian_rec)
                    if vege_rec["cuisine"] in agent.cuisine:
                        if vege_rec["type"] == "vegan":
                            agent_swipe = get_swipe_prob(agent.trans_mat[0][1])
                            data.append((agent.name,str(i + 1),
                                    agent.food_preference,
                                    agent.cuisine,
                                    vege_rec["title"],
                                    vege_rec["cuisine"],
                                    vege_rec["type"],
                                    vege_rec["time"],
                                    agent_swipe))


                        elif vege_rec["type"] == "vegetarian":
                            agent_swipe = get_swipe_prob(agent.trans_mat[1][1])
                            data.append((agent.name, str(i + 1),
                                     agent.food_preference,
                                     agent.cuisine,
                                     vege_rec["title"],
                                     vege_rec["cuisine"],
                                     vege_rec["type"],
                                     vege_rec["time"],
                                     agent_swipe))
                    else:
                        agent_swipe = get_swipe_prob(agent.trans_mat[0][0])
                        data.append((agent.name, str(i + 1),
                                 agent.food_preference,
                                 agent.cuisine,
                                 vege_rec["title"],
                                 vege_rec["cuisine"],
                                 vege_rec["type"],
                                 vege_rec["time"],
                                 agent_swipe))
                #print('{} - {}: {}'.format(agent.name, vege_rec['title'],'swipe right' if agent_swipe == 1 else 'swipe left'))
                elif agent.food_preference == "omnivore":
                    recipe = random.choice(recept)
                    if recipe["cuisine"] in agent.cuisine:
                        if recipe["type"] == "vegan":
                            agent_swipe = get_swipe_prob(agent.trans_mat[0][1])
                            data.append((agent.name, str(i + 1),
                                     agent.food_preference,
                                     agent.cuisine,
                                     recipe["title"],
                                     recipe["cuisine"],
                                     recipe["type"],
                                     recipe["time"],
                                     agent_swipe))
                        elif recipe["type"] == "vegetarian":
                            agent_swipe = get_swipe_prob(agent.trans_mat[1][1])
                            data.append((agent.name, str(i + 1),
                                     agent.food_preference,
                                     agent.cuisine,
                                     recipe["title"],
                                     recipe["cuisine"],
                                     recipe["type"],
                                     recipe["time"],
                                     agent_swipe))

                        elif recipe["type"] == "omnivore":
                            agent_swipe = get_swipe_prob(agent.trans_mat[2][1])
                            data.append((agent.name, str(i + 1),
                                     agent.food_preference,
                                     agent.cuisine,
                                     recipe["title"],
                                     recipe["cuisine"],
                                     recipe["type"],
                                     recipe["time"],
                                     agent_swipe))
                    else:
                        agent_swipe = get_swipe_prob(agent.trans_mat[0][0])
                        data.append((agent.name, str(i + 1),
                                 agent.food_preference,
                                 agent.cuisine,
                                 recipe["title"],
                                 recipe["cuisine"],
                                 recipe["type"],
                                 recipe["time"],
                                 agent_swipe))
                #print('{} - {}: {}'.format(agent.name,recipe['title'], 'swipe right' if agent_swipe == 1 else 'swipe left'))


    cols = ["User-ID","RunID","Agent_diet","Agent_cuisine","Recipe","Recipe_cuisine","Recipe_diet","Time","Choice"]
    result = pd.DataFrame(data,columns=cols)
    return result

results = simulation(agents,5)









