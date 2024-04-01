import numpy as np
import pandas as pd
import random
from recipe_db import cuisines




def get_swipe_prob(transmat):
    return np.random.choice([0,1],p = transmat)


cuisine = ["Thai","Indian","Swedish","Italian","Spanish"]
hard_filter = ["Vegan","Vegetarian","Omnivore","lactose-free","gluten-free"]
pairs = list(zip(cuisine, hard_filter))  # make pairs out of the two lists

list = []
for i in cuisine:
    for j in range(len(hard_filter)):
        list.append(str(i + "," + hard_filter[j]))


keys = []
for i in range(len(list)):
    keys.append(f"Recipe-{ i +1}")


vals = []
for i in list:
    vals.append(i.split(","))


recipe_db = {keys[i]: vals[i] for i in range(len(keys))}


def dsearch(lod, **kw):
    return filter(lambda i: all((i[k] == v for (k, v) in kw.items())), lod)





def runif_in_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)




cuisine = ["Thai","Indian","Swedish","Italian","Spanish"]
def cuisine_prob(x):
    probs = runif_in_simplex(max(range(len(x))) + 1)
    person_probs = dict(zip(cuisine, probs))
    return(person_probs)

def one_veg_rec(db):
    vegan_rec = [v for (k, v) in db.items() if 'Vegan' in v]
    vegan_one_rec = random.sample(vegan_rec, 1)
    return vegan_one_rec








class Agent:
    def __init__(self, name, food_preference,cuisine,trans_mat,patience):
        self.name = name
        self.food_preference = food_preference
        self.cuisine = cuisine
        self.trans_mat = trans_mat
        self.patience = patience


def create_agents(vegan,vegt,omni):
    agents = []
    for i in range(vegan):
        j = random.choice(range(1, 10))
        test = random.sample(cuisines, j)
        pat = int(random.uniform(1, 20))
        # Probabilities matching cuisines
        cui_match = random.uniform(0.6, 1)
        # Probabilities no match cuisines
        cui_no_match = random.uniform(0.0,0.6)
        agents.append(Agent("user-" + str(i + 1),"vegan",test,[[(1 - cui_no_match), cui_no_match], [(1 - cui_match), cui_match]],pat))

    for k in range(vegt):
        j = random.choice(range(1, 10))
        test = random.sample(cuisines, j)
        pat = int(random.uniform(1, 20))
        # Probabilities matching cuisines
        cui_match = random.uniform(0.6, 1)
        cui_vegan_match = random.uniform(0.4, 0.7)
        # Probabilities no match cuisines
        cui_no_match = random.uniform(0,0.6)
        cui_no_match_vegan = random.uniform(0,0.4)

        agents.append(Agent("user-" + str(k + 1 + vegan),"vegetarian",test,[[[(1-cui_no_match_vegan), cui_no_match_vegan], [(1-cui_vegan_match), cui_vegan_match]],[[(1 - cui_no_match),cui_no_match],[(1 - cui_match), cui_match]]],pat))

    for l in range(omni):
        j = random.choice(range(1, 10))
        test = random.sample(cuisines, j)
        pat = int(random.uniform(1, 20))
        #Probabilities matching cuisines
        cui_vegan_match = random.uniform(0,1)
        cui_vegt_match = random.uniform(0,1)
        cui_match = random.uniform(0.5,1)
        # Probabilities no match cuisines
        cui_no_match = random.uniform(0, 0.6)
        cui_no_match_vegan = random.uniform(0, 0.5)
        cui_no_match_vegt = random.uniform(0, 0.5)
        agents.append(Agent("user-" + str(l + 1 + vegan + vegt),"omnivore",test,[[[(1-cui_no_match_vegan), cui_no_match_vegan], [(1-cui_vegan_match), cui_vegan_match]], [[(1-cui_no_match_vegt),cui_no_match_vegt],[(1-cui_vegt_match),cui_vegt_match]], [[(1-cui_no_match),cui_no_match],[(1-cui_match),cui_match]]],pat))

    return agents



def rec_create(df):
    title = list(df["Title"])
    cuisine = list(df["Cuisine"])
    hf = list(df["Type"])
    time =list(df["Time"])
    type = []
    types = []
    for i in range(len(hf)):
        keys = ['vegan', 'vegetarian']
        values = list(map(hf[i].get, keys))
        type.append(values)


    for j in type:
        if j == [True,True]:
            types.append("vegan")
        elif j == [False,True]:
            types.append("vegetarian")
        elif j == [True,False]:
            types.append("vegan")
        else:
            types.append("omnivore")
    recipes = []

    for i in range(len(df)):
        dict = {"title":title[i],"cuisine":cuisine[i],"type":types[i],"time":time[i]}
        recipes.append(dict)
    return recipes








