
import pandas as pd
import random
data = pd.read_json("recipes.json",lines=True)






title = data["title"]
hf = data["filter"]
cuisine = data["cousine"]
time = data["time"]

d = {'Title':title,'Cuisine': cuisine, 'Type': hf,"Time":time}
df = pd.DataFrame(d)


filtered_df = df[df['Time'].notnull()]
cuisines = list(df["Cuisine"].unique())
cuisines = [i for i in cuisines if i is not None]


n = random.choice(range(1,10))

test = random.sample(cuisines,n)


def agent_generator(list,n):
    recipes_db = []
    cuisine_pref = []
    users = []
    #test = random.sample(cuisines, n)
    for j in range(n):
        i = random.choice(range(1, 10))
        test = random.sample(cuisines, i)
        cuisine_pref.append(test)
        users.append("User-" + str(j + 1))
    for l in range(n):
        dict = {"Name":users[l],"Cuisines": cuisine_pref[l],"Type":random.choice(["vegan","vegetarian","omnivore"])}
        recipes_db.append(dict)

    return(recipes_db)








