from functions import create_agents
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from recipe_db import cuisines
import plotly.graph_objects as go
import altair as alt
import time
from sklearn.cluster import KMeans
from Final import simulation
from matplotlib.patches import ConnectionPatch
import gower
from sklearn.cluster import DBSCAN

import numpy as np


@st.cache(suppress_st_warning=True)
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
def get_chart(data):
    hover = alt.selection_single(
        fields=["RunID"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Number of Right-swipes per iteration")
        .mark_line()
        .encode(
            x="RunID",
            y="Choice",
            color="Agent_diet",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="RunID",
            y="Choice",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("RunID", title="Iteration"),
                alt.Tooltip("Choice", title="Number of right swipes"),
            ],
        )
        .add_selection(hover)
    )
    return (lines + points + tooltips).interactive()


def matches(df):
    x = (results.apply(lambda row: row['Recipe_cuisine'] in row['Agent_cuisine'], axis='columns'))
    return x.astype(int)

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page',['Home','Simulation','Prediction','Visualization']) #two pages
############################################# HOME PAGE ################################################################
if app_mode == "Home":
    st.title("An agent-based model of user interaction on the app 'Kurr'")
    st.title("Step 1: Generate users")
    st.subheader("Dietary preferences")
    st.write("A user-specified number of different agents are generated with different dietary preferences.")
    st.markdown("The dietary preferences (or filters) are *vegan*, *vegetarian*, and *omnivore*.")
    st.write("Based on these dietary preferences, the agents interact differently with the recipes on the app. Vegans interact only with vegan recipes,vegetarians interact with both vegan and vegetarian recipes, while omnivores interact with all of them")
    st.subheader("Cuisine preferences")
    st.write("Each agent also have a random number of cuisine preferences. The cuisine preferences are based on the recipe database provided by 'Kurr'.")
    st.write("In total there are 18 different cuisines. For each agent we draw a random number (with equal probability) between 1 and 18. This number is then used to decide how many different cuisines the agent prefers.")
    st.write("For example, if the result of the random number generator is 3, we sample 3 cuisines from the full list of cuisines and assign those to the agent. Resulting in the following structure:")
    df = pd.DataFrame({"User":"User-1","Cuisines": ['Thai,Indian,Husmanskost'],"Filter":"Vegan"})
    fig = go.Figure(data=[go.Table(header=dict(values=['User', 'Cuisine',"Filter"]),
                 cells=dict(values=["User-1",'Thai,Indian,Husmanskost',"Vegan"]))
                     ])
    st.plotly_chart(fig)



#st.table(df)
    st.title("Step 2: Simulating interaction")
    st.subheader("Steps")
    st.write("At each step of the simulation, the agents are presented with a recipe and decides to either 'swipe left' or 'swipe right' on the recipe. The likelihood of 'swiping right' is determined by two factors.")
    st.markdown("- **Base probability:** Each agent are assigned a vector of probabilities that represent the likelihood of a left-or right swipe. The probability vector is constant in each of the agent types. Meaning, all vegans have the same base probability of liking a recipe")
    st.markdown("- **Conditional probability:** As mentioned earlier, the recipe might also match with the agents preferences of cuisine. If it does match then the agents have a higher probability to 'swipe right' on the recipe than if there were no match")
    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
    }
    </style>
    ''', unsafe_allow_html=True)
    st.write("To exemplify, look at the following table of probabilities:")
    st.latex(r'''
    \begin{matrix}
    & Left & Right \\
    Base & 0.5 & 0.5 \\
    Conditional & 0.2 & 0.8
    \end{matrix}''')
    st.write("In the base-probability case (corresponding to no match between agent cuisine and recipe cuisine), the probability to swipe left or right is 0.5 respectively. In the conditional probability case (corresponding to a match between recipe cuisine and agent cuisine preference), the probability to swipe right is higher. We implemented this feature to create a more realistic model. The probabilities can be changed using the probability sliders in the sidebar")
    st.write("To add further complexity to the model, the base -and conditional probabilities are different depending on the dietary preferences. For instance, a vegetarian most likely eats vegan food as well but prefers vegetarian because s/he likes dairy products. To capture such phenomena we added another vector of probabilities for vegetarians. The first vector represents the probabilities of left/right swipe on vegan recipes and the second vector represent the same probabilities but for vegetarian recipes.The same procedure was implemented for omnivores as well.")
    st.write("The probabilities of liking a given recipe for all the agent types are represented in the matrix below:")
    st.latex(r'''
        \begin{matrix}
        & & Vegan & & Vegetarian & & Omnivore &\\
        & &Left & Right & Left & Right & Left & Right\\
        Vegan & Base & 0.5 & 0.5 & & & &\\
        & Conditional & 0.3 & 0.7 & & & &\\
        Vegetarian & Base & 0.6 & 0.4 & 0.4 & 0.6 & &\\
        & Conditional & 0.5 & 0.5 & 0.3 & 0.7 & &\\
        Omnivore & Base & 0.8 & 0.2 & 0.7 & 0.3 & 0.6 & 0.4\\
        & Conditional & 0.7 & 0.3 & 0.6 & 0.4 & 0.1 & 0.9
        \end{matrix}''')
    st.title("Interacting with the model")
    st.subheader("Simulation page")
    st.markdown("**Step 1:** Specify model parameters. Use the sliders in the left sidebar to specify the number of iterations and agents.")
    st.markdown("**Step 2:** Hit the 'Generate agents' button. This will generate the agents and some descriptive statistics and plots of the population")
    st.markdown("**Step 3:** Hit the 'Run simulation' button. This will start the simulation and end at the specified maximum number of steps")
    st.write("When the simulation is finished, the user may choose to run prediction models and visualization by changing the 'Selected Page' to any of these")

############################################# SIMULATION PAGE ###############################################################
if 'result' not in st.session_state:
    st.session_state.result = None
if app_mode =="Simulation":


    st.sidebar.title("Model specifications")
    st.sidebar.subheader("Number of Steps (Iterations)")
    n = st.sidebar.slider("N",0,500,100,step=50)
    st.sidebar.subheader("Number of agents")
    n_vegan = st.sidebar.slider("Vegan",0,1000,100,step=50)
    n_vegetarian = st.sidebar.slider("Vegetarian",0,1000,100,step=50)
    n_omnivore = st.sidebar.slider("Omnivore",0,1000,100,step=50)
    #st.sidebar.button("Generate users")
    agents = create_agents(n_vegan, n_vegetarian, n_omnivore)
    sim = simulation(agents, n)

    if st.sidebar.button("Generate users"):
        # CREATING AGENTS



        # GETTING THEIR DIETARY PREFERENCES
        cuisines3 = []
        for agent in agents:
            cuisines3.append(agent.cuisine)
        res = []
        for cui in cuisines:
            res.append([cui, sum(x.count(cui) for x in cuisines3)])

        categories = []
        count = []
        for i in range(len(res)):
            categories.append(res[i][0])
            count.append(res[i][1])
        ##################################################################
        # PLOTTING
        fig, ax = plt.subplots()

        bar_labels = ['red', 'blue', '_red', 'orange']
        bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
        ax.bar(categories, count, )
        ax.set_title("Agent cuisine preferences")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        progress_bar = st.progress(0)
        for i in range(n):
            time.sleep(0.01)
            progress_bar.progress((i + 1) / n)
        st.success('Users generated!', icon="✅")

    if st.sidebar.button("Run simulation"):
        try:
            sim = simulation(agents, n)
            progress_bar = st.progress(0)
            for i in range(n):
                time.sleep(0.01)
                progress_bar.progress((i + 1) / n)
            st.success('Simulation complete!', icon="✅")
            results = sim
        finally:
            st.session_state.result = results




        #VISUALIZATION



        match = matches(results)

        results["Match"] = match
        #st.write(results)
        st.title("Descriptive statistics")
        data_csv = convert_df(results)
        st.download_button(label="Download Data",data = data_csv,file_name="Simulation-results.csv")

        groups = ["Vegan","Vegetarian","Omnivore"]

        #newdf2 = results.groupby(['Agent_diet'])['Choice'].mean()
        x1 = results[results["Agent_diet"] == "vegan"]["Choice"]
        x2 = results[results["Agent_diet"] == "vegetarian"]["Choice"]
        x3 = results[results["Agent_diet"] == "omnivore"]["Choice"]




        pie = pd.Series((x1.mean(),x2.mean(),x3.mean()))

        ############################################################################################################
        # make figure and assign axis objects
        fig2, (ax2,ax3) = plt.subplots(1,2,figsize = (9,5))
        fig2.subplots_adjust(wspace=0)
        x4 = [results["Choice"].mean(), (1 -results["Choice"].mean())]
        # pie chart parameters
        labels_x4 = ["Swipe right","Swipe left"]
        explode = [0.1,0]

        # rotate so that first wedge is split by the x-axis
        angle = -180 * x4[0]
        wedges, *_ = ax2.pie(x4,autopct='%1.1f%%', startangle=angle,
                     labels=labels_x4,explode = explode)
        # bar chart parameters
        bottom = sum(pie)
        width = .2

        # Adding from the top matches the legend.
        for j, (height, label) in enumerate(reversed([*zip(pie, groups)])):
            bottom -= height
            bc = ax3.bar(0, height, width, bottom=bottom, color='C0', label=label,
                         alpha=0.1 + 0.25 * j)
            ax3.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

        ax3.set_title('Proportion based on diet')
        ax3.legend()
        ax3.axis('off')
        ax3.set_xlim(- 2.5 * width, 2.5 * width)

        # use ConnectionPatch to draw lines between the two plots
        theta1, theta2 = wedges[0].theta1, wedges[0].theta2
        center, r = wedges[0].center, wedges[0].r
        bar_height = sum(pie)

        # draw top connecting line
        x = r * np.cos(np.pi / 180 * theta2) + center[0]
        y = r * np.sin(np.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax3.transData,
                              xyB=(x, y), coordsB=ax2.transData)

        con.set_color([0, 0, 0])
        con.set_linewidth(2)
        ax3.add_artist(con)

        # draw bottom connecting line
        x = r * np.cos(np.pi / 180 * theta1) + center[0]
        y = r * np.sin(np.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax3.transData,
                              xyB=(x, y), coordsB=ax2.transData)
        con.set_color([0, 0, 0])
        ax3.add_artist(con)
        con.set_linewidth(2)
        st.pyplot(fig2)

        ############################################################################################################
        rec_diet_df = results.groupby(["Agent_diet","Recipe_diet","Match"], as_index=False)['Choice'].sum()
        rec_diet_df["Match"] = np.where(rec_diet_df["Match"] == 0, "No match", "Match")




        bar_chart = alt.Chart(rec_diet_df).mark_bar().encode(
            x= alt.X("Match:N",title=None),
            y= alt.Y("Choice:Q",title = "Number of 'right-swipes'"),
            color="Recipe_diet:N",
            column = alt.Column("Agent_diet:O",title=None),
            tooltip=[
                alt.Tooltip("Match", title="Matching recipe"),
                alt.Tooltip("Recipe_diet", title="Recipe diet"),
                alt.Tooltip("Choice", title="Number of right swipes"),
            ]

        ).properties(title = "CHICKIIIES",width = 160,height=100)


        st.altair_chart(bar_chart)
        ############################################################################################################
        # Reformatting the data
        results['RunID'] = results['RunID'].astype(int)
        df1 = results.groupby(['RunID',"Agent_diet"], as_index=False)['Choice'].sum()
        #st.line_chart(time_data)
        ############################################################################################################



        chart = get_chart(df1)
        st.altair_chart(chart,use_container_width=True)
        st.write(results)
columns = ["Agent_diet","Match","Recipe_cuisine","Time","Choice"]
if app_mode =="Prediction":
    mode = st.sidebar.radio("Mode", ["EDA", "Clustering"])
    st.markdown("<h1 style='text-align: center; color: #ff0000;'>Analysis</h1>", unsafe_allow_html=True)
    st.markdown("# Mode: {}".format(mode), unsafe_allow_html=True)












