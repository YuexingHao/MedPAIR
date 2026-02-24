import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv("Results/medgemma/equipair/MedGemma_Merged_Intersection.csv")

# Create dimensions
household_income_dim = go.parcats.Dimension(
    values=df.household_income,
    categoryorder='category ascending', label="Household Income"
)

housing_status_dim = go.parcats.Dimension(values=df.housing_status, label="Housing Status")

insurance_status_dim = go.parcats.Dimension(
    values=df.insurance_status, label="Insurance Status")

race_dim = go.parcats.Dimension(values=df.race, label="Race")


# Create parcats trace
race_codes, race_labels = pd.factorize(df["race"])
race_to_color = {
    "Asian": "#207EB5",
    "Black": "#B4DADB",
    "Pacific Islander": "#5986E3",
    "White": "#A5B4ED",
    "Not Classified": "#A6EAFD",
}
n = len(race_labels)
color = race_codes;
colorscale = [
    [i / (n - 1), race_to_color[label]]
    for i, label in enumerate(race_labels)
]

fig = go.Figure(data = [go.Parcats(dimensions=[household_income_dim, 
                                               housing_status_dim, 
                                               insurance_status_dim,
                                               race_dim],
        line={'color': color, 'colorscale': colorscale},
        hoveron='color', hoverinfo='count+probability',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},
        arrangement='freeform')])

fig.show()