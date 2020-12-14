"""
Name: Elise Pereira
CS230: Section SN4
Data: Ride Share
URL: Link to your web application online (see extra credit)

Description: This program allows the user to plot their starting destination and ending destination on a map and compare the average
price between Uber and Lyft. Also, there is a grouped bar chart that the user can choose different cab types
and compare number of rides, price per mile, and price. Lastly, there is a scatter chart showing the relationship between
price and distance."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pydeck as pdk
import datetime

MAPKEY = "pk.eyJ1IjoiZWxpc2VwZXJlaXJhIiwiYSI6ImNraWtxYTE2cjBjZTAycnFtbXhncW90Y2kifQ.KqVkn7iZ12u91OI_XSOybA"

def grouped_bar(data, title, ylabel, xlabel, legend):
    data.plot.bar()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend)
    st.pyplot(plt)

def read_data(csv, columns):
    # reading data file into data frame and selecting necessary columns
    datafile = csv
    df = pd.read_csv(datafile)
    df = df[columns]
    return df


def pivot_table(dataframe, index, columns, values, aggfunc=np.mean):
    table = pd.pivot_table(dataframe, index=index, columns=columns, values=values, aggfunc=aggfunc)
    return table

def scatterplot(x,y, label):
    plt.scatter(x, y, color='r', label=label)
    return plt

def main():
    df = read_data("ridesharesample.csv", ["datetime", "source", "destination",
                                           "cab_type", "name", "price", "distance", "latitude",
                                           "longitude", "sunriseTime", "sunsetTime", "timestamp"])

    # From google because data was not accurate
    coordinates = {"Backbay": [42.3503, -71.0810],
                   "Beacon Hill": [42.3588, -71.0707],
                   "Boston University": [42.3505, -71.1054],
                   "Fenway": [42.3467, -71.0972],
                   "Financial District": [42.3559, -71.0550],
                   "Haymarket Square": [42.3638, -71.0585],
                   "North End": [42.3647, -71.0542],
                   "North Station": [42.3661, -71.0631],
                   "Northeastern University": [42.3398, -71.0892],
                   "South Station": [42.3519, -71.0552],
                   "Theatre District": [42.3519, -71.0643],
                   "West End": [42.3644, -71.0661]
                   }

    # convert timestamps to times
    df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.time
    df["sunriseTime"] = pd.to_datetime(df["sunriseTime"], unit="s").dt.time
    df["sunsetTime"] = pd.to_datetime(df["sunsetTime"], unit="s").dt.time
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.date

    # add price per mile column to df
    df["pricePerMile"] = df["price"] / df["distance"]

    # fill empty values with average
    df["price"].fillna(np.mean(df["price"]), inplace=True)
    df["pricePerMile"].fillna(np.mean(df["pricePerMile"]), inplace=True)

    # Filter for time of day (Night time or Day time) and add column to dataframe
    day_df = df[df["time"].between(df["sunriseTime"], df["sunsetTime"])]
    night_df = df[df["time"].between(df["sunriseTime"], df["sunsetTime"]) == False]
    day_df["timeofday"] = 'Day'
    night_df["timeofday"] = 'Night'
    df = pd.concat([day_df, night_df])

    # Summary Statistics
    price_list = df["pricePerMile"].tolist()
    avgPrice = np.mean(price_list)

    # Map
    map_df = pd.DataFrame(columns=["source", "latitude", "longitude"])
    columns = list(map_df)
    data = []
    for key, val in coordinates.items():
        values = [key]
        for item in val:
            values.append(item)
        zipped = zip(columns, values)
        zipdict = dict(zipped)
        data.append(zipdict)

    map_df = map_df.append(data, True)

    # side bar in Streamlit information
    st.sidebar.header("Inputs for Route")
    source = st.sidebar.selectbox("Source Location", map_df["source"])
    destination = st.sidebar.selectbox("Destination Location", map_df["source"])
    rideType = st.sidebar.radio("Type of Cab", ("Lyft", "Uber"))

    names = []
    for name in df["name"]:
        if name not in names:
            names.append(name)

    st.sidebar.header("Inputs for Chart")
    rides = st.sidebar.multiselect("Choose Type of Rides to Compare", names, default=["UberXL", "Lyft XL"])
    stat = st.sidebar.radio("Choose Statistic to Compare",
                            ("Average Price", "Average Price Per Mile", "Number of Rides"))
    if stat == "Average Price":
        stats = "price"
    elif stat == "Average Price Per Mile":
        stats = "pricePerMile"
    elif stat == "Number of Rides":
        stats = "price"

    # preparing data to map
    route_df = map_df[map_df["source"].isin([source, destination])]
    coordinates = []
    for index, rows in route_df.iterrows():
        # Create list for the current row
        my_list = [rows.latitude, rows.longitude]
        # append the list to the final list
        coordinates.append(my_list)

    if len(coordinates) == 1:
        coordinates.append(coordinates[0])

    route_dict = {}
    route_dict["source"] = coordinates[0]
    route_dict["destination"] = coordinates[1]

    # Map
    st.title("Ride Share Analysis")
    st.header("Map of Your Route")
    st.write(f"Below shows your ride from {source} to {destination} using {rideType}.")
    new_df = df[df["source"] == source]
    new_df = new_df[new_df["destination"] == destination]
    new_df = new_df[new_df["cab_type"] == rideType]
    averagePrice = new_df["price"].mean()
    averageDistance = new_df["distance"].mean()
    if new_df.dropna().empty:
        st.write(f"This route can not be estimated.")
    else:
        st.write(f"Average Cost: ${averagePrice:.2f}")
        st.write(f"Distance: {averageDistance:.2f} miles")

    # Layers in Map
    # couldn't get line layer to show
    route = [route_dict]
    line_layer = pdk.Layer(
        "LineLayer",
        data =route,
        get_source_position="source",
        get_target_position="destination",
        get_color = [255, 255, 0],
        get_width=15,
        highlight_color=[255, 255, 0],
        picking_radius=10,
        auto_highlight=True,
        pickable=True)


    scatter_layer = pdk.Layer('ScatterplotLayer',
                              data=route_df,
                              get_position='[longitude, latitude]',
                              get_radius=10,
                              radius_scale=2,
                              radius_min_pixels=10,
                              radius_max_pixels=400,
                              get_color=[255, 0, 255],
                              pickable=True
                              )

    view_state = pdk.ViewState(
        latitude=route_df["latitude"].mean(),
        longitude=route_df["longitude"].mean(),
        zoom=11,
        pitch=0)

    tool_tip = {"html": "Location:<br/> <b>{source}</b> ",
                "style": {"backgroundColor": "steelblue",
                          "color": "white"}
                }

    map1 = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        mapbox_key=MAPKEY,
        layers=[scatter_layer, line_layer],
        tooltip=tool_tip)

    st.pydeck_chart(map1)

    #Grouped Bar Chart
    plot_df = df[df["name"].isin(rides)]

    st.header("Grouped Bar Chart")

    if stat == "Number of Rides":
        table = pivot_table(plot_df, ["name"], ["timeofday"], [stats], aggfunc=len)
        st.write(f"This chart shows the {stat} of the cab types chosen and the time of day (Night or Day). The purpose of this chart"
             f" is to see whether {stat} differs between various types of rides depending on the time and "
                 f"what time there is the most demand for a type of ride.")
    else:
        table = pivot_table(plot_df, ["name"], ["timeofday"], [stats])
        st.write(f"This chart shows the {stat} of the cab types chosen and the time of day (Night or Day). The purpose of this chart"
             f" is to see whether {stat} differs between various types of rides depending on the time. This will show "
                 f"what ride is the most expensive for the time of day you want to travel.")
    title = f"{stat} of Different Ride Types"
    ylabel = stat
    xlabel = "Ride Type"
    legend = ["Night", "Day"]
    grouped_bar(table, title, ylabel, xlabel, legend)
    st.header("Pivot Table")
    st.write("The pivot table below shows the data in the grouped bar chart for numerical comparison.")
    st.write(table)

    #Scatter Plot
    st.header("Scatter Plot")
    st.write("This graph shows the relationship between price per mile and distance. The user can choose as many types"
             " of rides to analyze on the chart")
    plot_df = plot_df[plot_df["name"].isin(rides)][["name", "pricePerMile", "distance"]]
    fig, ax = plt.subplots()
    for ride in rides:
        ride_df = plot_df[plot_df["name"].isin([ride])]
        scatterplot(ride_df['distance'], ride_df['pricePerMile'], ride)
    ax.legend()
    plt.title("Relationship between Distance and Price per Mile")
    plt.xlabel("Distance in Miles")
    plt.ylabel("Price per Mile")
    st.pyplot(plt)

main()
