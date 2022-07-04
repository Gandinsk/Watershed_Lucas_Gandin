# ============================================= BAIN ADVANCED ANALYTICS - WATERSHED CHALLENGE ==============================================
# ---------------------------------------- LUCAS SANTIAGO GANDIN | ANALYTICS SR ASSOCIATE CANDIDATE ----------------------------------------



## Purpose of this code
# Support a new business strategy that determines how to deal with debt collection

## Summary
# 01. Intro
# 02. Cockpit
# 03. Data Manipulation
# 04. Data Exploration
# 05. Feature Selection
# 06. Model
# 07. Final Considerations

## Postscript (Read before running this code)
# Pycaret until 2022-06-25 would not run properly in the most updated version of scikit-learn, therefore downgrade it to 0.23.2
# This code was built in Python version 3.6.13
# Before running the code, check if all the libraries are installed in the chapter "01. Intro"
# Check which switches are on in the chapter "02. Cockpit" - if it is your first time running the code, you may like to turn all of them on

## Contact
# Lucas Santiago Gandin
# (11) 99686-6597
# https://www.linkedin.com/in/lucas-gandin



# --------------------------------------------------------------- 01. INTRO ----------------------------------------------------------------



# It is important to downgrade scikit-learn to a version before 0.23.2, otherwise pycaret will not be able to run 
# pip uninstall scikit-learn                         
# pip install scikit-learn==0.22.1

# Import libraries
import os                                                                                                                                   # File Management
import numpy as np                                                                                                                          # Mathematical functions
import pandas as pd                                                                                                                         # DataFrame Manipulation
from dfply import *                                                                                                                         # DataFrame Manipulation with synthax similar to R
import time                                                                                                                                 # Time related functions
from datetime import datetime                                                                                                               # Time related functions
from datetime import timedelta                                                                                                              # Time duration function
import seaborn as sns                                                                                                                       # Data Visualization tool
import matplotlib.pyplot as plt                                                                                                             # Data Visualization tool
import missingno as mn                                                                                                                      # Missing data visualization
from pycaret.classification import *                                                                                                        # Machine Learning workflow
from sklearn.model_selection import train_test_split                                                                                        # Split into Train and Test samples
from sklearn.ensemble import RandomForestClassifier                                                                                         # Random Forest Classifier model
from sklearn.feature_selection import SelectFromModel                                                                                       # Feature selection tool
import folium                                                                                                                               # Biblioteca para visualização de mapas
from folium import plugins                                                                                                                  # Parte da biblioteca para criar criculos
import branca
import branca.colormap as cm
from folium.plugins import FloatImage

# Clean memory
import gc
gc.collect()

# Clock startpoint
Start_Time = time.monotonic()

# Check where this file is saved
wdir = os.getcwd()                                                                                                                          # Determine working directory path
wdir = wdir.replace("\\", "/")                                                                                                              # Replace path's character to match python's synthax
os.chdir(wdir)                                                                                                                              # Pinpoint working directory

# Aditional path to store input and results
Inputs_Path = "/01. Inputs/"
Results_Path = "/02. Results/"                                                                          

# Set theme for some charts
sns.set_theme(style = "whitegrid")

# Ends this chapter
print("01. Intro | OK")



# -------------------------------------------------------------- 02. COCKPIT ---------------------------------------------------------------



# File Hard Inputs
Version_Save = datetime.now().strftime("%Y-%m-%d--%Hh%M")                                                                                   # Safe copy version filename extension
Base_File = wdir + Inputs_Path + "flux.csv"                                                                                                 # Historical database
Result_File = wdir + Results_Path + "LUCAS_GANDIN_PREDICTIONS.csv"                                                                          # Prediction file
Result_Copy = wdir + Results_Path + "LUCAS_GANDIN_PREDICTIONS_" + Version_Save + ".csv"                                                     # Prediction safe copy

# Switch On/Off
Index_Recent_Calculate = 1                                                                                                                  # Switch On (Value = 1) to calculate this dataframe or Off to open previously calculated in Inputs directory to save time  

# Ends this chapter
print("02. Cockpit | OK")



# --------------------------------------------------------- 03. DATA MANIPULATION ----------------------------------------------------------



# Clock chapter startpoint
start_time = time.monotonic()

# Dividing the Chile territory in regions
# For simplicity we aggregate some regions. The groups are:
## VRB: Valparaíso, Region Metropolitana, Bernardo O'Higgins
## MNB: Maule, Nuble, Bio-Bio
## ARL: Araucania, Rios, Lagos
## The others regions correpond real provinces' coordinates, without any modification
Regions = {"region":["Arica y Parinacota", "Tarapaca", "Antofagasta", "Atacama", "Coquimbo", "VRB", "MNB", "ARL", "Aysen", "Magallanes"],
         "lat_max":[-16,-19.11667682,-22.31999551,-27.4800423,-31.18002317,-32.75000486,-35.83999713,-38.74002684,-47.26627643,-52.64997882],
         "lat_min":[-19.11667681,-22.31999550,-27.4800422,-31.18002316,-32.75000485,-35.83999712,-38.74002683,-47.26627642,-52.64997881,-60]}

# Transform the Region Dictionary in DataFrame
Regions = ((pd.DataFrame(Regions) >>
            arrange(X.lat_min, ascending = True))
            .reset_index(drop = True))

# Create a list of the Regions
Region_List = list(Regions.region)


# List of seasons of the year
DM_Season = pd.DataFrame(
            {"season": ["Summer", "Fall", "Winter", "Spring"], 
            "url_icon": 
            ["https://cdn-icons-png.flaticon.com/128/2917/2917249.png", 
            "https://cdn-icons-png.flaticon.com/128/282/282520.png",
            "https://cdn-icons-png.flaticon.com/128/2834/2834554.png",
            "https://cdn-icons-png.flaticon.com/128/1875/1875383.png"]}) 

# Dictionary with months of the year and respective season in Chile
Dict_Season = {"January": "Summer",
                "February": "Summer",
                "March": "Fall", 
                "April": "Fall",
                "May": "Fall",
                "June": "Winter",
                "July": "Winter",
                "August": "Winter",
                "September": "Spring",
                "October": "Spring",
                "November": "Spring",
                "December": "Summer"}

# Dimension that explicit month name and season for each month number
DM_Month = (pd.DataFrame(Dict_Season.items(), columns = ["month", "season"]) >>
                mutate(month_num = range(1, 13, 1)))

# Read Historical Database
base = pd.read_csv(Base_File, sep = ",")

# Dimension that store specific information about each watershed
DM_Basin_Raw = (base >>
                group_by(X.basin_id,
                    X.gauge_name,
                    X.lat,
                    X.lon,
                    X.mean_elev,
                    X.area_km2) >>
                summarize_each(X.basin_id))

# Create a temporary copy of the original DM_Basin
DM_Basin_Temp = DM_Basin_Raw

# Create an empty DataFrame to 
DM_Basin = pd.DataFrame()

# Begin iteration in zero
i = 0

# Loop to designate proper region for each watershed in the Region_List
for region in Region_List:

    # Get the maximum latitude of the region of this iteration
    lat_max = Regions.lat_max[i]

    # Could be writen as i = i + 1 (would give the same result), but there is a difference in how python reads both methods                 # += operator implement the __iadd__() method therefore does not work well with strings (which is not this case)
    i += 1

    # Create a new temporary DM_Basin filtered by remaining watersheds bellow the most nothern latitude of that region
    DM_Basin_New = (DM_Basin_Temp >>
                    mutate(between_region = X.lat < lat_max) >>
                    mask(X.between_region == True) >>
                    mutate(region = region))

    # Remove the watersheds that have a region designated
    DM_Basin_Temp = (DM_Basin_Temp >>
                    mutate(between_region = X.lat < lat_max) >>
                    mask(X.between_region == False) >>
                    select(~X.between_region))

    # Append the watersheds with designated regions to the final Dimension
    DM_Basin = pd.concat([DM_Basin, DM_Basin_New])

# Dabase Transformation Part 1: New time related features, Filling missing values and merging season variable
Base = (base >>
        mutate(date = pd.to_datetime(base.date)) >>
        mutate(year = X.date.dt.year,
                month_num = X.date.dt.month,
                flux_lag2 = lag(X.flux, 2),
                temp_max_lag2 = lag(X.temp_max, 2),
                precip_lag2 = lag(X.precip, 2)) >> 
        mutate(temp_max_last1 = lag(X.precip, 1),                                                                                           # Insert a column with max temperature of the last day
                temp_max_next1 = lead(X.precip, 1)) >>                                                                                      # Insert a column with max temperature of the next day
        mutate(temp_max_mean_1dayrange = (X.temp_max_last1 + X.temp_max_next1)/2) >>                                                        # Average max temperature between last and next day
        mutate(temp_max = X.temp_max.fillna(X.temp_max_mean_1dayrange)) >>                                                                  # Fill maximum temperature missing values with the mean of last and next day's max temperature        
        left_join(DM_Month,                                                                                                     
            by = "month_num")) 

# Calculate the 95 percentile of extreme weather conditions of each season
Extreme_Conditions = (Base >>
                        group_by(X.season) >>
                        summarize(flux_extreme_num = X.flux.quantile(0.95),
                            precip_extreme_num = X.precip.quantile(0.95),
                            temp_extreme_num = X.temp_max.quantile(0.95)))

# Dabase Transformation Part 2: Merging extreme conditions values and defining if it ocorred at each basin_id-day
Database = (Base >>
            left_join(Extreme_Conditions,
                by = "season") >>
            left_join(DM_Basin,
                by = list(DM_Basin.columns).remove("region")) >>
            mutate(flux_extreme = 1 * (X.flux > X.flux_extreme_num),
                precip_extreme = 1 * (X.precip > X.precip_extreme_num),
                temp_extreme = 1 * (X.temp_max > X.temp_extreme_num)) >>
            mutate(flux_extreme_lag2 = lag(X.flux_extreme, 2),
                precip_extreme_lag2 = lag(X.precip_extreme, 2),
                temp_extreme_lag2 = lag(X.temp_extreme, 2)) >>
            mask(~X.flux_lag2.isnull(),
                ~X.temp_max_lag2.isnull(),
                ~X.precip_lag2.isnull()))

# Database only with numerical variables for ploting charts
Base_Num = (Database >>
            select(X.flux,
                X.precip,
                X.temp_max,
                X.mean_elev,
                X.area_km2))

# Database only with categorical and ordinal variables for ploting charts
Base_Cat = (Database >>
            select(X.year,
                X.month,
                X.season,
                X.gauge_name))

# Ends this chapter
end_time = time.monotonic()
print("03. Data Manipulation | OK")
print(f"Duration: {timedelta(seconds = end_time - start_time)}")
print(" ")



# ------------------------------------------------------------- 04. FUNCTIONS --------------------------------------------------------------



# This function plot the evolution of a choosen variable of a certain basin_id, through time, constrained by minimum and maximum date
def plot_one_timeserie(cod_station, variable, min_date, max_date):

    # Filter the Database accordingly to the parameters and select only the chosen variable as valid column aside from date
    Dataset_Temp = (Database >>
                    mask(X.basin_id == cod_station,
                        X.date >= min_date,
                        X.date <= max_date) >>
                    select(Database[variable],
                        X.date))

    # Chosen variable line plot
    sns.lineplot(data = Dataset_Temp, 
                x = "date", 
                y = variable)

# This function Plot the evolution of flux, precip and temp_max of a certain basin_id, through time, constrained by minimum and maximum date
def plot_three_timeseries(cod_station, min_date, max_date):

    # Filter the Database accordingly to the parameters and select only flux, precip and temp_max as valid columns aside from date
    Dataset_Temp = (Database >>
                    mask(X.basin_id == cod_station,
                        X.date >= min_date,
                        X.date <= max_date) >>
                    gather("variable", "value", ["flux", "precip", "temp_max"]))

    # Chosen variable line plot
    sns.lineplot(data = Dataset_Temp, 
                x = "date", 
                y = "value",
                hue = "variable")

# This function allow the user to set parameters to draw a map plot by points
def draw_map_buble_single(
    zoom = 6, 
    year = 2011, 
    season = "Winter", 
    buble_size = 7000, 
    tile = "CartoDB Positron"):                                                                                                             # Stamen Terrain, Stamen Toner, Stamen Water Color, CartoDB Positron...

    # Filter database according to parameters
    Locations = (Database >>
                mask(X.season == season,
                    X.year == year) >> 
                group_by(X.basin_id) >>
                summarize(total_count = n(X.flux_extreme),
                    variable_extreme_count = np.sum(X.flux_extreme)) >>
                mutate(perc_extreme = X.variable_extreme_count / X.total_count) >>
                left_join(DM_Basin,
                    by = "basin_id"))

    # Locate best start point for the map
    lat_mean = mean(Locations.lat)
    lon_mean = mean(Locations.lon)

    # Color Gradient
    colormap = cm.LinearColormap(colors = ["#009796", "#FF5A43"],
                                index = [0, 1],
                                vmin = 0, 
                                vmax = 1)

    # Save list of latitude, longitude and values that will be used to make the map
    lat = list(Locations.lat)
    lon = list(Locations.lon)
    val = list(Locations.perc_extreme)

    # Plot empty map with chosen parameters
    m = folium.Map(location = [lat_mean, lon_mean], 
                zoom_start = zoom, 
                control_scale = True,
                tiles = tile)

    # Iteration to create circle for each basin_id
    for loc, p in zip(zip(lat, lon), val):

        folium.Circle(
            location = loc,
            radius = buble_size,
            fill = True,
            color = colormap(p),
            fill_opacity = 0.7
        ).add_to(m)

    m.add_child(colormap)

    # Display final map
    display(m)

# This function allow the user to set parameters to draw a map plot by points
def draw_map_buble_season(
    zoom = 5, 
    year = 2011, 
    buble_size = 7000, 
    tile = "CartoDB Positron"):                                                                                                             # Stamen Terrain, Stamen Toner, Stamen Water Color, CartoDB Positron...

    # Create a temporary lists to compare season maps
    Season_List_temp = list(DM_Season.season)
    Icon_List_temp = list(DM_Season.url_icon)
    
    # Loop to plot all the seasons of the selected year to compare results
    while len(Season_List_temp) > 0:
        
        # Keep the name of 2 seasons to be compared and ploted
        season1 = Season_List_temp[0]
        season2 = Season_List_temp[1]

        # Remove from temporary
        Season_List_temp.remove(season1)
        Season_List_temp.remove(season2)

        # Catch season url
        url1 = Icon_List_temp[0]
        url2 = Icon_List_temp[1]

        # Remove from temporary
        Icon_List_temp.remove(url1)
        Icon_List_temp.remove(url2)

        # Filter database according to parameters for the map on the left
        Locations1 = (Database >>
                        mask(X.season == season1,
                            X.year == year) >> 
                        group_by(X.basin_id) >>
                        summarize(total_count = n(X.flux_extreme),
                            variable_extreme_count = np.sum(X.flux_extreme)) >>
                        mutate(perc_extreme = X.variable_extreme_count / X.total_count) >>
                        left_join(DM_Basin,
                            by = "basin_id"))

        # Filter database according to parameters for the map on the right
        Locations2 = (Database >>
                        mask(X.season == season2,
                            X.year == year) >> 
                        group_by(X.basin_id) >>
                        summarize(total_count = n(X.flux_extreme),
                            variable_extreme_count = np.sum(X.flux_extreme)) >>
                        mutate(perc_extreme = X.variable_extreme_count / X.total_count) >>
                        left_join(DM_Basin,
                            by = "basin_id"))

        # Locate best start point for the map (does not matter which season)
        lat_mean = mean(Locations1.lat)
        lon_mean = mean(Locations1.lon)

        # Color Gradient for leftside map
        colormap = cm.LinearColormap(colors = ["#009796", "#FF5A43"],
                                    index = [0, 1],
                                    vmin = 0, 
                                    vmax = 1)

        # Save list of latitude, longitude and values that will be used to make the map
        lat1 = list(Locations1.lat)
        lon1 = list(Locations1.lon)
        val1 = list(Locations1.perc_extreme)
        title1 = f"{season1} {year}"

        lat2 = list(Locations2.lat)
        lon2 = list(Locations2.lon)
        val2 = list(Locations2.perc_extreme)
        title2 = f"{season2} {year}"

        # Plot empty map with chosen parameters
        m = folium.plugins.DualMap(
                        location = [lat_mean, lon_mean], 
                        zoom_start = zoom, 
                        control_scale = True,
                        tiles = tile)

        # Map on the left | Iteration to create circle for each basin_id
        for loc, p in zip(zip(lat1, lon1), val1):

            folium.Circle(
                location = loc,
                radius = buble_size,
                fill = True,
                color = colormap(p),
                fill_opacity = 0.7
            ).add_to(m.m1)

        # Add season image to the bottom of the map
        FloatImage(url1, bottom = 0, left = 50).add_to(m.m1)

        # Map on the right | Iteration to create circle for each basin_id
        for loc, p in zip(zip(lat2, lon2), val2):

            folium.Circle(
                location = loc,
                radius = buble_size,
                fill = True,
                color = colormap(p),
                fill_opacity = 0.7
            ).add_to(m.m2)

        # Add season image to the bottom of the map
        FloatImage(url2, bottom = 0, left = 0).add_to(m.m2)

        # Gradient Color map design
        m.add_child(colormap)

        # Display final map
        folium.LayerControl(collapsed = False).add_to(m)
        display(m)

# Test plot_one_timeserie function
plot_one_timeserie(8114001, "flux", "2010-01-01", "2015-01-01")

# Test plot_three_timeseries function
plot_three_timeseries(8114001, "2010-01-01", "2015-01-01")

# Test draw_map_buble_single function
draw_map_buble_single()

# Test draw_map_buble_season function
draw_map_buble_season()

# Ends this chapter
end_time = time.monotonic()
print("04. Functions | OK")
print(f"Duration: {timedelta(seconds = end_time - start_time)}")
print(" ")



# ---------------------------------------------------------- 05. DATA EXPLORATION ----------------------------------------------------------



# Clock chapter startpoint
start_time = time.monotonic()



# Ends this chapter
end_time = time.monotonic()
print("05. Data Exploration | OK")
print(f"Duration: {timedelta(seconds = end_time - start_time)}")
print(" ")



# --------------------------------------------------------- 06. FEATURE SELECTION ----------------------------------------------------------



# Clock chapter startpoint
start_time = time.monotonic()

# Split data into train and test databases
X_Train, X_Test, Y_Train, Y_Test = train_test_split(Database, Database.flux_extreme, test_size = 0.2)

# Identify features that need to be encodes before using Random Forest 
Features_Encode = X_Train.columns[X_Train.dtypes == object].tolist()
Features_Encode.remove("gauge_name")                                                                                                        

# Transform every feature that needs to be encoded into dummies
X_Train_Encoded = pd.get_dummies(X_Train, prefix = Features_Encode, columns = ["month", "season", "region"])
X_Test_Encoded = pd.get_dummies(X_Test, prefix = Features_Encode, columns = ["month", "season", "region"])

# Remove Date and Collected (target variable) as possible variables of the model
X_Train_Encoded = (X_Train_Encoded >> 
                    select(~X.temp_extreme_num,                                                                                           # 
                        ~X.precip_extreme_num,                                                                                                       # 
                        ~X.flux_extreme_num,                                                                                  # 
                        ~X.temp_extreme,
                        ~X.precip_extreme,
                        ~X.flux_extreme,
                        ~X.temp_max_mean_1dayrange,                                                                                                           # 
                        ~X.temp_max_next1,                                                                                                       # 
                        ~X.temp_max_last1,
                        ~X.month_num,
                        ~X.date,
                        ~X.gauge_name,
                        ~X.flux,
                        ~X.precip,
                        ~X.temp_max,
                        ~X.basin_id))                                                                                                  # 

# The same for Test database
X_Test_Encoded = (X_Test_Encoded >> 
                    select(~X.temp_extreme_num,                                                                                           # 
                        ~X.precip_extreme_num,                                                                                                       # 
                        ~X.flux_extreme_num,                                                                                  # 
                        ~X.temp_extreme,
                        ~X.precip_extreme,
                        ~X.flux_extreme,
                        ~X.temp_max_mean_1dayrange,                                                                                                           # 
                        ~X.temp_max_next1,                                                                                                       # 
                        ~X.temp_max_last1,
                        ~X.month_num,
                        ~X.date,
                        ~X.gauge_name,
                        ~X.flux,
                        ~X.precip,
                        ~X.temp_max,
                        ~X.basin_id))                                                                                                  #

# Random Forest Options to help select best features for the model
Rf_Select = SelectFromModel(
                RandomForestClassifier(
                    min_samples_leaf = 50,                                                                                                  # Minimum sample sizes
                    n_estimators = 150,                                                                                                     # Choose quantity of estimators (150 should be enough for feature selection)
                    bootstrap = True,                                                                                                       # Allow Bootstrap procedeer
                    oob_score = True,                                                                                                       # Allow Out of Bag score to take data not utilized in Decision Tree to evaluate Forest precision
                    n_jobs = -1,                                                                                                            # Number of jos to run in parallel (-1 indicates all processors)
                    random_state = 123,                                                                                                     # Use a seed called 123 for replicabiliy
                    max_depth = 5,                                                                                                          # Maximum depth of the tree
                    max_features = 6))                                                                                                      # 

# Run Random Forest with the already encoded train database
Rf_Select.fit(X_Train_Encoded, Y_Train)

# True or false array that shows which features were selected
Rf_Select.get_support()

## Total bills and connection status are staple as most important features, followed by disconnection exxecuted by remote
## When dafaulting the bill becomes a decision (stratgy of financial management), it is more likely for debt to snowball to higher amounts
## As hypothesised before, if client is already disconnected and do not pay the debt, they probably bankrupt or don't live there anymore
## Remote disconnection is the most effective disconnection execution, therefore this dummy feature is next in terms of importance
# Translation of which features were selected to the model
Selected_Feat = X_Train_Encoded.columns[(Rf_Select.get_support())]

# Number of features selected - I would change if it get less than 3 or more than 6)
print("--------------------------------------------")
print(f"{len(Selected_Feat)} features were selected")
print("--------------------------------------------")

# Ends this chapter
end_time = time.monotonic()
print("06. Feature Selection | OK")
print(f"Duration: {timedelta(seconds = end_time - start_time)}")
print(" ")



# ---------------------------------------------------------------- 07. MODEL ---------------------------------------------------------------



# Clock chapter startpoint
start_time = time.monotonic()

# Build a Dataframe with selected features and target variable
Train = (X_Train_Encoded >> 
            select(Selected_Feat) >>                                                                                                        # Remove any feature not included in the feature selection
            mutate(Index = X.index) >>                                                                                                      # Create an auxiliary column that copy the index
            bind_cols(Y_Train) >>                                                                                                           # Insert the target variable to the complete train dataset
            select(~X.Index))

# Do the same to test sample, except it does not receive the target variable
Test = (X_Test_Encoded >> 
            select(Selected_Feat) >>                                                                                                        # Remove any feature not included in the feature selection
            mutate(Index = X.index) >>                                                                                                      # Create an auxiliary column that copy the index
            bind_cols(Y_Test)  >>                                                                                                           # Insert the target variable to the complete test dataset
            select(~X.Index))

# Nominate which variables are numeric and which are categorical
Base_Num_Cols = Train.columns[Train.dtypes == "float64"].tolist()                                                                           # Make a list of columns names with numeric like type
Base_Cat_Cols = Train.columns[Train.dtypes == "uint8"].tolist()                                                                             # Make a list of columns names with categorical like type

## The modelling strategy will be the following...
## Pycaret will run 12 models, with feature interaction of the features already selected with the train-test split already determined
## No ordinal features were identified
## Categorical and Numerical festures are explicit in the model
## K-fold number seleted rational was to balance variance and bias (and to be honest computational time was also taken into consideration)
## Pycaret will choose the best 3 models and create a hybrid model blending the top 3 models by soft voting
# Options to run caret (don't forget to press ENTER, when it asks to do so)
clf1 = setup(data = Train,                                                                                                                  # Use Train database as data input for pycaret
            target = "flux_extreme",                                                                                                        # Determine "flux_extreme" as the target variable
            categorical_features = Base_Cat_Cols,                                                                                           # Set which variables are categorical
            numeric_features = Base_Num_Cols,                                                                                               # Explicit which features are numeric
            test_data = Test,                                                                                                               # Insert Test database for cross-validation
            verbose = True,                                                                                                                 # Progress bar that shows how long it will take to finish the setup task
            fix_imbalance = False,                                                                                                          # Unfortunally, I was unable to turn fix imbalance on, because it not only took 80x more per model, but also could not do many models
            remove_outliers = False,                                                                                                        # Don't remove outliers from the set
            normalize = True,                                                                                                               # Standardize features so that every variable have the same weight
            feature_interaction = True,                                                                                                     # Interaction of features
            fold_strategy = "kfold",                                                                                                        # Cross validation strategy
            fold = 5,                                                                                                                       # Number of folds to be used in cross validation
            feature_selection = False,                                                                                                      # Relevant features for the model were already selected, therefore there is no need to do it again
            remove_multicollinearity = False)                                                                                               # Not able to remove multicolinearity from variables

# List of models that are suitable for this kind of prediction
Models_List = ["nb",                                                                                                                        # Naive Bayes
                "dt",                                                                                                                       # Decision Tree Classifier
                "rf",                                                                                                                       # Random Forest Classifier
                "lr",                                                                                                                       # Logistic Regression
                "lda",                                                                                                                      # Linear Discriminant Analysis
                "gbc",                                                                                                                      # Gradient Boosting Classifier
                "lightgbm",                                                                                                                 # Light Gradient Boosting
                "ada",                                                                                                                      # Ada Boost Classifier
                "ridge",                                                                                                                    # Ridge Classifier
                "et",                                                                                                                       # Extra Trees Classifier
                "qda",                                                                                                                      # Quadratic Discriminant Analysis
                "svm"]                                                                                                                      # SVM - Linear Kernel

# Highlight models with best performance by comparing with most used metrics
Best_Models = compare_models(include = Models_List,                                                                                         # Compare models of the list considered suitable for the problem
                            n_select = 3,                                                                                                   # Select top n models
                            sort = "auc")                                                                                                   # Select metric to choose top n models

# Show the best hiperparameter for each of the top models
print(Best_Models)

## If I had time to run a hybrid model I would run this code to blend vote the best 3 models
# # Tune Selected models
# Tuned_Best_Models = [tune_model(i) for i in Best_Models]                                                                                    # Tune each model (i) of the top model list (Best_Models)
# Tuned_Best_Models = [tune_model(i) for i in Best_Models]  

# Blend a hybrid model of the top models by Soft Voting their results to classify the target variable
# Final_Model = blend_models(Tuned_Best_Models)

# 
Final_Model = tune_model(Best_Models[0])

# Show the hiperparameters of the blended model
print(Final_Model) 

# This function output a dashboard with 20 charts regarding models performance                                                              # Feel free to explore the charts to evaluate the model
evaluate_model(Final_Model)                                                                                                                 # Four charts that are in this dashboard will be highlighted next

# Show the evolution of the Area Under the Curve (AUC), in other words measure True Positives and True Negatives through the model
plot_model(Final_Model, plot = "auc")                                                                                                       # AUC is a metric product from ROC curve (Receiver Operating Characteristic)

# Plot Precision-Recall chart
plot_model(Final_Model, plot = "pr")                                                                                                        # Shows the tradeoff between precision and recall for different threshold

# Plot Lift Curve that compare the potential of the model to find True Positives and True Negatives compared to a random prediction
plot_model(Final_Model, plot = "lift")                                                                                                      # Measure of the model's performance in comparison to a random classifier

# Confusion Matrix shows False Negatives, False Positives and corect predictions of the model
plot_model(Final_Model, plot = "confusion_matrix")                                                                                          # Quantity of True Positives, False Negatives, False Positives and False Negatives

# Plot Feature Importance chart of the best model
plot_model(Final_Model, plot = "feature")                                                                                          

# Predict collection in the test database and show metrics of how accurate the model is
Prediction = predict_model(Final_Model,
                            data = Test)

# Will help transform prediction to the submition format
Y_Test_aux = (Test >>
                mutate(Index = X.index,                                                                                                     # Create aux column for merging databases
                    Actual = X.flux_extreme) >>                                                                                                # Rename target variable to avoid conflict when merging databases
                select(X.Index, X.Actual))                                                                                                  # Select only Index and the target variable

# Transform prediction to the submition format
Prediction_to_Actual = (Prediction >>
                        mutate(Index = X.index) >>                                                                                          # Create aux column for merging databases
                        left_join(Y_Test_aux,                                                                                               # Merge with the target variable
                            by = "Index") >>                                                                                                # Use index to join databases
                        select(~X.Index, ~X.collected))                                                                                     # Remove Index column


# Ends this chapter
end_time = time.monotonic()
print("07. Model | OK")
print(f"Duration: {timedelta(seconds = end_time - start_time)}")
print(" ")




# ------------------------------------------------------------- 99. SCRAPYARD --------------------------------------------------------------


# # X
# Database = (base >>
#             mutate(date = pd.to_datetime(base.date).dt.date) >>
#             mutate(year = X.date.dt.year,
#                    temp_max_last1 = lag(X.precip, 1),                                                                                       # Insert a column with max temperature of the last day
#                    temp_max_next1 = lead(X.precip, 1)) >>                                                                                   # Insert a column with max temperature of the next day
#             mutate(temp_max_mean_1dayrange = (X.temp_max_last1 + X.temp_max_next1)/2) >>                                                    # Average max temperature between last and next day
#             mutate(temp_max = X.temp_max.fillna(X.temp_max_mean_1dayrange)) >>                                                              # Fill maximum temperature missing values with the mean of last and next day's max temperature
#             mask(X.precip != np.NaN))       









# # X
# Database = (base >>
#             mutate(date = pd.to_datetime(base.date)) >>
#             mutate(year = X.date.dt.year,
#                    month = X.date.dt.month) >>
#             unite("year_month", 
#                  ["year", "month"],
#                  sep = "-",
#                  remove = False)) >>
#             mutate(temp_max_last1 = lag(X.precip, 1),                                                                                       # Insert a column with max temperature of the last day
#                    temp_max_next1 = lead(X.precip, 1)) >>                                                                                   # Insert a column with max temperature of the next day
#             mutate(temp_max_mean_1dayrange = (X.temp_max_last1 + X.temp_max_next1)/2) >>                                                    # Average max temperature between last and next day
#             mutate(temp_max = X.temp_max.fillna(X.temp_max_mean_1dayrange)))                                                              # Fill maximum temperature missing values with the mean of last and next day's max temperature        











# import datetime
# date = '2021-05-21 11:22:03'
# datem = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
# date2 = '2021-05-21'
# datem2 = datetime.datetime.strptime(date2, "%Y-%m-%d")
# datem.day
# datem.dtypes
# d = {"Date": []}
# df = pd.DataFrame(date2)
# pd.concat(df, datem)




# # convert datetime column to just date
# Database['date'] = pd.to_datetime(Database['date']).dt.date


# Database.dtypes





# Result_Corredor = pd.DataFrame(columns = ["Date"])

# # Insere os resultados dessa iteração no conjunto de resutlados
# Result_Corredor = pd.concat([Result_Corredor, datem2])










# # 
# # XX
# NaN_Count = base[Database.columns].isna().sum()


# # base.precip.mmax


#     # Make the color list repeat itself until it satisfies enough color for the whole map
#     while len(Colors_List) < len(Locations_Values):

#         # Repeat color list
#         Colors_List = Colors_List.append(Colors_List)


    # # Cria centralização do mapa
    # map = folium.Map(location = [Locations_Mean["lat"], Locations_Mean["lon"]], 
    #                 zoom_start = zoom, 
    #                 control_scale = True,
    #                 tiles = tile)








#
# This function allow the user to set parameters to draw a map plot by points
# def draw_map_points(zoom, min_date, max_date, map_points, tile, variable):

#     # X
#     variable = "temp_max"
#     min_date = "2010-01-01"
#     max_date = "2015-01-01"
#     zoom = 7
#     map_points = 1                                                                                                                              # Icon = 0 | Circle = 1 | ClusterAPI = 2
#     tile = "CartoDB Positron"                                                                                                                     # Stamen Terrain, Stamen Toner, Stamen Water Color, CartoDB Positron...
#     variable_extreme = variable + "_extreme"

#     # Filtra as localizações da UF escolhida para o agrupamento escolhido
#     Locations = (Database >>
#                 mask(X.date >= min_date,
#                     X.date <= max_date))

#     # Versão somente com as localizações
#     Locations_Values = (Database >>
#                         select(X.lat,
#                             X.lon))

#     # Agrupa os valores da latitude e longitude para centralizar o mapa onde estão a maior parte dos postos
#     Locations_Mean = (Locations >>
#                         group_by(X.gauge_name) >>
#                         summarize(lat = mean(X.lat),
#                             lon = mean(X.lon)))

#     # Colors in list
#     Colors_List = [
#             "#009796", "#B29C86", "#7F7F7F", "#99D5D5", "#1D368E", \
#             "#43CFF6", "#FF5A43", "#FF8E2E", "#63277F", "#C6D02C", \
#             "#FF9C8E", "#E6DCD7", "#C0C0C0", \
            
#             "red", "blue", "green", "purple", "orange", "darkred", \
#             "lightred", "beige", "darkblue", "darkgreen", "cadetblue", \
#             "darkpurple", "pink", "lightblue", "lightgreen", "gray", \
#             "black", "lightgray", "red", "blue", "green", "purple", \
#             "orange", "darkred", "lightred", "beige", "darkblue", \
#             "darkgreen", "cadetblue", "darkpurple","pink", "lightblue", \
#             "lightgreen", "gray", "black", "lightgray"]

#     # Colors in Dataframe
#     Colors = (pd.DataFrame(Colors_List, columns = ["Color"]) >>
#                     mutate(Index = X.index))

#     # Cor para cada grupo
#     Color_Groups = (Locations >>
#                     group_by(X.gauge_name) >>
#                     summarize_each(X.gauge_name) >>
#                     mutate(Index = X.index) >>
#                     left_join(Colors,
#                         by = "Index"))

#     # Coloca as cores no Locations
#     Locations = (Locations >>
#                 left_join(Color_Groups,
#                     by = "gauge_name"))

#     # Cria centralização do mapa
#     map = folium.Map(location = [-38.8444, -71.6536], 
#                     zoom_start = zoom, 
#                     control_scale = True,
#                     tiles = tile)

#     # Mostra visualização do mapa
#     map

#     # Caso tenha sido selecionado o tipo de visualização de mapa com pontos em icones
#     if map_points == 0:

#         # Coloca os pontos em cada uma das coordenadas
#         for index, location_info in Locations.iterrows():
#             (folium.Marker([location_info["lat"],
#                             location_info["lon"]], 
#                             popup = location_info["basin_id"])
#             .add_to(map))

#     # Caso tenha sido selecionado o tipo de visualização de mapa com circulos segmentados por cluster
#     elif map_points == 1:
        
#         # Cria circulos de acordo com o cluster nas cores pré determinadas
#         for _, row in Locations.iterrows():
#             folium.CircleMarker(
#                 location = [row["lat"], row["lon"]],
#                 radius = 12, 
#                 weight = 2, 
#                 fill = True, 
#                 fill_color = row["Color"],
#                 color = row["Color"]
#             ).add_to(map)

#     # Caso tenha sido selecionado o tipo de visualização de mapa com API de clusterização de mapa
#     elif map_points == 2:

#         # Usa API de clusterização de localização
#         plugins.MarkerCluster(Locations_Values).add_to(map)

#     # Mostra visualização do mapa
#     map





# # 
# Check = (Database >>
#             group_by(X.basin_id) >>
#             summarize_each([n_distinct],
#                 X.area_km2,
#                 X.mean_elev,
#                 X.lon,
#                 X.lat,
#                 X.gauge_name))

# Check["area_km2_wrapper"].sum()
# Check["mean_elev_wrapper"].sum()
# Check["lon_wrapper"].sum()
# Check["lat_wrapper"].sum()
# Check["gauge_name_wrapper"].sum()









##########

# m = folium.plugins.DualMap(location=(52.1, 5.1), tiles=None, zoom_start=8)



#     folium.TileLayer("openstreetmap").add_to(m.m1)
#     folium.TileLayer("openstreetmap").add_to(m.m2)

#     HeatMap(list(zip(lat, lon))).add_to(folium.FeatureGroup(name='X').add_to(m))

#     HeatMap(list(zip(crime_robbery.LATITUDE.values, crime_robbery.LONGITUDE.values))).add_to(folium.FeatureGroup(name='Robbery').add_to(hmap))






##

    # url = (
    # "https://raw.githubusercontent.com/ocefpaf/secoora_assets_map/"
    # "a250729bbcf2ddd12f46912d36c33f7539131bec/secoora_icons/rose.png"
    # )




    # image_file = wdir + Inputs_Path + "Icon_Spring.jpg"



    # url_spring = (
    # "https://cdn-icons.flaticon.com/png/512/436/premium/436627.png?token=exp=1656864582~hmac=ea4464e93ee9f90a9a3c5ab5af2c0e29"
    # )
    # url_summer = (
    # "https://cdn-icons-png.flaticon.com/512/2917/2917249.png"
    # )
    #     url_fall = (
    # "https://cdn-icons-png.flaticon.com/512/3507/3507845.png"
    # )

    # url_winter = (
    # "https://cdn-icons-png.flaticon.com/512/642/642102.png"
    # )



    # req = requests.get(url_spring)
    # pilImage = Image.open(BytesIO(req.content))
    # url_spring = (wdir + Inputs_Path + "Spring.png")
    # image = Image.open(url_spring)
    # image.resize((100, 100))





# # 
# NaN_Search = (X_Train_Encoded >>
#             mutate(precip_na = X.precip.isna(),
#                 temp_max_na = X.temp_max.isna()) >>
#             mask((X.precip_na == 1) | (X.temp_max_na == 1)) >>
#             group_by(X.basin_id) >>
#             summarize_each([np.sum],
#                 X.precip_na,
#                 X.temp_max_na) >>
#             spread(X.basin_id,
#                 X.precip_na_sum))



# 
Table_Extreme = (Database >>
                group_by(X.basin_id) >>
                summarize(total_count = n(X.flux_extreme),
                    variable_extreme_count = np.sum(X.flux_extreme)) >>
                mutate(perc_extreme = X.variable_extreme_count / X.total_count))








Database >> mask(X.date == "2010-01-01") >> group_by(X.basin_id) >> summarize(count = n(X.gauge_name))
Database >> mask(X.date == "2010-01-01") >> group_by(X.gauge_name) >> summarize(count = n(X.basin_id))






X_Test_Encoded.isna().sum()




## Precipitation and max temperature are only available 
# 
NaN_Search = (base >>
            mutate(precip_na = X.precip.isna(),
                temp_max_na = X.temp_max.isna()) >>
            mask((X.precip_na == 1) | (X.temp_max_na == 1)) >>
            group_by(X.date, X.gauge_name) >>
            summarize_each([np.sum],
                X.precip_na,
                X.temp_max_na) >>
            spread(X.gauge_name,
                X.precip_na_sum))

NaN_Search

NaN_Search.temp_max_na.sum()