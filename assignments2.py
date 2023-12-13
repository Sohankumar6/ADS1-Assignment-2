''' import rquired libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import stats


def readfile(filename):
    '''
    Read csv file

    Parameters
    ----------
    filename : str
       This is the file path of the CSV file 

    Returns
    -------
    data1 : DataFrame
        Shows DataFrame containing the read data from the CSV file.
        

    '''
    #This function reads a CSV file and returns DataFrame
    data1 = pd.read_csv(filename, sep=',', skiprows=4)
    return data1

#Reading the CSV file and loading it into the dataframe
df = readfile(
    r"C:\Users\sohan\OneDrive\Desktop\API_19_DS2_en_csv_v2_5998250.csv")

#Displays first 5 rows and all columns 
print(df.head())
#Dropping specific columns 
df = df.drop(['Country Code', 'Indicator Code', 'Unnamed: 67'], axis=1)

#Listing required nations 
countries = ['India', 'United Kingdom',
             'United States', 'Australia', 'Germany', 'Pakistan']


def transpose(data):
    '''
    Transpose the DataFrame

    Parameters
    ----------
    data : Dataframe
        Input Dataframe to be transposed 

    Returns
    -------
    countries_df : DataFrame
        DataFrame with selected columns.
    years_df : DataFrame 
        DataFrame with data for specific countries 
        is given based on Country Name

    '''    
    #Transposes the data
    data_t = data.T
    
    #Assigning the first row as column after Transpose
    data_t.columns = data_t.iloc[0]
    
    #Removing the header 
    data_t = data_t[1:]
    
    #Resetting and renaming the index columns as Year
    data_t.reset_index(inplace=True)
    data_t.rename(columns={'index': 'Year'}, inplace=True)
    
    #Extracting columns for specified nations
    countries_df = data_t.copy().loc[:, countries]
    
    #Filterin data for specified countries
    years_df = data[data["Country Name"].isin(countries)]
    
    return countries_df, years_df

#Transposing df and assigning it to 'data_c' and 'data_y'
data_c, data_y = transpose(df)

#Dropping missing values
data_y.dropna()

#Displaying statistical discription of the data_y
print(data_y.describe())

#displaying statistical discription of data_c
print(data_c)
print(data_c.describe())

#Filtering data to obtain 'Urban population growth (annual %)' indicator
data_urben_pop_growth = data_y[data_y['Indicator Name']
                               == 'Urban population growth (annual %)']

#Selecting 'Country Name' and 'Indicator Name'
data_urben_pop_growth_n = data_urben_pop_growth.loc[:, [
    'Country Name', 'Indicator Name']]

#Filtering the data by dropping null columns and displaying it
print(data_urben_pop_growth_n)
data_urben_pop_growth = data_urben_pop_growth.dropna(axis=1)
print(data_urben_pop_growth)


def barplot_1():
    '''
    This function generates the grouped bar chat displaying the
    urban population growth for specified countries in year 2010,
    2015 and 2020

    Returns
    -------
    None.

    '''
    
    #filtering the dataframe for specified countries 
    bar_new_1 = data_urben_pop_growth[
        data_urben_pop_growth['Country Name'].isin([
            'India', 'United Kingdom', 'United States', 'Australia', 'Germany',
            'Pakistan'])]
    
    #Mentioning the width of each bar 
    bar_width = 0.3
    
    #Creating an array of indices  
    index = np.arange(len(bar_new_1['Country Name']))
    
    #Creating grouped bar for each year 
    plt.bar(index - bar_width/2, bar_new_1['2010'], width=bar_width,
            label='2013', color='c')
    plt.bar(index + bar_width/2, bar_new_1['2015'], width=bar_width,
            label='2015',  color='m')
    plt.bar(index + 3*bar_width/2, bar_new_1['2020'], width=bar_width,
            label='2020',  color='b')
    
    #Mentioning labels and title for the plot 
    plt.xlabel('Countries')
    plt.ylabel('Urban population growth (annual %)')
    plt.title('Urban Population growth (annual %) of different countries')
    
    #Setting the x-axis  ticks to country name and rotating it by 90 degree
    plt.xticks(index, bar_new_1['Country Name'], rotation=90)
    
    #Displaying the legend for the plot 
    plt.legend()
    
    #showing the plot 
    plt.show()

#Calling the function and displaying the bar plot 
barplot_1()

#Filtering data to obtain 'Population growth (annual %)' indicator
data_pop_growth = data_y[data_y['Indicator Name']
                         == 'Population growth (annual %)']

##Selecting 'Country Name' and 'Indicator Name' and displaying it
data_pop_growth_n = data_pop_growth.loc[:, ['Country Name', 'Indicator Name']]
print(data_pop_growth_n)

#Filtering the data by dropping null columns and displaying it
data_pop_growth = data_pop_growth.dropna(axis=1)
data_pop_growth


def barplot_2():
    '''
    This function generates the grouped bar chat displaying the
    population growth for specified countries in year 2010,
    2015 and 2020

    Returns
    -------
    None.

    '''
    
    #Filtering the dataframe for specified countries 
    bar_new = data_pop_growth[data_pop_growth['Country Name'].isin(
        ['India', 'United Kingdom', 'United States', 'Australia', 'Germany', 
         'Pakistan'])]
    
    #Mentioning the width of each bar 
    bar_width = 0.3
    
    #Creating an array of indices  
    index = np.arange(len(bar_new['Country Name']))
    
    #Creating grouped bar for each year 
    plt.bar(index - bar_width/2, bar_new['2010'], width=bar_width,
            label='2013', color='yellow')
    plt.bar(index + bar_width/2, bar_new['2015'], width=bar_width,
            label='2015',  color='r')
    plt.bar(index + 3*bar_width/2, bar_new['2020'], width=bar_width,
            label='2020',  color='green')
    
    #Mentioning labels and title 
    plt.xlabel('Countries')
    plt.ylabel('Population growth (annual %)')
    plt.title('Population growth (annual %) of different countries')
    
    #Setting the x-axis  ticks to country name and rotating it by 90 degree
    plt.xticks(index, bar_new['Country Name'], rotation=90)
    
    #Displaying the legend for the plot 
    plt.legend()
    
    #showing the plot 
    plt.show()

#Calling the function and displaying the bar plot 
barplot_2()

#Filtering data to obtain 'Mortality rate, (per 1,000 live births)' indicator
data_mortality_total = data_y[
    data_y['Indicator Name'] ==
    'Mortality rate, under-5 (per 1,000 live births)']

##Selecting 'Country Name' and 'Indicator Name' and displaying it
data_mortality_total = data_mortality_total.loc[:, [
    'Country Name', 'Indicator Name', '2010', '2015', '2020']]
print(data_mortality_total)


def scatter_plot(data, years):
    '''
    This function generates the scatter plot displaying the
    Mortality rate for specified countries in year 2010,
    2015 and 2020

    Parameters
    ----------
    data : DataFrame
        Data containing mortality rates for different countries and years
    years : list of string
        list of string containing years 

    Returns
    -------
    None.

    '''
    
    #Mentioning marker style and colors 
    markers = ['o', 's', '+']
    colors = ['blue', 'green', 'red']
    
    #Setting up the figure size
    plt.figure(figsize=(10, 6))
    
    #Iterating through the spicified years and creating a scatterplot  
    for i, year in enumerate(years):
        plt.scatter(data[year], data['Country Name'], label=str(
            year), marker=markers[i], color=colors[i])
        
    #labling the x and y axis and adding the title
    plt.xlabel('Mortality rate (per 1,000 live births)')
    plt.ylabel('Country')
    plt.title('Mortality rate of different countries per 1000 live births')
    
    #To display grid lines in plot
    plt.grid()
    
    #Displaying the legend
    plt.legend()
    
    #showing the plot
    plt.show()

#Assigning a variable to the list of years 
year_sp = ['2010', '2015', '2020']

#calling the function to create the scatter plot 
scatter_plot(data_mortality_total, year_sp)

#Filtering data to obtain 'Agriculture, forestry, and fishing, 
   #value added (% of GDP)' indicator
data_GDP = data_y[data_y['Indicator Name'] ==
                  'Agriculture, forestry, and fishing, value added (% of GDP)']

#Selecting 'Country Name' and 'Indicator Name' and displaying it
data_GDP_n = data_GDP.loc[:, ['Country Name', 'Indicator Name']]
print(data_GDP_n)

#Filtering the data by dropping null columns and selecting some columns 
data_GDP = data_GDP.dropna(axis=1)
data_GDP = data_GDP.loc[:, ['Country Name', '2011', '2012',
                            '2013', '2014', '2015', '2016', '2017', '2018',
                            '2019', '2020']]

#Displaying Dataframe 
print(data_GDP)


def plot_hist():
    '''
    The function generates a histogram displaying the distribution Agriculture,
    Forestry and fishing value added for GDP. It also calculates skewness and 
    kurtosis.
    
    Parameters
    ----------
    data_column : Pandas series 
        Series containing the data of Agriculture,
        Forestry and fishing value added for GDP for sletected years
    year : int
        the year for which the data is visualised 

    Returns
    -------
    None.

    '''
    
    #Calculating the skewness and Kurtosis for the data 
    skewness = stats.skew(data_GDP['2015'])
    kurtosis = stats.kurtosis(data_GDP['2015'])
    
    #Creating a new figure
    plt.figure()
    
    #Creating a histogram with specifyed bins and color
    plt.hist(data_GDP['2015'], bins=20, alpha=0.6, color='salmon')
    
    #labling x and y axis and titling the histogram with calculated values
    plt.xlabel(
        f'''Agriculture, Forestry and Fishing value added for GDP, 
                Percent for year {'2015'}''')
    plt.ylabel('Frequency')
    plt.title(f''''Histogram of Agriculture, Forestry and Fishing 
    value added for GDP in year 2015 \nSkewness={skewness:.2f} and
    Kurtosis={kurtosis:.2f}''')
              
    #Dysplaying the histogram 
    plt.show()

#Calling the function
plot_hist()

#Filtering data to obtain 'Renewable energy consumption' indicator
data_Renewable = data_y[
    data_y['Indicator Name'] ==
    'Renewable energy consumption (% of total final energy consumption)']

#Selecting some required columns 
data_Renewable_ = data_Renewable.loc[:, ['Country Name', 'Indicator Name',
                                         '2011', '2012', '2013', '2014', 
                                         '2015', '2016', '2017', '2018', 
                                         '2019', '2020']]

#Displaying the data
print(data_Renewable_)


def plot_hist2():
    '''
    The function generates a histogram displaying the distribution of Renewable
    energy consumption for year 2015. It also calculates skewness and 
    kurtosis.

    Parameters
    ----------
    data : pandas dataframe
        Dataframe consists the data for renewable energy consumption for year
        required year
    year : str
        particular year for which the data is visualized 

    Returns
    -------
    None.

    '''
    
    #Calculating Skewness and Kurtosis for required year 
    skewness = stats.skew(data_Renewable_['2015'])
    kurtosis = stats.kurtosis(data_Renewable_['2015'])

    #Creating a new figure
    plt.figure()
    
    #Generating histogram for year 2015 with bins and edgecolor
    plt.hist(data_Renewable_['2015'], bins=25, ec='red')
    
    #Labling x and y axis and adding title with calculated skew and kurtosis
    plt.xlabel('Renewable energy consumption, in %')
    plt.ylabel('Frequency')
    plt.title(f''''Histogram of Renewable energy 
 consumption in year 2015\nSkewness={skewness:.2f} and
 Kurtosis={kurtosis:.2f}''')
 
    #Display histogram
    plt.show()

#calling the function
plot_hist2()

#Filtering data to obtain 'Electricity production from oil sources' indicator
data_ele_oil = data_y[data_y['Indicator Name'] ==
                      'Electricity production from oil sources (% of total)']

#Selecting 'Country Name' and 'Indicator Name' and displaying it
data_ele_oil_n = data_ele_oil.loc[:, ['Country Name', 'Indicator Name']]
print(data_ele_oil_n)

#Dropping the null columns and displaying the outcome
data_ele_oil = data_ele_oil.dropna(axis=1)
print(data_ele_oil)

#Filtering data to obtain 'Electricity production from oil sources' indicator
data_ele_nuclear = data_y[
    data_y['Indicator Name'] ==
    'Electricity production from nuclear sources (% of total)']

#Selecting 'Country Name' and 'Indicator Name' and displaying it
data_ele_nuclear_n = data_ele_nuclear.loc[:, [
    'Country Name', 'Indicator Name']]
print(data_ele_nuclear_n)

#Dropping the null columns and displaying the outcome
data_ele_nuclear = data_ele_nuclear.dropna(axis=1)
print(data_ele_nuclear)

#Filtering data to obtain 'Electricity production from natural gas' indicator
data_ele_gas = data_y[
    data_y['Indicator Name'] == 
    'Electricity production from natural gas sources (% of total)']

#Selecting 'Country Name' and 'Indicator Name' and displaying it
data_ele_gas_n = data_ele_gas.loc[:, ['Country Name', 'Indicator Name']]
print(data_ele_gas_n)

#Dropping the null columns and displaying the outcome
data_ele_gas = data_ele_gas.dropna(axis=1)
print(data_ele_gas)

#Filtering data to obtain 'Electricity production from hydroelectric' indicator
data_ele_hydro = data_y[
    data_y['Indicator Name'] ==
    'Electricity production from hydroelectric sources (% of total)']

#Selecting 'Country Name' and 'Indicator Name' and displaying it
data_ele_hydro_n = data_ele_hydro.loc[:, ['Country Name', 'Indicator Name']]
print(data_ele_hydro_n)

#Dropping the null columns and displaying the outcome
data_ele_hydro = data_ele_hydro.dropna(axis=1)
print(data_ele_hydro)

#Filtering data to obtain 'Electricity production from coal sources' indicator
data_ele_coal = data_y[data_y['Indicator Name'] ==
                       'Electricity production from coal sources (% of total)']

#Selecting 'Country Name' and 'Indicator Name' and displaying it
data_ele_coal_n = data_ele_coal.loc[:, ['Country Name', 'Indicator Name']]
print(data_ele_coal_n)

#Dropping the null columns and displaying the outcome
data_ele_coal = data_ele_coal.dropna(axis=1)
print(data_ele_coal)


def pop_ele_source():
    '''
    This function creates a heatmap which shows correlation
   between different sources of electricity production and the population 
   growth in year 2014 across various countries.

    Returns
    -------
    None.

    '''
    
    #Selecting the required columns from both the data sets 
    pop_new = data_pop_growth[['Country Name', '2014']]
    col_new = data_ele_coal[['Country Name', '2014']]
    hydro_new = data_ele_hydro[['Country Name', '2014']]
    gas_new = data_ele_gas[['Country Name', '2014']]
    nuclear_new = data_ele_nuclear[['Country Name', '2014']]
    oil_new = data_ele_oil[['Country Name', '2014']]
    
    #Merging the dataframe on Country Name with different suffixes
    merge_data = pd.merge(
        pop_new, col_new, on='Country Name', suffixes=('_pop', '_col'))
    merge_data1 = pd.merge(merge_data, hydro_new,
                           on='Country Name', suffixes=('_hydro', '_hydro'))
    merge_data2 = pd.merge(merge_data1, gas_new,
                           on='Country Name', suffixes=('_hydro', '_gas'))
    merge_data3 = pd.merge(merge_data2, nuclear_new,
                           on='Country Name', suffixes=('_gas', '_nuclear'))
    merge_all_data = pd.merge(merge_data3, oil_new,
                              on='Country Name', suffixes=('_nuclear', '_oil'))
    
    #Calculating correlation matrix
    correlation_data = merge_all_data.corr()
    
    #Correlation produced using heatmap and adding title
    sns.heatmap(correlation_data, annot=True, cmap='viridis', fmt='.3f')
    plt.title(
        '''Correlation Heatmap of Countries and 
        source of electricity production''')
        
    #display the plot
    plt.show()

#Call the function
pop_ele_source()

#Filtering the dataframe and selecting the 'Forest area(sq.km)' indicator
data_forest = data_y[data_y['Indicator Name'] == 'Forest area (sq. km)']

#Dropping the null values and displaying it
data_forest = data_forest.dropna(axis=1)
print(data_forest)


def forest_pop():
    '''
    Visulazitaion is done in this function on merging population growth
    and forest area in year 1990, 2000 and 2010 and heatmap is produced 
    by correlating them

    '''
    #Selective columns are summened here 
    pop_1990 = data_pop_growth[['Country Name', '1990']]
    for_1990 = data_forest[['Country Name', '1990']]
    pop_2000 = data_pop_growth[['Country Name', '2000']]
    for_2000 = data_forest[['Country Name', '2000']]
    pop_2010 = data_pop_growth[['Country Name', '2010']]
    for_2010 = data_forest[['Country Name', '2010']]
    
    #Merging data based on Country Name
    merge_forest = pd.merge(
        pop_1990, for_1990, on='Country Name', suffixes=('_pop', '_for'))
    merge_forest1 = pd.merge(merge_forest, pop_2000,
                             on='Country Name', suffixes=('_pop', '_pop'))
    merge_forest2 = pd.merge(merge_forest1, for_2000,
                             on='Country Name', suffixes=('_pop', '_for'))
    merge_forest3 = pd.merge(merge_forest2, pop_2010,
                             on='Country Name', suffixes=('_pop', '_pop'))
    all_merge_forest = pd.merge(
        merge_forest3, for_2010, on='Country Name', suffixes=('_pop', '_for'))
    
    #Calculating correlation matrix
    correlation_forest = all_merge_forest.corr()
    
    #Correlation produced using heatmap and titling it
    sns.heatmap(correlation_forest, annot=True, cmap='cividis', fmt='.3f')
    plt.title('Correlation Heatmap of Countries and forest area')
    
    #display heatmap
    plt.show()

#Calling functiom
forest_pop()
