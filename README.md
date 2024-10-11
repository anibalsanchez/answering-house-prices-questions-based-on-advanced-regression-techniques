# Answering House Prices Questions using Advanced Regression Techniques

## Background

As a Udemy [Data Scientist](https://www.udacity.com/enrollment/nd025) Nanodegree Program student, I'm tasked with writing a blog post and a kernel following the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) process.

The Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition is a fantastic playground for budding data scientists like myself. It challenges us to predict house prices in Ames, Iowa, leveraging 79 predictor variables through machine learning models. This well-analyzed dataset has received over 20,000 submissions, making it an excellent resource for developing and showcasing our skills.

The notebook and source code is available here:

- Blog post: <https://blog.anibalhsanchez.com/en/blogging/85-answering-house-prices-questions-using-advanced-regression-techniques.html>
- Repository: <https://github.com/anibalsanchez/answering-house-prices-questions-based-on-advanced-regression-techniques>

## Objectives

In my blog post, I'll take a fresh approach by adhering to the CRISP-DM process to address three fundamental questions often posed in the housing markets, using the Ames dataset as a case study.

### What are the main house price ranges?

**Identify the primary price ranges for houses in the dataset**. It's essential to identify the specific price ranges encompassing most homes and their distribution. This information will help segment the housing market and tailor the analysis to the most relevant price ranges.

### Which areas can you locate these price ranges?

**Determine the areas or neighborhoods where these price ranges are concentrated**. It is crucial to identify the geographic areas or neighborhoods associated with different price ranges. I can uncover patterns and identify undervalued or overvalued regions by mapping price ranges to specific areas.

### What variables best predict the price range of each home?

**Identify the key variables that best predict the price range of each home**. The dataset contains numerous features describing various aspects of the houses, such as the number of bedrooms, bathrooms, lot size, construction materials, and neighborhood characteristics. Determining the most influential variables that accurately predict the price range for individual homes is vital. This information can guide feature engineering efforts and ensure the most relevant predictors are included in the modeling process.

Following the CRISP-DM process, I'll systematically analyze and preprocess the data, build predictive models, and present the findings in a comprehensive blog post and notebook. This project will allow me to showcase my skills, including data exploration, feature engineering, model selection, and result interpretation.

## Exploring the Data

To start the project, I imported the packages, defined the global variables, and read the Ames, Iowa train dataset CSV file.


```python
import numpy as np
import pandas as pd
from math import expm1
from IPython.display import display # Allows the use of display() for DataFrames
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go

head_n_of_records = 5
seed = 42

init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook

plt.style.use('bmh')
sns.set_style({'axes.grid':False})

# Ignore warnings
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

# Define a color-blind friendly palette
# Using a palette from ColorBrewer which is designed to be color-blind friendly
colorblind_palette = sns.color_palette("colorblind", n_colors=8)

# Show all rows and colums
pd.options.display.max_rows = None
pd.options.display.max_columns = None

%matplotlib inline

study_data = pd.read_csv("train.csv")
study_data_num_rows, study_data_num_columns = (
    study_data.shape
)

# Display markdown formatted output like bold, italic bold etc.'''
from IPython.display import Markdown
def display_md(string):
    display(Markdown(string))

display_md('### Preview of Train Data')
display(study_data.head(n=head_n_of_records))
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-2.35.2.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




### Preview of Train Data



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>196.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706</td>
      <td>Unf</td>
      <td>0</td>
      <td>150</td>
      <td>856</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>548</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978</td>
      <td>Unf</td>
      <td>0</td>
      <td>284</td>
      <td>1262</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>460</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>162.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486</td>
      <td>Unf</td>
      <td>0</td>
      <td>434</td>
      <td>920</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>608</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216</td>
      <td>Unf</td>
      <td>0</td>
      <td>540</td>
      <td>756</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3</td>
      <td>642</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655</td>
      <td>Unf</td>
      <td>0</td>
      <td>490</td>
      <td>1145</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>836</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>



```python
display_md('**Shape of our train data:**')
display(study_data.shape)

display_md('**Name of our variables:**')
display(study_data.columns.values)
```


**Shape of our train data:**



    (1460, 81)



**Name of our variables:**



    array(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
           'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
           'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl',
           'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
           'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
           'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
           'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
           'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
           'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
           'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
           'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
           'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition',
           'SalePrice'], dtype=object)


## Feature set Exploration

The Ames, Iowa train dataset has 79 variables in total, including the SalePrice variable, which is our target variable. The remaining variables are used for clustering and characterization.

In this list, I briefly describe each feature and its type (Categorical, Ordinal, or Numeric).

| Name | Description | Type |
|------|-------------|------|
| MSSubClass | Identifies the type of dwelling involved in the sale | Categorical |
| MSZoning | Identifies the general zoning classification of the sale | Categorical |
| LotFrontage | Linear feet of street connected to property | Numeric |
| LotArea | Lot size in square feet | Numeric |
| Street | Type of road access to property | Categorical |
| Alley | Type of alley access to property | Categorical |
| LotShape | General shape of property | Ordinal |
| LandContour | Flatness of the property | Categorical |
| Utilities | Type of utilities available | Categorical |
| LotConfig | Lot configuration | Categorical |
| LandSlope | Slope of property | Ordinal |
| Neighborhood | Physical locations within Ames city limits | Categorical |
| Condition1 | Proximity to various conditions | Categorical |
| Condition2 | Proximity to various conditions (if more than one is present) | Categorical |
| BldgType | Type of dwelling | Categorical |
| HouseStyle | Style of dwelling | Categorical |
| OverallQual | Rates the overall material and finish of the house | Ordinal |
| OverallCond | Rates the overall condition of the house | Ordinal |
| YearBuilt | Original construction date | Numeric |
| YearRemodAdd | Remodel date (same as construction date if no remodeling or additions) | Numeric |
| RoofStyle | Type of roof | Categorical |
| RoofMatl | Roof material | Categorical |
| Exterior1st | Exterior covering on house | Categorical |
| Exterior2nd | Exterior covering on house (if more than one material) | Categorical |
| MasVnrType | Masonry veneer type | Categorical |
| MasVnrArea | Masonry veneer area in square feet | Numeric |
| ExterQual | Evaluates the quality of the material on the exterior | Ordinal |
| ExterCond | Evaluates the present condition of the material on the exterior | Ordinal |
| Foundation | Type of foundation | Categorical |
| BsmtQual | Evaluates the height of the basement | Ordinal |
| BsmtCond | Evaluates the general condition of the basement | Ordinal |
| BsmtExposure | Refers to walkout or garden level walls | Ordinal |
| BsmtFinType1 | Rating of basement finished area | Ordinal |
| BsmtFinSF1 | Type 1 finished square feet | Numeric |
| BsmtFinType2 | Rating of basement finished area (if multiple types) | Ordinal |
| BsmtFinSF2 | Type 2 finished square feet | Numeric |
| BsmtUnfSF | Unfinished square feet of basement area | Numeric |
| TotalBsmtSF | Total square feet of basement area | Numeric |
| Heating | Type of heating | Categorical |
| HeatingQC | Heating quality and condition | Ordinal |
| CentralAir | Central air conditioning | Categorical |
| Electrical | Electrical system | Categorical |
| 1stFlrSF | First Floor square feet | Numeric |
| 2ndFlrSF | Second floor square feet | Numeric |
| LowQualFinSF | Low quality finished square feet (all floors) | Numeric |
| GrLivArea | Above grade (ground) living area square feet | Numeric |
| BsmtFullBath | Basement full bathrooms | Numeric |
| BsmtHalfBath | Basement half bathrooms | Numeric |
| FullBath | Full bathrooms above grade | Numeric |
| HalfBath | Half baths above grade | Numeric |
| BedroomAbvGr | Bedrooms above grade (does NOT include basement bedrooms) | Numeric |
| KitchenAbvGr | Kitchens above grade | Numeric |
| KitchenQual | Kitchen quality | Ordinal |
| TotRmsAbvGrd | Total rooms above grade (does not include bathrooms) | Numeric |
| Functional | Home functionality (Assume typical unless deductions are warranted) | Ordinal |
| Fireplaces | Number of fireplaces | Numeric |
| FireplaceQu | Fireplace quality | Ordinal |
| GarageType | Garage location | Categorical |
| GarageYrBlt | Year garage was built | Numeric |
| GarageFinish | Interior finish of the garage | Ordinal |
| GarageCars | Size of garage in car capacity | Numeric |
| GarageArea | Size of garage in square feet | Numeric |
| GarageQual | Garage quality | Ordinal |
| GarageCond | Garage condition | Ordinal |
| PavedDrive | Paved driveway | Ordinal |
| WoodDeckSF | Wood deck area in square feet | Numeric |
| OpenPorchSF | Open porch area in square feet | Numeric |
| EnclosedPorch | Enclosed porch area in square feet | Numeric |
| 3SsnPorch | Three season porch area in square feet | Numeric |
| ScreenPorch | Screen porch area in square feet | Numeric |
| PoolArea | Pool area in square feet | Numeric |
| PoolQC | Pool quality | Ordinal |
| Fence | Fence quality | Ordinal |
| MiscFeature | Miscellaneous feature not covered in other categories | Categorical |
| MiscVal | $Value of miscellaneous feature | Numeric |
| MoSold | Month Sold (MM) | Numeric |
| YrSold | Year Sold (YYYY) | Numeric |
| SaleType | Type of sale | Categorical |
| SaleCondition | Condition of sale | Categorical |


```python
# Definition of Features Types

categorical_features = ["MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "Foundation", "Heating", "CentralAir", "Electrical", "GarageType", "SaleType", "SaleCondition"]
numerical_features = ["LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]
ordinal_features = ["LotShape", "LandSlope", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive"]

```

## Outliers Removal

In the competition "Ames Iowa Housing Dataset - Special Notes", there is a specific requirement to remove outlier observations.

_SPECIAL NOTES: There are 5 observations that an instructor may wish to remove from the data set before giving it to students (a plot of SALE PRICE versus GR LIV AREA will indicate them quickly). Three of them are true outliers (Partial Sales that likely don’t represent actual market values) and two of them are simply unusual sales (very large houses priced relatively appropriately). I would recommend removing any houses with more than 4000 square feet from the data set (which eliminates these 5 unusual observations) before assigning it to students._

The following code takes care of this point.


```python
def scatter_plot(x, y, title, xaxis, yaxis, size, c_scale):
    trace = go.Scatter(
    x = x,
    y = y,
    mode = 'markers',
    marker = dict(color = y, size = size, showscale = True, colorscale = c_scale))
    layout = go.Layout(hovermode= 'closest', title = title, xaxis = dict(title = xaxis), yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig)

scatter_plot(study_data.GrLivArea, study_data.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
```


<div>                            <div id="76134c2c-35ab-49c6-aebe-06a10f82ee18" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("76134c2c-35ab-49c6-aebe-06a10f82ee18")) {                    Plotly.newPlot(                        "76134c2c-35ab-49c6-aebe-06a10f82ee18",                        [{"marker":{"color":[208500,181500,223500,140000,250000,143000,307000,200000,129900,118000,129500,345000,144000,279500,157000,132000,149000,90000,159000,139000,325300,139400,230000,129900,154000,256300,134800,306000,207500,68500,40000,149350,179900,165500,277500,309000,145000,153000,109000,82000,160000,170000,144000,130250,141000,319900,239686,249700,113000,127000,177000,114500,110000,385000,130000,180500,172500,196500,438780,124900,158000,101000,202500,140000,219500,317000,180000,226000,80000,225000,244000,129500,185000,144900,107400,91000,135750,127000,136500,110000,193500,153500,245000,126500,168500,260000,174000,164500,85000,123600,109900,98600,163500,133900,204750,185000,214000,94750,83000,128950,205000,178000,118964,198900,169500,250000,100000,115000,115000,190000,136900,180000,383970,217000,259500,176000,139000,155000,320000,163990,180000,100000,136000,153900,181000,84500,128000,87000,155000,150000,226000,244000,150750,220000,180000,174000,143000,171000,230000,231500,115000,260000,166000,204000,125000,130000,105000,222500,141000,115000,122000,372402,190000,235000,125000,79000,109500,269500,254900,320000,162500,412500,220000,103200,152000,127500,190000,325624,183500,228000,128500,215000,239000,163000,184000,243000,211000,172500,501837,100000,177000,200100,120000,200000,127000,475000,173000,135000,153337,286000,315000,184000,192000,130000,127000,148500,311872,235000,104000,274900,140000,171500,112000,149000,110000,180500,143900,141000,277000,145000,98000,186000,252678,156000,161750,134450,210000,107000,311500,167240,204900,200000,179900,97000,386250,112000,290000,106000,125000,192500,148000,403000,94500,128200,216500,89500,185500,194500,318000,113000,262500,110500,79000,120000,205000,241500,137000,140000,180000,277000,76500,235000,173000,158000,145000,230000,207500,220000,231500,97000,176000,276000,151000,130000,73000,175500,185000,179500,120500,148000,266000,241500,290000,139000,124500,205000,201000,141000,415298,192000,228500,185000,207500,244600,179200,164700,159000,88000,122000,153575,233230,135900,131000,235000,167000,142500,152000,239000,175000,158500,157000,267000,205000,149900,295000,305900,225000,89500,82500,360000,165600,132000,119900,375000,178000,188500,260000,270000,260000,187500,342643,354000,301000,126175,242000,87000,324000,145250,214500,78000,119000,139000,284000,207000,192000,228950,377426,214000,202500,155000,202900,82000,87500,266000,85000,140200,151500,157500,154000,437154,318061,190000,95000,105900,140000,177500,173000,134000,130000,280000,156000,145000,198500,118000,190000,147000,159000,165000,132000,162000,172400,134432,125000,123000,219500,61000,148000,340000,394432,179000,127000,187750,213500,76000,240000,192000,81000,125000,191000,426000,119000,215000,106500,100000,109000,129000,123000,169500,67000,241000,245500,164990,108000,258000,168000,150000,115000,177000,280000,339750,60000,145000,222000,115000,228000,181134,149500,239000,126000,142000,206300,215000,113000,315000,139000,135000,275000,109008,195400,175000,85400,79900,122500,181000,81000,212000,116000,119000,90350,110000,555000,118000,162900,172500,210000,127500,190000,199900,119500,120000,110000,280000,204000,210000,188000,175500,98000,256000,161000,110000,263435,155000,62383,188700,124000,178740,167000,146500,250000,187000,212000,190000,148000,440000,251000,132500,208900,380000,297000,89471,326000,374000,155000,164000,132500,147000,156000,175000,160000,86000,115000,133000,172785,155000,91300,34900,430000,184000,130000,120000,113000,226700,140000,289000,147000,124500,215000,208300,161000,124500,164900,202665,129900,134000,96500,402861,158000,265000,211000,234000,106250,150000,159000,184750,315750,176000,132000,446261,86000,200624,175000,128000,107500,39300,178000,107500,188000,111250,158000,272000,315000,248000,213250,133000,179665,229000,210000,129500,125000,263000,140000,112500,255500,108000,284000,113000,141000,108000,175000,234000,121500,170000,108000,185000,268000,128000,325000,214000,316600,135960,142600,120000,224500,170000,139000,118500,145000,164500,146000,131500,181900,253293,118500,325000,133000,369900,130000,137000,143000,79500,185900,451950,138000,140000,110000,319000,114504,194201,217500,151000,275000,141000,220000,151000,221000,205000,152000,225000,359100,118500,313000,148000,261500,147000,75500,137500,183200,105500,314813,305000,67000,240000,135000,168500,165150,160000,139900,153000,135000,168500,124000,209500,82500,139400,144000,200000,60000,93000,85000,264561,274000,226000,345000,152000,370878,143250,98300,155000,155000,84500,205950,108000,191000,135000,350000,88000,145500,149000,97500,167000,197900,402000,110000,137500,423000,230500,129000,193500,168000,137500,173500,103600,165000,257500,140000,148500,87000,109500,372500,128500,143000,159434,173000,285000,221000,207500,227875,148800,392000,194700,141000,755000,335000,108480,141500,176000,89000,123500,138500,196000,312500,140000,361919,140000,213000,55000,302000,254000,179540,109900,52000,102776,189000,129000,130500,165000,159500,157000,341000,128500,275000,143000,124500,135000,320000,120500,222000,194500,110000,103000,236500,187500,222500,131400,108000,163000,93500,239900,179000,190000,132000,142000,179000,175000,180000,299800,236000,265979,260400,98000,96500,162000,217000,275500,156000,172500,212000,158900,179400,290000,127500,100000,215200,337000,270000,264132,196500,160000,216837,538000,134900,102000,107000,114500,395000,162000,221500,142500,144000,135000,176000,175900,187100,165500,128000,161500,139000,233000,107900,187500,160200,146800,269790,225000,194500,171000,143500,110000,485000,175000,200000,109900,189000,582933,118000,227680,135500,223500,159950,106000,181000,144500,55993,157900,116000,224900,137000,271000,155000,224000,183000,93000,225000,139500,232600,385000,109500,189000,185000,147400,166000,151000,237000,167000,139950,128000,153500,100000,144000,130500,140000,157500,174900,141000,153900,171000,213000,133500,240000,187000,131500,215000,164000,158000,170000,127000,147000,174000,152000,250000,189950,131500,152000,132500,250580,148500,248900,129000,169000,236000,109500,200500,116000,133000,66500,303477,132250,350000,148000,136500,157000,187500,178000,118500,100000,328900,145000,135500,268000,149500,122900,172500,154500,165000,118858,140000,106500,142953,611657,135000,110000,153000,180000,240000,125500,128000,255000,250000,131000,174000,154300,143500,88000,145000,173733,75000,35311,135000,238000,176500,201000,145900,169990,193000,207500,175000,285000,176000,236500,222000,201000,117500,320000,190000,242000,79900,184900,253000,239799,244400,150900,214000,150000,143000,137500,124900,143000,270000,192500,197500,129000,119900,133900,172000,127500,145000,124000,132000,185000,155000,116500,272000,155000,239000,214900,178900,160000,135000,37900,140000,135000,173000,99500,182000,167500,165000,85500,199900,110000,139000,178400,336000,159895,255900,126000,125000,117000,395192,195000,197000,348000,168000,187000,173900,337500,121600,136500,185000,91000,206000,82000,86000,232000,136905,181000,149900,163500,88000,240000,102000,135000,100000,165000,85000,119200,227000,203000,187500,160000,213490,176000,194000,87000,191000,287000,112500,167500,293077,105000,118000,160000,197000,310000,230000,119750,84000,315500,287000,97000,80000,155000,173000,196000,262280,278000,139600,556581,145000,115000,84900,176485,200141,165000,144500,255000,180000,185850,248000,335000,220000,213500,81000,90000,110500,154000,328000,178000,167900,151400,135000,135000,154000,91500,159500,194000,219500,170000,138800,155900,126000,145000,133000,192000,160000,187500,147000,83500,252000,137500,197000,92900,160000,136500,146000,129000,176432,127000,170000,128000,157000,60000,119500,135000,159500,106000,325000,179900,274725,181000,280000,188000,205000,129900,134500,117000,318000,184100,130000,140000,133700,118400,212900,112000,118000,163900,115000,174000,259000,215000,140000,135000,93500,117500,239500,169000,102000,119000,94000,196000,144000,139000,197500,424870,80000,80000,149000,180000,174500,116900,143000,124000,149900,230000,120500,201800,218000,179900,230000,235128,185000,146000,224000,129000,108959,194000,233170,245350,173000,235000,625000,171000,163000,171900,200500,239000,285000,119500,115000,154900,93000,250000,392500,745000,120000,186700,104900,95000,262000,195000,189000,168000,174000,125000,165000,158000,176000,219210,144000,178000,148000,116050,197900,117000,213000,153500,271900,107000,200000,140000,290000,189000,164000,113000,145000,134500,125000,112000,229456,80500,91500,115000,134000,143000,137900,184000,145000,214000,147000,367294,127000,190000,132500,101800,142000,130000,138887,175500,195000,142500,265900,224900,248328,170000,465000,230000,178000,186500,169900,129500,119000,244000,171750,130000,294000,165400,127500,301500,99900,190000,151000,181000,128900,161500,180500,181000,183900,122000,378500,381000,144000,260000,185750,137000,177000,139000,137000,162000,197900,237000,68400,227000,180000,150500,139000,169000,132500,143000,190000,278000,281000,180500,119500,107500,162900,115000,138500,155000,140000,160000,154000,225000,177500,290000,232000,130000,325000,202500,138000,147000,179200,335000,203000,302000,333168,119000,206900,295493,208900,275000,111000,156500,72500,190000,82500,147000,55000,79000,130500,256000,176500,227000,132500,100000,125500,125000,167900,135000,52500,200000,128500,123000,155000,228500,177000,155835,108500,262500,283463,215000,122000,200000,171000,134900,410000,235000,170000,110000,149900,177500,315000,189000,260000,104900,156932,144152,216000,193000,127000,144000,232000,105000,165500,274300,466500,250000,239000,91000,117000,83000,167500,58500,237500,157000,112000,105000,125500,250000,136000,377500,131000,235000,124000,123000,163000,246578,281213,160000,137500,138000,137450,120000,193000,193879,282922,105000,275000,133000,112000,125500,215000,230000,140000,90000,257000,207000,175900,122500,340000,124000,223000,179900,127500,136500,274970,144000,142000,271000,140000,119000,182900,192140,143750,64500,186500,160000,174000,120500,394617,149700,197000,191000,149300,310000,121000,179600,129000,157900,240000,112000,92000,136000,287090,145000,84500,185000,175000,210000,266500,142125,147500],"colorscale":[[0.0,"rgb(150,0,90)"],[0.125,"rgb(0,0,200)"],[0.25,"rgb(0,25,255)"],[0.375,"rgb(0,152,255)"],[0.5,"rgb(44,255,150)"],[0.625,"rgb(151,255,0)"],[0.75,"rgb(255,234,0)"],[0.875,"rgb(255,111,0)"],[1.0,"rgb(255,0,0)"]],"showscale":true,"size":10},"mode":"markers","x":[1710,1262,1786,1717,2198,1362,1694,2090,1774,1077,1040,2324,912,1494,1253,854,1004,1296,1114,1339,2376,1108,1795,1060,1060,1600,900,1704,1600,520,1317,1228,1234,1700,1561,2452,1097,1297,1057,1152,1324,1328,884,938,1150,1752,2149,1656,1452,955,1470,1176,816,1842,1360,1425,1739,1720,2945,780,1158,1111,1370,1710,2034,2473,2207,1479,747,2287,2223,845,1718,1086,1605,988,952,1285,1768,1230,2142,1337,1563,1065,1474,2417,1560,1224,1526,990,1040,1235,964,2291,1786,1470,1588,960,835,1225,1610,1732,1535,1226,1818,1992,1047,789,1517,1844,1855,1430,2696,2259,2320,1458,1092,1125,3222,1456,988,1123,1080,1199,1586,754,958,840,1348,1053,2157,2054,1327,1296,1721,1682,1214,1959,1852,1764,864,1734,1385,1501,1728,1709,875,2035,1080,1344,969,1710,1993,1252,1200,1096,1040,1968,1947,2462,1232,2668,1541,882,1616,1355,1867,2161,1720,1707,1382,1656,1767,1362,1651,2158,2060,1920,2234,968,1525,1802,1340,2082,1252,3608,1217,1656,1224,1593,2727,1479,1431,1709,864,1456,1726,3112,2229,1713,1121,1279,1310,848,1284,1442,1696,1100,2062,1092,864,1212,1852,990,1392,1236,1436,1328,1954,1248,1498,2267,1552,864,2392,1302,2520,987,912,1555,1194,2794,987,894,1960,987,1414,1744,1694,1487,1566,866,1440,1217,2110,1872,1928,1375,1668,2144,1306,1625,1640,1302,1314,2291,1728,1604,1792,882,1382,2574,1212,1316,764,1422,1511,2192,778,1113,1939,1363,2270,1632,816,1548,1560,864,2121,2022,1982,1262,1314,1468,1575,1250,1734,858,900,1396,1919,1716,1716,2263,1644,1003,1558,1950,1743,1152,1336,2452,1541,894,3493,2000,2243,1406,861,1944,1501,972,1118,2036,1641,1432,2353,1959,2646,1472,2596,2468,2730,1163,2978,803,1719,1383,2134,1192,1728,1056,1629,1358,1638,1786,1922,1536,1621,1215,1908,841,1040,1684,1112,1577,958,1478,1626,2728,1869,1453,1111,720,1595,1200,1167,1142,1352,1924,912,1505,1922,987,1574,1344,1394,1431,1268,1287,1664,1588,752,1319,1928,904,914,2466,1856,1800,1691,1301,1797,784,1953,1269,1184,1125,1479,2332,1367,1961,882,788,1034,1144,894,1812,1077,1550,1288,1310,672,2263,1572,1620,1639,1680,2172,2078,1276,1056,1478,1028,2097,1340,1400,2624,1134,1056,1344,1602,988,2630,1196,1389,1644,907,1208,1412,987,1198,1365,1604,630,1661,1118,904,694,1196,2402,1440,1573,1258,1908,1689,1888,1886,1376,1183,813,1533,1756,1590,1728,1242,1344,1663,1666,1203,1935,1135,864,1660,1040,1414,1277,1644,1634,1710,1502,1969,1072,1976,1652,970,1493,2643,1718,1131,1850,1792,1826,1216,999,1113,1073,1484,2414,630,1304,1578,1456,1269,886,720,3228,1820,899,912,1218,1768,1214,1801,1322,1960,1911,1218,1378,1041,1363,1368,864,1080,789,2020,2119,2344,1796,2080,1294,1244,1664,4676,2398,1266,928,2713,605,2515,1509,1362,827,334,1414,1347,1724,864,1159,1601,1838,2285,1680,767,1496,2183,1635,768,825,2094,1069,928,1717,1126,2046,1048,1092,1336,1446,1557,1392,1389,996,1674,2295,1647,2504,1535,2132,943,1728,864,1692,1430,1109,1216,1477,1320,1392,1795,1429,2042,816,2775,1573,2028,838,860,1473,935,1582,2296,816,848,924,1826,1368,1402,1647,1556,1904,1375,1915,1200,1494,1986,1040,2008,3194,1029,2153,1032,1872,1120,630,1054,1509,832,1828,2262,864,2614,980,1512,1790,1116,1422,1520,2080,1350,1750,1554,1411,1056,1056,3395,800,1387,796,1567,1518,1929,2704,1620,1766,981,1048,1094,1839,630,1665,1510,1716,1469,2113,1092,1053,1502,1458,1486,1935,2448,1392,1181,2097,1936,2380,1679,1437,1180,1476,1369,1208,1839,1136,1441,1774,792,2046,988,923,1520,1291,1668,1839,2090,1761,1102,1419,1362,848,4316,2519,1073,1539,1137,616,1148,894,1391,1800,1164,2576,1812,1484,1092,1824,1324,1456,904,729,1178,1228,960,1479,1350,2554,1178,2418,971,1742,848,864,1470,1698,864,1680,1232,1776,1208,1616,1146,2031,1144,948,1768,1040,1801,1200,1728,1432,912,1349,1464,1337,2715,2256,2640,1720,1529,1140,1320,1494,2098,1026,1471,1768,1386,1501,2531,864,1301,1547,2365,1494,1506,1714,1750,1836,3279,858,1220,1117,912,1973,1204,1614,894,2020,1004,1253,1603,1430,1110,1484,1342,1652,2084,901,2087,1145,1062,2013,1496,1895,1564,1285,773,3140,1768,1688,1196,1456,2822,1128,1428,980,1576,1086,2138,1309,848,1044,1442,1250,1661,1008,1689,1052,1358,1640,936,1733,1489,1489,2084,784,1434,2126,1223,1392,1200,1829,1516,1144,1067,1559,987,1099,1200,1482,1539,1165,1800,1416,1701,1775,864,2358,1855,848,1456,1646,1445,1779,1040,1026,1481,1370,2654,1426,1039,1097,1148,1372,1002,1646,1120,2320,1949,894,1682,910,1268,1131,2610,1040,2224,1155,864,1090,1717,1593,2230,892,1709,1712,1393,2217,1505,924,1683,1068,1383,1535,1796,951,2240,2364,1236,858,1306,1509,1670,902,1063,1636,2057,902,1484,2274,1268,1015,2002,1224,1092,480,1229,2127,1414,1721,2200,1316,1617,1686,1126,2374,1978,1788,2236,1466,925,1905,1500,2069,747,1200,1971,1962,2403,1728,2060,1440,1632,1344,1869,1144,1629,1776,1381,864,965,768,1968,980,1958,1229,1057,1337,1416,858,2872,1548,1800,1894,1484,1308,1098,968,1095,1192,1626,918,1428,2019,1382,869,1241,894,1121,999,2612,1266,2290,1734,1164,1635,1940,2030,1576,2392,1742,1851,1500,1718,1230,1050,1442,1077,1208,944,691,1574,1680,1504,985,1657,1092,1710,1522,1271,1664,1502,1022,1082,1665,1504,1360,1472,1506,1132,1220,1248,1504,2898,882,1264,1646,1376,1218,1928,3082,2520,1654,954,845,1620,2263,1344,630,1803,1632,1306,2329,2524,1733,2868,990,1771,930,1302,1316,1977,1526,1989,1523,1364,1850,2184,1991,1338,894,2337,1103,1154,2260,1571,1611,2521,893,1048,1556,1456,1426,1240,1740,1466,1096,848,990,1258,1040,1459,1251,1498,996,1092,1953,1709,1247,1040,1252,1694,1200,936,1314,1355,1088,1324,1601,438,950,1134,1194,1302,2622,1442,2021,1690,1836,1658,1964,816,1008,833,1734,1419,894,1601,1040,1012,1552,960,698,1482,1005,1555,1530,1959,936,1981,974,2210,2020,1600,986,1252,1020,1567,1167,952,1868,2828,1006,924,1576,1298,1564,1111,1482,932,1466,1811,816,1820,1437,1265,1314,1580,1876,1456,1640,894,1258,1432,1502,1694,1671,2108,3627,1118,1261,1250,3086,2345,2872,923,1224,1343,1124,2514,1652,4476,1130,1572,1221,1699,1624,1660,1804,1622,1441,1472,1224,1352,1456,1863,1690,1212,1382,864,1779,1348,1630,1074,2196,1056,1700,1283,1660,1845,1752,672,960,999,894,1902,1314,912,1218,912,1211,1846,2136,1490,1138,1933,912,1702,1507,2620,1190,1224,1188,1964,1784,1626,1948,1141,1484,1768,1689,1173,2076,1517,1868,1553,1034,2058,988,2110,1405,874,2167,1656,1367,1987,864,1166,1054,1675,1050,1788,1824,1337,1452,1889,2018,3447,1524,1524,1489,935,1357,1250,1920,1395,1724,2031,1128,1573,1339,1040,1824,2447,1412,1328,1582,1659,1970,1152,1302,2372,1664,864,1052,1128,1072,5642,1246,1983,1494,2526,1616,1708,1652,1368,990,1122,1294,1902,1274,2810,2599,948,2112,1630,1352,1787,948,1478,720,1923,708,1795,796,774,816,2792,1632,1588,954,816,1360,1365,1334,1656,693,1861,864,872,1114,2169,1913,1456,960,2156,1776,1494,2358,2634,1716,1176,3238,1865,1920,892,1078,1573,1980,2601,1530,1738,1412,1200,1674,1790,1475,848,1668,1374,1661,2097,2633,1958,1571,790,1604,987,1394,864,2117,1762,1416,1258,1154,2784,2526,1746,1218,1525,1584,900,1912,1500,2482,1687,1513,1904,1608,1158,1593,1294,1464,1214,1646,768,833,1363,2093,1840,1668,1040,1844,1848,1569,2290,2450,1144,1844,1416,1069,848,2201,1344,1252,2127,1558,804,1440,1838,958,968,1792,1126,1537,864,1932,1236,1725,2555,848,2007,952,1422,913,1188,2090,1346,630,1792,1578,1072,1140,1221,1647,2073,2340,1078,1256],"y":[208500,181500,223500,140000,250000,143000,307000,200000,129900,118000,129500,345000,144000,279500,157000,132000,149000,90000,159000,139000,325300,139400,230000,129900,154000,256300,134800,306000,207500,68500,40000,149350,179900,165500,277500,309000,145000,153000,109000,82000,160000,170000,144000,130250,141000,319900,239686,249700,113000,127000,177000,114500,110000,385000,130000,180500,172500,196500,438780,124900,158000,101000,202500,140000,219500,317000,180000,226000,80000,225000,244000,129500,185000,144900,107400,91000,135750,127000,136500,110000,193500,153500,245000,126500,168500,260000,174000,164500,85000,123600,109900,98600,163500,133900,204750,185000,214000,94750,83000,128950,205000,178000,118964,198900,169500,250000,100000,115000,115000,190000,136900,180000,383970,217000,259500,176000,139000,155000,320000,163990,180000,100000,136000,153900,181000,84500,128000,87000,155000,150000,226000,244000,150750,220000,180000,174000,143000,171000,230000,231500,115000,260000,166000,204000,125000,130000,105000,222500,141000,115000,122000,372402,190000,235000,125000,79000,109500,269500,254900,320000,162500,412500,220000,103200,152000,127500,190000,325624,183500,228000,128500,215000,239000,163000,184000,243000,211000,172500,501837,100000,177000,200100,120000,200000,127000,475000,173000,135000,153337,286000,315000,184000,192000,130000,127000,148500,311872,235000,104000,274900,140000,171500,112000,149000,110000,180500,143900,141000,277000,145000,98000,186000,252678,156000,161750,134450,210000,107000,311500,167240,204900,200000,179900,97000,386250,112000,290000,106000,125000,192500,148000,403000,94500,128200,216500,89500,185500,194500,318000,113000,262500,110500,79000,120000,205000,241500,137000,140000,180000,277000,76500,235000,173000,158000,145000,230000,207500,220000,231500,97000,176000,276000,151000,130000,73000,175500,185000,179500,120500,148000,266000,241500,290000,139000,124500,205000,201000,141000,415298,192000,228500,185000,207500,244600,179200,164700,159000,88000,122000,153575,233230,135900,131000,235000,167000,142500,152000,239000,175000,158500,157000,267000,205000,149900,295000,305900,225000,89500,82500,360000,165600,132000,119900,375000,178000,188500,260000,270000,260000,187500,342643,354000,301000,126175,242000,87000,324000,145250,214500,78000,119000,139000,284000,207000,192000,228950,377426,214000,202500,155000,202900,82000,87500,266000,85000,140200,151500,157500,154000,437154,318061,190000,95000,105900,140000,177500,173000,134000,130000,280000,156000,145000,198500,118000,190000,147000,159000,165000,132000,162000,172400,134432,125000,123000,219500,61000,148000,340000,394432,179000,127000,187750,213500,76000,240000,192000,81000,125000,191000,426000,119000,215000,106500,100000,109000,129000,123000,169500,67000,241000,245500,164990,108000,258000,168000,150000,115000,177000,280000,339750,60000,145000,222000,115000,228000,181134,149500,239000,126000,142000,206300,215000,113000,315000,139000,135000,275000,109008,195400,175000,85400,79900,122500,181000,81000,212000,116000,119000,90350,110000,555000,118000,162900,172500,210000,127500,190000,199900,119500,120000,110000,280000,204000,210000,188000,175500,98000,256000,161000,110000,263435,155000,62383,188700,124000,178740,167000,146500,250000,187000,212000,190000,148000,440000,251000,132500,208900,380000,297000,89471,326000,374000,155000,164000,132500,147000,156000,175000,160000,86000,115000,133000,172785,155000,91300,34900,430000,184000,130000,120000,113000,226700,140000,289000,147000,124500,215000,208300,161000,124500,164900,202665,129900,134000,96500,402861,158000,265000,211000,234000,106250,150000,159000,184750,315750,176000,132000,446261,86000,200624,175000,128000,107500,39300,178000,107500,188000,111250,158000,272000,315000,248000,213250,133000,179665,229000,210000,129500,125000,263000,140000,112500,255500,108000,284000,113000,141000,108000,175000,234000,121500,170000,108000,185000,268000,128000,325000,214000,316600,135960,142600,120000,224500,170000,139000,118500,145000,164500,146000,131500,181900,253293,118500,325000,133000,369900,130000,137000,143000,79500,185900,451950,138000,140000,110000,319000,114504,194201,217500,151000,275000,141000,220000,151000,221000,205000,152000,225000,359100,118500,313000,148000,261500,147000,75500,137500,183200,105500,314813,305000,67000,240000,135000,168500,165150,160000,139900,153000,135000,168500,124000,209500,82500,139400,144000,200000,60000,93000,85000,264561,274000,226000,345000,152000,370878,143250,98300,155000,155000,84500,205950,108000,191000,135000,350000,88000,145500,149000,97500,167000,197900,402000,110000,137500,423000,230500,129000,193500,168000,137500,173500,103600,165000,257500,140000,148500,87000,109500,372500,128500,143000,159434,173000,285000,221000,207500,227875,148800,392000,194700,141000,755000,335000,108480,141500,176000,89000,123500,138500,196000,312500,140000,361919,140000,213000,55000,302000,254000,179540,109900,52000,102776,189000,129000,130500,165000,159500,157000,341000,128500,275000,143000,124500,135000,320000,120500,222000,194500,110000,103000,236500,187500,222500,131400,108000,163000,93500,239900,179000,190000,132000,142000,179000,175000,180000,299800,236000,265979,260400,98000,96500,162000,217000,275500,156000,172500,212000,158900,179400,290000,127500,100000,215200,337000,270000,264132,196500,160000,216837,538000,134900,102000,107000,114500,395000,162000,221500,142500,144000,135000,176000,175900,187100,165500,128000,161500,139000,233000,107900,187500,160200,146800,269790,225000,194500,171000,143500,110000,485000,175000,200000,109900,189000,582933,118000,227680,135500,223500,159950,106000,181000,144500,55993,157900,116000,224900,137000,271000,155000,224000,183000,93000,225000,139500,232600,385000,109500,189000,185000,147400,166000,151000,237000,167000,139950,128000,153500,100000,144000,130500,140000,157500,174900,141000,153900,171000,213000,133500,240000,187000,131500,215000,164000,158000,170000,127000,147000,174000,152000,250000,189950,131500,152000,132500,250580,148500,248900,129000,169000,236000,109500,200500,116000,133000,66500,303477,132250,350000,148000,136500,157000,187500,178000,118500,100000,328900,145000,135500,268000,149500,122900,172500,154500,165000,118858,140000,106500,142953,611657,135000,110000,153000,180000,240000,125500,128000,255000,250000,131000,174000,154300,143500,88000,145000,173733,75000,35311,135000,238000,176500,201000,145900,169990,193000,207500,175000,285000,176000,236500,222000,201000,117500,320000,190000,242000,79900,184900,253000,239799,244400,150900,214000,150000,143000,137500,124900,143000,270000,192500,197500,129000,119900,133900,172000,127500,145000,124000,132000,185000,155000,116500,272000,155000,239000,214900,178900,160000,135000,37900,140000,135000,173000,99500,182000,167500,165000,85500,199900,110000,139000,178400,336000,159895,255900,126000,125000,117000,395192,195000,197000,348000,168000,187000,173900,337500,121600,136500,185000,91000,206000,82000,86000,232000,136905,181000,149900,163500,88000,240000,102000,135000,100000,165000,85000,119200,227000,203000,187500,160000,213490,176000,194000,87000,191000,287000,112500,167500,293077,105000,118000,160000,197000,310000,230000,119750,84000,315500,287000,97000,80000,155000,173000,196000,262280,278000,139600,556581,145000,115000,84900,176485,200141,165000,144500,255000,180000,185850,248000,335000,220000,213500,81000,90000,110500,154000,328000,178000,167900,151400,135000,135000,154000,91500,159500,194000,219500,170000,138800,155900,126000,145000,133000,192000,160000,187500,147000,83500,252000,137500,197000,92900,160000,136500,146000,129000,176432,127000,170000,128000,157000,60000,119500,135000,159500,106000,325000,179900,274725,181000,280000,188000,205000,129900,134500,117000,318000,184100,130000,140000,133700,118400,212900,112000,118000,163900,115000,174000,259000,215000,140000,135000,93500,117500,239500,169000,102000,119000,94000,196000,144000,139000,197500,424870,80000,80000,149000,180000,174500,116900,143000,124000,149900,230000,120500,201800,218000,179900,230000,235128,185000,146000,224000,129000,108959,194000,233170,245350,173000,235000,625000,171000,163000,171900,200500,239000,285000,119500,115000,154900,93000,250000,392500,745000,120000,186700,104900,95000,262000,195000,189000,168000,174000,125000,165000,158000,176000,219210,144000,178000,148000,116050,197900,117000,213000,153500,271900,107000,200000,140000,290000,189000,164000,113000,145000,134500,125000,112000,229456,80500,91500,115000,134000,143000,137900,184000,145000,214000,147000,367294,127000,190000,132500,101800,142000,130000,138887,175500,195000,142500,265900,224900,248328,170000,465000,230000,178000,186500,169900,129500,119000,244000,171750,130000,294000,165400,127500,301500,99900,190000,151000,181000,128900,161500,180500,181000,183900,122000,378500,381000,144000,260000,185750,137000,177000,139000,137000,162000,197900,237000,68400,227000,180000,150500,139000,169000,132500,143000,190000,278000,281000,180500,119500,107500,162900,115000,138500,155000,140000,160000,154000,225000,177500,290000,232000,130000,325000,202500,138000,147000,179200,335000,203000,302000,333168,119000,206900,295493,208900,275000,111000,156500,72500,190000,82500,147000,55000,79000,130500,256000,176500,227000,132500,100000,125500,125000,167900,135000,52500,200000,128500,123000,155000,228500,177000,155835,108500,262500,283463,215000,122000,200000,171000,134900,410000,235000,170000,110000,149900,177500,315000,189000,260000,104900,156932,144152,216000,193000,127000,144000,232000,105000,165500,274300,466500,250000,239000,91000,117000,83000,167500,58500,237500,157000,112000,105000,125500,250000,136000,377500,131000,235000,124000,123000,163000,246578,281213,160000,137500,138000,137450,120000,193000,193879,282922,105000,275000,133000,112000,125500,215000,230000,140000,90000,257000,207000,175900,122500,340000,124000,223000,179900,127500,136500,274970,144000,142000,271000,140000,119000,182900,192140,143750,64500,186500,160000,174000,120500,394617,149700,197000,191000,149300,310000,121000,179600,129000,157900,240000,112000,92000,136000,287090,145000,84500,185000,175000,210000,266500,142125,147500],"type":"scatter"}],                        {"hovermode":"closest","template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"GrLivArea vs SalePrice"},"xaxis":{"title":{"text":"GrLivArea"}},"yaxis":{"title":{"text":"SalePrice"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('76134c2c-35ab-49c6-aebe-06a10f82ee18');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
# Drop observations where GrLivArea is greater than 4000 sq.ft
study_data.drop(study_data[study_data.GrLivArea>4000].index, inplace = True)
study_data.reset_index(drop = True, inplace = True)
```


```python
scatter_plot(study_data.GrLivArea, study_data.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
```


<div>                            <div id="cb4d11aa-0aa0-4f21-b46d-ae8cbf86653b" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("cb4d11aa-0aa0-4f21-b46d-ae8cbf86653b")) {                    Plotly.newPlot(                        "cb4d11aa-0aa0-4f21-b46d-ae8cbf86653b",                        [{"marker":{"color":[208500,181500,223500,140000,250000,143000,307000,200000,129900,118000,129500,345000,144000,279500,157000,132000,149000,90000,159000,139000,325300,139400,230000,129900,154000,256300,134800,306000,207500,68500,40000,149350,179900,165500,277500,309000,145000,153000,109000,82000,160000,170000,144000,130250,141000,319900,239686,249700,113000,127000,177000,114500,110000,385000,130000,180500,172500,196500,438780,124900,158000,101000,202500,140000,219500,317000,180000,226000,80000,225000,244000,129500,185000,144900,107400,91000,135750,127000,136500,110000,193500,153500,245000,126500,168500,260000,174000,164500,85000,123600,109900,98600,163500,133900,204750,185000,214000,94750,83000,128950,205000,178000,118964,198900,169500,250000,100000,115000,115000,190000,136900,180000,383970,217000,259500,176000,139000,155000,320000,163990,180000,100000,136000,153900,181000,84500,128000,87000,155000,150000,226000,244000,150750,220000,180000,174000,143000,171000,230000,231500,115000,260000,166000,204000,125000,130000,105000,222500,141000,115000,122000,372402,190000,235000,125000,79000,109500,269500,254900,320000,162500,412500,220000,103200,152000,127500,190000,325624,183500,228000,128500,215000,239000,163000,184000,243000,211000,172500,501837,100000,177000,200100,120000,200000,127000,475000,173000,135000,153337,286000,315000,184000,192000,130000,127000,148500,311872,235000,104000,274900,140000,171500,112000,149000,110000,180500,143900,141000,277000,145000,98000,186000,252678,156000,161750,134450,210000,107000,311500,167240,204900,200000,179900,97000,386250,112000,290000,106000,125000,192500,148000,403000,94500,128200,216500,89500,185500,194500,318000,113000,262500,110500,79000,120000,205000,241500,137000,140000,180000,277000,76500,235000,173000,158000,145000,230000,207500,220000,231500,97000,176000,276000,151000,130000,73000,175500,185000,179500,120500,148000,266000,241500,290000,139000,124500,205000,201000,141000,415298,192000,228500,185000,207500,244600,179200,164700,159000,88000,122000,153575,233230,135900,131000,235000,167000,142500,152000,239000,175000,158500,157000,267000,205000,149900,295000,305900,225000,89500,82500,360000,165600,132000,119900,375000,178000,188500,260000,270000,260000,187500,342643,354000,301000,126175,242000,87000,324000,145250,214500,78000,119000,139000,284000,207000,192000,228950,377426,214000,202500,155000,202900,82000,87500,266000,85000,140200,151500,157500,154000,437154,318061,190000,95000,105900,140000,177500,173000,134000,130000,280000,156000,145000,198500,118000,190000,147000,159000,165000,132000,162000,172400,134432,125000,123000,219500,61000,148000,340000,394432,179000,127000,187750,213500,76000,240000,192000,81000,125000,191000,426000,119000,215000,106500,100000,109000,129000,123000,169500,67000,241000,245500,164990,108000,258000,168000,150000,115000,177000,280000,339750,60000,145000,222000,115000,228000,181134,149500,239000,126000,142000,206300,215000,113000,315000,139000,135000,275000,109008,195400,175000,85400,79900,122500,181000,81000,212000,116000,119000,90350,110000,555000,118000,162900,172500,210000,127500,190000,199900,119500,120000,110000,280000,204000,210000,188000,175500,98000,256000,161000,110000,263435,155000,62383,188700,124000,178740,167000,146500,250000,187000,212000,190000,148000,440000,251000,132500,208900,380000,297000,89471,326000,374000,155000,164000,132500,147000,156000,175000,160000,86000,115000,133000,172785,155000,91300,34900,430000,184000,130000,120000,113000,226700,140000,289000,147000,124500,215000,208300,161000,124500,164900,202665,129900,134000,96500,402861,158000,265000,211000,234000,106250,150000,159000,315750,176000,132000,446261,86000,200624,175000,128000,107500,39300,178000,107500,188000,111250,158000,272000,315000,248000,213250,133000,179665,229000,210000,129500,125000,263000,140000,112500,255500,108000,284000,113000,141000,108000,175000,234000,121500,170000,108000,185000,268000,128000,325000,214000,316600,135960,142600,120000,224500,170000,139000,118500,145000,164500,146000,131500,181900,253293,118500,325000,133000,369900,130000,137000,143000,79500,185900,451950,138000,140000,110000,319000,114504,194201,217500,151000,275000,141000,220000,151000,221000,205000,152000,225000,359100,118500,313000,148000,261500,147000,75500,137500,183200,105500,314813,305000,67000,240000,135000,168500,165150,160000,139900,153000,135000,168500,124000,209500,82500,139400,144000,200000,60000,93000,85000,264561,274000,226000,345000,152000,370878,143250,98300,155000,155000,84500,205950,108000,191000,135000,350000,88000,145500,149000,97500,167000,197900,402000,110000,137500,423000,230500,129000,193500,168000,137500,173500,103600,165000,257500,140000,148500,87000,109500,372500,128500,143000,159434,173000,285000,221000,207500,227875,148800,392000,194700,141000,335000,108480,141500,176000,89000,123500,138500,196000,312500,140000,361919,140000,213000,55000,302000,254000,179540,109900,52000,102776,189000,129000,130500,165000,159500,157000,341000,128500,275000,143000,124500,135000,320000,120500,222000,194500,110000,103000,236500,187500,222500,131400,108000,163000,93500,239900,179000,190000,132000,142000,179000,175000,180000,299800,236000,265979,260400,98000,96500,162000,217000,275500,156000,172500,212000,158900,179400,290000,127500,100000,215200,337000,270000,264132,196500,160000,216837,538000,134900,102000,107000,114500,395000,162000,221500,142500,144000,135000,176000,175900,187100,165500,128000,161500,139000,233000,107900,187500,160200,146800,269790,225000,194500,171000,143500,110000,485000,175000,200000,109900,189000,582933,118000,227680,135500,223500,159950,106000,181000,144500,55993,157900,116000,224900,137000,271000,155000,224000,183000,93000,225000,139500,232600,385000,109500,189000,185000,147400,166000,151000,237000,167000,139950,128000,153500,100000,144000,130500,140000,157500,174900,141000,153900,171000,213000,133500,240000,187000,131500,215000,164000,158000,170000,127000,147000,174000,152000,250000,189950,131500,152000,132500,250580,148500,248900,129000,169000,236000,109500,200500,116000,133000,66500,303477,132250,350000,148000,136500,157000,187500,178000,118500,100000,328900,145000,135500,268000,149500,122900,172500,154500,165000,118858,140000,106500,142953,611657,135000,110000,153000,180000,240000,125500,128000,255000,250000,131000,174000,154300,143500,88000,145000,173733,75000,35311,135000,238000,176500,201000,145900,169990,193000,207500,175000,285000,176000,236500,222000,201000,117500,320000,190000,242000,79900,184900,253000,239799,244400,150900,214000,150000,143000,137500,124900,143000,270000,192500,197500,129000,119900,133900,172000,127500,145000,124000,132000,185000,155000,116500,272000,155000,239000,214900,178900,160000,135000,37900,140000,135000,173000,99500,182000,167500,165000,85500,199900,110000,139000,178400,336000,159895,255900,126000,125000,117000,395192,195000,197000,348000,168000,187000,173900,337500,121600,136500,185000,91000,206000,82000,86000,232000,136905,181000,149900,163500,88000,240000,102000,135000,100000,165000,85000,119200,227000,203000,187500,160000,213490,176000,194000,87000,191000,287000,112500,167500,293077,105000,118000,160000,197000,310000,230000,119750,84000,315500,287000,97000,80000,155000,173000,196000,262280,278000,139600,556581,145000,115000,84900,176485,200141,165000,144500,255000,180000,185850,248000,335000,220000,213500,81000,90000,110500,154000,328000,178000,167900,151400,135000,135000,154000,91500,159500,194000,219500,170000,138800,155900,126000,145000,133000,192000,160000,187500,147000,83500,252000,137500,197000,92900,160000,136500,146000,129000,176432,127000,170000,128000,157000,60000,119500,135000,159500,106000,325000,179900,274725,181000,280000,188000,205000,129900,134500,117000,318000,184100,130000,140000,133700,118400,212900,112000,118000,163900,115000,174000,259000,215000,140000,135000,93500,117500,239500,169000,102000,119000,94000,196000,144000,139000,197500,424870,80000,80000,149000,180000,174500,116900,143000,124000,149900,230000,120500,201800,218000,179900,230000,235128,185000,146000,224000,129000,108959,194000,233170,245350,173000,235000,625000,171000,163000,171900,200500,239000,285000,119500,115000,154900,93000,250000,392500,120000,186700,104900,95000,262000,195000,189000,168000,174000,125000,165000,158000,176000,219210,144000,178000,148000,116050,197900,117000,213000,153500,271900,107000,200000,140000,290000,189000,164000,113000,145000,134500,125000,112000,229456,80500,91500,115000,134000,143000,137900,184000,145000,214000,147000,367294,127000,190000,132500,101800,142000,130000,138887,175500,195000,142500,265900,224900,248328,170000,465000,230000,178000,186500,169900,129500,119000,244000,171750,130000,294000,165400,127500,301500,99900,190000,151000,181000,128900,161500,180500,181000,183900,122000,378500,381000,144000,260000,185750,137000,177000,139000,137000,162000,197900,237000,68400,227000,180000,150500,139000,169000,132500,143000,190000,278000,281000,180500,119500,107500,162900,115000,138500,155000,140000,154000,225000,177500,290000,232000,130000,325000,202500,138000,147000,179200,335000,203000,302000,333168,119000,206900,295493,208900,275000,111000,156500,72500,190000,82500,147000,55000,79000,130500,256000,176500,227000,132500,100000,125500,125000,167900,135000,52500,200000,128500,123000,155000,228500,177000,155835,108500,262500,283463,215000,122000,200000,171000,134900,410000,235000,170000,110000,149900,177500,315000,189000,260000,104900,156932,144152,216000,193000,127000,144000,232000,105000,165500,274300,466500,250000,239000,91000,117000,83000,167500,58500,237500,157000,112000,105000,125500,250000,136000,377500,131000,235000,124000,123000,163000,246578,281213,160000,137500,138000,137450,120000,193000,193879,282922,105000,275000,133000,112000,125500,215000,230000,140000,90000,257000,207000,175900,122500,340000,124000,223000,179900,127500,136500,274970,144000,142000,271000,140000,119000,182900,192140,143750,64500,186500,160000,174000,120500,394617,149700,197000,191000,149300,310000,121000,179600,129000,157900,240000,112000,92000,136000,287090,145000,84500,185000,175000,210000,266500,142125,147500],"colorscale":[[0.0,"rgb(150,0,90)"],[0.125,"rgb(0,0,200)"],[0.25,"rgb(0,25,255)"],[0.375,"rgb(0,152,255)"],[0.5,"rgb(44,255,150)"],[0.625,"rgb(151,255,0)"],[0.75,"rgb(255,234,0)"],[0.875,"rgb(255,111,0)"],[1.0,"rgb(255,0,0)"]],"showscale":true,"size":10},"mode":"markers","x":[1710,1262,1786,1717,2198,1362,1694,2090,1774,1077,1040,2324,912,1494,1253,854,1004,1296,1114,1339,2376,1108,1795,1060,1060,1600,900,1704,1600,520,1317,1228,1234,1700,1561,2452,1097,1297,1057,1152,1324,1328,884,938,1150,1752,2149,1656,1452,955,1470,1176,816,1842,1360,1425,1739,1720,2945,780,1158,1111,1370,1710,2034,2473,2207,1479,747,2287,2223,845,1718,1086,1605,988,952,1285,1768,1230,2142,1337,1563,1065,1474,2417,1560,1224,1526,990,1040,1235,964,2291,1786,1470,1588,960,835,1225,1610,1732,1535,1226,1818,1992,1047,789,1517,1844,1855,1430,2696,2259,2320,1458,1092,1125,3222,1456,988,1123,1080,1199,1586,754,958,840,1348,1053,2157,2054,1327,1296,1721,1682,1214,1959,1852,1764,864,1734,1385,1501,1728,1709,875,2035,1080,1344,969,1710,1993,1252,1200,1096,1040,1968,1947,2462,1232,2668,1541,882,1616,1355,1867,2161,1720,1707,1382,1656,1767,1362,1651,2158,2060,1920,2234,968,1525,1802,1340,2082,1252,3608,1217,1656,1224,1593,2727,1479,1431,1709,864,1456,1726,3112,2229,1713,1121,1279,1310,848,1284,1442,1696,1100,2062,1092,864,1212,1852,990,1392,1236,1436,1328,1954,1248,1498,2267,1552,864,2392,1302,2520,987,912,1555,1194,2794,987,894,1960,987,1414,1744,1694,1487,1566,866,1440,1217,2110,1872,1928,1375,1668,2144,1306,1625,1640,1302,1314,2291,1728,1604,1792,882,1382,2574,1212,1316,764,1422,1511,2192,778,1113,1939,1363,2270,1632,816,1548,1560,864,2121,2022,1982,1262,1314,1468,1575,1250,1734,858,900,1396,1919,1716,1716,2263,1644,1003,1558,1950,1743,1152,1336,2452,1541,894,3493,2000,2243,1406,861,1944,1501,972,1118,2036,1641,1432,2353,1959,2646,1472,2596,2468,2730,1163,2978,803,1719,1383,2134,1192,1728,1056,1629,1358,1638,1786,1922,1536,1621,1215,1908,841,1040,1684,1112,1577,958,1478,1626,2728,1869,1453,1111,720,1595,1200,1167,1142,1352,1924,912,1505,1922,987,1574,1344,1394,1431,1268,1287,1664,1588,752,1319,1928,904,914,2466,1856,1800,1691,1301,1797,784,1953,1269,1184,1125,1479,2332,1367,1961,882,788,1034,1144,894,1812,1077,1550,1288,1310,672,2263,1572,1620,1639,1680,2172,2078,1276,1056,1478,1028,2097,1340,1400,2624,1134,1056,1344,1602,988,2630,1196,1389,1644,907,1208,1412,987,1198,1365,1604,630,1661,1118,904,694,1196,2402,1440,1573,1258,1908,1689,1888,1886,1376,1183,813,1533,1756,1590,1728,1242,1344,1663,1666,1203,1935,1135,864,1660,1040,1414,1277,1644,1634,1710,1502,1969,1072,1976,1652,970,1493,2643,1718,1131,1850,1792,1826,1216,999,1113,1073,1484,2414,630,1304,1578,1456,1269,886,720,3228,1820,899,912,1218,1768,1214,1801,1322,1960,1911,1218,1378,1041,1363,1368,864,1080,789,2020,2119,2344,1796,2080,1294,1244,1664,2398,1266,928,2713,605,2515,1509,1362,827,334,1414,1347,1724,864,1159,1601,1838,2285,1680,767,1496,2183,1635,768,825,2094,1069,928,1717,1126,2046,1048,1092,1336,1446,1557,1392,1389,996,1674,2295,1647,2504,1535,2132,943,1728,864,1692,1430,1109,1216,1477,1320,1392,1795,1429,2042,816,2775,1573,2028,838,860,1473,935,1582,2296,816,848,924,1826,1368,1402,1647,1556,1904,1375,1915,1200,1494,1986,1040,2008,3194,1029,2153,1032,1872,1120,630,1054,1509,832,1828,2262,864,2614,980,1512,1790,1116,1422,1520,2080,1350,1750,1554,1411,1056,1056,3395,800,1387,796,1567,1518,1929,2704,1620,1766,981,1048,1094,1839,630,1665,1510,1716,1469,2113,1092,1053,1502,1458,1486,1935,2448,1392,1181,2097,1936,2380,1679,1437,1180,1476,1369,1208,1839,1136,1441,1774,792,2046,988,923,1520,1291,1668,1839,2090,1761,1102,1419,1362,848,2519,1073,1539,1137,616,1148,894,1391,1800,1164,2576,1812,1484,1092,1824,1324,1456,904,729,1178,1228,960,1479,1350,2554,1178,2418,971,1742,848,864,1470,1698,864,1680,1232,1776,1208,1616,1146,2031,1144,948,1768,1040,1801,1200,1728,1432,912,1349,1464,1337,2715,2256,2640,1720,1529,1140,1320,1494,2098,1026,1471,1768,1386,1501,2531,864,1301,1547,2365,1494,1506,1714,1750,1836,3279,858,1220,1117,912,1973,1204,1614,894,2020,1004,1253,1603,1430,1110,1484,1342,1652,2084,901,2087,1145,1062,2013,1496,1895,1564,1285,773,3140,1768,1688,1196,1456,2822,1128,1428,980,1576,1086,2138,1309,848,1044,1442,1250,1661,1008,1689,1052,1358,1640,936,1733,1489,1489,2084,784,1434,2126,1223,1392,1200,1829,1516,1144,1067,1559,987,1099,1200,1482,1539,1165,1800,1416,1701,1775,864,2358,1855,848,1456,1646,1445,1779,1040,1026,1481,1370,2654,1426,1039,1097,1148,1372,1002,1646,1120,2320,1949,894,1682,910,1268,1131,2610,1040,2224,1155,864,1090,1717,1593,2230,892,1709,1712,1393,2217,1505,924,1683,1068,1383,1535,1796,951,2240,2364,1236,858,1306,1509,1670,902,1063,1636,2057,902,1484,2274,1268,1015,2002,1224,1092,480,1229,2127,1414,1721,2200,1316,1617,1686,1126,2374,1978,1788,2236,1466,925,1905,1500,2069,747,1200,1971,1962,2403,1728,2060,1440,1632,1344,1869,1144,1629,1776,1381,864,965,768,1968,980,1958,1229,1057,1337,1416,858,2872,1548,1800,1894,1484,1308,1098,968,1095,1192,1626,918,1428,2019,1382,869,1241,894,1121,999,2612,1266,2290,1734,1164,1635,1940,2030,1576,2392,1742,1851,1500,1718,1230,1050,1442,1077,1208,944,691,1574,1680,1504,985,1657,1092,1710,1522,1271,1664,1502,1022,1082,1665,1504,1360,1472,1506,1132,1220,1248,1504,2898,882,1264,1646,1376,1218,1928,3082,2520,1654,954,845,1620,2263,1344,630,1803,1632,1306,2329,2524,1733,2868,990,1771,930,1302,1316,1977,1526,1989,1523,1364,1850,2184,1991,1338,894,2337,1103,1154,2260,1571,1611,2521,893,1048,1556,1456,1426,1240,1740,1466,1096,848,990,1258,1040,1459,1251,1498,996,1092,1953,1709,1247,1040,1252,1694,1200,936,1314,1355,1088,1324,1601,438,950,1134,1194,1302,2622,1442,2021,1690,1836,1658,1964,816,1008,833,1734,1419,894,1601,1040,1012,1552,960,698,1482,1005,1555,1530,1959,936,1981,974,2210,2020,1600,986,1252,1020,1567,1167,952,1868,2828,1006,924,1576,1298,1564,1111,1482,932,1466,1811,816,1820,1437,1265,1314,1580,1876,1456,1640,894,1258,1432,1502,1694,1671,2108,3627,1118,1261,1250,3086,2345,2872,923,1224,1343,1124,2514,1652,1130,1572,1221,1699,1624,1660,1804,1622,1441,1472,1224,1352,1456,1863,1690,1212,1382,864,1779,1348,1630,1074,2196,1056,1700,1283,1660,1845,1752,672,960,999,894,1902,1314,912,1218,912,1211,1846,2136,1490,1138,1933,912,1702,1507,2620,1190,1224,1188,1964,1784,1626,1948,1141,1484,1768,1689,1173,2076,1517,1868,1553,1034,2058,988,2110,1405,874,2167,1656,1367,1987,864,1166,1054,1675,1050,1788,1824,1337,1452,1889,2018,3447,1524,1524,1489,935,1357,1250,1920,1395,1724,2031,1128,1573,1339,1040,1824,2447,1412,1328,1582,1659,1970,1152,1302,2372,1664,864,1052,1128,1072,1246,1983,1494,2526,1616,1708,1652,1368,990,1122,1294,1902,1274,2810,2599,948,2112,1630,1352,1787,948,1478,720,1923,708,1795,796,774,816,2792,1632,1588,954,816,1360,1365,1334,1656,693,1861,864,872,1114,2169,1913,1456,960,2156,1776,1494,2358,2634,1716,1176,3238,1865,1920,892,1078,1573,1980,2601,1530,1738,1412,1200,1674,1790,1475,848,1668,1374,1661,2097,2633,1958,1571,790,1604,987,1394,864,2117,1762,1416,1258,1154,2784,2526,1746,1218,1525,1584,900,1912,1500,2482,1687,1513,1904,1608,1158,1593,1294,1464,1214,1646,768,833,1363,2093,1840,1668,1040,1844,1848,1569,2290,2450,1144,1844,1416,1069,848,2201,1344,1252,2127,1558,804,1440,1838,958,968,1792,1126,1537,864,1932,1236,1725,2555,848,2007,952,1422,913,1188,2090,1346,630,1792,1578,1072,1140,1221,1647,2073,2340,1078,1256],"y":[208500,181500,223500,140000,250000,143000,307000,200000,129900,118000,129500,345000,144000,279500,157000,132000,149000,90000,159000,139000,325300,139400,230000,129900,154000,256300,134800,306000,207500,68500,40000,149350,179900,165500,277500,309000,145000,153000,109000,82000,160000,170000,144000,130250,141000,319900,239686,249700,113000,127000,177000,114500,110000,385000,130000,180500,172500,196500,438780,124900,158000,101000,202500,140000,219500,317000,180000,226000,80000,225000,244000,129500,185000,144900,107400,91000,135750,127000,136500,110000,193500,153500,245000,126500,168500,260000,174000,164500,85000,123600,109900,98600,163500,133900,204750,185000,214000,94750,83000,128950,205000,178000,118964,198900,169500,250000,100000,115000,115000,190000,136900,180000,383970,217000,259500,176000,139000,155000,320000,163990,180000,100000,136000,153900,181000,84500,128000,87000,155000,150000,226000,244000,150750,220000,180000,174000,143000,171000,230000,231500,115000,260000,166000,204000,125000,130000,105000,222500,141000,115000,122000,372402,190000,235000,125000,79000,109500,269500,254900,320000,162500,412500,220000,103200,152000,127500,190000,325624,183500,228000,128500,215000,239000,163000,184000,243000,211000,172500,501837,100000,177000,200100,120000,200000,127000,475000,173000,135000,153337,286000,315000,184000,192000,130000,127000,148500,311872,235000,104000,274900,140000,171500,112000,149000,110000,180500,143900,141000,277000,145000,98000,186000,252678,156000,161750,134450,210000,107000,311500,167240,204900,200000,179900,97000,386250,112000,290000,106000,125000,192500,148000,403000,94500,128200,216500,89500,185500,194500,318000,113000,262500,110500,79000,120000,205000,241500,137000,140000,180000,277000,76500,235000,173000,158000,145000,230000,207500,220000,231500,97000,176000,276000,151000,130000,73000,175500,185000,179500,120500,148000,266000,241500,290000,139000,124500,205000,201000,141000,415298,192000,228500,185000,207500,244600,179200,164700,159000,88000,122000,153575,233230,135900,131000,235000,167000,142500,152000,239000,175000,158500,157000,267000,205000,149900,295000,305900,225000,89500,82500,360000,165600,132000,119900,375000,178000,188500,260000,270000,260000,187500,342643,354000,301000,126175,242000,87000,324000,145250,214500,78000,119000,139000,284000,207000,192000,228950,377426,214000,202500,155000,202900,82000,87500,266000,85000,140200,151500,157500,154000,437154,318061,190000,95000,105900,140000,177500,173000,134000,130000,280000,156000,145000,198500,118000,190000,147000,159000,165000,132000,162000,172400,134432,125000,123000,219500,61000,148000,340000,394432,179000,127000,187750,213500,76000,240000,192000,81000,125000,191000,426000,119000,215000,106500,100000,109000,129000,123000,169500,67000,241000,245500,164990,108000,258000,168000,150000,115000,177000,280000,339750,60000,145000,222000,115000,228000,181134,149500,239000,126000,142000,206300,215000,113000,315000,139000,135000,275000,109008,195400,175000,85400,79900,122500,181000,81000,212000,116000,119000,90350,110000,555000,118000,162900,172500,210000,127500,190000,199900,119500,120000,110000,280000,204000,210000,188000,175500,98000,256000,161000,110000,263435,155000,62383,188700,124000,178740,167000,146500,250000,187000,212000,190000,148000,440000,251000,132500,208900,380000,297000,89471,326000,374000,155000,164000,132500,147000,156000,175000,160000,86000,115000,133000,172785,155000,91300,34900,430000,184000,130000,120000,113000,226700,140000,289000,147000,124500,215000,208300,161000,124500,164900,202665,129900,134000,96500,402861,158000,265000,211000,234000,106250,150000,159000,315750,176000,132000,446261,86000,200624,175000,128000,107500,39300,178000,107500,188000,111250,158000,272000,315000,248000,213250,133000,179665,229000,210000,129500,125000,263000,140000,112500,255500,108000,284000,113000,141000,108000,175000,234000,121500,170000,108000,185000,268000,128000,325000,214000,316600,135960,142600,120000,224500,170000,139000,118500,145000,164500,146000,131500,181900,253293,118500,325000,133000,369900,130000,137000,143000,79500,185900,451950,138000,140000,110000,319000,114504,194201,217500,151000,275000,141000,220000,151000,221000,205000,152000,225000,359100,118500,313000,148000,261500,147000,75500,137500,183200,105500,314813,305000,67000,240000,135000,168500,165150,160000,139900,153000,135000,168500,124000,209500,82500,139400,144000,200000,60000,93000,85000,264561,274000,226000,345000,152000,370878,143250,98300,155000,155000,84500,205950,108000,191000,135000,350000,88000,145500,149000,97500,167000,197900,402000,110000,137500,423000,230500,129000,193500,168000,137500,173500,103600,165000,257500,140000,148500,87000,109500,372500,128500,143000,159434,173000,285000,221000,207500,227875,148800,392000,194700,141000,335000,108480,141500,176000,89000,123500,138500,196000,312500,140000,361919,140000,213000,55000,302000,254000,179540,109900,52000,102776,189000,129000,130500,165000,159500,157000,341000,128500,275000,143000,124500,135000,320000,120500,222000,194500,110000,103000,236500,187500,222500,131400,108000,163000,93500,239900,179000,190000,132000,142000,179000,175000,180000,299800,236000,265979,260400,98000,96500,162000,217000,275500,156000,172500,212000,158900,179400,290000,127500,100000,215200,337000,270000,264132,196500,160000,216837,538000,134900,102000,107000,114500,395000,162000,221500,142500,144000,135000,176000,175900,187100,165500,128000,161500,139000,233000,107900,187500,160200,146800,269790,225000,194500,171000,143500,110000,485000,175000,200000,109900,189000,582933,118000,227680,135500,223500,159950,106000,181000,144500,55993,157900,116000,224900,137000,271000,155000,224000,183000,93000,225000,139500,232600,385000,109500,189000,185000,147400,166000,151000,237000,167000,139950,128000,153500,100000,144000,130500,140000,157500,174900,141000,153900,171000,213000,133500,240000,187000,131500,215000,164000,158000,170000,127000,147000,174000,152000,250000,189950,131500,152000,132500,250580,148500,248900,129000,169000,236000,109500,200500,116000,133000,66500,303477,132250,350000,148000,136500,157000,187500,178000,118500,100000,328900,145000,135500,268000,149500,122900,172500,154500,165000,118858,140000,106500,142953,611657,135000,110000,153000,180000,240000,125500,128000,255000,250000,131000,174000,154300,143500,88000,145000,173733,75000,35311,135000,238000,176500,201000,145900,169990,193000,207500,175000,285000,176000,236500,222000,201000,117500,320000,190000,242000,79900,184900,253000,239799,244400,150900,214000,150000,143000,137500,124900,143000,270000,192500,197500,129000,119900,133900,172000,127500,145000,124000,132000,185000,155000,116500,272000,155000,239000,214900,178900,160000,135000,37900,140000,135000,173000,99500,182000,167500,165000,85500,199900,110000,139000,178400,336000,159895,255900,126000,125000,117000,395192,195000,197000,348000,168000,187000,173900,337500,121600,136500,185000,91000,206000,82000,86000,232000,136905,181000,149900,163500,88000,240000,102000,135000,100000,165000,85000,119200,227000,203000,187500,160000,213490,176000,194000,87000,191000,287000,112500,167500,293077,105000,118000,160000,197000,310000,230000,119750,84000,315500,287000,97000,80000,155000,173000,196000,262280,278000,139600,556581,145000,115000,84900,176485,200141,165000,144500,255000,180000,185850,248000,335000,220000,213500,81000,90000,110500,154000,328000,178000,167900,151400,135000,135000,154000,91500,159500,194000,219500,170000,138800,155900,126000,145000,133000,192000,160000,187500,147000,83500,252000,137500,197000,92900,160000,136500,146000,129000,176432,127000,170000,128000,157000,60000,119500,135000,159500,106000,325000,179900,274725,181000,280000,188000,205000,129900,134500,117000,318000,184100,130000,140000,133700,118400,212900,112000,118000,163900,115000,174000,259000,215000,140000,135000,93500,117500,239500,169000,102000,119000,94000,196000,144000,139000,197500,424870,80000,80000,149000,180000,174500,116900,143000,124000,149900,230000,120500,201800,218000,179900,230000,235128,185000,146000,224000,129000,108959,194000,233170,245350,173000,235000,625000,171000,163000,171900,200500,239000,285000,119500,115000,154900,93000,250000,392500,120000,186700,104900,95000,262000,195000,189000,168000,174000,125000,165000,158000,176000,219210,144000,178000,148000,116050,197900,117000,213000,153500,271900,107000,200000,140000,290000,189000,164000,113000,145000,134500,125000,112000,229456,80500,91500,115000,134000,143000,137900,184000,145000,214000,147000,367294,127000,190000,132500,101800,142000,130000,138887,175500,195000,142500,265900,224900,248328,170000,465000,230000,178000,186500,169900,129500,119000,244000,171750,130000,294000,165400,127500,301500,99900,190000,151000,181000,128900,161500,180500,181000,183900,122000,378500,381000,144000,260000,185750,137000,177000,139000,137000,162000,197900,237000,68400,227000,180000,150500,139000,169000,132500,143000,190000,278000,281000,180500,119500,107500,162900,115000,138500,155000,140000,154000,225000,177500,290000,232000,130000,325000,202500,138000,147000,179200,335000,203000,302000,333168,119000,206900,295493,208900,275000,111000,156500,72500,190000,82500,147000,55000,79000,130500,256000,176500,227000,132500,100000,125500,125000,167900,135000,52500,200000,128500,123000,155000,228500,177000,155835,108500,262500,283463,215000,122000,200000,171000,134900,410000,235000,170000,110000,149900,177500,315000,189000,260000,104900,156932,144152,216000,193000,127000,144000,232000,105000,165500,274300,466500,250000,239000,91000,117000,83000,167500,58500,237500,157000,112000,105000,125500,250000,136000,377500,131000,235000,124000,123000,163000,246578,281213,160000,137500,138000,137450,120000,193000,193879,282922,105000,275000,133000,112000,125500,215000,230000,140000,90000,257000,207000,175900,122500,340000,124000,223000,179900,127500,136500,274970,144000,142000,271000,140000,119000,182900,192140,143750,64500,186500,160000,174000,120500,394617,149700,197000,191000,149300,310000,121000,179600,129000,157900,240000,112000,92000,136000,287090,145000,84500,185000,175000,210000,266500,142125,147500],"type":"scatter"}],                        {"hovermode":"closest","template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"GrLivArea vs SalePrice"},"xaxis":{"title":{"text":"GrLivArea"}},"yaxis":{"title":{"text":"SalePrice"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('cb4d11aa-0aa0-4f21-b46d-ae8cbf86653b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


##  Normalization of Skewed Distributions

### Skewness of SalePrice

Since algorithms can be sensitive to skewed distributions and may underperform if the data range isn't properly normalized, I show the **skewness of SalePrice** and its normalization using the natural logarithm transformation, specifically log(1 + x) (base e).


```python
def plot_histogram(x, title, yaxis, color):
    trace = go.Histogram(x = x,
                        marker = dict(color = color))
    layout = go.Layout(hovermode= 'closest', title = title, yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig)

sale_price_array = study_data.SalePrice
title = 'Distribution of SalePrice with skewness (skewness: {:0.4f})'.format(sale_price_array.skew())
plot_histogram(sale_price_array, title, 'Abs Frequency', 'darkred')

sale_price_array = np.log1p(sale_price_array)

# Update the sales prices with the normalized value
study_data['SalePrice'] = sale_price_array

title = 'Distribution of SalePrice removing skewness (skewness: {:0.4f})'.format(sale_price_array.skew())
plot_histogram(sale_price_array, title, 'Abs Frequency', 'green')
```


<div>                            <div id="62d2a6d6-3bdf-450a-a790-b36f427dec3f" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("62d2a6d6-3bdf-450a-a790-b36f427dec3f")) {                    Plotly.newPlot(                        "62d2a6d6-3bdf-450a-a790-b36f427dec3f",                        [{"marker":{"color":"darkred"},"x":[208500,181500,223500,140000,250000,143000,307000,200000,129900,118000,129500,345000,144000,279500,157000,132000,149000,90000,159000,139000,325300,139400,230000,129900,154000,256300,134800,306000,207500,68500,40000,149350,179900,165500,277500,309000,145000,153000,109000,82000,160000,170000,144000,130250,141000,319900,239686,249700,113000,127000,177000,114500,110000,385000,130000,180500,172500,196500,438780,124900,158000,101000,202500,140000,219500,317000,180000,226000,80000,225000,244000,129500,185000,144900,107400,91000,135750,127000,136500,110000,193500,153500,245000,126500,168500,260000,174000,164500,85000,123600,109900,98600,163500,133900,204750,185000,214000,94750,83000,128950,205000,178000,118964,198900,169500,250000,100000,115000,115000,190000,136900,180000,383970,217000,259500,176000,139000,155000,320000,163990,180000,100000,136000,153900,181000,84500,128000,87000,155000,150000,226000,244000,150750,220000,180000,174000,143000,171000,230000,231500,115000,260000,166000,204000,125000,130000,105000,222500,141000,115000,122000,372402,190000,235000,125000,79000,109500,269500,254900,320000,162500,412500,220000,103200,152000,127500,190000,325624,183500,228000,128500,215000,239000,163000,184000,243000,211000,172500,501837,100000,177000,200100,120000,200000,127000,475000,173000,135000,153337,286000,315000,184000,192000,130000,127000,148500,311872,235000,104000,274900,140000,171500,112000,149000,110000,180500,143900,141000,277000,145000,98000,186000,252678,156000,161750,134450,210000,107000,311500,167240,204900,200000,179900,97000,386250,112000,290000,106000,125000,192500,148000,403000,94500,128200,216500,89500,185500,194500,318000,113000,262500,110500,79000,120000,205000,241500,137000,140000,180000,277000,76500,235000,173000,158000,145000,230000,207500,220000,231500,97000,176000,276000,151000,130000,73000,175500,185000,179500,120500,148000,266000,241500,290000,139000,124500,205000,201000,141000,415298,192000,228500,185000,207500,244600,179200,164700,159000,88000,122000,153575,233230,135900,131000,235000,167000,142500,152000,239000,175000,158500,157000,267000,205000,149900,295000,305900,225000,89500,82500,360000,165600,132000,119900,375000,178000,188500,260000,270000,260000,187500,342643,354000,301000,126175,242000,87000,324000,145250,214500,78000,119000,139000,284000,207000,192000,228950,377426,214000,202500,155000,202900,82000,87500,266000,85000,140200,151500,157500,154000,437154,318061,190000,95000,105900,140000,177500,173000,134000,130000,280000,156000,145000,198500,118000,190000,147000,159000,165000,132000,162000,172400,134432,125000,123000,219500,61000,148000,340000,394432,179000,127000,187750,213500,76000,240000,192000,81000,125000,191000,426000,119000,215000,106500,100000,109000,129000,123000,169500,67000,241000,245500,164990,108000,258000,168000,150000,115000,177000,280000,339750,60000,145000,222000,115000,228000,181134,149500,239000,126000,142000,206300,215000,113000,315000,139000,135000,275000,109008,195400,175000,85400,79900,122500,181000,81000,212000,116000,119000,90350,110000,555000,118000,162900,172500,210000,127500,190000,199900,119500,120000,110000,280000,204000,210000,188000,175500,98000,256000,161000,110000,263435,155000,62383,188700,124000,178740,167000,146500,250000,187000,212000,190000,148000,440000,251000,132500,208900,380000,297000,89471,326000,374000,155000,164000,132500,147000,156000,175000,160000,86000,115000,133000,172785,155000,91300,34900,430000,184000,130000,120000,113000,226700,140000,289000,147000,124500,215000,208300,161000,124500,164900,202665,129900,134000,96500,402861,158000,265000,211000,234000,106250,150000,159000,315750,176000,132000,446261,86000,200624,175000,128000,107500,39300,178000,107500,188000,111250,158000,272000,315000,248000,213250,133000,179665,229000,210000,129500,125000,263000,140000,112500,255500,108000,284000,113000,141000,108000,175000,234000,121500,170000,108000,185000,268000,128000,325000,214000,316600,135960,142600,120000,224500,170000,139000,118500,145000,164500,146000,131500,181900,253293,118500,325000,133000,369900,130000,137000,143000,79500,185900,451950,138000,140000,110000,319000,114504,194201,217500,151000,275000,141000,220000,151000,221000,205000,152000,225000,359100,118500,313000,148000,261500,147000,75500,137500,183200,105500,314813,305000,67000,240000,135000,168500,165150,160000,139900,153000,135000,168500,124000,209500,82500,139400,144000,200000,60000,93000,85000,264561,274000,226000,345000,152000,370878,143250,98300,155000,155000,84500,205950,108000,191000,135000,350000,88000,145500,149000,97500,167000,197900,402000,110000,137500,423000,230500,129000,193500,168000,137500,173500,103600,165000,257500,140000,148500,87000,109500,372500,128500,143000,159434,173000,285000,221000,207500,227875,148800,392000,194700,141000,335000,108480,141500,176000,89000,123500,138500,196000,312500,140000,361919,140000,213000,55000,302000,254000,179540,109900,52000,102776,189000,129000,130500,165000,159500,157000,341000,128500,275000,143000,124500,135000,320000,120500,222000,194500,110000,103000,236500,187500,222500,131400,108000,163000,93500,239900,179000,190000,132000,142000,179000,175000,180000,299800,236000,265979,260400,98000,96500,162000,217000,275500,156000,172500,212000,158900,179400,290000,127500,100000,215200,337000,270000,264132,196500,160000,216837,538000,134900,102000,107000,114500,395000,162000,221500,142500,144000,135000,176000,175900,187100,165500,128000,161500,139000,233000,107900,187500,160200,146800,269790,225000,194500,171000,143500,110000,485000,175000,200000,109900,189000,582933,118000,227680,135500,223500,159950,106000,181000,144500,55993,157900,116000,224900,137000,271000,155000,224000,183000,93000,225000,139500,232600,385000,109500,189000,185000,147400,166000,151000,237000,167000,139950,128000,153500,100000,144000,130500,140000,157500,174900,141000,153900,171000,213000,133500,240000,187000,131500,215000,164000,158000,170000,127000,147000,174000,152000,250000,189950,131500,152000,132500,250580,148500,248900,129000,169000,236000,109500,200500,116000,133000,66500,303477,132250,350000,148000,136500,157000,187500,178000,118500,100000,328900,145000,135500,268000,149500,122900,172500,154500,165000,118858,140000,106500,142953,611657,135000,110000,153000,180000,240000,125500,128000,255000,250000,131000,174000,154300,143500,88000,145000,173733,75000,35311,135000,238000,176500,201000,145900,169990,193000,207500,175000,285000,176000,236500,222000,201000,117500,320000,190000,242000,79900,184900,253000,239799,244400,150900,214000,150000,143000,137500,124900,143000,270000,192500,197500,129000,119900,133900,172000,127500,145000,124000,132000,185000,155000,116500,272000,155000,239000,214900,178900,160000,135000,37900,140000,135000,173000,99500,182000,167500,165000,85500,199900,110000,139000,178400,336000,159895,255900,126000,125000,117000,395192,195000,197000,348000,168000,187000,173900,337500,121600,136500,185000,91000,206000,82000,86000,232000,136905,181000,149900,163500,88000,240000,102000,135000,100000,165000,85000,119200,227000,203000,187500,160000,213490,176000,194000,87000,191000,287000,112500,167500,293077,105000,118000,160000,197000,310000,230000,119750,84000,315500,287000,97000,80000,155000,173000,196000,262280,278000,139600,556581,145000,115000,84900,176485,200141,165000,144500,255000,180000,185850,248000,335000,220000,213500,81000,90000,110500,154000,328000,178000,167900,151400,135000,135000,154000,91500,159500,194000,219500,170000,138800,155900,126000,145000,133000,192000,160000,187500,147000,83500,252000,137500,197000,92900,160000,136500,146000,129000,176432,127000,170000,128000,157000,60000,119500,135000,159500,106000,325000,179900,274725,181000,280000,188000,205000,129900,134500,117000,318000,184100,130000,140000,133700,118400,212900,112000,118000,163900,115000,174000,259000,215000,140000,135000,93500,117500,239500,169000,102000,119000,94000,196000,144000,139000,197500,424870,80000,80000,149000,180000,174500,116900,143000,124000,149900,230000,120500,201800,218000,179900,230000,235128,185000,146000,224000,129000,108959,194000,233170,245350,173000,235000,625000,171000,163000,171900,200500,239000,285000,119500,115000,154900,93000,250000,392500,120000,186700,104900,95000,262000,195000,189000,168000,174000,125000,165000,158000,176000,219210,144000,178000,148000,116050,197900,117000,213000,153500,271900,107000,200000,140000,290000,189000,164000,113000,145000,134500,125000,112000,229456,80500,91500,115000,134000,143000,137900,184000,145000,214000,147000,367294,127000,190000,132500,101800,142000,130000,138887,175500,195000,142500,265900,224900,248328,170000,465000,230000,178000,186500,169900,129500,119000,244000,171750,130000,294000,165400,127500,301500,99900,190000,151000,181000,128900,161500,180500,181000,183900,122000,378500,381000,144000,260000,185750,137000,177000,139000,137000,162000,197900,237000,68400,227000,180000,150500,139000,169000,132500,143000,190000,278000,281000,180500,119500,107500,162900,115000,138500,155000,140000,154000,225000,177500,290000,232000,130000,325000,202500,138000,147000,179200,335000,203000,302000,333168,119000,206900,295493,208900,275000,111000,156500,72500,190000,82500,147000,55000,79000,130500,256000,176500,227000,132500,100000,125500,125000,167900,135000,52500,200000,128500,123000,155000,228500,177000,155835,108500,262500,283463,215000,122000,200000,171000,134900,410000,235000,170000,110000,149900,177500,315000,189000,260000,104900,156932,144152,216000,193000,127000,144000,232000,105000,165500,274300,466500,250000,239000,91000,117000,83000,167500,58500,237500,157000,112000,105000,125500,250000,136000,377500,131000,235000,124000,123000,163000,246578,281213,160000,137500,138000,137450,120000,193000,193879,282922,105000,275000,133000,112000,125500,215000,230000,140000,90000,257000,207000,175900,122500,340000,124000,223000,179900,127500,136500,274970,144000,142000,271000,140000,119000,182900,192140,143750,64500,186500,160000,174000,120500,394617,149700,197000,191000,149300,310000,121000,179600,129000,157900,240000,112000,92000,136000,287090,145000,84500,185000,175000,210000,266500,142125,147500],"type":"histogram"}],                        {"hovermode":"closest","template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Distribution of SalePrice with skewness (skewness: 1.5660)"},"yaxis":{"title":{"text":"Abs Frequency"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('62d2a6d6-3bdf-450a-a790-b36f427dec3f');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



<div>                            <div id="2cd510c3-b725-437f-8dac-667bb573d91d" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("2cd510c3-b725-437f-8dac-667bb573d91d")) {                    Plotly.newPlot(                        "2cd510c3-b725-437f-8dac-667bb573d91d",                        [{"marker":{"color":"green"},"x":[12.24769911637256,12.109016442313738,12.317171167298682,11.849404844423074,12.429220196836383,11.870606902224585,12.634606283893019,12.206077645517674,11.774527900858466,11.678448377988163,11.771443882099637,12.75130259456002,11.87757552297847,12.540761149388421,11.964007453736912,11.790564777297387,11.91170829631447,11.407576060361786,11.976665770490767,11.842236406331555,12.692506186542177,11.845109950883979,12.345838935721968,11.774527900858466,11.944714374881176,12.454107814022272,11.811554895830453,12.631343648903034,12.242891437918386,11.134603622683905,10.596659732783579,11.914054519303125,12.100161978566867,12.016732516077072,12.533579815765737,12.6410997921206,11.884495917930654,11.938199736300925,11.599112335481124,11.314486721293981,11.982935344196433,12.043559598368038,11.87757552297847,11.777218637129327,11.856522261533737,12.675766851909591,12.387089184491993,12.428019481065611,11.635151947212842,11.751950239425476,12.083910661257521,11.648338835562747,11.608244735642321,12.861001210669144,11.775297421715825,12.103491596905931,12.058158312538197,12.188422799337028,12.991755706536901,11.735276702486741,11.970356641102999,11.522885696764481,12.218500103788141,11.849404844423074,12.29911206729564,12.666660207428574,12.100717685412471,12.328294703023394,11.289794413577894,12.323860125621124,12.404927602627597,11.771443882099637,12.128116509451258,11.883806029596208,11.58432477200052,11.418625774449596,11.818577604251553,11.751950239425476,11.824087219587643,11.608244735642321,12.173037959397275,11.941462360646149,12.409017571151187,11.748005492256807,12.034696963475044,12.468440756144114,12.066816325306588,12.010671928196441,11.350418300109132,11.724813914587886,11.60733523953139,11.498836682527127,12.004574385508691,11.804855999911272,12.229549885708273,12.128116509451258,12.273735966890266,11.459007676978688,11.326607934898927,11.767187766223199,12.230770136157428,12.08954444723597,11.686584611185529,12.200562492481511,12.040614105490253,12.429220196836383,11.51293546492023,11.652696102959753,11.652696102959753,12.154784614286667,11.82701331585176,12.100717685412471,12.858322307884688,12.287657240806908,12.466515835145202,12.07824495582233,11.842236406331555,11.951186847493474,12.676079399771027,12.007566827251699,12.100717685412471,11.51293546492023,11.820417517632333,11.944064817531656,12.10625783509458,11.344518647594766,11.759793355371237,11.373674891823535,11.951186847493474,11.918397239722838,12.328294703023394,12.404927602627597,11.923384748066601,12.301387370778713,12.100717685412471,12.066816325306588,11.870606902224585,12.049424683420913,12.345838935721968,12.352339472153524,11.652696102959753,12.468440756144114,12.019749091416921,12.225880174775122,11.736077016252437,11.775297421715825,11.561725152903833,12.312686874960354,11.856522261533737,11.652696102959753,11.711784520403112,12.827731880206521,12.154784614286667,12.367345048436391,11.736077016252437,11.277215789596891,11.603688960617083,12.504327379899443,12.448630513463613,12.676079399771027,11.998439434579147,12.929993908996359,12.301387370778713,11.544433821905132,11.931642378754141,11.755879486687116,12.154784614286667,12.693501691489768,12.119975396053194,12.337105293891872,11.763691965388299,12.278397958261774,12.384223015005313,12.001511614739405,12.122696471358962,12.400820837540556,12.259618151783465,12.058158312538197,13.126032637428713,11.51293546492023,12.083910661257521,12.206577518060586,11.695255355062795,12.206077645517674,11.751950239425476,13.07107218827772,12.06105265381003,11.813037464800539,11.940399914116648,12.563750586299372,12.660331092405904,12.122696471358962,12.165255859329688,11.775297421715825,11.751950239425476,11.908346971208951,12.650351332641707,12.367345048436391,11.552155793461898,12.524166311833152,11.849404844423074,12.052344376474922,11.626263078808801,11.91170829631447,11.608244735642321,12.103491596905931,11.87688084212164,11.856522261533737,12.531776395271262,11.884495917930654,11.49273296168228,12.13350732902497,12.439875187430006,11.957617696467539,11.993813394724253,11.808955099378537,12.25486757159303,11.580593459194764,12.649157827477366,12.027191164883382,12.230282214644559,12.206077645517674,12.100161978566867,11.48247656671073,12.864242696187505,11.626263078808801,12.577639650232573,11.571203807011969,11.736077016252437,12.167856627501678,11.904974309480181,12.90669432231532,11.456365695436427,11.761354623750297,12.285350445351659,11.402005077384885,12.130815551850676,12.178192583415596,12.669809806411457,11.635151947212842,12.478010170530368,11.612779849672751,11.277215789596891,11.695255355062795,12.230770136157428,12.394628892852941,11.827743504053695,11.849404844423074,12.100717685412471,12.531776395271262,11.245059091624615,12.367345048436391,12.06105265381003,11.970356641102999,11.884495917930654,12.345838935721968,12.242891437918386,12.301387370778713,12.352339472153524,11.48247656671073,12.07824495582233,12.528159767881128,11.92504173829169,11.775297421715825,11.19822841866684,12.075400019877522,12.128116509451258,12.097936057925592,11.699413330633599,11.904974309480181,12.491255347155265,12.394628892852941,12.577639650232573,11.842236406331555,11.732069026983156,12.230770136157428,12.211065162153215,11.856522261533737,12.936754021718041,12.165255859329688,12.339295865674435,12.128116509451258,12.242891437918386,12.407383590534293,12.096263359864539,12.011886987792716,11.976665770490767,11.385103457032141,11.711784520403112,11.941950837487406,12.359784657728284,11.81968195846389,11.782960235741939,12.367345048436391,12.025755079404915,11.867104296210078,11.931642378754141,12.384223015005313,12.072546967175038,11.973516181427835,11.964007453736912,12.495007682693725,12.230770136157428,11.917730355182513,12.59473402514672,12.631016799175821,12.323860125621124,11.402005077384885,11.320565693461432,12.793862088206213,12.017336559562406,11.790564777297387,11.694421681264394,12.834683971615659,12.08954444723597,12.146858590895919,12.468440756144114,12.506180941677357,12.468440756144114,12.141539457711714,12.74444728628135,12.777055016970678,12.61486886598463,11.745433036677353,12.39669713736169,11.373674891823535,12.688501881189456,11.886218559372042,12.276069679344003,11.264476926102367,11.686887175419702,11.842236406331555,12.556733038263904,12.240478903153711,12.165255859329688,12.34126328584159,12.841132451487937,12.273735966890266,12.218500103788141,11.951186847493474,12.220473454337986,11.314486721293981,11.379405500851828,12.491255347155265,11.350418300109132,11.850832386224807,11.928347504569842,11.967187086434018,11.944714374881176,12.988043102265232,12.670001611311875,12.154784614286667,11.461642696843066,11.570259974415546,11.849404844423074,12.086731521684554,12.06105265381003,11.80560254159177,11.775297421715825,12.54254845357358,11.957617696467539,11.884495917930654,12.198549416880068,11.678448377988163,12.154784614286667,11.898194668458823,11.976665770490767,12.013706813470412,11.790564777297387,11.995357787034974,12.057578437658943,11.808821212646512,11.736077016252437,11.719947764402805,12.29911206729564,11.0186455364637,11.904974309480181,12.73670383776449,12.885204569609956,12.095146671399466,11.751950239425476,12.142871895843887,12.271396795480596,11.23850177707664,12.388398368982115,12.165255859329688,11.30221677925738,11.736077016252437,12.160033942617154,12.962196972666593,11.686887175419702,12.278397958261774,11.575909653758895,11.51293546492023,11.599112335481124,11.767575435251747,11.719947764402805,12.040614105490253,11.112462823634855,12.392556361841777,12.411056299528171,12.013646205940514,11.58989576532275,12.460718739865234,12.031725210748633,11.918397239722838,11.652696102959753,12.083910661257521,12.54254845357358,12.73596827534975,11.002116507732017,11.884495917930654,12.310437165348775,11.652696102959753,12.337105293891872,12.106997888588934,11.915058360753717,12.384223015005313,11.744045122410057,11.863589378812122,12.237091546357338,12.278397958261774,11.635151947212842,12.660331092405904,11.842236406331555,11.813037464800539,12.524530013005732,11.599185726610003,12.182809136284991,12.072546967175038,11.355113089309977,11.288543647320601,11.715874472198905,12.10625783509458,11.30221677925738,12.264346270624156,11.661354090740998,11.686887175419702,11.411457364027745,11.608244735642321,13.22672519452875,11.678448377988163,12.000897933306714,12.058158312538197,12.25486757159303,11.755879486687116,12.154784614286667,12.20557752297723,11.691080018519525,11.695255355062795,11.608244735642321,12.54254845357358,12.225880174775122,12.25486757159303,12.144202560946875,12.075400019877522,11.49273296168228,12.45293662970407,11.989165855127435,11.608244735642321,12.481565733211923,11.951186847493474,11.041064111222378,12.147919030759663,11.728044909070784,12.093693109638318,12.025755079404915,11.89478753335453,12.429220196836383,12.138869243416007,12.264346270624156,12.154784614286667,11.904974309480181,12.994532278619134,12.433212202169729,11.794345471549745,12.249615734512615,12.847929163278053,12.601490784782534,11.401681006153426,12.694655727828803,12.832013750189885,11.951186847493474,12.00762780434872,11.794345471549745,11.898194668458823,11.957617696467539,12.072546967175038,11.982935344196433,11.362114203075018,11.652696102959753,11.798111925972616,12.059809113551395,11.951186847493474,11.421917019425596,10.460270761075149,12.971542813248435,12.122696471358962,11.775297421715825,11.695255355062795,11.635151947212842,12.331387247601903,11.849404844423074,12.574185427296195,11.898194668458823,11.732069026983156,12.278397958261774,12.246739428005663,11.989165855127435,11.732069026983156,12.013100572810686,12.219314582801099,11.774527900858466,11.80560254159177,11.477308649967686,12.90634935052389,11.970356641102999,12.487488878546145,12.259618151783465,12.36308066783498,11.573559498507079,11.918397239722838,11.976665770490767,12.662709207270272,12.07824495582233,11.790564777297387,13.008661502470739,11.362114203075018,12.209192772866418,12.072546967175038,11.759793355371237,11.58525542883217,10.579005242826247,12.08954444723597,11.58525542883217,12.144202560946875,11.619544188752133,11.970356641102999,12.513561021741964,12.660331092405904,12.421188057397053,12.2702251546747,11.798111925972616,12.098854850641,12.34148164933907,12.25486757159303,11.771443882099637,11.736077016252437,12.47991311343404,11.849404844423074,11.630717389475995,12.450981602512561,11.58989576532275,12.556733038263904,11.635151947212842,11.856522261533737,11.58989576532275,12.072546967175038,12.36308066783498,11.707677772181544,12.043559598368038,11.58989576532275,12.128116509451258,12.498745990829315,11.759793355371237,12.691583538230217,12.273735966890266,12.665397584983756,11.820123358887447,11.867805799560465,11.695255355062795,12.32163544049751,12.043559598368038,11.842236406331555,11.68267667834028,11.884495917930654,12.010671928196441,11.891368749982085,11.78676973513378,12.111217862017215,12.442306148381936,11.68267667834028,12.691583538230217,11.798111925972616,12.820990681250251,11.775297421715825,11.827743504053695,11.870606902224585,11.283524879179666,12.132969552931215,13.02132904585825,11.835016210489899,11.849404844423074,11.608244735642321,12.672949516558306,11.648373769145277,12.176654133447505,12.289958727201455,11.92504173829169,12.524530013005732,11.856522261533737,12.301387370778713,11.92504173829169,12.30592250537653,12.230770136157428,11.931642378754141,12.323860125621124,12.791358964949923,11.68267667834028,12.653961664405365,11.904974309480181,12.47419338657158,11.898194668458823,11.231901180182513,11.831386468789589,12.118339189722553,11.566475710526321,12.659737267217572,12.628070334272698,11.112462823634855,12.388398368982115,11.813037464800539,12.034696963475044,12.014615485902027,11.982935344196433,11.848690308590855,11.938199736300925,11.813037464800539,12.034696963475044,11.728044909070784,12.252483791602627,11.320565693461432,11.845109950883979,11.87757552297847,12.206077645517674,11.002116507732017,11.440365524765754,11.350418300109132,12.485830907348902,12.520887034998584,12.328294703023394,12.75130259456002,11.931642378754141,12.823631142839984,11.872353615355411,11.495789479023493,11.951186847493474,11.951186847493474,11.344518647594766,12.2353935553999,11.58989576532275,12.160033942617154,11.813037464800539,12.765691290604371,11.385103457032141,11.887938238422299,11.91170829631447,11.487617913343598,12.025755079404915,12.19552218463499,12.904209855160254,11.608244735642321,11.831386468789589,12.955129822091815,12.348010479804223,11.767575435251747,12.173037959397275,12.031725210748633,11.831386468789589,12.063938642041201,11.548302261270587,12.013706813470412,12.458778882573533,11.849404844423074,11.908346971208951,11.373674891823535,11.603688960617083,12.827995001361906,11.763691965388299,11.870606902224585,11.979391594635791,12.06105265381003,12.560247968016562,12.30592250537653,12.242891437918386,12.336556900342153,11.910365121788654,12.879019669789754,12.179220327453931,11.856522261533737,12.721888795877375,11.59433032142082,11.860062063178265,12.07824495582233,11.39640288460621,11.724004532183379,11.83863282480007,12.185875040240456,12.652362948153472,11.849404844423074,12.799178471909892,11.849404844423074,12.269052139516221,10.915106645867501,12.618185607609803,12.445093483000797,12.09815887308463,11.60733523953139,10.859018228147885,11.54031687156214,12.149507585033072,11.767575435251747,11.779136168550774,12.013706813470412,11.979805470779858,11.964007453736912,12.739640688812672,11.763691965388299,12.524530013005732,11.870606902224585,11.732069026983156,11.813037464800539,12.676079399771027,11.699413330633599,12.310437165348775,12.178192583415596,11.608244735642321,11.542493975902508,12.373707715234994,12.141539457711714,12.312686874960354,11.786008995353765,11.58989576532275,12.001511614739405,11.44572741040675,12.3879816172226,12.095146671399466,12.154784614286667,11.790564777297387,11.863589378812122,12.095146671399466,12.072546967175038,12.100717685412471,12.610874200202108,12.371591321286905,12.491176396967154,12.469978029554953,11.49273296168228,11.477308649967686,11.995357787034974,12.287657240806908,12.526346537332584,11.957617696467539,12.058158312538197,12.264346270624156,11.97603664577121,12.097378802727306,12.577639650232573,11.755879486687116,11.51293546492023,12.279327754099118,12.727841176689147,12.506180941677357,12.484208043148731,12.188422799337028,11.982935344196433,12.28690580994122,13.195615697878253,11.812296455066814,11.532737896139919,11.580593459194764,11.648338835562747,12.886643575525623,11.995357787034974,12.308182383129822,11.867104296210078,11.87757552297847,11.813037464800539,12.07824495582233,12.077676615757813,12.139403856983334,12.016732516077072,11.759793355371237,11.992266613576142,11.842236406331555,12.358798024384122,11.588969419043872,12.141539457711714,11.984184555794167,11.89683320712845,12.505402864156405,12.323860125621124,12.178192583415596,12.049424683420913,11.874097282798646,11.608244735642321,13.091906231773164,12.072546967175038,12.206077645517674,11.60733523953139,12.149507585033072,13.275829251377,11.678448377988163,12.335700805443338,11.816734299348461,12.317171167298682,11.982622797311857,11.571203807011969,12.10625783509458,11.881041706925902,10.932999821119937,11.969723533345908,11.661354090740998,12.323415584358152,11.827743504053695,12.50987778989193,11.951186847493474,12.319405795112926,12.117246896289501,11.440365524765754,12.323860125621124,11.845827048676645,12.357079818283598,12.861001210669144,11.603688960617083,12.149507585033072,12.128116509451258,11.900912042974875,12.019749091416921,11.92504173829169,12.37581963951765,12.025755079404915,11.849047640327145,11.759793355371237,11.941462360646149,11.51293546492023,11.87757552297847,11.779136168550774,11.849404844423074,11.967187086434018,12.071975378543236,11.856522261533737,11.944064817531656,12.049424683420913,12.269052139516221,11.80186424743109,12.388398368982115,12.138869243416007,11.78676973513378,12.278397958261774,12.00762780434872,11.970356641102999,12.043559598368038,11.751950239425476,11.898194668458823,12.066816325306588,11.931642378754141,12.429220196836383,12.154521423145214,11.78676973513378,11.931642378754141,11.794345471549745,12.43153750053306,11.908346971208951,12.424810506025395,11.767575435251747,12.037659911047466,12.371591321286905,11.603688960617083,12.208574513247495,11.661354090740998,11.798111925972616,11.104972264124866,12.623064399210183,11.792456911128632,12.765691290604371,11.904974309480181,11.824087219587643,11.964007453736912,12.141539457711714,12.08954444723597,11.68267667834028,11.51293546492023,12.70351207261035,11.884495917930654,11.816734299348461,12.498745990829315,11.915058360753717,11.719134432217524,12.058158312538197,11.9479558477909,12.013706813470412,11.685693195625962,11.849404844423074,11.575909653758895,11.870278179170796,13.323928581764987,11.813037464800539,11.608244735642321,11.938199736300925,12.100717685412471,12.388398368982115,11.74006900564972,11.759793355371237,12.449022745701502,12.429220196836383,11.782960235741939,12.066816325306588,11.946660519211651,11.874097282798646,11.385103457032141,11.884495917930654,12.065280672855897,11.225256725762893,10.471978128496517,11.813037464800539,12.380030154325457,12.081081821061817,12.211065162153215,11.890683588496984,12.043500773454495,12.170450649220749,12.242891437918386,12.072546967175038,12.560247968016562,12.07824495582233,12.373707715234994,12.310437165348775,12.211065162153215,11.674202123168433,12.676079399771027,12.154784614286667,12.39669713736169,11.288543647320601,12.127575825689417,12.441148720271016,12.387560521575551,12.406565597924233,11.924379271639218,12.273735966890266,11.918397239722838,11.870606902224585,11.831386468789589,11.735276702486741,11.870606902224585,12.506180941677357,12.167856627501678,12.193498926601634,11.767575435251747,11.694421681264394,11.804855999911272,12.055255569732177,11.755879486687116,11.884495917930654,11.728044909070784,11.790564777297387,12.128116509451258,11.951186847493474,11.665655135642039,12.513561021741964,11.951186847493474,12.384223015005313,12.277932735946903,12.094587859196112,11.982935344196433,11.813037464800539,10.542732775946709,11.849404844423074,11.813037464800539,12.06105265381003,11.507922973347437,12.111767460549332,12.02874460037869,12.013706813470412,11.356283350762887,12.20557752297723,11.608244735642321,11.842236406331555,12.091789104493502,12.724869415131389,11.982278882874377,12.45254592991672,11.744045122410057,11.736077016252437,11.669937760751914,12.887129532148373,12.180759965737863,12.190964083849373,12.7599606323157,12.031725210748633,12.138869243416007,12.066241450757122,12.729323752253295,11.7085004721646,11.824087219587643,12.128116509451258,11.418625774449596,12.235636302128867,11.314486721293981,11.362114203075018,12.354496960983985,11.827049837927534,12.10625783509458,11.917730355182513,12.004574385508691,11.385103457032141,12.388398368982115,11.532737896139919,11.813037464800539,11.51293546492023,12.013706813470412,11.350418300109132,11.68856642283994,12.33270970174018,12.220966184120165,12.141539457711714,11.982935344196433,12.271349956195543,12.07824495582233,12.175618592671356,11.373674891823535,12.160033942617154,12.567240979056246,11.630717389475995,12.02874460037869,12.588194064175278,11.561725152903833,11.678448377988163,11.982935344196433,12.190964083849373,12.644330802262576,12.345838935721968,11.693169865968983,11.338583982516495,12.661917130530489,12.567240979056246,11.48247656671073,11.289794413577894,11.951186847493474,12.06105265381003,12.185875040240456,12.477171727089518,12.535379989788606,11.846543632608535,13.229569788348703,11.884495917930654,11.652696102959753,11.349241150793087,12.080996832096167,12.206782393599413,12.013706813470412,11.881041706925902,12.449022745701502,12.100717685412471,12.13270055639427,12.421188057397053,12.721888795877375,12.301387370778713,12.271396795480596,11.30221677925738,11.407576060361786,11.612779849672751,11.944714374881176,12.70077193614212,12.08954444723597,12.031129798974042,11.927687224983487,11.813037464800539,11.813037464800539,11.944714374881176,11.42410518016564,11.979805470779858,12.175618592671356,12.29911206729564,12.043559598368038,11.840796531639839,11.95697646939348,11.744045122410057,11.884495917930654,11.798111925972616,12.165255859329688,11.982935344196433,12.141539457711714,11.898194668458823,11.332613886815139,12.437188334739655,11.831386468789589,12.190964083849373,11.439289689006642,11.982935344196433,11.824087219587643,11.891368749982085,11.767575435251747,12.080696479888445,11.751950239425476,12.043559598368038,11.759793355371237,11.964007453736912,11.002116507732017,11.691080018519525,11.813037464800539,11.979805470779858,11.571203807011969,12.691583538230217,12.100161978566867,12.52352951631214,12.10625783509458,12.54254845357358,12.144202560946875,12.230770136157428,11.774527900858466,11.80932691294063,11.669937760751914,12.669809806411457,12.123239799036943,11.775297421715825,11.849404844423074,11.803361242493626,11.681832447342321,12.268582547911404,11.626263078808801,11.678448377988163,12.007017865994577,11.652696102959753,12.066816325306588,12.464587201678082,12.278397958261774,11.849404844423074,11.813037464800539,11.44572741040675,11.674202123168433,12.386312871189734,12.037659911047466,11.532737896139919,11.686887175419702,11.451060699493427,12.185875040240456,11.87757552297847,11.842236406331555,12.193498926601634,12.959540872420414,11.289794413577894,11.289794413577894,11.91170829631447,12.100717685412471,12.069685751267224,11.669082701743504,11.870606902224585,11.728044909070784,11.917730355182513,12.345838935721968,11.699413330633599,12.215037342290755,12.292254928916668,12.100161978566867,12.345838935721968,12.367889578686164,12.128116509451258,11.891368749982085,12.319405795112926,11.767575435251747,11.59873612138334,12.175618592671356,12.359527368950229,12.41044511731981,12.06105265381003,12.367345048436391,13.345508528717257,12.049424683420913,12.001511614739405,12.05467400868968,12.208574513247495,12.384223015005313,12.560247968016562,11.691080018519525,11.652696102959753,11.950541482162043,11.440365524765754,12.429220196836383,12.880294363972055,11.695255355062795,12.137263685701864,11.560772327227415,11.461642696843066,12.476103599529843,12.180759965737863,12.149507585033072,12.031725210748633,12.066816325306588,11.736077016252437,12.013706813470412,11.970356641102999,12.07824495582233,12.29779001527865,11.87757552297847,12.08954444723597,11.904974309480181,11.661785028640898,12.19552218463499,11.669937760751914,12.269052139516221,11.941462360646149,12.513193308436527,11.580593459194764,12.206077645517674,11.849404844423074,12.577639650232573,12.149507585033072,12.00762780434872,11.635151947212842,11.884495917930654,11.80932691294063,11.736077016252437,11.626263078808801,12.343470927093518,11.296024885689746,11.42410518016564,11.652696102959753,11.80560254159177,11.870606902224585,11.834291315386716,12.122696471358962,11.884495917930654,12.273735966890266,11.898194668458823,12.813920618864572,11.751950239425476,12.154784614286667,11.794345471549745,11.530775206233024,11.863589378812122,11.775297421715825,11.841423131921784,12.075400019877522,12.180759965737863,11.867104296210078,12.490879338036374,12.323415584358152,12.42250975887691,12.043559598368038,13.049794835104814,12.345838935721968,12.08954444723597,12.136191879981927,12.042971193457888,11.771443882099637,11.686887175419702,12.404927602627597,12.053801032483948,11.775297421715825,12.591338447675579,12.016128107502665,11.755879486687116,12.616528611893461,11.511934974596555,12.154784614286667,11.92504173829169,12.10625783509458,11.766799946849087,11.992266613576142,12.103491596905931,12.10625783509458,12.122152848315528,11.711784520403112,12.843973993864076,12.850557278777309,11.87757552297847,12.468440756144114,12.132162346145595,11.827743504053695,12.083910661257521,11.842236406331555,11.827743504053695,11.995357787034974,12.19552218463499,12.37581963951765,11.133142723386813,12.33270970174018,12.100717685412471,11.921725007667265,11.842236406331555,12.037659911047466,11.794345471549745,11.870606902224585,12.154784614286667,12.535379989788606,12.546113507028412,12.103491596905931,11.691080018519525,11.58525542883217,12.000897933306714,11.652696102959753,11.83863282480007,11.951186847493474,11.849404844423074,11.944714374881176,12.323860125621124,12.086731521684554,12.577639650232573,12.354496960983985,11.775297421715825,12.691583538230217,12.218500103788141,11.835016210489899,11.898194668458823,12.096263359864539,12.721888795877375,12.220966184120165,12.618185607609803,12.716405147731708,11.686887175419702,12.239995696974733,12.596403811053621,12.249615734512615,12.524530013005732,11.6172944892629,11.960817678718287,11.19135563385109,12.154784614286667,11.320565693461432,11.898194668458823,10.915106645867501,11.277215789596891,11.779136168550774,12.45293662970407,12.081081821061817,12.33270970174018,11.794345471549745,11.51293546492023,11.74006900564972,11.736077016252437,12.031129798974042,11.813037464800539,10.868587496017359,12.206077645517674,11.763691965388299,11.719947764402805,11.951186847493474,12.339295865674435,12.083910661257521,11.95655945119582,11.59451466851004,12.478010170530368,12.554840409959793,12.278397958261774,11.711784520403112,12.206077645517674,12.049424683420913,11.812296455066814,12.923914877701906,12.367345048436391,12.043559598368038,11.608244735642321,11.917730355182513,12.086731521684554,12.660331092405904,12.149507585033072,12.468440756144114,11.560772327227415,11.963574241653713,11.878630514504515,12.283038316285214,12.170450649220749,11.751950239425476,11.87757552297847,12.354496960983985,11.561725152903833,12.016732516077072,12.521981322562505,13.05301544288996,12.429220196836383,12.384223015005313,11.418625774449596,11.669937760751914,11.326607934898927,12.02874460037869,10.97679912709094,12.377927112974284,11.964007453736912,11.626263078808801,11.561725152903833,11.74006900564972,12.429220196836383,11.820417517632333,12.841328496674329,11.782960235741939,12.367345048436391,11.728044909070784,11.719947764402805,12.001511614739405,12.415437707935368,12.546871224308067,11.982935344196433,11.831386468789589,11.835016210489899,11.831022768939789,11.695255355062795,12.170450649220749,12.174994689959313,12.552930054797935,11.561725152903833,12.524530013005732,11.798111925972616,11.626263078808801,11.74006900564972,12.278397958261774,12.345838935721968,11.849404844423074,11.407576060361786,12.45683525492037,12.240478903153711,12.077676615757813,11.715874472198905,12.73670383776449,11.728044909070784,12.314931534737134,12.100161978566867,11.755879486687116,11.824087219587643,12.524420916542532,11.87757552297847,11.863589378812122,12.50987778989193,11.849404844423074,11.686887175419702,12.116700301832573,12.165984756488543,11.875837915157138,11.07443600653965,12.136191879981927,11.982935344196433,12.066816325306588,11.699413330633599,12.885673487343764,11.916395250412128,12.190964083849373,12.160033942617154,11.913719681428494,12.644330802262576,11.703554089007538,12.098493002763465,11.767575435251747,11.969723533345908,12.388398368982115,11.626263078808801,11.42955472553732,11.820417517632333,12.567554517655415,11.884495917930654,11.344518647594766,12.128116509451258,12.072546967175038,12.25486757159303,12.493133274926212,11.864469267087891,11.901590234400047],"type":"histogram"}],                        {"hovermode":"closest","template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Distribution of SalePrice removing skewness (skewness: 0.0655)"},"yaxis":{"title":{"text":"Abs Frequency"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('2cd510c3-b725-437f-8dac-667bb573d91d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


### Skewness of Explanatory Variables

Now, it is time to analyze the skewness of explanatory variables. I show the skewness of explanatory variables in the following bar plot.


```python
def bar_plot(x, y, title, yaxis, c_scale):
    trace = go.Bar(
    x = x,
    y = y,
    marker = dict(color = y, colorscale = c_scale))
    layout = go.Layout(hovermode= 'closest', title = title, yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig)

def show_explanetory_variables_skewness(process_data):
    skew_study_data = process_data[numerical_features].skew()
    skew_merged = pd.DataFrame(data = skew_study_data, columns = ['Skewness'])
    skew_merged_sorted = skew_merged.sort_values(ascending = False, by = 'Skewness')
    bar_plot(skew_merged_sorted.index, skew_merged_sorted.Skewness, 'Skewness in Explanatory Variables', 'Skewness', 'Bluered')

show_explanetory_variables_skewness(study_data)
```


<div>                            <div id="fd7f9cb8-2a27-4cd1-bf8b-1c10db2d32a3" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("fd7f9cb8-2a27-4cd1-bf8b-1c10db2d32a3")) {                    Plotly.newPlot(                        "fd7f9cb8-2a27-4cd1-bf8b-1c10db2d32a3",                        [{"marker":{"color":[24.443364355103984,17.52261328656046,12.587561473397649,10.289865655700298,8.998564223097551,4.481366385352843,4.248586900519406,4.128966618050054,4.115641120236566,3.084453658776395,2.648986842579437,2.3398461730491924,1.5512706094546915,0.9217587594500611,0.8670808651211774,0.8351923886551447,0.7778663580365494,0.7448554856296865,0.6842230842005776,0.6614164666720442,0.6326775818360327,0.5911523018088685,0.48639533694703774,0.2178834913024472,0.21506674449586063,0.13299116169635294,0.09321383457628772,0.017693587867073528,-0.34347539404012006,-0.4998309619319798,-0.6100865132247322,-0.6458205729752212],"colorscale":[[0.0,"rgb(0,0,255)"],[1.0,"rgb(255,0,0)"]]},"x":["MiscVal","PoolArea","LotArea","3SsnPorch","LowQualFinSF","KitchenAbvGr","BsmtFinSF2","BsmtHalfBath","ScreenPorch","EnclosedPorch","MasVnrArea","OpenPorchSF","WoodDeckSF","BsmtUnfSF","1stFlrSF","GrLivArea","2ndFlrSF","BsmtFinSF1","HalfBath","TotRmsAbvGrd","Fireplaces","BsmtFullBath","TotalBsmtSF","MoSold","BedroomAbvGr","GarageArea","YrSold","FullBath","GarageCars","YearRemodAdd","YearBuilt","GarageYrBlt"],"y":[24.443364355103984,17.52261328656046,12.587561473397649,10.289865655700298,8.998564223097551,4.481366385352843,4.248586900519406,4.128966618050054,4.115641120236566,3.084453658776395,2.648986842579437,2.3398461730491924,1.5512706094546915,0.9217587594500611,0.8670808651211774,0.8351923886551447,0.7778663580365494,0.7448554856296865,0.6842230842005776,0.6614164666720442,0.6326775818360327,0.5911523018088685,0.48639533694703774,0.2178834913024472,0.21506674449586063,0.13299116169635294,0.09321383457628772,0.017693587867073528,-0.34347539404012006,-0.4998309619319798,-0.6100865132247322,-0.6458205729752212],"type":"bar"}],                        {"hovermode":"closest","template":{"data":{"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Skewness in Explanatory Variables"},"yaxis":{"title":{"text":"Skewness"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('fd7f9cb8-2a27-4cd1-bf8b-1c10db2d32a3');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


The graph shows that many variables are skewed, predominantly to the right. To address this, I apply the natural logarithm transformation to variables with skewness greater than 0.75, using this threshold to improve their distributions.


```python
display_md('**Features to be transformed (skewness>0.75):**')

skew_study_data = study_data[numerical_features].skew()
filtered_skew_study_data = skew_study_data[skew_study_data>0.75]
skewed_columns = filtered_skew_study_data.index.values

display(skewed_columns)

for col in skewed_columns:
    col_values = np.log1p(study_data[col])
    study_data[col] = col_values
```


**Features to be transformed (skewness>0.75):**



    array(['LotArea', 'MasVnrArea', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF',
           '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath',
           'KitchenAbvGr', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
           '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'], dtype=object)


##  Assessing Missing Data in Each Column

Several columns have a high proportion of missing values, making them outliers. To address this, I apply a 6% threshold to filter out mostly empty columns.


```python
column_missing_data_percentage = {}
column_missing_data_threshold = 6

print(
    f"* The threshold for acceptable missing data in a column is {column_missing_data_threshold}%. Most features meet this criteria, with only a few exceeding it. Those features that exceed the threshold are notably distinct from the majority."
)

for index, feature in enumerate(study_data.columns):
    counts = study_data[feature].value_counts(dropna=False)
    percentages = counts / study_data_num_rows * 100

    if np.nan not in percentages.index:
        column_missing_data_percentage[feature] = 0
        continue

    column_missing_data_percentage[feature] = percentages[np.nan]

    print(
        f"{feature} attribute with more than {percentages[np.nan] :03.2f}% NaN"
    )

    # Investigate patterns in the amount of missing data in each column.
    if percentages[np.nan] < column_missing_data_threshold:
        continue

    print(
        f"---> {index} - {feature} with missing data that exceeds the threshold of {column_missing_data_threshold}%"
    )

    # plt.figure()
    # counts.plot.bar(title=feature, grid=True)
```

    * The threshold for acceptable missing data in a column is 6%. Most features meet this criteria, with only a few exceeding it. Those features that exceed the threshold are notably distinct from the majority.
    LotFrontage attribute with more than 17.74% NaN
    ---> 3 - LotFrontage with missing data that exceeds the threshold of 6%
    Alley attribute with more than 93.49% NaN
    ---> 6 - Alley with missing data that exceeds the threshold of 6%
    MasVnrType attribute with more than 59.66% NaN
    ---> 25 - MasVnrType with missing data that exceeds the threshold of 6%
    MasVnrArea attribute with more than 0.55% NaN
    BsmtQual attribute with more than 2.53% NaN
    BsmtCond attribute with more than 2.53% NaN
    BsmtExposure attribute with more than 2.60% NaN
    BsmtFinType1 attribute with more than 2.53% NaN
    BsmtFinType2 attribute with more than 2.60% NaN
    Electrical attribute with more than 0.07% NaN
    FireplaceQu attribute with more than 47.26% NaN
    ---> 57 - FireplaceQu with missing data that exceeds the threshold of 6%
    GarageType attribute with more than 5.55% NaN
    GarageYrBlt attribute with more than 5.55% NaN
    GarageFinish attribute with more than 5.55% NaN
    GarageQual attribute with more than 5.55% NaN
    GarageCond attribute with more than 5.55% NaN
    PoolQC attribute with more than 99.38% NaN
    ---> 72 - PoolQC with missing data that exceeds the threshold of 6%
    Fence attribute with more than 80.55% NaN
    ---> 73 - Fence with missing data that exceeds the threshold of 6%
    MiscFeature attribute with more than 96.03% NaN
    ---> 74 - MiscFeature with missing data that exceeds the threshold of 6%


As a result of the previous analysis, the following columns are identified to be dropped:

- Alley
- Fence
- FireplaceQu
- LotFrontage
- MasVnrType
- MiscFeature
- PoolQC


```python
removed90_NaN_feats_study_data = pd.DataFrame(study_data)

for feature in ["LotFrontage", "Alley", "MasVnrType", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]:
    removed90_NaN_feats_study_data.drop(feature, axis=1, inplace=True)

display(removed90_NaN_feats_study_data.head(n=head_n_of_records))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>9.042040</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>5.283204</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>5.017280</td>
      <td>856</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>6.753438</td>
      <td>6.751101</td>
      <td>0.0</td>
      <td>7.444833</td>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0.693147</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>548</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0.000000</td>
      <td>4.127134</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.247699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>9.169623</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>0.000000</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>5.652489</td>
      <td>1262</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>7.141245</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.141245</td>
      <td>0</td>
      <td>0.693147</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0.693147</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>460</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>5.700444</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.109016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>9.328212</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>5.093750</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>6.075346</td>
      <td>920</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>6.825460</td>
      <td>6.765039</td>
      <td>0.0</td>
      <td>7.488294</td>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0.693147</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>608</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0.000000</td>
      <td>3.761200</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.317171</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>9.164401</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>0.000000</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>6.293419</td>
      <td>756</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>6.869014</td>
      <td>6.629363</td>
      <td>0.0</td>
      <td>7.448916</td>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0.693147</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3</td>
      <td>642</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0.000000</td>
      <td>3.583519</td>
      <td>5.609472</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>11.849405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>9.565284</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>5.860786</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>6.196444</td>
      <td>1145</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>7.044033</td>
      <td>6.960348</td>
      <td>0.0</td>
      <td>7.695758</td>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0.693147</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>836</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>5.262690</td>
      <td>4.442651</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.429220</td>
    </tr>
  </tbody>
</table>
</div>


### Imputing Missing Values

After the previous steps, the columns now contain 94% meaningful values.

I applied a simple imputer to complete the dataset to ensure compatibility with estimators that require all values to be numerical.

Numerical features are imputed with their mean value, while the remaining features are imputed with the most frequent value.


```python
from sklearn.impute import SimpleImputer

numerical_imputer = SimpleImputer(strategy="mean")
ordinal_imputer = SimpleImputer(strategy="most_frequent")
categorical_imputer = SimpleImputer(strategy="most_frequent")

numerical_feats_with_NaN = [
    "MasVnrArea", # attribute with more than 0.55% NaN
    "GarageYrBlt", # attribute with more than 5.55% NaN
]

ordinal_feats_with_NaN = [
    "BsmtQual", # attribute with more than 2.53% NaN
    "BsmtCond", # attribute with more than 2.53% NaN
    "BsmtExposure", # attribute with more than 2.60% NaN
    "BsmtFinType1", # attribute with more than 2.53% NaN
    "BsmtFinType2", # attribute with more than 2.60% NaN
    "GarageFinish", # attribute with more than 5.55% NaN
    "GarageQual", # attribute with more than 5.55% NaN
    "GarageCond" # attribute with more than 5.55% NaN
]

categorical_feats_with_NaN = [
    "Electrical", # attribute with more than 0.07% NaN
    "GarageType", # attribute with more than 5.55% NaN
]

all_feats_with_NaN = numerical_feats_with_NaN + ordinal_feats_with_NaN + categorical_feats_with_NaN

non_id_study_data = removed90_NaN_feats_study_data.reset_index(drop=True)
for feature in all_feats_with_NaN:
    non_id_study_data.drop(feature, axis=1, inplace=True)

numerical_imputed_study_data = numerical_imputer.fit_transform(removed90_NaN_feats_study_data[numerical_feats_with_NaN])
ordinal_imputed_study_data = ordinal_imputer.fit_transform(removed90_NaN_feats_study_data[ordinal_feats_with_NaN])
categorical_imputed_study_data = categorical_imputer.fit_transform(removed90_NaN_feats_study_data[categorical_feats_with_NaN])

imputed_study_data  = pd.DataFrame(
    np.column_stack([numerical_imputed_study_data, ordinal_imputed_study_data, categorical_imputed_study_data]),
    columns=all_feats_with_NaN
)

imputed_study_data = pd.concat([
    non_id_study_data,
    imputed_study_data
], axis=1)

columns_order = list(removed90_NaN_feats_study_data.columns)
imputed_study_data = imputed_study_data[columns_order]

display(imputed_study_data.head(n=head_n_of_records))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>9.042040</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>5.283204</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>5.017280</td>
      <td>856</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>6.753438</td>
      <td>6.751101</td>
      <td>0.0</td>
      <td>7.444833</td>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0.693147</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>548</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0.000000</td>
      <td>4.127134</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.247699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>9.169623</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>5.652489</td>
      <td>1262</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>7.141245</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.141245</td>
      <td>0</td>
      <td>0.693147</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0.693147</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>460</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>5.700444</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.109016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>9.328212</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>5.09375</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>6.075346</td>
      <td>920</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>6.825460</td>
      <td>6.765039</td>
      <td>0.0</td>
      <td>7.488294</td>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0.693147</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>608</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0.000000</td>
      <td>3.761200</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.317171</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>9.164401</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>6.293419</td>
      <td>756</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>6.869014</td>
      <td>6.629363</td>
      <td>0.0</td>
      <td>7.448916</td>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0.693147</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3</td>
      <td>642</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0.000000</td>
      <td>3.583519</td>
      <td>5.609472</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>11.849405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>9.565284</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>5.860786</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>6.196444</td>
      <td>1145</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>7.044033</td>
      <td>6.960348</td>
      <td>0.0</td>
      <td>7.695758</td>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0.693147</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>836</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>5.262690</td>
      <td>4.442651</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.429220</td>
    </tr>
  </tbody>
</table>
</div>



```python
missing_values_per_row = imputed_study_data.isnull().sum(axis=1)
print(f"To confirm {missing_values_per_row.max()} Missing Values per Row")
```

    To confirm 0 Missing Values per Row


## Normalizing Numerical Features

In addition to transforming highly skewed features, it is good practice to scale numerical features to a [0, 1] range. While this normalization does not alter the shape of each feature's distribution, it ensures that all features are treated equally when applying supervised learners.


```python
from sklearn.preprocessing import StandardScaler

scaled_study_data = pd.DataFrame(imputed_study_data)
min_max_scaler = StandardScaler()  # default=(0, 1)
scaled_study_data[numerical_features] = min_max_scaler.fit_transform(scaled_study_data[numerical_features])

display(scaled_study_data.head(n=head_n_of_records))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>-0.127817</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1.053769</td>
      <td>0.880629</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>1.207011</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>0.625446</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>-0.339062</td>
      <td>-0.472456</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>-0.805990</td>
      <td>1.185669</td>
      <td>-0.133789</td>
      <td>0.548227</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>0.800349</td>
      <td>1.231823</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>Gd</td>
      <td>0.927120</td>
      <td>Typ</td>
      <td>-0.951673</td>
      <td>Attchd</td>
      <td>1.023874</td>
      <td>RFn</td>
      <td>0.315804</td>
      <td>0.360672</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>-0.943983</td>
      <td>0.849493</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>-1.603837</td>
      <td>0.137472</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.247699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>0.120797</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>0.159469</td>
      <td>-0.427190</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>-0.811344</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>1.257846</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>0.003303</td>
      <td>0.512947</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>0.433256</td>
      <td>-0.867410</td>
      <td>-0.133789</td>
      <td>-0.378408</td>
      <td>-0.819275</td>
      <td>4.040898</td>
      <td>0.800349</td>
      <td>-0.758781</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>TA</td>
      <td>-0.314155</td>
      <td>Typ</td>
      <td>0.610487</td>
      <td>Attchd</td>
      <td>-0.101720</td>
      <td>RFn</td>
      <td>0.315804</td>
      <td>-0.054591</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1.253310</td>
      <td>-1.070556</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>-0.491667</td>
      <td>-0.615009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.109016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>0.429834</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>0.987524</td>
      <td>0.832191</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>1.134633</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>0.113946</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>0.231214</td>
      <td>-0.317122</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>-0.575842</td>
      <td>1.189908</td>
      <td>-0.133789</td>
      <td>0.680880</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>0.800349</td>
      <td>1.231823</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>Gd</td>
      <td>-0.314155</td>
      <td>Typ</td>
      <td>0.610487</td>
      <td>Attchd</td>
      <td>0.940496</td>
      <td>RFn</td>
      <td>0.315804</td>
      <td>0.643806</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>-0.943983</td>
      <td>0.679251</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>0.991227</td>
      <td>0.137472</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.317171</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>0.110623</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>-1.860986</td>
      <td>-0.717817</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>-0.811344</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>-0.513805</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>0.348751</td>
      <td>-0.715166</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>-0.436663</td>
      <td>1.148647</td>
      <td>-0.133789</td>
      <td>0.560689</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>-1.026153</td>
      <td>-0.758781</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>Gd</td>
      <td>0.306482</td>
      <td>Typ</td>
      <td>0.610487</td>
      <td>Detchd</td>
      <td>0.815430</td>
      <td>Unf</td>
      <td>1.656362</td>
      <td>0.804249</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>-0.943983</td>
      <td>0.596590</td>
      <td>2.840004</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>-1.603837</td>
      <td>-1.367490</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>11.849405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>0.891805</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>0.954402</td>
      <td>0.735316</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>1.427666</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>0.506871</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>0.296484</td>
      <td>0.228976</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>0.122612</td>
      <td>1.249303</td>
      <td>-0.133789</td>
      <td>1.314119</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>0.800349</td>
      <td>1.231823</td>
      <td>1.392121</td>
      <td>-0.207905</td>
      <td>Gd</td>
      <td>1.547757</td>
      <td>Typ</td>
      <td>0.610487</td>
      <td>Attchd</td>
      <td>0.898808</td>
      <td>RFn</td>
      <td>1.656362</td>
      <td>1.719716</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1.084573</td>
      <td>0.996280</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>2.103397</td>
      <td>0.137472</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.429220</td>
    </tr>
  </tbody>
</table>
</div>


## Bivariate Analysis of the SalePrice

To gain a deeper understanding of the Ames, Iowa dataset, I conducted a bivariate analysis of SalePrice. The results provide valuable insights into the features that could significantly influence the analysis.

This analysis highlights the top 10 variables with the highest correlation to SalePrice, both positive and negative. For this analysis, categorical and ordinal features are encoded to the [0, 1] range.



```python
from sklearn.preprocessing import LabelEncoder

df_corr = pd.DataFrame(study_data)

object_columns = df_corr.select_dtypes(include=['object']).columns

for col in object_columns:
    label_encoder = LabelEncoder()
    df_corr[col] = label_encoder.fit_transform(df_corr[col])

display(df_corr.head(n=head_n_of_records))

df_corr = df_corr.corr()

display_md('**Best 10 Positively Correlated Variables:**')
display(df_corr['SalePrice'].sort_values(ascending = False)[:11])

display_md('**Best 10 Negatively Correlated Variables:**')
display(df_corr['SalePrice'].sort_values(ascending = False)[-10:])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>3</td>
      <td>65.0</td>
      <td>9.042040</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>13</td>
      <td>1</td>
      <td>5.283204</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>706</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.017280</td>
      <td>856</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>6.753438</td>
      <td>6.751101</td>
      <td>0.0</td>
      <td>7.444833</td>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0.693147</td>
      <td>2</td>
      <td>8</td>
      <td>6</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2003.0</td>
      <td>1</td>
      <td>2</td>
      <td>548</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>0.000000</td>
      <td>4.127134</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>0.0</td>
      <td>2</td>
      <td>2008</td>
      <td>8</td>
      <td>4</td>
      <td>12.247699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>3</td>
      <td>80.0</td>
      <td>9.169623</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>24</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>8</td>
      <td>3</td>
      <td>0.000000</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>978</td>
      <td>5</td>
      <td>0.0</td>
      <td>5.652489</td>
      <td>1262</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7.141245</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>7.141245</td>
      <td>0</td>
      <td>0.693147</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0.693147</td>
      <td>3</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1976.0</td>
      <td>1</td>
      <td>2</td>
      <td>460</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>5.700444</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>0.0</td>
      <td>5</td>
      <td>2007</td>
      <td>8</td>
      <td>4</td>
      <td>12.109016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>3</td>
      <td>68.0</td>
      <td>9.328212</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>13</td>
      <td>1</td>
      <td>5.093750</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>486</td>
      <td>5</td>
      <td>0.0</td>
      <td>6.075346</td>
      <td>920</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>6.825460</td>
      <td>6.765039</td>
      <td>0.0</td>
      <td>7.488294</td>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0.693147</td>
      <td>2</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2001.0</td>
      <td>1</td>
      <td>2</td>
      <td>608</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>0.000000</td>
      <td>3.761200</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>0.0</td>
      <td>9</td>
      <td>2008</td>
      <td>8</td>
      <td>4</td>
      <td>12.317171</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>3</td>
      <td>60.0</td>
      <td>9.164401</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>15</td>
      <td>3</td>
      <td>0.000000</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>216</td>
      <td>5</td>
      <td>0.0</td>
      <td>6.293419</td>
      <td>756</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>6.869014</td>
      <td>6.629363</td>
      <td>0.0</td>
      <td>7.448916</td>
      <td>1</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0.693147</td>
      <td>2</td>
      <td>7</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>1998.0</td>
      <td>2</td>
      <td>3</td>
      <td>642</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>0.000000</td>
      <td>3.583519</td>
      <td>5.609472</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>0.0</td>
      <td>2</td>
      <td>2006</td>
      <td>8</td>
      <td>0</td>
      <td>11.849405</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>3</td>
      <td>84.0</td>
      <td>9.565284</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>15</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>13</td>
      <td>1</td>
      <td>5.860786</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>655</td>
      <td>5</td>
      <td>0.0</td>
      <td>6.196444</td>
      <td>1145</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>7.044033</td>
      <td>6.960348</td>
      <td>0.0</td>
      <td>7.695758</td>
      <td>1</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0.693147</td>
      <td>2</td>
      <td>9</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2000.0</td>
      <td>1</td>
      <td>3</td>
      <td>836</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>5.262690</td>
      <td>4.442651</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>0.0</td>
      <td>12</td>
      <td>2008</td>
      <td>8</td>
      <td>4</td>
      <td>12.429220</td>
    </tr>
  </tbody>
</table>
</div>



**Best 10 Positively Correlated Variables:**



    SalePrice       1.000000
    OverallQual     0.819240
    GrLivArea       0.732807
    GarageCars      0.680408
    GarageArea      0.655212
    TotalBsmtSF     0.641553
    1stFlrSF        0.611030
    FullBath        0.590919
    YearBuilt       0.588977
    YearRemodAdd    0.568986
    GarageYrBlt     0.544005
    Name: SalePrice, dtype: float64



**Best 10 Negatively Correlated Variables:**



    LotShape       -0.273934
    BsmtExposure   -0.299405
    MasVnrType     -0.310619
    HeatingQC      -0.425864
    FireplaceQu    -0.465384
    GarageType     -0.504519
    KitchenQual    -0.530470
    ExterQual      -0.584138
    BsmtQual       -0.588815
    GarageFinish   -0.604917
    Name: SalePrice, dtype: float64


## Reaching the Final Train Data

At this point, the dataset is mainly ready to target the objectives of this analysis. To enrich the dataset further and help reach valuable conclusions, I added the following features based on the existing ones:

- TotalSF: This feature represents the total living area in the house by adding up the basement, first-floor, and second-floor square footage.
- YrBltAndRemod: This combines the years since the house was built and any major renovations, giving a comprehensive view of its age and updates.
- Total_Bathrooms: This feature sums up the total number of bathrooms, including full and half bathrooms, in both the main living area and the basement.
- QualCond: This combines the house's overall quality and overall condition into a single score.
- ExterQualCond: This merges the quality and condition of the house's exterior into one feature.
- GarageQualCond: This combines the quality and condition of the garage into a single feature.
- BsmtQualCond: This merges the quality and condition of the basement into one feature.
- hasPool: This binary feature indicates whether the house has a pool.
- hasGarage: This binary feature indicates whether the house has a garage.
- hasBsmt: This binary feature indicates whether the house has a basement.
- hasFireplace: This binary feature indicates whether the house has a fireplace.
- house_age: This calculates the age of the house by subtracting the year it was built from the year it was sold.
- garage_age: This calculates the age of the garage by subtracting the year it was built from the year it was sold.
- old_house: This binary feature indicates whether the house was built before 1900.


```python
final_study_data = pd.DataFrame(scaled_study_data)

final_study_data['TotalSF'] = final_study_data['TotalBsmtSF'] + final_study_data['1stFlrSF'] + final_study_data['2ndFlrSF']
final_study_data['YrBltAndRemod'] = final_study_data['YearBuilt'] + final_study_data['YearRemodAdd']
final_study_data['Total_Bathrooms'] = (final_study_data['FullBath']
                               + (0.5 * final_study_data['HalfBath'])
                               + final_study_data['BsmtFullBath']
                               + (0.5 * final_study_data['BsmtHalfBath'])
                              )

final_study_data['QualCond'] = final_study_data.OverallQual * 100 + final_study_data.OverallCond
final_study_data['ExterQualCond'] = final_study_data.ExterQual + final_study_data.ExterCond
final_study_data['GarageQualCond'] = final_study_data.GarageQual + final_study_data.GarageCond
final_study_data['BsmtQualCond'] = final_study_data.BsmtQual + final_study_data.BsmtCond

final_study_data['hasPool'] = final_study_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
final_study_data['hasGarage'] = final_study_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
final_study_data['hasBsmt'] = final_study_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
final_study_data['hasFireplace'] = final_study_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

final_study_data['house_age'] = final_study_data.YrSold - final_study_data.YearBuilt
final_study_data['garage_age'] = final_study_data.YrSold - final_study_data.GarageYrBlt
final_study_data['old_house'] = np.where(final_study_data.YearBuilt < 1900, 1, 0)

display(final_study_data.head(n=head_n_of_records))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
      <th>TotalSF</th>
      <th>YrBltAndRemod</th>
      <th>Total_Bathrooms</th>
      <th>QualCond</th>
      <th>ExterQualCond</th>
      <th>GarageQualCond</th>
      <th>BsmtQualCond</th>
      <th>hasPool</th>
      <th>hasGarage</th>
      <th>hasBsmt</th>
      <th>hasFireplace</th>
      <th>house_age</th>
      <th>garage_age</th>
      <th>old_house</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>-0.127817</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1.053769</td>
      <td>0.880629</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>1.207011</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>0.625446</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>-0.339062</td>
      <td>-0.472456</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>-0.805990</td>
      <td>1.185669</td>
      <td>-0.133789</td>
      <td>0.548227</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>0.800349</td>
      <td>1.231823</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>Gd</td>
      <td>0.927120</td>
      <td>Typ</td>
      <td>-0.951673</td>
      <td>Attchd</td>
      <td>1.023874</td>
      <td>RFn</td>
      <td>0.315804</td>
      <td>0.360672</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>-0.943983</td>
      <td>0.849493</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>-1.603837</td>
      <td>0.137472</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.247699</td>
      <td>-0.092777</td>
      <td>1.934398</td>
      <td>2.409470</td>
      <td>705</td>
      <td>GdTA</td>
      <td>TATA</td>
      <td>GdTA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-0.916296</td>
      <td>-0.886401</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>0.120797</td>
      <td>Pave</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>0.159469</td>
      <td>-0.427190</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>-0.811344</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>1.257846</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>0.003303</td>
      <td>0.512947</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>0.433256</td>
      <td>-0.867410</td>
      <td>-0.133789</td>
      <td>-0.378408</td>
      <td>-0.819275</td>
      <td>4.040898</td>
      <td>0.800349</td>
      <td>-0.758781</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>TA</td>
      <td>-0.314155</td>
      <td>Typ</td>
      <td>0.610487</td>
      <td>Attchd</td>
      <td>-0.101720</td>
      <td>RFn</td>
      <td>0.315804</td>
      <td>-0.054591</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1.253310</td>
      <td>-1.070556</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>-0.491667</td>
      <td>-0.615009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.109016</td>
      <td>0.078793</td>
      <td>-0.267722</td>
      <td>1.622132</td>
      <td>608</td>
      <td>TATA</td>
      <td>TATA</td>
      <td>GdTA</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-0.774477</td>
      <td>-0.513288</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>0.429834</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>0.987524</td>
      <td>0.832191</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>1.134633</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>0.113946</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>0.231214</td>
      <td>-0.317122</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>-0.575842</td>
      <td>1.189908</td>
      <td>-0.133789</td>
      <td>0.680880</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>0.800349</td>
      <td>1.231823</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>Gd</td>
      <td>-0.314155</td>
      <td>Typ</td>
      <td>0.610487</td>
      <td>Attchd</td>
      <td>0.940496</td>
      <td>RFn</td>
      <td>0.315804</td>
      <td>0.643806</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>-0.943983</td>
      <td>0.679251</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>0.991227</td>
      <td>0.137472</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.317171</td>
      <td>0.296944</td>
      <td>1.819716</td>
      <td>2.409470</td>
      <td>705</td>
      <td>GdTA</td>
      <td>TATA</td>
      <td>GdTA</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.850052</td>
      <td>-0.803024</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>0.110623</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>-1.860986</td>
      <td>-0.717817</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>-0.811344</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>-0.513805</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>0.348751</td>
      <td>-0.715166</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>-0.436663</td>
      <td>1.148647</td>
      <td>-0.133789</td>
      <td>0.560689</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>-1.026153</td>
      <td>-0.758781</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>Gd</td>
      <td>0.306482</td>
      <td>Typ</td>
      <td>0.610487</td>
      <td>Detchd</td>
      <td>0.815430</td>
      <td>Unf</td>
      <td>1.656362</td>
      <td>0.804249</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>-0.943983</td>
      <td>0.596590</td>
      <td>2.840004</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>-1.603837</td>
      <td>-1.367490</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>11.849405</td>
      <td>-0.003182</td>
      <td>-2.578803</td>
      <td>-0.412333</td>
      <td>705</td>
      <td>TATA</td>
      <td>TATA</td>
      <td>TAGd</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.493497</td>
      <td>-2.182920</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>0.891805</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>0.954402</td>
      <td>0.735316</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>1.427666</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>0.506871</td>
      <td>Unf</td>
      <td>-0.355892</td>
      <td>0.296484</td>
      <td>0.228976</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>0.122612</td>
      <td>1.249303</td>
      <td>-0.133789</td>
      <td>1.314119</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>0.800349</td>
      <td>1.231823</td>
      <td>1.392121</td>
      <td>-0.207905</td>
      <td>Gd</td>
      <td>1.547757</td>
      <td>Typ</td>
      <td>0.610487</td>
      <td>Attchd</td>
      <td>0.898808</td>
      <td>RFn</td>
      <td>1.656362</td>
      <td>1.719716</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>1.084573</td>
      <td>0.996280</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>2.103397</td>
      <td>0.137472</td>
      <td>WD</td>
      <td>Normal</td>
      <td>12.429220</td>
      <td>1.600891</td>
      <td>1.689718</td>
      <td>2.409470</td>
      <td>805</td>
      <td>GdTA</td>
      <td>TATA</td>
      <td>GdTA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-0.816930</td>
      <td>-0.761335</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


### Preparing Categorical Variables for Supervised Learning

Finally, I need to convert non-numeric columns (without any inherent order) into numerical values to complete the data preprocessing. This conversion is crucial for the learning algorithms to work effectively. I use one-hot encoding, which creates binary (0 or 1) columns for each category, ensuring no ordinal relationship is implied. This can be done conveniently with the pandas `get_dummies` method.


```python
def one_hot_encode(final_data):
    categorical_features = [
        "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "Foundation", "Heating", "CentralAir", "Electrical", "GarageType", "SaleType", "SaleCondition"
    ]
    ordinal_features = [
        "LotShape", "LandSlope", "OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive",
        # Engineered Qual Features
        "QualCond", "ExterQualCond", "GarageQualCond", "BsmtQualCond"
    ]

    onehot1_study_data = pd.get_dummies(data = final_data, columns = categorical_features)
    return pd.get_dummies(data = onehot1_study_data, columns = ordinal_features)

encoded_study_data = one_hot_encode(final_study_data)
display(encoded_study_data.head(n=head_n_of_records))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>LotArea</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
      <th>TotalSF</th>
      <th>YrBltAndRemod</th>
      <th>Total_Bathrooms</th>
      <th>hasPool</th>
      <th>hasGarage</th>
      <th>hasBsmt</th>
      <th>hasFireplace</th>
      <th>house_age</th>
      <th>garage_age</th>
      <th>old_house</th>
      <th>MSSubClass_20</th>
      <th>MSSubClass_30</th>
      <th>MSSubClass_40</th>
      <th>MSSubClass_45</th>
      <th>MSSubClass_50</th>
      <th>MSSubClass_60</th>
      <th>MSSubClass_70</th>
      <th>MSSubClass_75</th>
      <th>MSSubClass_80</th>
      <th>MSSubClass_85</th>
      <th>MSSubClass_90</th>
      <th>MSSubClass_120</th>
      <th>MSSubClass_160</th>
      <th>MSSubClass_180</th>
      <th>MSSubClass_190</th>
      <th>MSZoning_C (all)</th>
      <th>MSZoning_FV</th>
      <th>MSZoning_RH</th>
      <th>MSZoning_RL</th>
      <th>MSZoning_RM</th>
      <th>Street_Grvl</th>
      <th>Street_Pave</th>
      <th>LandContour_Bnk</th>
      <th>LandContour_HLS</th>
      <th>LandContour_Low</th>
      <th>LandContour_Lvl</th>
      <th>Utilities_AllPub</th>
      <th>Utilities_NoSeWa</th>
      <th>LotConfig_Corner</th>
      <th>LotConfig_CulDSac</th>
      <th>LotConfig_FR2</th>
      <th>LotConfig_FR3</th>
      <th>LotConfig_Inside</th>
      <th>Neighborhood_Blmngtn</th>
      <th>Neighborhood_Blueste</th>
      <th>Neighborhood_BrDale</th>
      <th>Neighborhood_BrkSide</th>
      <th>Neighborhood_ClearCr</th>
      <th>Neighborhood_CollgCr</th>
      <th>Neighborhood_Crawfor</th>
      <th>Neighborhood_Edwards</th>
      <th>Neighborhood_Gilbert</th>
      <th>Neighborhood_IDOTRR</th>
      <th>Neighborhood_MeadowV</th>
      <th>Neighborhood_Mitchel</th>
      <th>Neighborhood_NAmes</th>
      <th>Neighborhood_NPkVill</th>
      <th>Neighborhood_NWAmes</th>
      <th>Neighborhood_NoRidge</th>
      <th>Neighborhood_NridgHt</th>
      <th>Neighborhood_OldTown</th>
      <th>Neighborhood_SWISU</th>
      <th>Neighborhood_Sawyer</th>
      <th>Neighborhood_SawyerW</th>
      <th>Neighborhood_Somerst</th>
      <th>Neighborhood_StoneBr</th>
      <th>Neighborhood_Timber</th>
      <th>Neighborhood_Veenker</th>
      <th>Condition1_Artery</th>
      <th>Condition1_Feedr</th>
      <th>Condition1_Norm</th>
      <th>Condition1_PosA</th>
      <th>Condition1_PosN</th>
      <th>Condition1_RRAe</th>
      <th>Condition1_RRAn</th>
      <th>Condition1_RRNe</th>
      <th>Condition1_RRNn</th>
      <th>Condition2_Artery</th>
      <th>Condition2_Feedr</th>
      <th>Condition2_Norm</th>
      <th>Condition2_PosA</th>
      <th>Condition2_PosN</th>
      <th>Condition2_RRAe</th>
      <th>Condition2_RRAn</th>
      <th>Condition2_RRNn</th>
      <th>BldgType_1Fam</th>
      <th>BldgType_2fmCon</th>
      <th>BldgType_Duplex</th>
      <th>BldgType_Twnhs</th>
      <th>BldgType_TwnhsE</th>
      <th>HouseStyle_1.5Fin</th>
      <th>HouseStyle_1.5Unf</th>
      <th>HouseStyle_1Story</th>
      <th>HouseStyle_2.5Fin</th>
      <th>HouseStyle_2.5Unf</th>
      <th>HouseStyle_2Story</th>
      <th>HouseStyle_SFoyer</th>
      <th>HouseStyle_SLvl</th>
      <th>RoofStyle_Flat</th>
      <th>RoofStyle_Gable</th>
      <th>RoofStyle_Gambrel</th>
      <th>RoofStyle_Hip</th>
      <th>RoofStyle_Mansard</th>
      <th>RoofStyle_Shed</th>
      <th>RoofMatl_CompShg</th>
      <th>RoofMatl_Membran</th>
      <th>RoofMatl_Metal</th>
      <th>RoofMatl_Roll</th>
      <th>RoofMatl_Tar&amp;Grv</th>
      <th>RoofMatl_WdShake</th>
      <th>RoofMatl_WdShngl</th>
      <th>Exterior1st_AsbShng</th>
      <th>Exterior1st_AsphShn</th>
      <th>Exterior1st_BrkComm</th>
      <th>Exterior1st_BrkFace</th>
      <th>Exterior1st_CBlock</th>
      <th>Exterior1st_CemntBd</th>
      <th>Exterior1st_HdBoard</th>
      <th>Exterior1st_ImStucc</th>
      <th>Exterior1st_MetalSd</th>
      <th>Exterior1st_Plywood</th>
      <th>Exterior1st_Stone</th>
      <th>Exterior1st_Stucco</th>
      <th>Exterior1st_VinylSd</th>
      <th>Exterior1st_Wd Sdng</th>
      <th>Exterior1st_WdShing</th>
      <th>Exterior2nd_AsbShng</th>
      <th>Exterior2nd_AsphShn</th>
      <th>Exterior2nd_Brk Cmn</th>
      <th>Exterior2nd_BrkFace</th>
      <th>Exterior2nd_CBlock</th>
      <th>Exterior2nd_CmentBd</th>
      <th>Exterior2nd_HdBoard</th>
      <th>Exterior2nd_ImStucc</th>
      <th>Exterior2nd_MetalSd</th>
      <th>Exterior2nd_Other</th>
      <th>Exterior2nd_Plywood</th>
      <th>Exterior2nd_Stone</th>
      <th>Exterior2nd_Stucco</th>
      <th>Exterior2nd_VinylSd</th>
      <th>Exterior2nd_Wd Sdng</th>
      <th>Exterior2nd_Wd Shng</th>
      <th>Foundation_BrkTil</th>
      <th>Foundation_CBlock</th>
      <th>Foundation_PConc</th>
      <th>Foundation_Slab</th>
      <th>Foundation_Stone</th>
      <th>Foundation_Wood</th>
      <th>Heating_Floor</th>
      <th>Heating_GasA</th>
      <th>Heating_GasW</th>
      <th>Heating_Grav</th>
      <th>Heating_OthW</th>
      <th>Heating_Wall</th>
      <th>CentralAir_N</th>
      <th>CentralAir_Y</th>
      <th>Electrical_FuseA</th>
      <th>Electrical_FuseF</th>
      <th>Electrical_FuseP</th>
      <th>Electrical_Mix</th>
      <th>Electrical_SBrkr</th>
      <th>GarageType_2Types</th>
      <th>GarageType_Attchd</th>
      <th>GarageType_Basment</th>
      <th>GarageType_BuiltIn</th>
      <th>GarageType_CarPort</th>
      <th>GarageType_Detchd</th>
      <th>SaleType_COD</th>
      <th>SaleType_CWD</th>
      <th>SaleType_Con</th>
      <th>SaleType_ConLD</th>
      <th>SaleType_ConLI</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
      <th>LotShape_IR1</th>
      <th>LotShape_IR2</th>
      <th>LotShape_IR3</th>
      <th>LotShape_Reg</th>
      <th>LandSlope_Gtl</th>
      <th>LandSlope_Mod</th>
      <th>LandSlope_Sev</th>
      <th>OverallQual_1</th>
      <th>OverallQual_2</th>
      <th>OverallQual_3</th>
      <th>OverallQual_4</th>
      <th>OverallQual_5</th>
      <th>OverallQual_6</th>
      <th>OverallQual_7</th>
      <th>OverallQual_8</th>
      <th>OverallQual_9</th>
      <th>OverallQual_10</th>
      <th>OverallCond_1</th>
      <th>OverallCond_2</th>
      <th>OverallCond_3</th>
      <th>OverallCond_4</th>
      <th>OverallCond_5</th>
      <th>OverallCond_6</th>
      <th>OverallCond_7</th>
      <th>OverallCond_8</th>
      <th>OverallCond_9</th>
      <th>ExterQual_Ex</th>
      <th>ExterQual_Fa</th>
      <th>ExterQual_Gd</th>
      <th>ExterQual_TA</th>
      <th>ExterCond_Ex</th>
      <th>ExterCond_Fa</th>
      <th>ExterCond_Gd</th>
      <th>ExterCond_Po</th>
      <th>ExterCond_TA</th>
      <th>BsmtQual_Ex</th>
      <th>BsmtQual_Fa</th>
      <th>BsmtQual_Gd</th>
      <th>BsmtQual_TA</th>
      <th>BsmtCond_Fa</th>
      <th>BsmtCond_Gd</th>
      <th>BsmtCond_Po</th>
      <th>BsmtCond_TA</th>
      <th>BsmtExposure_Av</th>
      <th>BsmtExposure_Gd</th>
      <th>BsmtExposure_Mn</th>
      <th>BsmtExposure_No</th>
      <th>BsmtFinType1_ALQ</th>
      <th>BsmtFinType1_BLQ</th>
      <th>BsmtFinType1_GLQ</th>
      <th>BsmtFinType1_LwQ</th>
      <th>BsmtFinType1_Rec</th>
      <th>BsmtFinType1_Unf</th>
      <th>BsmtFinType2_ALQ</th>
      <th>BsmtFinType2_BLQ</th>
      <th>BsmtFinType2_GLQ</th>
      <th>BsmtFinType2_LwQ</th>
      <th>BsmtFinType2_Rec</th>
      <th>BsmtFinType2_Unf</th>
      <th>HeatingQC_Ex</th>
      <th>HeatingQC_Fa</th>
      <th>HeatingQC_Gd</th>
      <th>HeatingQC_Po</th>
      <th>HeatingQC_TA</th>
      <th>KitchenQual_Ex</th>
      <th>KitchenQual_Fa</th>
      <th>KitchenQual_Gd</th>
      <th>KitchenQual_TA</th>
      <th>Functional_Maj1</th>
      <th>Functional_Maj2</th>
      <th>Functional_Min1</th>
      <th>Functional_Min2</th>
      <th>Functional_Mod</th>
      <th>Functional_Sev</th>
      <th>Functional_Typ</th>
      <th>GarageFinish_Fin</th>
      <th>GarageFinish_RFn</th>
      <th>GarageFinish_Unf</th>
      <th>GarageQual_Ex</th>
      <th>GarageQual_Fa</th>
      <th>GarageQual_Gd</th>
      <th>GarageQual_Po</th>
      <th>GarageQual_TA</th>
      <th>GarageCond_Ex</th>
      <th>GarageCond_Fa</th>
      <th>GarageCond_Gd</th>
      <th>GarageCond_Po</th>
      <th>GarageCond_TA</th>
      <th>PavedDrive_N</th>
      <th>PavedDrive_P</th>
      <th>PavedDrive_Y</th>
      <th>QualCond_101</th>
      <th>QualCond_103</th>
      <th>QualCond_203</th>
      <th>QualCond_205</th>
      <th>QualCond_302</th>
      <th>QualCond_303</th>
      <th>QualCond_304</th>
      <th>QualCond_305</th>
      <th>QualCond_306</th>
      <th>QualCond_307</th>
      <th>QualCond_308</th>
      <th>QualCond_402</th>
      <th>QualCond_403</th>
      <th>QualCond_404</th>
      <th>QualCond_405</th>
      <th>QualCond_406</th>
      <th>QualCond_407</th>
      <th>QualCond_408</th>
      <th>QualCond_409</th>
      <th>QualCond_502</th>
      <th>QualCond_503</th>
      <th>QualCond_504</th>
      <th>QualCond_505</th>
      <th>QualCond_506</th>
      <th>QualCond_507</th>
      <th>QualCond_508</th>
      <th>QualCond_509</th>
      <th>QualCond_603</th>
      <th>QualCond_604</th>
      <th>QualCond_605</th>
      <th>QualCond_606</th>
      <th>QualCond_607</th>
      <th>QualCond_608</th>
      <th>QualCond_609</th>
      <th>QualCond_703</th>
      <th>QualCond_704</th>
      <th>QualCond_705</th>
      <th>QualCond_706</th>
      <th>QualCond_707</th>
      <th>QualCond_708</th>
      <th>QualCond_709</th>
      <th>QualCond_804</th>
      <th>QualCond_805</th>
      <th>QualCond_806</th>
      <th>QualCond_807</th>
      <th>QualCond_808</th>
      <th>QualCond_809</th>
      <th>QualCond_902</th>
      <th>QualCond_905</th>
      <th>QualCond_1005</th>
      <th>QualCond_1009</th>
      <th>ExterQualCond_ExEx</th>
      <th>ExterQualCond_ExGd</th>
      <th>ExterQualCond_ExTA</th>
      <th>ExterQualCond_FaFa</th>
      <th>ExterQualCond_FaTA</th>
      <th>ExterQualCond_GdGd</th>
      <th>ExterQualCond_GdTA</th>
      <th>ExterQualCond_TAEx</th>
      <th>ExterQualCond_TAFa</th>
      <th>ExterQualCond_TAGd</th>
      <th>ExterQualCond_TAPo</th>
      <th>ExterQualCond_TATA</th>
      <th>GarageQualCond_ExEx</th>
      <th>GarageQualCond_ExTA</th>
      <th>GarageQualCond_FaFa</th>
      <th>GarageQualCond_FaPo</th>
      <th>GarageQualCond_FaTA</th>
      <th>GarageQualCond_GdGd</th>
      <th>GarageQualCond_GdTA</th>
      <th>GarageQualCond_PoPo</th>
      <th>GarageQualCond_TAFa</th>
      <th>GarageQualCond_TAGd</th>
      <th>GarageQualCond_TATA</th>
      <th>BsmtQualCond_ExGd</th>
      <th>BsmtQualCond_ExTA</th>
      <th>BsmtQualCond_FaFa</th>
      <th>BsmtQualCond_FaPo</th>
      <th>BsmtQualCond_FaTA</th>
      <th>BsmtQualCond_GdFa</th>
      <th>BsmtQualCond_GdGd</th>
      <th>BsmtQualCond_GdTA</th>
      <th>BsmtQualCond_TAFa</th>
      <th>BsmtQualCond_TAGd</th>
      <th>BsmtQualCond_TATA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-0.127817</td>
      <td>1.053769</td>
      <td>0.880629</td>
      <td>1.207011</td>
      <td>0.625446</td>
      <td>-0.355892</td>
      <td>-0.339062</td>
      <td>-0.472456</td>
      <td>-0.805990</td>
      <td>1.185669</td>
      <td>-0.133789</td>
      <td>0.548227</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>0.800349</td>
      <td>1.231823</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>0.927120</td>
      <td>-0.951673</td>
      <td>1.023874</td>
      <td>0.315804</td>
      <td>0.360672</td>
      <td>-0.943983</td>
      <td>0.849493</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>-1.603837</td>
      <td>0.137472</td>
      <td>12.247699</td>
      <td>-0.092777</td>
      <td>1.934398</td>
      <td>2.409470</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-0.916296</td>
      <td>-0.886401</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.120797</td>
      <td>0.159469</td>
      <td>-0.427190</td>
      <td>-0.811344</td>
      <td>1.257846</td>
      <td>-0.355892</td>
      <td>0.003303</td>
      <td>0.512947</td>
      <td>0.433256</td>
      <td>-0.867410</td>
      <td>-0.133789</td>
      <td>-0.378408</td>
      <td>-0.819275</td>
      <td>4.040898</td>
      <td>0.800349</td>
      <td>-0.758781</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>-0.314155</td>
      <td>0.610487</td>
      <td>-0.101720</td>
      <td>0.315804</td>
      <td>-0.054591</td>
      <td>1.253310</td>
      <td>-1.070556</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>-0.491667</td>
      <td>-0.615009</td>
      <td>12.109016</td>
      <td>0.078793</td>
      <td>-0.267722</td>
      <td>1.622132</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-0.774477</td>
      <td>-0.513288</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.429834</td>
      <td>0.987524</td>
      <td>0.832191</td>
      <td>1.134633</td>
      <td>0.113946</td>
      <td>-0.355892</td>
      <td>0.231214</td>
      <td>-0.317122</td>
      <td>-0.575842</td>
      <td>1.189908</td>
      <td>-0.133789</td>
      <td>0.680880</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>0.800349</td>
      <td>1.231823</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>-0.314155</td>
      <td>0.610487</td>
      <td>0.940496</td>
      <td>0.315804</td>
      <td>0.643806</td>
      <td>-0.943983</td>
      <td>0.679251</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>0.991227</td>
      <td>0.137472</td>
      <td>12.317171</td>
      <td>0.296944</td>
      <td>1.819716</td>
      <td>2.409470</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-0.850052</td>
      <td>-0.803024</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.110623</td>
      <td>-1.860986</td>
      <td>-0.717817</td>
      <td>-0.811344</td>
      <td>-0.513805</td>
      <td>-0.355892</td>
      <td>0.348751</td>
      <td>-0.715166</td>
      <td>-0.436663</td>
      <td>1.148647</td>
      <td>-0.133789</td>
      <td>0.560689</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>-1.026153</td>
      <td>-0.758781</td>
      <td>0.165909</td>
      <td>-0.207905</td>
      <td>0.306482</td>
      <td>0.610487</td>
      <td>0.815430</td>
      <td>1.656362</td>
      <td>0.804249</td>
      <td>-0.943983</td>
      <td>0.596590</td>
      <td>2.840004</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>-1.603837</td>
      <td>-1.367490</td>
      <td>11.849405</td>
      <td>-0.003182</td>
      <td>-2.578803</td>
      <td>-0.412333</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.493497</td>
      <td>-2.182920</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.891805</td>
      <td>0.954402</td>
      <td>0.735316</td>
      <td>1.427666</td>
      <td>0.506871</td>
      <td>-0.355892</td>
      <td>0.296484</td>
      <td>0.228976</td>
      <td>0.122612</td>
      <td>1.249303</td>
      <td>-0.133789</td>
      <td>1.314119</td>
      <td>1.114055</td>
      <td>-0.241689</td>
      <td>0.800349</td>
      <td>1.231823</td>
      <td>1.392121</td>
      <td>-0.207905</td>
      <td>1.547757</td>
      <td>0.610487</td>
      <td>0.898808</td>
      <td>1.656362</td>
      <td>1.719716</td>
      <td>1.084573</td>
      <td>0.996280</td>
      <td>-0.404890</td>
      <td>-0.128701</td>
      <td>-0.293206</td>
      <td>-0.058688</td>
      <td>-0.190752</td>
      <td>2.103397</td>
      <td>0.137472</td>
      <td>12.429220</td>
      <td>1.600891</td>
      <td>1.689718</td>
      <td>2.409470</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-0.816930</td>
      <td>-0.761335</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


## Findings in the Ames, Iowa Housing Market

### What are the main house price ranges?

To determine the main house price ranges in the Ames, Iowa dataset, I used three clustering approaches to analyze the SalePrice values. Clustering helps to group similar house prices, and these methods can provide different insights:

- **K-means Clustering + Elbow Optimization**: This method uses K-means clustering to group houses based on their prices. The Elbow Optimization technique helps determine the optimal number of clusters by finding where adding more doesn't significantly improve the fit.
- **K-means Clustering + Silhouette Optimization**: Similar to the above method, K-means clustering is used. However, it leverages Silhouette Optimization to measure how similar each house is to its own cluster compared to others. This helps identify the ideal number of clusters for the best grouping.
- **Gaussian Mixture Model + Bayesian Information Criterion (BIC) Score**: This method uses a Gaussian Mixture Model (GMM) to group house prices. GMM allows clusters to take various shapes, unlike K-means which assumes spherical clusters. The BIC score helps to select the best model by balancing the fit and complexity, ensuring the most appropriate number of clusters.



```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

clustering_data = pd.DataFrame(final_study_data)
sale_prices = clustering_data[['SalePrice']].values

# Before clustering, we need to scale the SalePrices
standard_scaler = StandardScaler()
scaled_sale_prices = standard_scaler.fit_transform(sale_prices)
```

#### K-means Clustering + Elbow Optimization of Ames Housing Prices


```python
inertias = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(scaled_sale_prices)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Based on the elbow curve, let's choose an appropriate number of clusters
# For this example, let's say we choose 4 clusters
n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
kmeans.fit(scaled_sale_prices)

# Add cluster labels to the original dataframe
clustering_data = pd.DataFrame(final_study_data)
clustering_data['Cluster'] = kmeans.labels_

# Calculate the mean price for each cluster
cluster_means = clustering_data.groupby('Cluster')['SalePrice'].mean().sort_values()

for cluster, mean_price in cluster_means.items():
    print(f"Cluster {cluster}: Mean Price = ${expm1(mean_price):.2f}")

# Visualize the clusters
plt.figure(figsize=(12, 6))
for i in range(n_clusters):
    cluster_data = clustering_data[clustering_data['Cluster'] == i]
    plt.scatter(cluster_data.index, cluster_data['SalePrice'], label=f'Cluster {i}', color=colorblind_palette[i])

plt.xlabel('House Index')
plt.ylabel('Sale Price')
plt.title('K-means Clustering of Ames Housing Prices')
plt.legend()
plt.show()
```



![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_41_0.png)



    Cluster 2: Mean Price = $83958.98
    Cluster 1: Mean Price = $133851.67
    Cluster 0: Mean Price = $194839.43
    Cluster 3: Mean Price = $312264.64




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_41_2.png)



#### K-means Clustering + Silhouette Optimization of Ames Housing Prices


```python
from sklearn.metrics import silhouette_score

# Calculate silhouette scores for different numbers of clusters
silhouette_scores = []
K = range(2, 11)  # Start from 2 clusters as silhouette score is not defined for 1 cluster

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(scaled_sale_prices)
    score = silhouette_score(scaled_sale_prices, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method for Optimal k')
plt.show()

# Find the optimal number of clusters (highest silhouette score)
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of clusters: {optimal_k}")

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=seed)
kmeans.fit(scaled_sale_prices)

# Add cluster labels to the original dataframe
clustering_data = pd.DataFrame(final_study_data)
clustering_data['Cluster'] = kmeans.labels_

# Calculate the mean price for each cluster
cluster_means = clustering_data.groupby('Cluster')['SalePrice'].mean().sort_values()

# Print the mean prices for each cluster
for cluster, mean_price in cluster_means.items():
    print(f"Cluster {cluster}: Mean Price = ${expm1(mean_price):.2f}")

# Visualize the clusters
plt.figure(figsize=(12, 6))
for i in range(optimal_k):
    cluster_data = clustering_data[clustering_data['Cluster'] == i]
    plt.scatter(cluster_data.index, cluster_data['SalePrice'], label=f'Cluster {i}', color=colorblind_palette[i])

plt.xlabel('House Index')
plt.ylabel('Sale Price')
plt.title(f'K-means Clustering of Ames Housing Prices (k={optimal_k})')
plt.legend()
plt.show()
```



![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_43_0.png)



    Optimal number of clusters: 2
    Cluster 1: Mean Price = $125567.91
    Cluster 0: Mean Price = $233574.83




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_43_2.png)



## Gaussian Mixture Model + BIC Score of Ames Housing Prices


```python
from sklearn.mixture import GaussianMixture

# Calculate BIC and silhouette scores for different numbers of components
n_components_range = range(2, 11)
bic = []
silhouette_scores = []

for n_components in n_components_range:
    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    gmm.fit(scaled_sale_prices)

    # Calculate BIC score
    bic.append(gmm.bic(scaled_sale_prices))

# Plot the BIC scores
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic, 'bo-')
plt.xlabel('Number of components')
plt.ylabel('BIC')
plt.title('BIC Score vs. Number of GMM Components')
plt.show()

# Find the optimal number of components (lowest BIC score)
optimal_n_components = n_components_range[bic.index(min(bic))]
print(f"The optimal number of components based on BIC: {optimal_n_components}\n")

# Fit the final GMM model with the optimal number of components
gmm = GaussianMixture(n_components=optimal_n_components, random_state=seed)
gmm.fit(scaled_sale_prices)

# Add cluster labels to the original dataframe
gmm_clustering_data = pd.DataFrame(final_study_data)
gmm_clustering_data['Cluster'] = gmm.predict(scaled_sale_prices)

# Calculate the mean price for each cluster
cluster_sale_price = gmm_clustering_data.groupby('Cluster')['SalePrice']
cluster_means = cluster_sale_price.mean().sort_values()
cluster_mins = cluster_sale_price.min()
cluster_maxs = cluster_sale_price.max()

# Visualize the clusters
plt.figure(figsize=(12, 6))
for i in range(optimal_n_components):
    cluster_data = gmm_clustering_data[gmm_clustering_data['Cluster'] == i]
    plt.scatter(cluster_data.index, cluster_data['SalePrice'], label=f'Cluster {i}', color=colorblind_palette[i])

plt.xlabel('House Index')
plt.ylabel('Sale Price')
plt.title(f'Gaussian Mixture Model of Ames Housing Prices (n_components={optimal_n_components})')
plt.legend()
plt.show()
```



![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_45_0.png)



    The optimal number of components based on BIC: 3





![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_45_2.png)



### SalePrice Clustering Results

Using three different clustering methods to analyze the SalePrice values in the Ames, Iowa dataset, I identified various clusters:

- K-means Clustering + Elbow Optimization: Grouped houses into 4 clusters.
- K-means Clustering + Silhouette Optimization: Grouped houses into 2 clusters.
- Gaussian Mixture Model + Bayesian Information Criterion (BIC) Score: Grouped houses into 3 clusters.

To simplify the analysis and provide a clearer view of the housing market, I chose to use the results from the GMM+BIC method, defining three distinct housing segments based on their sale price ranges:

- **Luxury Homes Segment**: These are the high-end, more expensive homes.
- **Mid-Range Homes Segment**: These average-priced homes fall in the middle range.
- **Budget Homes Segment**: These are the more affordable, lower-priced homes.


```python
#
# Define cluster labels to present the business conclussion
#
# cluster 1: Mean Price = $89484.06 [$35K - $111K]
# cluster 0: Mean Price = $155502.32 [$112K - $218K]
# cluster 2: Mean Price = $282697.23 [$219K - $625K]
#
cluster_labels = [
    "Mid-Range Homes Segment",
    "Budget Homes Segment",
    "Luxury Homes Segment",
]

# Print the mean prices for each cluster
for cluster, mean_price in cluster_means.items():
    print(f"{cluster_labels[cluster]} (cluster {cluster}): Mean Price = ${expm1(mean_price)/1000:.0f}K [${expm1(cluster_mins[cluster])/1000:.0f}K - ${expm1(cluster_maxs[cluster])/1000:.0f}K]")

```

    Budget Homes Segment (cluster 1): Mean Price = $89K [$35K - $111K]
    Mid-Range Homes Segment (cluster 0): Mean Price = $156K [$112K - $218K]
    Luxury Homes Segment (cluster 2): Mean Price = $283K [$219K - $625K]


#### Main House Price Ranges in Ames, Iowa

This classification helps understand the housing market by segmenting it into easily understandable price ranges.

Segment | Mean Price | Minimum Price | Maximum Price
---------|----------|---------|---------
 Luxury Homes | $283K | $219K | $625K
 Mid-Range Homes | $156K | $112K | $218K
 Budget Homes | $89K | $35K | $111K

## Which Areas Can You Locate These Price Ranges?

Based on the segmentation of budget, mid-range, and luxury homes, I can now identify the top neighborhoods where these houses are located. This helps us understand where different types of homes are most commonly found in the Ames, Iowa, housing market.


```python
# Analyze the distribution of clusters across neighborhoods
neighborhood_clusters = gmm_clustering_data.groupby('Neighborhood')['Cluster'].value_counts(normalize=True).unstack()

# Fill NaN values with 0
neighborhood_clusters = neighborhood_clusters.fillna(0)

# Sort neighborhoods by the most prevalent cluster
dominant_cluster = neighborhood_clusters.idxmax(axis=1)
neighborhood_clusters['Dominant_Cluster'] = dominant_cluster
neighborhood_clusters = neighborhood_clusters.sort_values(by=['Dominant_Cluster'] + list(neighborhood_clusters.columns[:-1]))

# Identify top neighborhoods for each cluster
top_neighborhoods = {}
for cluster in range(3):
    top_neighborhoods[cluster] = neighborhood_clusters[cluster].nlargest(5)

for cluster, neighborhoods in top_neighborhoods.items():
    print(f"\nTop 5 neighborhoods for {cluster_labels[cluster]}:")
    for neighborhood, proportion in neighborhoods.items():
        print(f"  {neighborhood}: {proportion:.2f}")

# Plot the distribution of clusters across neighborhoods
plt.figure(figsize=(15, 10))
ax = sns.heatmap(neighborhood_clusters.iloc[:, :-1])
ax.set_xticklabels(cluster_labels)
plt.title('Distribution of Segments Across Neighborhoods')
plt.ylabel('Neighborhood')
plt.xlabel('Segment')
plt.tight_layout()
plt.show()

```


    Top 5 neighborhoods for Mid-Range Homes Segment:
      Blueste: 1.00
      NPkVill: 1.00
      Mitchel: 0.90
      Sawyer: 0.89
      SWISU: 0.88

    Top 5 neighborhoods for Budget Homes Segment:
      MeadowV: 0.71
      IDOTRR: 0.62
      BrDale: 0.56
      OldTown: 0.39
      Edwards: 0.38

    Top 5 neighborhoods for Luxury Homes Segment:
      NoRidge: 0.97
      NridgHt: 0.82
      StoneBr: 0.72
      Timber: 0.58
      Somerst: 0.52




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_50_1.png)



These are the top 5 neighborhoods for each segment:

- **Luxury Homes Segment**: These neighborhoods are known for their expensive houses.
    - NoRidge
    - NridgHt
    - StoneBr
    - Timber
    - Somerst
- **Mid-Range Homes Segment**: These neighborhoods feature homes with average prices that fall in the middle range.
    - Blueste
    - NPkVill
    - Mitchel
    - Sawyer
    - SWISU
- **Budget Homes Segment**: These neighborhoods are characterized by more affordable, lower-priced houses.
    - MeadowV
    - IDOTRR
    - BrDale
    - OldTown
    - Edwards

## What features best predict the price range of each home?

To determine which features most accurately predict the price range of each home, I analyzed the overall housing market and the previously identified segments: budget, mid-range, and luxury homes. This approach allows me to identify the main factors influencing the entire market and the specific segments.

I used the Random Forest Regressor, AdaBoost Regressor, and a Decision Tree Regressor to pinpoint the key features. For this analysis, I used the default settings of these regressors, but further studies could explore other models and fine-tune the parameters for even better results.


```python
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Separate features and target
X = encoded_study_data.drop('SalePrice', axis=1)
y = encoded_study_data['SalePrice']

def regresor_fit_and_print(title, X, y, regressor):
    """
    Fit and print a regressor results
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    regressor.fit(X_train, y_train)

    importances = regressor.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.values[:5], y=feature_importances.index[:5])
    plt.title(title + ' - Top 5 Most Important Features for Predicting House Prices')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()

    print(title + " - Top 5 most important features:")
    for i, (feature, importance) in enumerate(feature_importances[:5].items(), 1):
        print(f"{i}. {feature}: {importance:.4f}")

random_forest_regressor = RandomForestRegressor(random_state=seed)
regresor_fit_and_print("Random Forest Regressor", X, y, random_forest_regressor)

ada_boost_regressor = AdaBoostRegressor(random_state=seed)
regresor_fit_and_print("AdaBoost Regressor", X, y, ada_boost_regressor)

decision_tree_regressor = DecisionTreeRegressor(random_state=seed)
regresor_fit_and_print("Decision Tree Regressor", X, y, decision_tree_regressor)
```



![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_54_0.png)



    Random Forest Regressor - Top 5 most important features:
    1. TotalSF: 0.5387
    2. ExterQual_TA: 0.0587
    3. YrBltAndRemod: 0.0585
    4. GrLivArea: 0.0321
    5. Total_Bathrooms: 0.0320




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_54_2.png)



    AdaBoost Regressor - Top 5 most important features:
    1. TotalSF: 0.3212
    2. GrLivArea: 0.2062
    3. Total_Bathrooms: 0.0593
    4. GarageCars: 0.0524
    5. Fireplaces: 0.0379




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_54_4.png)



    Decision Tree Regressor - Top 5 most important features:
    1. TotalSF: 0.5877
    2. YrBltAndRemod: 0.0880
    3. ExterQual_TA: 0.0751
    4. GrLivArea: 0.0227
    5. Total_Bathrooms: 0.0205


According to the results from these regressors, here are the top 7 features that drive housing prices in the general market:

- Random Forest Regressor:
    - TotalSF: Total square footage of the house.
    - ExterQual_TA: Quality of the exterior material (Average/Typical).
    - YrBltAndRemod: Combined years since the house was built and any renovations.
    - GrLivArea: Above-ground living area.
    - Total_Bathrooms: Total number of bathrooms in the house.
    - YearBuilt: The year the house was built.
    - GarageArea: Size of the garage.
- AdaBoost Regressor:
    - TotalSF: Total square footage of the house.
    - GrLivArea: Above-ground living area.
    - Total_Bathrooms: Total number of bathrooms in the house.
    - GarageCars: Number of cars the garage can hold.
    - Fireplaces: Number of fireplaces in the house.
    - GarageYrBlt: The year the garage was built.
    - GarageArea: Size of the garage.
- Decision Tree Regressor:
    - TotalSF: Total square footage of the house.
    - YrBltAndRemod: Combined years since the house was built and any renovations.
    - ExterQual_TA: Quality of the exterior material.
    - GrLivArea: Above-ground living area.
    - Total_Bathrooms: Total number of bathrooms in the house.
    - KitchenAbvGr: Number of kitchens above ground.
    - BsmtFinSF1: Finished square footage of the basement.

### Conclusions

Based on the results from these regressors, I conclude that several key factors strongly influence the sale price of homes:

- **House Surface**: Total square footage and above-ground living area are crucial in determining home value.
- **Garage Characteristics**: The size of the garage, the number of cars it can hold, and the year it was built are significant predictors of price.
- **House Age**: The year the house was built and the combined years since any renovations are influential factors.
- **Total Bathrooms**: The total number of bathrooms in the house also significantly predicts price.

Understanding these features helps to focus on what truly matters when evaluating home prices.

### Sale Price Predictors for Budget, Mid-Range, and Luxury Homes

I repeated the analysis for each category to understand the different factors influencing the sale price of homes in the budget, mid-range, and luxury segments. This advanced study reveals the various considerations that buyers might prioritize within each segment.


```python
for i in range(optimal_n_components):
    cluster_data = gmm_clustering_data[gmm_clustering_data['Cluster'] == i]
    final_cluster_data = one_hot_encode(cluster_data)
    X_cluster = final_cluster_data.drop(columns=['SalePrice'])
    y_cluster = final_cluster_data['SalePrice']

    random_forest_regressor = RandomForestRegressor(random_state=seed)
    regresor_fit_and_print(f"{cluster_labels[i]}: Random Forest Regressor", X_cluster, y_cluster, random_forest_regressor)

    ada_boost_regressor = AdaBoostRegressor(random_state=seed)
    regresor_fit_and_print(f"{cluster_labels[i]}: AdaBoost Regressor", X_cluster, y_cluster, ada_boost_regressor)

    decision_tree_regressor = DecisionTreeRegressor(random_state=seed)
    regresor_fit_and_print(f"{cluster_labels[i]}: Decision Tree Regressor", X_cluster, y_cluster, decision_tree_regressor)
```



![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_57_0.png)



    Mid-Range Homes Segment: Random Forest Regressor - Top 5 most important features:
    1. FullBath: 0.1648
    2. TotalSF: 0.1285
    3. Total_Bathrooms: 0.0758
    4. YearBuilt: 0.0640
    5. YrBltAndRemod: 0.0631




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_57_2.png)



    Mid-Range Homes Segment: AdaBoost Regressor - Top 5 most important features:
    1. TotalSF: 0.1554
    2. Total_Bathrooms: 0.0618
    3. GrLivArea: 0.0548
    4. LotArea: 0.0467
    5. YearBuilt: 0.0412




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_57_4.png)



    Mid-Range Homes Segment: Decision Tree Regressor - Top 5 most important features:
    1. FullBath: 0.3068
    2. TotalSF: 0.1332
    3. YrBltAndRemod: 0.1247
    4. GarageType_Detchd: 0.0299
    5. GrLivArea: 0.0290




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_57_6.png)



    Budget Homes Segment: Random Forest Regressor - Top 5 most important features:
    1. 1stFlrSF: 0.0982
    2. TotalSF: 0.0924
    3. GrLivArea: 0.0788
    4. ExterCond_Fa: 0.0767
    5. OpenPorchSF: 0.0622




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_57_8.png)



    Budget Homes Segment: AdaBoost Regressor - Top 5 most important features:
    1. 1stFlrSF: 0.1304
    2. GrLivArea: 0.0897
    3. MSZoning_C (all): 0.0724
    4. GarageArea: 0.0612
    5. OpenPorchSF: 0.0532




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_57_10.png)



    Budget Homes Segment: Decision Tree Regressor - Top 5 most important features:
    1. TotalSF: 0.2474
    2. OpenPorchSF: 0.1096
    3. ExterCond_Fa: 0.1031
    4. Heating_Grav: 0.0772
    5. GrLivArea: 0.0728




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_57_12.png)



    Luxury Homes Segment: Random Forest Regressor - Top 5 most important features:
    1. TotalSF: 0.3759
    2. BsmtQual_Ex: 0.0930
    3. GrLivArea: 0.0551
    4. BsmtFinSF1: 0.0438
    5. TotalBsmtSF: 0.0403




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_57_14.png)



    Luxury Homes Segment: AdaBoost Regressor - Top 5 most important features:
    1. TotalSF: 0.4139
    2. TotRmsAbvGrd: 0.0821
    3. GarageCars: 0.0514
    4. BsmtQual_Ex: 0.0488
    5. BsmtFinSF1: 0.0467




![png](answering-house-prices-questions-based-on-advanced-regression-techniques-files/answering-house-prices-questions-based-on-advanced-regression-techniques_57_16.png)



    Luxury Homes Segment: Decision Tree Regressor - Top 5 most important features:
    1. TotalSF: 0.4106
    2. BsmtQual_Ex: 0.1165
    3. YearBuilt: 0.1103
    4. GrLivArea: 0.0718
    5. BsmtFinSF1: 0.0489


### Conclusions

From the top 5 most essential features for budget, mid-range, and luxury homes, we notice some variables previously identified for the general market. By excluding these general commonalities, we can delve deeper and identify patterns that specifically increase the sale price within each segment. Here are my findings:

- **Luxury Homes Segment**
    - **Livable Basement Finished Surface** (BsmtFinSF1 and TotalBsmtSF) and the height of the basement (BsmtQual). Excellent quality basements (100+ inches) add significant value.
    - **Total Rooms Above Grade**: More rooms (excluding bathrooms) increase the home's value.
- **Mid-Range Homes Segment**
    - **Full Bathrooms Above Grade** (FullBath): More full bathrooms on the main floors significantly impact the sale price.
    - **Detached Garage Location** (GarageType Detchd): Detached garages are valuable in this segment.
- **Budget Homes Segment**:
    - **Open Porch Area** (OpenPorchSF): Larger open porch areas contribute to higher prices.
    - **First Floor Area** (1stFlrSF): A more extensive first-floor area adds value.
    - **Exterior Material Condition** (ExterCond): The condition of the exterior material, mainly if it's excellent or fair, impacts the sale price.

Understanding these specific drivers within each segment helps us better predict and evaluate home prices based on various characteristics and amenities. Further studies could refine these insights and explore additional features influencing housing prices.

### Parting Thoughts

What started as a simple assignment for the Udemy Data Scientist Nanodegree Program was an incredible learning journey. I applied various data science techniques, including:

- Setting clear objectives
- Exploring and visualizing data
- Identifying and removing outliers
- Addressing skewed distributions
- Handling missing data
- Normalizing numerical features
- Conducting bivariate analysis
- Extracting insights from the Ames, Iowa housing market

This experience deepened my understanding of the data science workflow and the CRISP-DM process. It answered the initial questions and sparked my curiosity about further exploration in the field.

## LICENSE

Attribution 4.0 International - CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/anibalsanchez/answering-house-prices-questions-based-on-advanced-regression-techniques">Answering House Prices Questions using Advanced Regression Techniques</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://www.linkedin.com/in/anibalsanchez/">Anibal H. Sanchez Perez</a> is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Creative Commons Attribution 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""></a></p>
