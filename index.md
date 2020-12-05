# Various Player Predictions Using FIFA 19 Game Data

## Introduction

For this project, our team has chosen to focus on using FIFA video game player data to make various predictions such as what position they are most likely to play, what a players salary is, what their in game rating is, and what country they are from. All of these predictions are made using the in-game stats such as what their dribbling, shooting, or defending levels are, what their position chemistry would be, and many other factors. Different machine learning methods are used such as clustering, (**insert whatever other things we use here**). We will begin by going over how we had to clean our data to make it usable. Following that, each feature we will be predicting will receive its own section going over the methods and algorithms used for predictions and a conclusion. Finally, an overall closing conclusion will be made on the effectiveness of these features we used in relation to making predictions.

## Motivation

FIFA video games contain a plethora of data within them not just about the matches being played and the users playing the game, but on each and every soccer player at the professional level. All of this data fascinated us, and there were so many possibilities with it. Of course, video game data does not match up well with real life scenarios, but perhaps there is a way to make some sort of close predictions using the numbers that Electronic Arts (the game developer) have manufactured for the game. Will these numbers relating to skill and chemistry hold any weight when it comes to making predictions, or are they totally arbitrary and only useful for balancing gameplay? As sports and video game lovers, we knew we had to find this out as it was something we found exciting and needed to know the answer to.

## Dataset

Our data came from [Kaggle](https://www.kaggle.com/karangadiya/fifa19) which we found while browsing cool and intersting data we could work with. We chose this set because there were many features for us to use and most of it was already in a nice to use numerical form.

### Cleaning Data

In order to get our data into a usable form, we had to do a few things including removing replacing some NaN values with zeroes, removing the other NaN filled rows, and converting strings that were meant to be used as numbers into usable integers or floats.

Initially, the goal was to get rid of any players who have NaN values populating a majority of their columns as they won't be of much use to us when making predictions. An issue with that is that if a player is a goalkeeper (GK), then every single column with a position chemistry was NaN. It didn't seem right to just throw out the goalkeepers, so we began by replacing those specific NaN values with 0 to indicate they had no chemistry in other positions. To do that, we only dropped rows that had NaN in two of the columns, Club and Position. When we did this, we were left with only NaN values relating to goalkeeper missing values.

```
clean_fifa_df = clean_fifa_df.dropna(subset=['Club', 'Position'])
```

Once the bad rows were taken out, we had to replace all of the NaN values with 0 which was a simple fix. We reached a new issue when looking at the values populating those position columns. Each one has a number, but it is paired with a '+' and another value. To keep things simple, we kept only the base value that appears before the plus sign. Once this was accomplished, we could then convert the columns to an 'int' datatype.

```
positions = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
for pos in positions:
  clean_fifa_df[pos] = clean_fifa_df[pos].str.split('+').str[0]
  clean_fifa_df[pos] = clean_fifa_df[pos].astype(int)
clean_fifa_df.head()
```
Now there were no NaN values and the positions were all integers, but there were other columns such as height, weight, value, and wages that were all in string format when they needed to be something numeric. This required the removal of the euro symbol for any monetary columns, removing any letters, converting height to inches, and correctly parsing wages and value into numbers based on whether it was in thousands (K) or millions (M).

```
# This parses the height into inches
def parse_ht(ht):
    ht_ = ht.split("'")
    ft_ = float(ht_[0])
    in_ = float(ht_[1].replace("\"",""))
    return (12*ft_) + in_

clean_fifa_df['Height'] = clean_fifa_df['Height'].apply(lambda x:parse_ht(x))
clean_fifa_df['Weight'] = clean_fifa_df['Weight'].str.split('l').str[0]

# Removing the euro symbol
clean_fifa_df['Value'] = clean_fifa_df['Value'].str.replace('€', '')
clean_fifa_df['Wage'] = clean_fifa_df['Wage'].str.replace('€', '')

# Turning value and wages into usable numbers
clean_fifa_df['Value'] = (clean_fifa_df['Value'].replace(r'[KM]+$', '', regex=True).astype(float) * clean_fifa_df['Value'].str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))
clean_fifa_df['Wage'] = (clean_fifa_df['Wage'].replace(r'[KM]+$', '', regex=True).astype(float) * clean_fifa_df['Wage'].str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(int))

# Turning everything into ints
clean_fifa_df['Value'] = clean_fifa_df['Value'].astype(int)
clean_fifa_df['Wage'] = clean_fifa_df['Wage'].astype(int)
clean_fifa_df['Weight'] = clean_fifa_df['Weight'].astype(int)
```

One final bit of cleaning was done by converting the club, nationality, position, and preferred foot columns to something along the lines of one hot encoding. Each unique value corresponded with a numerical value and the string was replaced with this number. This way, we could still include the string based columns when making our predictions since they are very likely to make a difference.

```
# 27 different positions seen with y.unique(). Also need to make things numeric for the clustering
y = fifa_df['Position']

class_mapping = {label:idx for idx,label in 
                 enumerate(np.unique(y))}
fifa_df['Position'] = fifa_df['Position'].map(class_mapping)

y = fifa_df['Position']

X = fifa_df.drop(['Position'], axis=1)

class_mapping = {label:idx for idx,label in 
                 enumerate(np.unique(X['Nationality']))}
X['Nationality'] = X['Nationality'].map(class_mapping)

class_mapping = {label:idx for idx,label in 
                 enumerate(np.unique(X['Club']))}
X['Club'] = X['Club'].map(class_mapping)

class_mapping = {label:idx for idx,label in 
                 enumerate(np.unique(X['Preferred Foot']))}
X['Preferred Foot'] = X['Preferred Foot'].map(class_mapping)
```

### Exploring Data

Prior to totally removing any strings in the data, we explored it by looking at some fun graphs showing many of the max values relating to wages, values, and total player counts. Each graph shows the top 20 highest values.

First we checked out how many players there were in each country.

![country_count](images/image1.png)

Second, we saw how the average wages looked for each club.

![wage_count](images/image2.png)

Last, we wanted to know which club had the highest average value of players.

![value_count](images/image3.png)

Looking at this data shows that nationality and club play a big factor into someones success. If you play for Juventus, there's a much higher chance that your valuation is above the mean player in the game.





You can use the [editor on GitHub](https://github.com/noahplacke/BDS_FIFA/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
