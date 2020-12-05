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

![image]()

### Exploring Data







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
