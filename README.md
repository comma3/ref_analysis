# Wisdom of the Fans: Determining the distribution of incorrect officiating based on fan commentary

Sports fans have a seemingly insatiable desire for statistics and talking points. Sports fans also complain. A lot. Especially about officiating.

This project uses fan comments about officiating to measure which teams are affected by bad (incorrect) calls the most. Statistical NLP is used to  categorize comments based on their subjects and timestamps. Then, the sentiments of these comments (i.e., claiming a call was bad vs. merely commenting on the call and then if later comments agree or disagree that there was a bad call) are determined. The approach depends on the “wisdom of the crowds”, which is the idea that the truth is close to the mean of a large number of biased opinions.

These comments will initially be collected from Reddit, which has a voting system and a “flair” system that will aid in determining the biases of the users and the number of users in the thread. From these data, I will determine whether a call was actually bad and extrapolate other interesting phenomena, such as which fan bases complain the most and/or don’t know the rules as well (this is an entertainment product after all, and statistics that allow fans to “trash talk” their rivals are very popular).

As an aside, I think officials and their review processes are very good. I do not believe that there are intrinsic biases against any teams. However, statistically, some teams should benefit from bad calls and other teams should be harmed by bad calls simply because officials are human and make mistakes.

***
## Data

Example of a sorry from the team that had the advantage

![Sorry](/writeup/sorry2_ore.png)

Example of an excuse from a team that had the advantage

![Excuse](/writeup/excuse2_ore.png)

Example of a bailed out comment.

![Excuse](/writeup/bailed.png)


Data are modified by substituting "hometeamtrack" and "awayteamtrack" (sufficiently unlikely terms to occur naturally) for names and nicknames of various teams, allowing the model to identify these 'Sorry', 'Excuse', and 'Bailed' cases without consideration of the team name.

## Methods

Training data are hand labeled with ~30 potential classes (5 special and 25 for different rules). A One vs Rest approach is taken where one Naive Bayes model is trained for each class and the appropriate tags are added to each call.

The precision and

 Metric      | Score
 ---         | ---
 Precision   | 0.833         
 Recall      | 0.974         
 Accuracy    | 0.941

Comments are then clustered based on the time of the

 ## Results

 ![Sample image](/writeup/sample.png)
