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

![Bailed out](/writeup/bailed.png)


Data are modified by substituting "hometeamtrack" and "awayteamtrack" (sufficiently unlikely terms to occur naturally) for names and nicknames of various teams, allowing the model to identify these 'Sorry', 'Excuse', and 'Bailed' cases without consideration of the team name.

Comments are also lemmatized and a custom stop word list is used to reduce the feature space.

## Methods

Training data are hand labeled with ~30 potential classes (5 special and 25 for different rules). A One vs Rest approach is taken where one Naive Bayes model is trained for each class and the appropriate tags are added to each call.

The precision, recall and accuracy of the model are quite good, although I expect the unseen data to be
slightly worse because all of the training/test data have accurate team nickname entries, while the unseen
data likely miss some nicknames. The results below are macro accuracy: Each label is given equal weight,
so minority classes (i.e., rare calls) are still represented strongly and common calls (e.g., off topic) do not
overly contribute.

 Metric      | Score
 ---         | ---
 Precision   | 0.833         
 Recall      | 0.974         
 Accuracy    | 0.941

Comments are then clustered based on the time of the comment and a custom distance metric designed based on the type of call. Certain calls are related and therefore less distance is added based on these labels. As an example, false start and offsides are often discussed together and likely are referring to the same play.

Hierarchical and kmeans clustering are both available, with the clustering automatically selected based on Silhouette score.

![Sample image](/writeup/dendrogram.png)


Each cluster is then analyzed individual, and the bad call is determined using various heuristics to predict the correct call (notably the examples mentioned above). Accuracy is approximated by considering the class imbalance (higher difference between the two classes indicates greater certainly of the result).

 ## Results

**Placeholder** Image showing average call differential against "argumentative", which is a measure of how disagreeable a fan base is based on percentage of comments that are downvoted.

"Whininess" is also calculated, which is the rate that fans complain about calls that are deemed to not be bad

 ![Sample image](/writeup/sample.png)


Other interesting statistics arise from these results:

 Award                    | Team                  | Value
 ---                      | ---                   | ---
 Most bad calls against   | Colorado              | 104
 Most bad calls for       | Colorado State        |  96    
 Worst officiated game    | Utah vs UCLA (2016)   | 15
 etc..                    |                       |


 ## Future Directions

 #### Cunningham's Law:
 "The best way to get the right answer on the internet is not to ask a question; it's to post the wrong answer."

 The r/football community has over 350,000 members, many of whom are quite fanatical about college football and their teams. Hand labeling the training data is very, very time consuming. Therefore, I intend to create a web app and leverage this community to label some of the data for me.

 Additionally, this model will require substantial iteration to improve the heuristics involved in labeling data. Manual examination of the clusters is required to determine if the prediction is accurate.
