# Wisdom of the Fans: Determining the distribution of incorrect officiating based on fan commentary

Sports fans have a seemingly insatiable desire for statistics and talking points. Sports fans also complain. A lot. Especially about officiating.

This project uses fan comments about officiating to measure which teams are affected by bad (incorrect) calls the most. Statistical NLP is used to  categorize comments based on their subjects and timestamps. Then, the sentiments of these comments (i.e., claiming a call was bad vs. merely commenting on the call and then if later comments agree or disagree that there was a bad call) are determined. The approach depends on the “wisdom of the crowds”, which is the idea that the truth is close to the mean of a large number of biased opinions.

These comments will initially be collected from Reddit, which has a voting system and a “flair” system that will aid in determining the biases of the users and the number of users in the thread. From these data, I will determine whether a call was actually bad and extrapolate other interesting phenomena, such as which fan bases complain the most and/or don’t know the rules as well (this is an entertainment product after all, and statistics that allow fans to “trash talk” their rivals are very popular).

As an aside, I think officials and their review processes are very good. I do not believe that there are intrinsic biases against any teams. However, statistically, some teams should benefit from bad calls and other teams should be harmed by bad calls simply because officials are human and make mistakes.

***
## Data



![Sorry](../ref_analysis/writeup/sorry2_ore.png?raw=true)

![Sample image](writeup/sample.png?raw=true)



## Methods




 Metric      | Score
 ---         | ---
 Precision   | 0.833         
 Recall      | 0.974         
 Accuracy    | 0.941
