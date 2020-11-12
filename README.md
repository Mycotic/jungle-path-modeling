# jungle-path-modeling

The goal of this project is to build a model that tells a player where they should go next based on what a professional player would do in their position. This tool could be useful for newer players trying to learn the game. Eventually it could be used to talk about player behavior when preparing for professional games.

## Installation

need to get an api key before running code from here:
https://developer.riotgames.com/
and save it in a text file api-key.txt in data

run order:
get-champion-ids
request-clid-match-history
pull-timelines (need to run request functs with a delay between them for now)
EDA
window-making-with-keras-guide-code

## About League of Legends and Junglers

League of Legends is a 5v5 "Multiplayer Online Battle Arena" or MOBA where players fight for resources across a map.

![resource locations](https://raw.githubusercontent.com/Mycotic/jungle-path-modeling/main/images/map%20camps.png)

Four of the five players on each team tend to compete for resources in lanes, but the fith player, the jungler, roams across the map collecting resources in many places while helping their team.

![player position map](https://raw.githubusercontent.com/Mycotic/jungle-path-modeling/main/images/player-pos.png)


## Data & Data preprocessing

I chose to work with games played by one player on one character to keep the model simple minimize variance from different playstyles. With more processing power and time different players and characters could be added as features to improve learning. I chose to train my model on Clid because Clid was arguably the best jungler in 2019 and is still among the best in 2020. The Riot API also doesn't support the Chinese ranked ladder outside of China (currently the strongest region professionally). As for the champion, I chose Lee Sin. Clid has an all time winrate of 60% on Lee Sin professionally over almost 100 games, and has a 56% winrate on the ranked ladder over more than 200 games. Even at the ranked ladder's highest level Lee Sin only has a 49% winrate despite being played in 20% of games. This indicates how good Clid is on Lee Sin and potential room for the playerbase as a whole. Additionally, Lee Sin is a fairly mobile jungler which means playing Lee Sin has more options for how to path.

I requested the timelines of Clid's 200 most recent Lee Sin games, which are given in json format. Match information is requested separately to identify which side of the map each player is assigned to as well as to identify which participantid is Clid's. I then created dataframes from each game and tracked the current gold, total gold, level, jungle camps taken, and the x and y position of each player at each minute. Current gold was chosen because it indicates how much stronger a player would be if they went back to base to shop (momentarily removing pressure from that part of the map). Total gold and level indicate the overall strength of a player. Jungle camps taken helps indicate what resources are currently available because they aren't directly indicated by the timeline. X and Y position indicate a lot of things, like what player has pressure or what resources are being taken by the enemy team etc. This adds up to 61 features, 6 for each player and 1 for which side of the map Clid's team is on. It wasn't easily feasible to make columns role specific, so other than Clid's stats columns are team specific.


Here are some boxplots of these stats for Clid vs a random player on his team based on all the games in the train set:
![current gold boxplot](https://raw.githubusercontent.com/Mycotic/jungle-path-modeling/main/images/current-gold-plot.png)
![total gold boxplot](https://raw.githubusercontent.com/Mycotic/jungle-path-modeling/main/images/total-gold-plot.png)
![level boxplot](https://raw.githubusercontent.com/Mycotic/jungle-path-modeling/main/images/level-plot.png)

I then split the games into a train, validation, and test set. I also trained only on the first 20 minutes (all if the game is shorter than 20min) of each game because gameplay in this period is more self-similar than the rest of the game (effectively improving stationarity). Each game was then chopped up into "windows" where each window has the feature values over a certain period of time and the value that the model is supposed to predict (in this case the x and y position at the next time step). Models that predicted based on fewer time steps therefore effectively had more data to train on because more windows could be selected from each game. For example, 20 minutes of a game with a 3 minute lookback would be split into 17 windows of 3 time steps with 61 features and 1 time step of 2 features (the predicted x and y). On the other hand, a 2 minute lookback would result in an feature set of shape (18,2,61) and a label set of shape (18,1,2). The windows of each of these can then be combined into their respective train, val, and test sets. For the test sets, models with smaller lookback used a slice of the test set of the model with the largest lookback so that the models were all tested on the same test set. For example, a model with lookback of 4 might have a label test set of shape (300,4,61), so even though a 2min lookback model could have a test set of shape (>300,2,61), it uses a slice of the 4min lookback model (300,2:,61).

## Loss Function and Metric

Mean Absolute Distance Error was used as my loss function and evaluation metric because using RMSE encouraged models to heavily predict the center of the map to minimize incorrect predictions. This is still an issue for MAE, and one of the things I experimented was using the log of MAE or RMSE, but I'm not confident enough on the metric and haven't done enough research on it to present those results yet. The tensorflow guide on time series did use RMSE as its log function despite using MAE as its evaluation, but I'm not sure if that's generally advisable or exactly what the value of doing that is, and when I tried it, it performed less well than just using the same loss function as metric.

## Models 

I implemented linear, dense, multistep dense, and convolutional models, and compared them against the baseline of predicting the center of the map as each label. The linear model basically behaves like a linear regression, assigning a different weight to each feature (I don't know exactly how linear regressions function with multiple targets but I assume it's fairly similar). The dense model adds a couple dense layers to the model, which basically lets the model create feature crosses and selects which ones are important based on the weights that perform best with based on the loss function. Multistep dense models flattens the set of timesteps into one column each and then performs a normal dense model based on the result set of features (a lookback of 2 would implement a dense model on 122 features - team side gets doubled). From what I've read the multistep dense model performs similarly to the convolutional model but can only work on windows of the right shape, whereas the convolutional model can make its predictions on larger samples, but still depends on a set lookback like multistep models do. However, my convolutional models performed consistently better despite having the same number of cells for each dense layer etc. Maybe these models are randomly performing better, or something about the way that the lookback kernel is applied is more efficient than flattening (maybe player columns from different time steps are being recognized as belonging to the same player?).

I used 30 epochs for each model because it performed better than the other values I tested, but I didn't experiment extensively. It's also possible that with a better metric using more epochs will perform better, or after collecting more data. I did not experiment any with using different numbers of cells per layer due to not having enough time, but I believe it shouldn't have too much if the ideal number of cells were lower than the number I used due to the algorithms method of killing cells (on that note I used relu as my activation function). Of the models I tested, only the multistep dense and convolutional models with a lookback of 2 performed better than predicting the center of the map, and both models' predictions were heavily biased toward the center of the map. Other models seemed to make some very good predictions, but it seems that the punishment for incorrect predictions still made them perform worse than predicting the center of the map.

| Model         | MAE  |
|---------------|------|
| Baseline      | 4186 |
| Dense         | 4369 |
| Convolution 2 | 4177 |
| Convolution 3 | 4224 |
| Convolution 4 | 4258 |
| Multi Dense 2 | 4178 |
| Multi Dense 3 | 4243 |
| Multi Dense 4 | 4292 |

Example predictions for convolutional model with lookback 2 (white numbers are true position of Clid during the given duration, orange numbers are the model's prediction of where Clid will be at each moment):

![30-2 conv model](https://raw.githubusercontent.com/Mycotic/jungle-path-modeling/main/images/30-2-conv.png)

For lookback 4:

![30-4 conv model](https://raw.githubusercontent.com/Mycotic/jungle-path-modeling/main/images/30-4-conv.png)

For dense model:

![30-1 dense model](https://raw.githubusercontent.com/Mycotic/jungle-path-modeling/main/images/30-1-dense.png)


## Conclusion and Potential Improvements

The convolutional model that predicts based on the two previous minutes performed best, but only barely performed better than the baseline model. The fact that the two minute model performed best is reasonably explained by the fact that most camps respawn after two minutes, so information from before then has less impact. The two minute models do also seem to predict the center a lot more than other models (I don't know why this is the case), which could also be the reason this model is performing best, given the error metric's natural punishment of predictions away from the center. Models with 30 epochs performed best compared to those with 15 and 60 (which certainly has room for more testing).

I've come up with two potential reasons models with 30 epochs are performing better than models with more epochs. The first is that, after a certain number of epochs, further epochs result in overfitting the validation set. This would explain why the models with more epochs aren't predicting the center as much, but are performing worse. One thing to check if this is occurring would be to see how the models are performing on the validation set. An alternative explanation is that these models are making better predictions but are being punished by a potentially flawed metric. Applying log to our loss function and metric could be a solution to this. Another potential metric would be to cluster locations somehow and turn the problem into a classification one.

One important thing I realized halfway into this project was that, unlike normal time series, time isn't a redundant feature, because the time isn't linearly increasing in the train set (it strictly would range from 0 to 20). This would let the model learn the timings of objectives and make predictions based on it. More generally, and redundant feature of a time series in a game has the potential to be significant in this combined time series dataset. I did actually make use of this by using the side indicator, but there are so many more potential features that could be added that are game specific, like which champion each player is playing as.

There's also a lot more information in the timelines given by the API, but many of them are observed separately from the time frames given to us - integrating them into the model would likely mean adding their events to the previous or next time step, which would be fairly complicated.

Finally, the model could be trained on more players and more champions, and then make predictions based on what champion a player is using. This more general model could also be with transfer learning to learn on a new player or champion.





