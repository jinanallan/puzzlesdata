#Puzzle (high-level) data analysis

library(rjson)
library(tidyverse)
library(ggplot2)

# LOAD just the 'high (puzzle) level data, for all puzzles and P's together ####
##update to pull from Dropbox!##
data_files <- list.files("/Users/jinanallan/Dropbox/pilot logs/P4", pattern="*.json", full.names=TRUE)
#data_files <- list.files("/home/jinan/Dropbox/SCIoI/logs/all data", pattern="*.json", full.names=TRUE)  #Ubuntu

# as a dataframe
high_level_data <- purrr::map_df(data_files, function(x) { 
  purrr::map(jsonlite::fromJSON(x), function(y) ifelse(is.null(y), NA, y)) 
})
#delete events data, because unnecessarily messy
high_level_data$events = NULL

#subset puzzles that are fully solved.
solved_puzzles<-subset(high_level_data, solved==TRUE)
#and which also provide a rating (second run through)
solved_with_rating<-subset(solved_puzzles, rating>=0)


# BASIC PUZZLE-LEVEL ANALYSES ####
high_level_data %>% count(solved)

#plot the number of correctly solved puzzles
ggplot(high_level_data, aes(x = solved)) +
  geom_bar()

#plot histogram of time taken on all puzzles (solved and unsolved)
#hist(high_level_data$`total-time`)
ggplot(high_level_data, aes(x=`total-time`)) + 
  geom_histogram(colour="black", fill="white") 

ggplot(high_level_data, aes(x=`total-time`)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white")+
  geom_density(alpha=.2, fill="#FF6666")

#plot histogram of time taken to solve puzzles
ggplot(solved_puzzles, aes(x=`time-solved`)) + 
  geom_histogram(colour="black", fill="white") 

ggplot(solved_puzzles, aes(x=`time-solved`)) + 
  geom_histogram(colour="black", fill="white") + 
  geom_vline(aes(xintercept=mean(`time-solved`)),
             color="blue", linetype="dashed", size=1)

#pull apart by run 1 and run 2
##for some reason the code to overlay the two histograms isn't working...
ggplot(solved_puzzles, aes(x=`time-solved`))+
  geom_histogram(color="black", fill="white")+
  facet_grid(run ~ .)


##calculate average time and standard deviation of time... 
##subset this by puzzle, once have enough participants. 



#subjective difficulty rating of each puzzle (subj IRT?)
ggplot(solved_with_rating, aes(x=rating))+
  geom_histogram(color="black", fill="white")

#solved/not solved dichotomous IRT for the full set of puzzles 
##will need to change format of data so P's as rows and each puzzle (solved/not) as a column with a column for each rating

