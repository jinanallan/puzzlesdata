#Events data analysis

library(rjson)
library(tidyverse)

# Load MULTIPLE .json files, as separate lists (one for each .json file) ####
##update to pull from Dropbox!##
data_files <- list.files("/Users/jinanallan/Dropbox/pilot logs/P4", pattern="*.json")  # Identify file names
data_files  # Print file names
#iterate through to load the files 
i=1 
for(i in 1:length(data_files)) {                              # Head of for-loop
  assign(paste0("data", i),                                   # Read and store data frames
         fromJSON(file = paste0("/Users/jinanallan/Dropbox/pilot logs/P4/",
                                data_files[i])))
}


# EVENTS-LEVEL ANALYSIS #####
## So far, must select one puzzle file at a time (e.g., data10)
##integrate map and pluck, to select one tasks events / codes 
data_aslist <- map(data10$events, pluck, "description") #returns a list
dataframe_char <- map_chr(data10$events, pluck, "description") %>%   #selects just 1 puzzle events; returns a dataframe
  as_tibble() 
dataframe_char <- dplyr::rename(dataframe_char, events_desc = value) #renames the column/variable name (from 'value' to 'events_code')
dataframe_code <- map_dbl(data10$events, pluck, "code") %>%   #uses 'codes' instead of movement 'descriptions' 
  as_tibble() 

#get count of each type of event code.
table(dataframe_char$events_desc)
dataframe_char %>% count(events_desc) 

#remove 'moving started' actions; not meaningful
df2<-subset(dataframe_char, events_desc!="Moving started ")
df2<-subset(df2, events_desc!="Moving not possible")
df2 %>% count(events_desc)

#plot the number of meaningful actions (in one puzzle)
ggplot(df2, aes(x = events_desc)) +
  geom_bar()



#ORDER THE MEANINGFUL ACTIONS BY TIME. #####



