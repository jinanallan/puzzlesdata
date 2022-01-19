
import os, json
import pandas as pd
print('Reading data')

path_to_json = 'logs/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)

jsons_data = pd.DataFrame(columns=['end-status','end-time', 'start-time', 'solved', 'total-time'])

# we need both the json and an index number so use enumerate()
for index, js in enumerate(json_files):
  with open(os.path.join(path_to_json, js)) as json_file:
    json_text = json.load(json_file)
#    print(json_text)

    endstatus = json_text['end-status']
    starttime = json_text['start-time']
    endtime = json_text['end-time']
    totaltime = json_text['total-time']
    solved = json_text['solved']
    puzzlefile = json_text['puzzle-file']

    jsons_data.loc[index] = [endstatus, endtime, starttime, solved, totaltime]

print(jsons_data)

