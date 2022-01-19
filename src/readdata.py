
import os, json
import pandas as pd
print('Reading data')

path_to_json = 'logs/'
json_files = [pos_json for pos_json in sorted(os.listdir(path_to_json)) if pos_json.endswith('.json')]
print(json_files)

jsons_data = pd.DataFrame(columns=['end-status','end-time', 'start-time', 'solved', 'total-time'])
cntSolved = 0
testDict = {}
testDict['prtcpID'] = 0
testDict['puzzles'] = {}

# we need both the json and an index number so use enumerate()
for index, js in enumerate(json_files):
  print(js)
  tokens = js.split('_')
  puzzleID = tokens[1]
  puzzleAttempts = []

  if( puzzleID in testDict['puzzles']):
    print('list already contains this puzzle, add a new attempt')
    puzzleAttempts = testDict['puzzles'][puzzleID]
  else:
    print('Creating a new list of puzzle attempts')
    
  print(tokens)
  with open(os.path.join(path_to_json, js)) as json_file:
    json_text = json.load(json_file)
    ###testDict['puzzles'][puzzleID].append(json_text)
#    print(json_text)

    endstatus = json_text['end-status']
    starttime = json_text['start-time']
    endtime = json_text['end-time']
    totaltime = json_text['total-time']
    solved = json_text['solved']
    puzzlefile = json_text['puzzle-file']

    puzzleAttempts.append(totaltime)
    testDict['puzzles'][puzzleID] = puzzleAttempts
#    testDict['puzzles'][puzzleID].append(totaltime)

    if solved == True:
      cntSolved = cntSolved + 1

    jsons_data.loc[index] = [endstatus, endtime, starttime, solved, totaltime]

print(jsons_data)
#print('Solved', cntSolved)
#print(testDict)

