import math
import os, json
import pandas as pd

#returns unix time in microseconds from the timestamp
def unixTimeFromTimestamp(timestamp):
    dateTokens = timestamp.split('-')
    unixTime = dateTokens[0]
    return unixTime

def getDistance(end, start):
#    dist = (end[0] - start[0])*(end[0] - start[0]) + (end[1] - start[1])*(end[1]-start[1])

    dist = math.sqrt((end[0] - start[0])*(end[0] - start[0]) + (end[1] - start[1])*(end[1]-start[1]))
    return dist


path_to_json = 'logs/'
json_files_events = [pos_json for pos_json in sorted(os.listdir(path_to_json)) if pos_json.endswith('.json')]
json_files_frames = [pos_json for pos_json in sorted(os.listdir(path_to_json + "/frames/")) if pos_json.endswith('.json')]

#print(json_files_events)
#print(json_files_frames)

#get times of interactions
#get times of movement
#get duration of interaction
#get duration of movement
#get areas visited (todo)

jsons_data = pd.DataFrame(columns=['end-status','end-time', 'start-time', 'solved', 'total-time'])
cntSolved = 0
testDict = {}
testDict['prtcpID'] = 0
testDict['puzzles'] = {}
clusters = {}
clustersPick = {}


# we need both the json and an index number so use enumerate()
for index, js in enumerate(json_files_events):
  #print(js)
  tokens = js.split('_')
  timeSession = tokens[0]
  partID = tokens[1]
  runNr = tokens[2]
  puzzleID = tokens[3]
  attemptNr = tokens[4]
  puzzleAttempts = []

  puzzleName = partID + "_" + puzzleID
  if( puzzleName in testDict['puzzles']):
    print('list already contains this puzzle, add a new attempt')
    puzzleAttempts = testDict['puzzles'][puzzleName]
  else:
    print('Creating a new list of puzzle attempts')
    

  with open(os.path.join(path_to_json, js)) as json_file:
    json_text = json.load(json_file)    
    #print(json_text)

    endstatus = json_text['end-status']
    starttime = json_text['start-time']
    endtime = json_text['end-time']
    totaltime = json_text['total-time']
    solved = json_text['solved']
    puzzlefile = json_text['puzzle-file']
    events = json_text['events']

    puzzleAttempts.append(totaltime)
    testDict['puzzles'][puzzleName] = puzzleAttempts

    if solved == True:
      cntSolved = cntSolved + 1

    inters = [] #interactions or movements
    intersPick = []

    #parse the events
    lastMoveTime = 0
    lastPickTime = 0
    moveStep = 50000 #50ms
    moveDuration = 0
    interDuration = 0
    pickDuration = 0
    grasped = False
    moveStarted = 0
    moveEnded = 0
    pathLength = 0
    distance = 0
    moveStartEvent = events[0]
    objectGrasped = ""

    
    for ev in events:
        #print(ev)
        code = ev['code']
        desc = ev['description']
        #egoPos = ev['ego']
        evTime = ev['timestamp']
        timeUnix = int(unixTimeFromTimestamp(evTime))
        

        if code == 7: 
            timePassed = timeUnix - lastMoveTime
            #print(timePassed, timeUnix, lastMoveTime)
            if timePassed < moveStep:
                moveDuration = moveDuration + timePassed
                lastMoveTime = timeUnix
                moveEnded = timeUnix #update the last time every step
                moveStartEvent = ev

            else:
                                
                if moveStartEvent['code'] == 7:
                    distance = getDistance(ev['coordinates']['end'], moveStartEvent['coordinates']['ego']) 

                else:
                    distance = 0

                moveInteraction = {}
                moveInteraction['id'] = js
                moveInteraction['type'] = "move"                            
                if grasped :
                    moveInteraction['type'] = "carry"
                moveInteraction['duration'] = moveDuration
                moveInteraction['start'] = moveStarted
                moveInteraction['end'] = moveEnded                
                moveInteraction['dist'] = distance
                moveInteraction['obj'] = objectGrasped
                
                lastMoveTime = timeUnix
                moveStarted = timeUnix
                moveEnded = timeUnix
                distance = 0
                if moveDuration > 0:
                    inters.append(moveInteraction)
                moveDuration = 0

        elif code == 3:
            #attach
            lastPickTime = timeUnix
            pickDuration = 0
            grasped = True
            objectGrasped = ev['description'][7:]

        elif code == 4:
            #release
            pickDuration = timeUnix - lastPickTime
            pickInteraction = {}
            pickInteraction['type'] = "pnp"
            pickInteraction['id'] = js
            pickInteraction['duration'] = pickDuration
            pickInteraction['obj'] = ev["description"][8:] #the text after "Release "
            grasped = False
            if pickDuration > 0:
                intersPick.append(pickInteraction)
            pickDuration = 0
            objectGrasped = ""


    moveIn = {}
    moveIn['id'] = js
    moveIn['type'] = "move"        
    if grasped :
        moveIn['type'] = "carry"
 
    moveIn['duration'] = moveDuration
    moveIn['start'] = moveStarted
    moveIn['end'] = moveEnded
    moveIn['dist'] = distance
    moveIn['obj'] = objectGrasped

 
    if moveDuration > 0:
        inters.append(moveIn)
              
  moveNr = len(inters)
  if moveNr not in clusters:
      clusters[moveNr] = []
  clusters[moveNr].append(inters)
  
  pickNr = len(intersPick)
  if pickNr not in clustersPick:
      clustersPick[pickNr] = []
  clustersPick[pickNr].append(intersPick)


#print(testDict)
#print(inters)
print("Clusters by move number")

for clSize in clusters.keys():
    print("size", clSize, len(clusters[clSize]))
    for test in clusters[clSize]:
        durations = ""
        for t in test:
            durations += str(t['duration']/1000000.0) + " " + t['obj'] + " "

 
        print(test[0]['id'], durations)
        
print("Clusters by pick number")
allcharts = []

#print(clustersPick)
for clSize in clustersPick.keys():
    print("size", clSize, len(clustersPick[clSize]))
    for test in clustersPick[clSize]:
        allcharts.append(test)
        durations = ""
        for t in test:
            durations += str(t['duration']/1000000.0) + " " + t['obj'] + " "
        if clSize > 0:
            print(test[0]['id'], durations)
            

forplotnr = [] # string of lengths
forplotobj = [] #string of objects
forplotid = []

actionNr = 0

maxActionNr = max(clustersPick.keys())

while actionNr < maxActionNr:
    durations = "["
    obj = "["
    pid = "["

    for test in allcharts:
        if len(test) == 0:
            continue
        
        if actionNr >= len(test):
            durations += "0.0, "
            obj += '"box1", '            
            pid += '"' + test[0]['id'][18:22] + '", '
        else:            
            t = test[actionNr]
            durations += str(t['duration']/1000000.0) + ", "
            obj += '"' + t['obj'] + '", '
            pid += '"' + t['id'][18:22] + '", '

    forplotnr.append(durations)
    forplotobj.append(obj)
    forplotid.append(pid)
    actionNr = actionNr + 1
    durations = durations[:len(durations)-2] + "]"
    obj = obj[:len(obj)-2] + "]"
    pid = pid[:len(pid)-2] + "]"

#    print(durations)
#    print(obj)
#    print(pid)
    cmd = 'plotdata.append(pd.DataFrame({"pnp":' + durations + ','
    cmd += '"obj":' + obj + '},'
    cmd += 'index=' + pid + '))'
    print(cmd)



            

