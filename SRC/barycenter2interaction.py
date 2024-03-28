import numpy as np 
import matplotlib.pyplot as plt
import json
import pandas as pd

def coloring(object,dummy = False):
    if dummy:
        if object=='box1':
            return (0,0,1) 
        elif object=='box2':
            return (0,1,0) 
        elif object=='obj1':
            return (1,0,0) 
        elif object=='obj1_a':
            return (1,0.5,0)
        elif object=='obj2':
            return (1,0,1) 
        elif object=='obj3':
            return (1,1,0) 
        elif object=='obj4':
            return (0,1,1) 
        elif object=='ego':
            return (0,0,0) 
    else:
        if object=='box1':
            return [(0,0,1,c) for c in np.linspace(0,1,100)]
        elif object=='box2':
            return [(0,1,0,c) for c in np.linspace(0,1,100)]
        elif object=='obj1':
            return [(1,0,0,c) for c in np.linspace(0,1,100)]
        elif object=='obj1_a':
            return [(1,0.5,0,c) for c in np.linspace(0,1,100)]
        elif object=='obj2':
            return [(1,0,1,c) for c in np.linspace(0,1,100)]
        elif object=='obj3':
            return [(1,1,0,c) for c in np.linspace(0,1,100)]
        elif object=='obj4':
            return [(0,1,1,c) for c in np.linspace(0,1,100)]
        elif object=='ego':
            return [(0,0,0,c) for c in np.linspace(0,1,100)]

def displacement(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
# this is an example of interaction extraction from the barycenter

barycenter_dir= "./Plots_Text/clustering/puzzle1/Cluster1_puzzle1_softbarycenter.json"
present_objects_dir = "./Plots_Text/clustering/puzzle1/present_objects_puzzle1.json"

#loading the barycenter
with open(barycenter_dir) as f:
    data = json.load(f)

data= np.array(data)
# print(data.shape)

#smoothing the barycenter by low pass filter twice
window_size=5
for i in range(data.shape[1]):
    data[:,i]=np.convolve(np.convolve(data[:,i], np.ones((window_size,))/window_size, mode='same'), np.ones((window_size,))/window_size, mode='same')
data = data[window_size:-window_size]

#loading the present objects
with open(present_objects_dir) as f:
    present_objects = json.load(f)
  
sub_columns = pd.MultiIndex.from_product([present_objects.keys(), ['x', 'y']], names=['ID', 'position'])

data=pd.DataFrame(data, columns=sub_columns)

#computing the velocity
velocity_vector = data.diff()
velocity_vector = velocity_vector.drop(0)
velocity_vector = velocity_vector.reset_index(drop=True)
v=np.zeros((len(velocity_vector),len(present_objects)))

for i, object_i in enumerate(present_objects):
    # print (i, object_i)
    object_i_name = present_objects[object_i]
    vx_i = velocity_vector[data.columns[i*2][0],'x']
    vx_i=np.array(vx_i, dtype=np.float64)
    vy_i = velocity_vector[data.columns[i*2][0],'y']
    vy_i=np.array(vy_i, dtype=np.float64)
    v_temp=np.sqrt(vx_i**2 + vy_i**2)
    v[:,i]=v_temp

T = len(v)  

#plotting the velocity profile and mean velocity and attachment
fig, ax = plt.subplots(len(present_objects),1,figsize=(10,3*len(present_objects)+4))

for i in range(len(present_objects)):
    
    nl=0
    ax[i].plot(v[:,i], label=present_objects[data.columns[i*2][0]], color=coloring(present_objects[data.columns[i*2][0]], True))
    ax[i].axhline(y=np.mean(v[:,i]), color='r', linestyle='--', label='mean')
    ax[i].set_title(present_objects[data.columns[i*2][0]]+" velocity profile")
    ax[i].legend()
    fig.tight_layout(pad=5.0)


# modifed attachmnet based on displacement
    
fig, ax1 = plt.subplots(figsize=(10,10))

for i, object_i in enumerate(present_objects):
    v_temp = v[:,i]
    
    d_total = displacement(data[object_i,'x'][0], data[object_i,'y'][0], data[object_i,'x'][T-1], data[object_i,'y'][T-1])
    # print(f"total displacement of {present_objects[object_i]} is {d_total}")
    attachment = []
    start_time = None
    
    first_time = np.where(v_temp>np.mean(v_temp))
    first_time = first_time if len(first_time[0])==0 else first_time[0][0]

    if v_temp.any() > 0:
        for t in np.arange(int(first_time), T):
            if v_temp[t] > np.mean(v_temp):
                if start_time is None:
                    start_time = t
            else:
                if start_time is not None:
                    attachment.append([start_time, t])
                    start_time = None
        # print(attachment)
        
        #merging close attachments
        if attachment != []:
            start_time = attachment[0][0]
            end_time = attachment[0][1]
            modified_attachment = []

            for attach_index in range(len(attachment)):
                if attachment[attach_index][0] - end_time < 50:
                    end_time = attachment[attach_index][1]
                else:
                    modified_attachment.append([start_time, end_time])
                    start_time = attachment[attach_index][0]
                    end_time = attachment[attach_index][1]
            modified_attachment.append([start_time, end_time])
            # print(modified_attachment)
            
        # mark the attachment on the velocity profile
        for attach in modified_attachment:
            ax[i].axvline(x=attach[0], color='r', linestyle='--')
            ax[i].axvline(x=attach[1], color='r', linestyle='--')
        
        d_attachment = []
        #filtering the interaction that has meaningless displacement defined less than 5% of the total displacement
        for attach in modified_attachment:
            d=displacement(data[object_i,'x'][attach[0]], data[object_i,'y'][attach[0]], data[object_i,'x'][attach[1]], data[object_i,'y'][attach[1]])
            # print(f"displacement of {present_objects[object_i]} is {d}")
            d_attachment.append(d)
        
        for attach, d in zip(modified_attachment, d_attachment):
            if d >d_total/20:
                modified_attachment.remove(attach)
                ax1.barh(y=i/2, width=attach[1]-attach[0], left=attach[0], height=0.5, color=coloring(present_objects[object_i], True))

ax1.set_xlabel('Time [s]',fontsize=16)
ax1.set_ylabel('Object name',fontsize=16)
ax1.set_yticks(np.arange(len(present_objects))/2, present_objects.values(), fontsize=14)
ax1.set_xticks(np.arange(0,T, 1000), np.arange(0,T/100, 10), fontsize=14)
plt.show()   