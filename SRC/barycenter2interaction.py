import numpy as np 
import matplotlib.pyplot as plt
import json
import pandas as pd
import os

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

for p in range(7,8):

    clustering_dir= f"./Plots_Text/clustering/puzzle{p}/"
    present_objects_dir = clustering_dir + f"/present_objects_puzzle{p}.json"
    extracted_info_dir = clustering_dir + "extracted_info/"
    if not os.path.exists(extracted_info_dir):
        os.makedirs(extracted_info_dir)
    box_size= 4

    for file in os.listdir(clustering_dir):
        if file.endswith("softbarycenter.json"):

            cluster_no = file.split("_")[0]
            puzzle_no = file.split("_")[1]

            barycenter_dir = clustering_dir + file
            info_file_path = extracted_info_dir + file.split("_")[0] + "_interaction.json"

            info= dict()
            info["moved_objects"] = [] # objects that moved
            info["interactions"] = dict() # set of attachment

            #loading the barycenter
            with open(barycenter_dir) as f:
                data = json.load(f)

            data= np.array(data)

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
                object_i_name = present_objects[object_i]
                vx_i = velocity_vector[data.columns[i*2][0],'x']
                vx_i=np.array(vx_i, dtype=np.float64)
                vy_i = velocity_vector[data.columns[i*2][0],'y']
                vy_i=np.array(vy_i, dtype=np.float64)
                v_temp=np.sqrt(vx_i**2 + vy_i**2)
                v[:,i]=v_temp

            T = len(v)  

            #plotting the velocity profile and mean velocity and attachment
            fig1, ax = plt.subplots(len(present_objects),1,figsize=(10,3*len(present_objects)+4))

            for i in range(len(present_objects)):
                
                nl=0
                ax[i].plot(v[:,i], label=present_objects[data.columns[i*2][0]], color=coloring(present_objects[data.columns[i*2][0]], True))
                ax[i].axhline(y=np.mean(v[:,i]), color='r', linestyle='--', label='mean')
                ax[i].set_title(present_objects[data.columns[i*2][0]]+" velocity profile")
                ax[i].set_xlabel('Time [s]')
                if T < 1000:
                    ax[i].set_xticks(np.arange(0,T, 100), np.arange(0,T/100, 1))
                else:
                    ax[i].set_xticks(np.arange(0,T, 1000), np.arange(0,T/100, 10))
                ax[i].legend()
                ax[i].set_ylabel('Velocity [1/s]')
                # ax[i].set_ylim(0, 1.5*np.max(v[:,i]))
            
            fig1.suptitle(f"Velocity profiles of  {cluster_no} barycenter of {puzzle_no}", fontsize=16)
            fig1.tight_layout(pad=5.0)

            # modifed attachmnet based on displacement
            fig2, ax1 = plt.subplots(figsize=(10,10))

            for i, object_i in enumerate(present_objects):
                v_temp = v[:,i]
                
                d_total = displacement(data[object_i,'x'][0], data[object_i,'y'][0], 
                                    data[object_i,'x'][T-1], data[object_i,'y'][T-1])
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

                    #merging close attachments
                    if attachment != []:
                        start_time = attachment[0][0]
                        end_time = attachment[0][1]
                        temporal_modified_attachment = []
                        spatial_modified_attachment = []

                        for attach_index in range(len(attachment)):
                            if attachment[attach_index][0] - end_time < 50:
                                end_time = attachment[attach_index][1]
                            else:
                                temporal_modified_attachment.append([start_time, end_time])
                                start_time = attachment[attach_index][0]
                                end_time = attachment[attach_index][1]
                        temporal_modified_attachment.append([start_time, end_time])
                        # print(modified_attachment)
                        
                    # mark the attachment on the velocity profile
                    for attach in temporal_modified_attachment:
                        ax[i].axvline(x=attach[0], color='r', linestyle='--')
                        ax[i].axvline(x=attach[1], color='r', linestyle='--')
                    
                    d_attachment = []
                    #filtering the interaction that has meaningless displacement defined less than 5% of the total displacement
                    for attach in temporal_modified_attachment:
                        d=displacement(data[object_i,'x'][attach[0]], data[object_i,'y'][attach[0]], data[object_i,'x'][attach[1]], data[object_i,'y'][attach[1]])
                        # print(f"displacement of {present_objects[object_i]} is {d}")
                        d_attachment.append(d)
                
                    for attach, d in zip(temporal_modified_attachment, d_attachment):
                        if d >d_total/20 and d > box_size/10:
                            ax1.barh(y=i/2, width=attach[1]-attach[0], left=attach[0], height=0.5, color=coloring(present_objects[object_i], True))
                            spatial_modified_attachment.append(attach)
                    
                    if len(spatial_modified_attachment) > 0  :
                        
                        interaction_coordinate = dict()
                        
                        for attach in spatial_modified_attachment:
                            interaction_coordinate[str(attach[0]/100)
                                                    + " s to "+str(attach[1]/100)+" s"] = (data[object_i,'x'][attach[0]], data[object_i,'y'][attach[0]], data[object_i,'x'][attach[1]], data[object_i,'y'][attach[1]])

                        info["moved_objects"].append(present_objects[object_i])
                    
                        info["interactions"][present_objects[object_i]] = interaction_coordinate
            
            fig1.savefig(extracted_info_dir+"velocity_profile_"+cluster_no+".png")
            plt.close(fig1)

            ego_interaction_time = []
            print(cluster_no,puzzle_no)
            if info["interactions"].get("ego") is not None:
                for value in info["interactions"]["ego"].keys():
                    s,f = value.split(" to ")
                    s = float(s.split(" ")[0])
                    f = float(f.split(" ")[0])
                    ego_interaction_time.append([s,f])

            info["ego_interaction"] = dict()

            ego_key = [key for key in present_objects.keys() if present_objects[key] == "ego"][0]
            #check which object interacted with the ego by checking the intersection of the time
            for key in list(info["interactions"].keys()):
                if key != "ego":
                    for value in list(info["interactions"][key].keys()):
                        s,f = value.split(" to ")
                        s = float(s.split(" ")[0])
                        f = float(f.split(" ")[0])
                        for time in ego_interaction_time:
                            if time[0] < f and time[1] > s:
                                info["ego_interaction"][str(time[0]) + " s to "+str(s) + " s"] = {"mode":"free", "start-end coordinates":(
                                    data[ego_key,'x'][int(time[0]*100)], data[ego_key,'y'][int(time[0]*100)], data[ego_key,'x'][int(s*100)], data[ego_key,'y'][int(s*100)])}
                                info["ego_interaction"][str(s) + " s to "+str(f) + " s"] = {"mode":"attached to "+key, "start-end coordinates":(
                                    data[ego_key,'x'][int(s*100)], data[ego_key,'y'][int(s*100)], data[ego_key,'x'][int(f*100)], data[ego_key,'y'][int(f*100)])}
                                info["ego_interaction"][str(f) + " s to "+str(time[1]) + " s"] = {"mode":"free", "start-end coordinates":(
                                    data[ego_key,'x'][int(f*100)], data[ego_key,'y'][int(f*100)], data[ego_key,'x'][int(time[1]*100)], data[ego_key,'y'][int(time[1]*100)])}

            sorted_ego_interaction_keys = sorted(info["ego_interaction"].keys(), key=lambda x: float(x.split(" ")[0]))
            info["ego_interaction"] = {key: info["ego_interaction"][key] for key in sorted_ego_interaction_keys}

                
            #saving the information
            with open(info_file_path, 'w') as f:
                json.dump(info, f, indent=4)

            ax1.set_xlabel('Time [s]',fontsize=16)
            ax1.set_ylabel('Object name',fontsize=16)
            ax1.set_yticks(np.arange(len(present_objects))/2, present_objects.values(), fontsize=14)
            #T is in centisecond
            if T < 1000:
                ax1.set_xticks(np.arange(0,T, 100), np.arange(0,T/100, 1), fontsize=14)
            else:
                ax1.set_xticks(np.arange(0,T, 1000), np.arange(0,T/100, 10), fontsize=14)
            
            ax1.set_title(f"Attachment of {cluster_no} barycenter of {puzzle_no}", fontsize=16)
            fig2.tight_layout(pad=5.0)
            fig2.savefig( extracted_info_dir+"attachment_"+cluster_no+".png")
            plt.close(fig2)



    
