import movementTracker
import HMPlotter
import os
import json


# get description of the event of puzzle 24
folder = '/home/erfan/Downloads/pnp'
for filename in os.listdir(folder):
            if filename.endswith('.json'):
                participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)

                if puzzle == 24:
                    # Load the JSON file
                    with open(os.path.join(folder, filename)) as json_file:
                        data = json.load(json_file)
                        df=movementTracker.df_from_json(data)
                        print(movementTracker.get_descriptions(df))