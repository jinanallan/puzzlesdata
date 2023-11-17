import os
import imageio

def convert_gif_to_video(root_folder, output_format='mp4'):
    # Iterate through the directory tree using os.walk
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            # Check if the file is a GIF
            if filename.lower().endswith('.gif'):
                input_path = os.path.join(root, filename)
                output_filename = os.path.splitext(filename)[0] + '.' + output_format
                output_path = os.path.join(root, output_filename)

                # Read GIF and write video
                with imageio.get_reader(input_path) as gif_reader:
                    # Get fps from the metadata or use a default value (e.g., 10 fps)
                    fps = gif_reader.get_meta_data().get('fps', 2)

                    # Create a video writer
                    with imageio.get_writer(output_path, format=output_format, fps=fps) as video_writer:
                        for frame in gif_reader:
                            video_writer.append_data(frame)

                print(f"Converted {filename} to {output_filename} with fps={fps}")

root_folder = './Plots_Text/clustering/'

convert_gif_to_video(root_folder)

