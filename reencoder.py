# # reencoder.py

# import subprocess
# import logging

# logging.basicConfig(level=logging.DEBUG)

# def reencode_to_h264(input_filepath, output_filepath):
#     command = [
#         'ffmpeg', '-y', '-i', input_filepath, '-vcodec', 'libx264', '-acodec', 'aac', output_filepath
#     ]
#     try:
#         result = subprocess.run(command, check=True, capture_output=True, text=True)
#         logging.debug(f"FFmpeg output: {result.stdout}")
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Error during FFmpeg re-encoding: {e.stderr}")
#         raise
import subprocess
import logging

logging.basicConfig(level=logging.DEBUG)

def reencode_to_h264(input_filepath, output_filepath):
    ffmpeg_path = 'C:\\ffmpeg\\bin\\ffmpeg'  # Full path to ffmpeg executable
    command = [
        ffmpeg_path, '-y', '-i', input_filepath, '-vcodec', 'libx264', '-acodec', 'aac', output_filepath
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.debug(f"FFmpeg output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during FFmpeg re-encoding: {e.stderr}")
        raise
