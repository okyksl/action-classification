import os
import sys
import subprocess
import argparse
import json
import urllib.request
import numpy as np
from re import search
from datetime import datetime

def extract_frames(file, output, start=None, end=None):
    prog = [
        'ffmpeg',
        '-i',
        file,
    ]
    
    select = None
    if start is None and end is not None:
        select = ['-vf', 'select=lte(n\\,{})'.format(end)]
    elif start is not None and end is None:
        select = ['-vf', 'select=gte(n\\,{})'.format(start)]
    elif start is not None and end is not None:
        select = ['-vf', 'select=between(n\\,{}\\,{})'.format(start, end)]
    
    if select is not None:
        prog = prog + select
    
    prog = prog + [
        output
    ]
    #print(' '.join(prog))
    
    #print( os.path.dirname(os.path.abspath(output)) )
    if not os.path.isdir( os.path.dirname(os.path.abspath(output)) ):
        os.makedirs( os.path.dirname(os.path.abspath(output)) )

    try:
        subprocess.check_output(prog)
    except subprocess.CalledProcessError as e:
        print(e)
        
def extract_random_frames(file, count, output, start=None, end=None):
    # Determine random points
    prog = [
        'ffmpeg',
        '-i',
        file,
        '-map',
        '0:v:0',
        '-c',
        'copy',
        '-f',
        'null',
        '-'
    ]
        
    try:
        info = subprocess.check_output(prog, stderr=subprocess.STDOUT)
        match = search('[\s\S]+frame=\s*(\d*)[\s\S]+time=(\S+)', str(info))
                
        frame_count = int(match.group(1))
        #print('FRAME COUNT: ' + str(frame_count))
        
        if start is None:
            start = 0
        if end is None:
            end = frame_count-1
        
        frames = np.random.choice(np.arange(start, end+1), count, replace=False)
        
        for frame in frames:
            output_file = output % (frame)
            extract_frames(file=file, start=frame, end=frame, output=output_file)
    except subprocess.CalledProcessError as e:
        print(e)    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts frames from given video source')
    parser.add_argument('--file', dest='file', type=str, required=True, help='Input video file')
    parser.add_argument('--count', dest='count', type=int, required=False, default=1, help='How many frames required')
    parser.add_argument('--random', dest='random', type=bool, required=False, default=False, help='Take random frames or not')
    parser.add_argument('--start', dest='start', type=float, required=False, default=None, help='Start frame in the video')
    parser.add_argument('--end', dest='end', type=float, required=False, default=None, help='End frame in the video')
    parser.add_argument('--output', dest='output', type=str, required=True, help='Output folder')

    args = parser.parse_args()
    if args.random:
        extract_random_frames(file=args.file, count=args.count, start=args.start, end=args.end, output=args.output)
    else:
        extract_frames(file=args.file, start=args.start, end=args.end, output=args.output)
