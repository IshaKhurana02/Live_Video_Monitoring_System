#stream:rtsp://admin:123456@192.168.1.205:554/avstream/channel=1/stream=0.sdp

import os
import json
import cv2
import numpy as np
import base64
import zmq
from datetime import datetime,timedelta
from ultralytics import YOLO
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
import sys
from pathlib import Path
from tracker import Tracker
from sort import Sort
import time
import logging
import platform
import helpers
from collections import deque
import math

print(" Starting main.py...")
sys.stdout.flush()

if platform.system()=="Windows":
    gst_install_dir = Path(r'C:\gstreamer-python')
    gst_bin_dir = gst_install_dir / 'bin'

    sys.path.append(str(gst_install_dir / 'lib' / 'site-packages'))

    os.environ['PATH'] += os.pathsep + 'C:\\gstreamer-python\\lib\\gstreamer-1.0'
    os.environ['PATH']+=os.pathsep + 'C:\\GSTREAMER\\gstreamer\\build\\subprojects\\gstreamer\\libs\\gst\\helpers\\gst-plugin-scanner.exe'

    os.add_dll_directory(str(gst_bin_dir))
else:
    pass

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(sys.argv)

#-------load configuration file function-----------
def load_config(confi_path="config.json"):
    with open(confi_path,'r') as file:
        return json.load(file)
    
#------------call above function------------
config=load_config()

#------------load class name file-----------
def cls_name(file_path="class_name.json"):
    with open(file_path,'r') as file:
        data=json.load(file)
        return data["CLASS_NAMES"]
    
clsname=cls_name()

#---------initialize zmq context for publisher and subscriber-------#
zmq_context=zmq.Context()
zmq_socket=zmq_context.socket(zmq.PUB)
zmq_socket.connect(config["zmq"]["publisher"]["address"])

#-------------------initialize device---------------#
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')

#-----------------load models---------------------#
yolo_model=YOLO("best.pt")
yolo_model.to(device)
fall=YOLO("fall2.pt")
fall.to(device)
fireNsmoke=YOLO("fireandsmoke.pt")
fireNsmoke.to(device)
pose_model=YOLO("pose.pt")
csrnet_model_path = "CSRNET_PartAmodel_best.pth"
csrnet_model = helpers.load_csrnet_model(csrnet_model_path,device)
person_attributes_model_path = "Person_attributes_resnet50_peta.pth"
person_attributes_model = helpers.load_person_attributes_resnet_model(person_attributes_model_path,device)

#---------------initailize json lock--------------#
json_lock=threading.Lock()
lock=threading.Lock()

# -----------------variables for ai-----------------#
intrusion_cls=config["detection"].get("intrusion_classes",[0])

#-------------------mongodb connection----------------#
mongo_uri = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/")
client = MongoClient(mongo_uri)
# client=MongoClient("mongodb://localhost:27017/")
db=client["VM_database"]
alert_db=db["alert_collection"]

#--------------------dictonaries  or lists--------------#
stream_flag={}
active_threads={}
entry_tym={}
pipeline_dict={}
center_pt_track={}
person_counters={}
direction_history={}
crossing_record = {}
has_crossed={}
waving_keypoint_history={}
crowd_start_times = {}
crowd_dispersion_start_time={}
waiting_time_entry={}
waving_alert_time={}
waiting_time_alert_time={}  
fall_alert_time={} 
crowd_formation_alert_time={}
crowd_dispersion_alert_time={}
crowd_estimation_alert_time={}
crowd_estimation_start_time={}
consecutive_fall_count={}
wrong_direction_history={}
wrong_direction_alert_time={}
directional_arrow_alert_time={}
person_last_states={}

FIRE_SMOKE_CLASS_NAMES = {
    "0": "fire",
    "1": "smoke"
}
FALL_CLASS_NAMES = {"0": "fall"}

# Create LOGS directory if not exists
if not os.path.exists('LOGS'):
    os.makedirs('LOGS')

# Function to configure logging for a specific RTSP stream
def configure_logging(rtsp_id):
    logger = logging.getLogger(rtsp_id)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        fh = logging.FileHandler(f'LOGS/{rtsp_id}.log')
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

#--------------------load stream.json-----------------#
def load_stream_json_file(stream_path="stream.json"):
    try:
        with open(stream_path,'r') as file:
            data=json.load(file)
            streams={}
            analytics={}
            stream_metadata={}

            for stream in data["streams"]:
                rtsp_id=stream['rtsp_id']
                streams[rtsp_id]=stream['rtsp_url']
                analytics[rtsp_id]=stream.get("analytics",[])
                stream_metadata[rtsp_id]={
                    "name": stream.get('name', f"Stream {rtsp_id}"),
                    "fps": stream.get('fps', 2),  # Default FPS to 30 if not provided
                    "username": stream.get('username', ''),
                    "password": stream.get('password', ''),
                    "loitering_threshold" : stream.get("loitering_threshold",30),
                    "crowd_formation_threshold" : stream.get("crowd_formation_threshold", 5),
                    "crowd_formation_duration" : stream.get("crowd_formation_duration",10),
                    "crowd_estimation_threshold" : stream.get("crowd_estimation_threshold",15),
                    "crowd_estimation_duration" : stream.get("crowd_estimation_duration",10),
                    "entry_line_type" : stream.get("entry_line_type"),
                    "exit_line_type" : stream.get("exit_line_type"),
                    "direction": stream.get("direction", "Left to Right"),
                    "crowd_dispersion_threshold" : stream.get("crowd_dispersion_threshold",10),
                    "crowd_dispersion_duration" : stream.get("crowd_dispersion_duration",10)
                }

        return streams,analytics,stream_metadata

    except Exception as e:
        print("could not load becuase of ",e)
        return {},{},{}

def save_data_to_json(file,streams,analytics,stream_metadata):
    try:
        with json_lock:
            data={
                "streams":[]
            }
            for rtsp_id,rtsp_url in streams.items():
                new_data={
                    "id": rtsp_id,
                    "url": rtsp_url,
                    "analytics": analytics.get(rtsp_id, []),
                    "name": stream_metadata.get(rtsp_id, {}).get("name", f"Stream {rtsp_id}"),
                    "fps": stream_metadata.get(rtsp_id, {}).get("fps", 3),
                    "username": stream_metadata.get(rtsp_id, {}).get("username", ""),
                    "password": stream_metadata.get(rtsp_id, {}).get("password", ""),
                    "loitering_threshold" : stream_metadata.get(rtsp_id, {}).get("loitering_threshold") ,
                    "crowd_formation_threshold" : stream_metadata.get(rtsp_id, {}).get("crowd_formation_threshold"),
                    "crowd_formation_duration" : stream_metadata.get(rtsp_id, {}).get("crowd_formation_duration"),
                    "crowd_estimation_threshold": stream_metadata.get(rtsp_id, {}).get("crowd_estimation_threshold"),
                    "crowd_estimation_duration" : stream_metadata.get(rtsp_id, {}).get("crowd_estimation_duration"),
                    "entry_line_type" : stream_metadata.get(rtsp_id,{}).get("entry_line_type"),
                    "exit_line_type" : stream_metadata.get(rtsp_id,{}).get("exit_line_type"),
                    "direction" : stream_metadata.get(rtsp_id,{}).get("direction"),
                    "crowd_dispersion_threshold" : stream_metadata.get(rtsp_id,{}).get("crowd_dispersion_threshold"),
                    "crowd_dispersion_duration" : stream_metadata.get(rtsp_id,{}).get("crowd_dispersion_duration")
                }
                
                data["streams"].append(new_data)

            with open(file,'w') as f:
                json.dump(data,f,indent=4)

    except Exception as e:
        print(f"error {e} in saving into json")

def start_stream(rtsp_id,rtsp_url,analytics_data,meta_data,executor):
    
    stream_flag[rtsp_id]=True
    activeThread=executor.submit(process_stream,rtsp_id,rtsp_url,analytics_data,meta_data)
    active_threads[rtsp_id]=activeThread

def stop_stream(rtsp_id):
    stream_flag[rtsp_id]=False
    if rtsp_id in active_threads:
        future=active_threads[rtsp_id]
        if future.running():
            try:
                future.result(timeout=2)
            except Exception as e:
                print(f"Thread with id {rtsp_id} enmcountered an error:{e}")
        active_threads.pop(rtsp_id,None)
        print(f"Thread removed with id :{rtsp_id}")
        #pipeline flush
        if rtsp_id in pipeline_dict:
            pipeline=pipeline_dict[rtsp_id]
            pipeline.send_event(Gst.Event.new_eos())
            pipeline.set_state(Gst.State.NULL)
            del pipeline_dict[rtsp_id]

def save_and_send_intrusion_data(rtsp_id,frame,logger=None):
    try:
        folder_path=os.path.join("Intrusion",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)
        
        filename=f"{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Intrusion Detected",
            "Base64_img":base64_img,
            "Remark": None,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters" : None
        }
        mongo_data={
                "DeviceID":playload['Device_id'],
                "Event":playload['Event_detected'],
                "timestamp":playload['timestamp'],
                "Parameters":playload['Parameters']
            }
        try:
            alert_db.insert_one(mongo_data)
            print("sending Intrusion data into mongodb")
        except Exception as e:
            print(f"error in saving intrusion data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)
            print("sending  intrusion alert\n")
        except Exception as e:
            print(f"error in sending intrusion data using zmq socket :{e}")

    except Exception as e:
        print(f"error in saving Intrusion: {e}")
    del buffer,base64_img,frame,json_data,mongo_data

def save_and_send_intrusion_attribute_alert(rtsp_id,frame,person_id,additional_data,logger=None):
    try:
        folder_path=os.path.join("Intrusion_with_attributes",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)
        
        filename=f"{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Intrusion Detected",
            "Base64_img":base64_img,
            "Remark": None,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters" : additional_data
        }
        mongo_data={
                "DeviceID":playload['Device_id'],
                "Event":playload['Event_detected'],
                "timestamp":playload['timestamp'],
                "Parameters":playload['Parameters']
            }
        try:
            alert_db.insert_one(mongo_data)
            print("sending Intrusion_attribute data into mongodb")
        except Exception as e:
            print(f"error in saving Intrusion_attribute data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)
            print("sending  Intrusion_attribute alert\n")
        except Exception as e:
            print(f"error in sending Intrusion_attribute data using zmq socket :{e}")

    except Exception as e:
        print(f"error in saving Intrusion_attribute: {e}")
    del buffer,base64_img,frame,json_data,mongo_data

def save_and_send_direction_arrow_data(rtsp_id,frame,person_id,direction,logger=None):
    try:
        folder_path=os.path.join("Directional_alarm",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)
        
        filename=f"{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Directional Alarms Detected",
            "Base64Image":base64_img,
            "Remark":f"Person ID {person_id} moving in direction: {direction}",
            "Timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters": {
                "type": "Person",
                "attributes": {
                    "person_id": int(person_id),
                    "direction": direction
                }
            }
        }
        mongo_data={
                "DeviceID":playload['Device_id'],
                "Event":playload['Event_detected'],
                "timestamp":playload['Timestamp'],
                "Parameters":playload['Parameters']
            }
        try:
            alert_db.insert_one(mongo_data)
            print("sending direction_arrow data into mongodb")
        except Exception as e:
            print(f"error in saving direction_arrow data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)
            print("sending  direction_arrow alert\n")
        except Exception as e:
            print(f"error in sending direction_arrow data using zmq socket :{e}")

    except Exception as e:
        print(f"error in saving direction_arrow: {e}")
    del buffer,base64_img,frame,json_data,mongo_data

def save_and_send_Loitering_data(rtsp_id,frame,person_id,logger=None):
    try:
        folder_path=os.path.join("Loitering",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"Loitering_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Loitering Detected",
            "Base64_img":base64_img,
            "Remark": None,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters" : None
        }
        
        mongo_data={
                "DeviceID":playload['Device_id'],
                "Event":playload['Event_detected'],
                "timestamp":playload['timestamp'],
                "parameters":playload['Parameters']
            }
        try:
            alert_db.insert_one(mongo_data)
            print("sending loitering data into mongodb")
        except Exception as e:
            print(f"error in saving Loitering data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)
            print("sending  loitering alert\n")
        except Exception as e:
            print(f"error in sending loitering data using zmq socket :{e}")
    except Exception as e:
        print(f"Error in saving Loitering:{e}")
    del buffer,base64_img,frame,json_data,mongo_data


def save_and_send_Fall_data(rtsp_id,frame,logger=None):
    try:
        folder_path=os.path.join("Fall",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"Fall_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Fall_Detected",
            "Base64_img":base64_img,
            "Remark": None,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters": None
        }
        
        mongo_data={
                "DeviceID":playload['Device_id'],
                "Event":playload['Event_detected'],
                "timestamp":playload['timestamp'],
                "parameters":playload['Parameters']
            }
        try:
            alert_db.insert_one(mongo_data)
            print("sending fall data into mongodb")
        except Exception as e:
            print(f"error in saving Fall data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)
            print("sending  fall alert\n")
        except Exception as e:
            print(f"error in sending fall data using zmq socket :{e}")
    except Exception as e:
        print(f"Error in saving fall:{e}")
    del buffer,base64_img,frame,json_data,mongo_data

def save_and_send_FireAndSmoke_data(rtsp_id,frame,detection_type,logger=None):
    try:
        folder_path=os.path.join("Fire_and_smoke",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"FireNsmoke_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Fire_Smoke_Detected",
            "Base64_img":base64_img,
            "Detection_name":f"{detection_type} Detected",
            "Remark": None,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters": None
        }
        
        mongo_data={
                "DeviceID":playload['Device_id'],
                "Event":playload['Event_detected'],
                "timestamp":playload['timestamp']
            }
        try:
            alert_db.insert_one(mongo_data)
            print("sending fire and smoke data into mongodb")
        except Exception as e:
            print(f"error in saving Fire and smoke data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)

            print("sending  Fire and smoke alert\n")
        except Exception as e:
            print(f"error in sending Fire and smoke data using zmq socket :{e}")
    except Exception as e:
        print(f"Error in saving fire and smoke alerts:{e}")
    del buffer,base64_img,frame,json_data,mongo_data

def save_and_send_wrong_direction_data(rtsp_id,frame,person_id,direction,logger=None):
    try:
        folder_path=os.path.join("Wrong_direction",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"wrong_direction_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Wrong Direction Detected",
            "Base64_img":base64_img,
            "Remark": f"Moving in wrong direction: {direction}",
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters": None
        }
        
        mongo_data={
                    "DeviceID":playload['Device_id'],
                    "Event":playload['Event_detected'],
                    "timestamp":playload['timestamp'],
                    "parameters":playload['Parameters']
                    }
        try:
            alert_db.insert_one(mongo_data)
            print("sending wrong direction data into mongodb")
        except Exception as e:
            print(f"error in saving wrong direction data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)

            print("sending  wrong direction alert\n")
        except Exception as e:
            print(f"error in sending wrong direction data using zmq socket :{e}")     
    except Exception as e:
        print(f"Error in saving wrong direction:{e}")
    del buffer,base64_img,frame,json_data,mongo_data


def save_and_send_waving_hand_data(rtsp_id,frame,person_id,logger=None):
    try:
        folder_path=os.path.join("waving_hands",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"waving_hand_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Waving Detected",
            "Base64_img":base64_img,
            "Remark": None,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters": None
        }
        
        mongo_data={
                    "DeviceID":playload['Device_id'],
                    "Event":playload['Event_detected'],
                    "timestamp":playload['timestamp'],
                    "parameters":playload['Parameters']
                    }
        try:
            alert_db.insert_one(mongo_data)
            print("sending waving hand data into mongodb")
        except Exception as e:
            print(f"error in saving waving hand data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)

            print("sending  waving hand alert\n")
        except Exception as e:
            print(f"error in sending waving hand data using zmq socket :{e}")     
    except Exception as e:
        print(f"Error in saving waving hand :{e}")
    del buffer,base64_img,frame,json_data,mongo_data

def save_frame_and_send_crowd_formation_alert(rtsp_id,frame,count,logger=None):
    try:
        folder_path=os.path.join("crowd_formation",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"crowd_formation_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Crowd Formation Detected",
            "Base64_img":base64_img,
            "Remark": None,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters" : {
                "type": "Person",
                "attributes": {
                    "count" : count
                }
            }
        }
        
        mongo_data={
                    "DeviceID":playload['Device_id'],
                    "Event":playload['Event_detected'],
                    "timestamp":playload['timestamp'],
                    "Parameters":playload['Parameters']
                    }
        try:
            alert_db.insert_one(mongo_data)
            print("sending crowd_formation data into mongodb")
        except Exception as e:
            print(f"error in saving crowd_formation data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)

            print("sending  crowd_formation alert\n")
        except Exception as e:
            print(f"error in sending crowd_formation data using zmq socket :{e}")     
    except Exception as e:
        print(f"Error in saving crowd_formation :{e}")
    del buffer,base64_img,frame,json_data,mongo_data

def save_and_send_crowd_estimation_alert(rtsp_id,frame,count,density_map_resized,logger=None):
    try:
        folder_path=os.path.join("crowd_estimation",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"crowd_estimation_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Crowd_Estimation_Detected",
            "Base64_img":base64_img,
            "Remark": None,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters" : {
                "type": "Person",
                "attributes": {
                    "count" : count
                }
            }
        }
        
        mongo_data={
                    "DeviceID":playload['Device_id'],
                    "Event":playload['Event_detected'],
                    "timestamp":playload['timestamp'],
                    "Parameters":playload['Parameters']
                    }
        try:
            alert_db.insert_one(mongo_data)
            print("sending crowd_estimation data into mongodb")
        except Exception as e:
            print(f"error in saving crowd_estimation data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)

            print("sending  crowd_estimation alert\n")
        except Exception as e:
            print(f"error in sending crowd_estimation data using zmq socket :{e}")     
    except Exception as e:
        print(f"Error in saving crowd_estimation :{e}")
    del buffer,base64_img,frame,json_data,mongo_data

def save_and_send_crowd_dispersion_alert(rtsp_id,frame,count,logger=None):
    try:
        folder_path=os.path.join("crowd_dispersion",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"crowd_dispersion_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Crowd Dispersion Detected",
            "Base64_img":base64_img,
            "Remark": None,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters" : {
                "type": "Person",
                "attributes": {
                    "count" : count
                }
            }
        }
        
        mongo_data={
                    "DeviceID":playload['Device_id'],
                    "Event":playload['Event_detected'],
                    "timestamp":playload['timestamp'],
                    "Parameters":playload['Parameters']

                    }
        try:
            alert_db.insert_one(mongo_data)
            print("sending crowd_dispersion data into mongodb")
        except Exception as e:
            print(f"error in saving crowd_dispersion data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)

            print("sending  crowd_dispersion alert\n")
        except Exception as e:
            print(f"error in sending crowd_dispersion data using zmq socket :{e}")     
    except Exception as e:
        print(f"Error in saving crowd_dispersion :{e}")
    del buffer,base64_img,frame,json_data,mongo_data
    
def save_and_send_waiting_time_alert(rtsp_id,frame,person_id,waiting_time,logger=None):
    try:
        folder_path=os.path.join("waiting_time",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"waiting_time_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        additional_data = {
            "type": "Person",
            "attributes": {
                "person_id": int(person_id),
                "waiting_time_seconds": round(waiting_time, 2)
            }
        }

        playload={
            "Device_id":rtsp_id,
            "Event_detected":"Waiting time detected",
            "Base64_img":base64_img,
            "Remark": f"Person ID {person_id} waited for {round(waiting_time, 2)} seconds",
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True,
            "Parameters": additional_data
        }
        
        mongo_data={
                    "DeviceID":playload['Device_id'],
                    "Event":playload['Event_detected'],
                    "timestamp":playload['timestamp'],
                    "Parameters":playload['Parameters']
                    }
        try:
            alert_db.insert_one(mongo_data)
            print("sending waiting_time data into mongodb")
        except Exception as e:
            print(f"error in saving waiting_time data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)

            print("sending  waiting_time alert\n")
        except Exception as e:
            print(f"error in sending waiting_time data using zmq socket :{e}")     
    except Exception as e:
        print(f"Error in saving waiting_time:{e}")
    del buffer,base64_img,frame,json_data,mongo_data


def save_and_send_in_out_data(rtsp_id,frame,entry_count,exit_count,logger=None):
    try:
        folder_path=os.path.join("Person_in_out",datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(folder_path,exist_ok=True)

        filename=f"Person_in_out_{rtsp_id}_{datetime.now().strftime('%H-%M-%S')}.jpg"
        filepath=os.path.join(folder_path,filename)
        cv2.imwrite(filepath,frame)

        _,buffer=cv2.imencode(".jpg",frame)
        base64_img=base64.b64encode(buffer).decode('utf-8')

        event_type="Person_In_Out_detected"

        playload={
            "Device_id":rtsp_id,
            "Event_detected":event_type,
            "Base64_img":base64_img,
            "timestamp":datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "Detection":True
        }
        
        mongo_data={
                    "DeviceID":playload['Device_id'],
                    "Event":playload['Event_detected'],
                    "timestamp":playload['timestamp']
                    }
        try:
            alert_db.insert_one(mongo_data)
            print("sending Person_in_out data into mongodb")
        except Exception as e:
            print(f"error in saving Person_in_out data into mongodb:{e}")

        json_data=json.dumps(playload)
        
        try:
            zmq_socket.send_string(json_data)

            print("sending  Person_in_out alert\n")
        except Exception as e:
            print(f"error in sending Person_in_out data using zmq socket :{e}")     
    except Exception as e:
        print(f"Error in saving Person_in_out:{e}")
    del buffer,base64_img,frame,json_data,mongo_data


def process_stream(rtsp_id,rtsp_url,analytics,metadata):

    print(f"starting main thread for rtsp_id : {rtsp_id}")
    logger = configure_logging(rtsp_id)
    logger.info(f"Starting processing stream {rtsp_id} URL: {rtsp_url}")
    
    stream_flag[rtsp_id] = True
    
    loit_threshold = metadata.get("loitering_threshold")
    crowd_formation_threshold = metadata.get("crowd_formation_threshold")
    crowd_formation_duration = metadata.get("crowd_formation_duration")
    crowd_estimation_threshold = metadata.get("crowd_estimation_threshold")
    crowd_estimation_duration = metadata.get("crowd_estimation_duration")
    crowd_dispersion_threshold = metadata.get("crowd_dispersion_threshold")
    crowd_dispersion_duration = metadata.get("crowd_dispersion_duration")
    entry_count = 0
    exit_count = 0
    consecutive_fire_smoke_count = 0
    consecutive_fall_count[rtsp_id] = 0
    
    intrusion_alert_time=datetime.now() - timedelta(seconds=5)
    loiter_alert_time=datetime.now() - timedelta(seconds=5)
    fall_alert_time[rtsp_id]=datetime.now() - timedelta(seconds=5)
    fireandsmoke_alert_time=datetime.now() - timedelta(seconds=5)
    Intrusion_attributes_alert_time = datetime.now() - timedelta(seconds=5)# Initialize loitering alert time
    crowd_formation_alert_time[rtsp_id] = datetime.now() - timedelta(seconds=crowd_formation_duration)
    crowd_estimation_alert_time[rtsp_id] = datetime.now() - timedelta(seconds=crowd_estimation_duration)
    crowd_dispersion_alert_time[rtsp_id] = datetime.now() - timedelta(seconds = crowd_dispersion_duration)
    waiting_time_alert_time[rtsp_id] = datetime.now() - timedelta(seconds=5)

    #create object of class Tracker
    tracker=Sort()
    dir_tracker=Sort()
    wrong_direction_tracker=Sort()
    waving_tracker=Sort()
    waiting_time_tracker=Sort()
    in_out_tracker=Sort()
   
    #frame timeout
    frame_timeout=10

    #fetch roi of each analytics
    intr_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="Intrusion"),[]),dtype=np.int32).reshape(-1,1,2)
    intrusion_with_attributes_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="intrusion_with_attributes"),[]),dtype=np.int32).reshape(-1,1,2)
    loit_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="Loitering"),[]),dtype=np.int32).reshape(-1,1,2)
    fall_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="fall"),[]),dtype=np.int32).reshape(-1,1,2) 
    fireandsmoke_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="fireandsmoke"),[]),dtype=np.int32).reshape(-1,1,2)
    directional_arrow_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="Direction_arrow"),[]),dtype=np.int32).reshape(-1,1,2)
    waving_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="person_waving_hand"),[]),dtype=np.int32).reshape(-1,1,2)
    crowd_formation_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="crowd_formation"),[]),dtype=np.int32).reshape(-1,1,2)
    crowd_dispersion_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="crowd_dispersion"),[]),dtype=np.int32).reshape(-1,1,2)
    waiting_time_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="waiting_time_in_roi"),[]),dtype=np.int32).reshape(-1,1,2)
    wrong_direction_roi=np.array(next((analytic['roi'] for analytic in analytics if analytic['type']=="wrong_direction"),[]),dtype=np.int32).reshape(-1,1,2)
    
    while stream_flag.get(rtsp_id,True):
        try:
            fps = metadata.get('fps',2)

            gst_pipeline=(
                # f"rtspsrc location={rtsp_url} latency=100 ! "
                # "rtph264depay ! h264parse ! avdec_h264 ! "
                f"filesrc location={rtsp_url} ! "
                "qtdemux ! "
                "decodebin ! "
                "videoconvert ! videoscale ! videorate ! "
                f"video/x-raw,format=BGR,width=640,height=640,framerate={fps}/1 ! "
                "appsink name=sink emit-signals=True max-buffers=1 drop=True"
            )
            
            pipeline=Gst.parse_launch(gst_pipeline)  
            appsink=pipeline.get_by_name('sink')
            
            if not appsink:
                print(f"[ERROR] Appsink not found for stream {rtsp_id}")
                break
            
            bus=pipeline.get_bus()
            bus.add_signal_watch()
            
            pipeline.set_state(Gst.State.PLAYING)
            pipeline_dict[rtsp_id] = pipeline
            last_frame_time=time.time()
            
            while stream_flag.get(rtsp_id,True):
                
                msg = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg:
                    if msg.type == Gst.MessageType.ERROR:
                        err, debug = msg.parse_error()
                        print(f"[GSTREAMER ERROR] {err.message} — Stream ID: {rtsp_id}")
                        if debug:
                            print(f"[DEBUG INFO] {debug}")
                        break
                    elif msg.type == Gst.MessageType.EOS:
                        print(f"[INFO] End of Stream for {rtsp_id}")
                        break
                
                appsink.set_property("drop", True)
                try:
                    sample = appsink.emit("pull-sample")

                except Exception as e:
                    print(f"Error pulling sample: {e}")
                    continue
                
                if sample is None:
                    print("no sample")
                    break
                    # if time.time()-last_frame_time>frame_timeout:
                    #     print(f"No frame recieved for Stream {rtsp_id}. Retrying...")
                    #     continue
                    # else:
                    #     time.sleep(5)
                    #     continue
                # else:   
                #     continue
                
                buff=sample.get_buffer()
                caps=sample.get_caps()
                height=caps.get_structure(0).get_value('height')
                width=caps.get_structure(0).get_value('width')
                
                success,map_info=buff.map(Gst.MapFlags.READ)
            
                if success:
                    try:
                        detection=[]
                        person_bbox=[]
                        frame=np.ndarray(shape=(height,width,3),dtype=np.uint8,buffer=map_info.data)
                        frame=cv2.resize(frame,(640,640))
                        yolo_results=yolo_model(frame,verbose=False,classes=intrusion_cls,conf=0.5,iou=0.25)
                        
                        for result in yolo_results:
                            if len(result)>0 and len(result.boxes)>0:
                                for box in result.boxes:
                                    
                                    x1,y1,x2,y2=map(int,box.xyxy[0])
                                    conf=float(box.conf[0])
                                    cls=int(box.cls[0])
                                    class_name=clsname.get(str(cls),"unknown")
                                    detection.append({"bbox":[x1,y1,x2,y2],"class":class_name,"cls":cls})

                                    #personbbox
                                    if cls==0:
                                        if conf>=0.7:
                                            person_bbox.append([x1,y1,x2,y2,conf])


                        if "Intrusion" in [analytic["type"] for analytic in analytics]:
                            for object in detection:
                                x1,y1,x2,y2=object['bbox']
                                class_name=object['class']
                                cls=object['cls']
                                center=((x1+x2)//2,(y1+y2)//2)
                                is_inside_roi=cv2.pointPolygonTest(intr_roi,center,False)>=0

                                if is_inside_roi and cls in intrusion_cls:
                                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                                    text = f"{class_name}"
                                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                current_time=datetime.now()
                            if len(detection)>0 and (current_time-intrusion_alert_time>=timedelta(seconds=5)):    
                                logger.info(f"Intrusion detected for class {class_name} in stream {rtsp_id}")
                                save_and_send_intrusion_data(rtsp_id,frame,logger=logger)
                                intrusion_alert_time=current_time

                        if "intrusion_with_attributes" in [analytic["type"] for analytic in analytics]:
                            for obj in detection:
                                x1, y1, x2, y2 = obj['bbox']
                                class_name=obj['class']
                                class_id=obj['cls']

                                # Draw bounding box and label
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                text = f"{class_name}"
                                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                # Check if object is inside the ROI
                                point_to_check = ((x1 + x2) // 2, (y1 + y2) // 2)
                                is_inside_roi = cv2.pointPolygonTest(intrusion_with_attributes_roi, point_to_check, False) >= 0
                                
                                # Check if the object belongs to intrusion classes
                                if is_inside_roi and class_id in intrusion_cls:
                                    current_time = datetime.now()
                                    if current_time - Intrusion_attributes_alert_time >= timedelta(seconds=5):  # Alert interval of 5 seconds
                                        # Initialize additional data for the alert
                                        attributes_data = {}
            
                                        # Detect color for all intrusion classes
                                        color_info = helpers.detect_dominant_color_hsv(frame, [x1, y1, x2, y2])
                                        attributes_data["color"] = color_info

                                        # Extract attributes only if the class is 'person' (class_id == 0)
                                        if class_id == 0:  # Person
                                            person_attributes = helpers.extract_attributes(person_attributes_model, frame, [x1, y1, x2, y2], device)
                                            attributes_data.update(person_attributes)

                                        additional_data = {
                                            "type": class_name,
                                            "attributes": attributes_data
                                        }
                                        print(additional_data)
                                        logger.info(f"Intrusion  with attributes detected for class {class_name} in stream {rtsp_id}")
                                        save_and_send_intrusion_attribute_alert(rtsp_id, frame, f'{class_name}_{rtsp_id}', additional_data,logger=logger)
                                        Intrusion_attributes_alert_time = current_time

                        if "Direction_arrow" in [analytic["type"] for analytic in analytics]:
    
                            if len(person_bbox) == 0:
                                continue
                            if np.array(person_bbox).shape[1] != 5:
                                continue

                            tracked_objects = dir_tracker.update(np.array(person_bbox))
                            
                            current_time = datetime.now()
                            for obj in tracked_objects:
                                bbox=obj[:4]
                                x1,y1,x2,y2=map(int,bbox)
                                person_id=obj[4]
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                point_to_check = (cx, cy)

                                is_inside_roi = cv2.pointPolygonTest(directional_arrow_roi, point_to_check, False) >= 0
                                if not is_inside_roi:
                                    continue

                                if person_id not in direction_history:
                                    direction_history[person_id] = deque(maxlen=10)
                                direction_history[person_id].append((cx, cy))

                                cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
                                cv2.putText(frame, f'ID: {person_id}', (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                if len(direction_history[person_id]) >= 2:
                                    prev_cx, prev_cy = direction_history[person_id][-2]
                                    curr_cx, curr_cy = direction_history[person_id][-1]

                                    # Compute direction vector
                                    dx = curr_cx - prev_cx
                                    dy = curr_cy - prev_cy
                                    magnitude = math.hypot(dx, dy)

                                    # Default direction
                                    direction = "Stationary"

                                    # Only proceed if movement
                                    if magnitude > 2:  #  filter noise
                                        # Normalize direction vector
                                        unit_dx = dx / magnitude
                                        unit_dy = dy / magnitude

                                        # Set fixed arrow length
                                        arrow_length = 50  # pixels
                                        arrow_end_x = int(curr_cx + unit_dx * arrow_length)
                                        arrow_end_y = int(curr_cy + unit_dy * arrow_length)

                                        # Draw arrow with fixed length in direction of movement
                                        cv2.arrowedLine(frame, (curr_cx, curr_cy), (arrow_end_x, arrow_end_y),
                                                        (0, 0, 255), 3, tipLength=0.5)

                                        # Direction label (textual)
                                        if abs(dx) > abs(dy):
                                            direction = "Right" if dx > 0 else "Left"
                                        else:
                                            direction = "Down" if dy > 0 else "Up"

                                    last_alert_time = directional_arrow_alert_time.get(rtsp_id)
                                    if direction != "Stationary" and (last_alert_time is None or current_time - last_alert_time >= timedelta(seconds=5)):
                                        logger.info(f"Directional arrow detected for person ID {person_id} in stream {rtsp_id}: Direction {direction}")
                                        save_and_send_direction_arrow_data(rtsp_id, frame,person_id,direction,logger=logger)
                                        directional_arrow_alert_time[rtsp_id] = current_time
                                    logger.debug(f"Person ID {person_id} moving {direction} in stream {rtsp_id}")

                        # Loitering detection
                        if "Loitering" in [analytic["type"] for analytic in analytics]:
                            current_time=datetime.now()
                            object_bbox_id=[]

                            if len(person_bbox) == 0:
                                continue
                            if np.array(person_bbox).shape[1] != 5:
                                continue

                            object_bbox_id=tracker.update(np.array(person_bbox))

                            for obj in object_bbox_id:
                                bbox=obj[:4]
                                x1,y1,x2,y2=map(int,bbox)
                                person_id=obj[4]
                                center=((x1+x2)//2,(y1+y2)//2)
                                if person_id not in center_pt_track:
                                    center_pt_track[person_id]=[]
                                center_pt_track[person_id].append(center)
                                
                                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                                text = f"Person:{person_id}"
                                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                                cv2.putText(frame,text,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)


                                if cv2.pointPolygonTest(loit_roi,center,False)>=0:
                                    if person_id not in entry_tym:    
                                        entry_tym[person_id]=current_time
                                    elapsed_time=current_time-entry_tym[person_id]
                                    loit_thresh=timedelta(seconds=loit_threshold)
                                    # If object stays in ROI beyond loitering threshold, save alert
                                    if elapsed_time>=loit_thresh:
                                        current_time=datetime.now()
                                        
                                        if current_time-loiter_alert_time>=timedelta(seconds=5):
                                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),2)
                                            text = f"Person_{person_id}"
                                            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 255), -1)
                                            cv2.putText(frame,text,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

                                            track=center_pt_track.get(person_id,[])
                                            if len(track)>=2:
                                                for i in range(1,len(track)):
                                                    cv2.line(frame,track[i-1],track[i],(0,0,0),2)
                                            for (x,y) in track:
                                                cv2.circle(frame,(x,y),radius=2,color=(0,255,0),thickness=-1)
                                            logger.info(f"Loitering detected for person ID {person_id} in stream {rtsp_id} (time in ROI: {elapsed_time})")
                                            save_and_send_Loitering_data(rtsp_id,frame,person_id,logger=logger)
                                            print("loitering alerts")
                                            loiter_alert_time=current_time
                                            if person_id in center_pt_track:
                                                del center_pt_track[person_id] 
                                            
                                else:
                                    if person_id in entry_tym:
                                        del entry_tym[person_id]
                                    if person_id in center_pt_track:
                                        del center_pt_track[person_id]
                    
                
                        if "crowd_formation" in [analytic["type"] for analytic in analytics]:
                            person_in_roi = 0 
                            
                            for obj in person_bbox:
                                x1_p, y1_p, x2_p, y2_p=obj[:4]
                                point_to_check = (int((x1_p + x2_p)/2), int((y1_p + y2_p ) / 2))
                                is_person_inside_roi = cv2.pointPolygonTest(crowd_formation_roi, point_to_check, False) >= 0 
                                if is_person_inside_roi:
                                    person_in_roi += 1

                            current_time = datetime.now()

                            if person_in_roi > crowd_formation_threshold:
                                # Crowd detected, check if we already started tracking this event
                                if rtsp_id not in crowd_start_times:
                                    # Start tracking this crowd formation
                                    crowd_start_times[rtsp_id] = current_time
                                else:
                                    # Check how long the crowd has been present
                                    time_elapsed = (current_time - crowd_start_times[rtsp_id]).total_seconds()
                                    
                                    # Alert every 5 seconds while the condition persists
                                    if time_elapsed >= crowd_formation_duration:
                                        # Use modulo to trigger alert every 5 seconds
                                        last_alert_at = crowd_formation_alert_time.get(rtsp_id, None)
                                        if last_alert_at is None or (current_time - last_alert_at).total_seconds() >= 5:
                                            print(f"Crowd formation detected at {rtsp_id}")
                                            logger.info(f"Crowd formation detected with {person_in_roi} persons in ROI for {rtsp_id}")
                                            save_frame_and_send_crowd_formation_alert(rtsp_id, frame, person_in_roi,logger=logger)
                                            crowd_formation_alert_time[rtsp_id] = current_time
                            else:
                                # Reset timer if the crowd is gone
                                if rtsp_id in crowd_start_times:
                                    del crowd_start_times[rtsp_id]
                                if rtsp_id in crowd_formation_alert_time:
                                    del crowd_formation_alert_time[rtsp_id]
                        
                        if "crowd_dispersion" in [analytic["type"] for analytic in analytics]:
                            dis_person_in_roi = 0
                            
                            for obj in person_bbox:
                                x1_p, y1_p, x2_p, y2_p=obj[:4]
                                point_to_check = (int((x1_p + x2_p) / 2), int((y1_p + y2_p) / 2))
                                is_person_inside_roi = cv2.pointPolygonTest(crowd_dispersion_roi, point_to_check, False) >= 0
                                if is_person_inside_roi:
                                    dis_person_in_roi += 1

                            current_time = datetime.now()

                            if dis_person_in_roi < crowd_dispersion_threshold:
                                
                                if rtsp_id not in crowd_dispersion_start_time:
                                    crowd_dispersion_start_time[rtsp_id] = current_time

                                # Time since dispersion started
                                time_elapsed = (current_time - crowd_dispersion_start_time[rtsp_id]).total_seconds()

                                # Only allow alerting if crowd has been dispersed for at least crowd_dispersion_duration
                                if time_elapsed >= crowd_dispersion_duration:
                                    last_alert_at = crowd_dispersion_alert_time.get(rtsp_id, None)

                                    # Alert every 5 seconds while condition persists
                                    if last_alert_at is None or (current_time - last_alert_at).total_seconds() >= 5:
                                        logger.info(f"Crowd dispersion detected with {dis_person_in_roi} persons in ROI for {rtsp_id}.")
                                        save_and_send_crowd_dispersion_alert(rtsp_id, frame, dis_person_in_roi,logger=logger)
                                        crowd_dispersion_alert_time[rtsp_id] = current_time

                            else:
                                # Reset if above threshold again
                                if rtsp_id in crowd_dispersion_start_time:
                                    del crowd_dispersion_start_time[rtsp_id]
                                if rtsp_id in crowd_dispersion_alert_time:
                                    del crowd_dispersion_alert_time[rtsp_id]
                        
                        if "crowd_estimation" in [analytic["type"] for analytic in analytics]:
                            crowd_count_in_roi, density_map = helpers.crowd_estimation_with_csrnet(frame, csrnet_model, device)
                            print(f"Crowd Count: {crowd_count_in_roi}")

                            density_map_normalized = (density_map / density_map.max() * 255).astype(np.uint8)
                            density_map_colored = cv2.applyColorMap(density_map_normalized, cv2.COLORMAP_JET)
                            density_map_resized = cv2.resize(density_map_colored, (640, 640))

                            current_time = datetime.now()

                            if crowd_count_in_roi > crowd_estimation_threshold:
                                # Crowd threshold met or exceeded

                                # Start tracking if not already doing so
                                if rtsp_id not in crowd_estimation_start_time:
                                    crowd_estimation_start_time[rtsp_id] = current_time

                                # Time since threshold was crossed
                                time_elapsed = (current_time - crowd_estimation_start_time[rtsp_id]).total_seconds()

                                # Only allow alerting if crowd has been present for at least crowd_estimation_duration
                                if time_elapsed >= crowd_estimation_duration:
                                    last_alert_at = crowd_estimation_alert_time.get(rtsp_id, None)

                                    # Alert every 5 seconds while condition persists
                                    if last_alert_at is None or (current_time - last_alert_at).total_seconds() >= 5:
                                        logger.info(f"Crowd estimation detected with {crowd_count_in_roi} persons in ROI for {rtsp_id}.")
                                        save_and_send_crowd_estimation_alert(rtsp_id, frame, crowd_count_in_roi, density_map_resized,logger=logger)
                                        crowd_estimation_alert_time[rtsp_id] = current_time

                            else:
                                # Reset if below threshold
                                if rtsp_id in crowd_estimation_start_time:
                                    del crowd_estimation_start_time[rtsp_id]
                                if rtsp_id in crowd_estimation_alert_time:
                                    del crowd_estimation_alert_time[rtsp_id]

                        if "fall" in [analytic["type"] for analytic in analytics]:
                            fall_results = fall(frame, iou=0.25, conf=0.5, verbose=False)
                            print("fall detection start")
                            fall_detections = [
                                {"bbox": box.xyxy[0], "class_id": int(box.cls[0])} for box in fall_results[0].boxes
                            ]
                            detection_in_roi = False
            
                            for detection in fall_detections:
                                x1, y1, x2, y2 = map(int, detection["bbox"])
                                class_id = detection["class_id"]
                                class_name = FALL_CLASS_NAMES.get(str(class_id), f"Class {class_id}")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for fall
                                text = f"{class_name}"
                                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                point_to_check = ((x1 + x2) // 2, (y1 + y2) // 2)
                                is_inside_roi = cv2.pointPolygonTest(fall_roi, point_to_check, False) >= 0
                                if is_inside_roi:
                                    detection_in_roi = True
                        
                                    break
                            current_time = datetime.now()
                            if detection_in_roi:
                                consecutive_fall_count[rtsp_id] += 1
                                if consecutive_fall_count[rtsp_id] >= 2:
                                    if current_time - fall_alert_time[rtsp_id] >= timedelta(seconds=5):
                                        logger.info(f"Fall detected for 4 consecutive frames in stream {rtsp_id}")
                                        save_and_send_Fall_data(rtsp_id, frame ,logger=logger)
                                        fall_alert_time[rtsp_id] = current_time
                                        consecutive_fall_count[rtsp_id] = 0
                            else:
                                consecutive_fall_count[rtsp_id] = 0
                                                
                        if "fireandsmoke" in [analytic["type"] for analytic in analytics]:
                           
                            fire_smoke_results = fireNsmoke(frame, iou=0.25, conf=0.5, verbose=False)
                            fire_smoke_detections = [
                                {"bbox": box.xyxy[0], "class_id": int(box.cls[0])} for box in fire_smoke_results[0].boxes
                            ]
                            
                            # Check if any fire/smoke detection is within ROI
                            detection_in_roi = False
                            last_class_name = None  # Track the class name for the alert
                            for detection in fire_smoke_detections:
                                x1, y1, x2, y2 = map(int, detection["bbox"])
                                class_id = detection["class_id"]
                                class_name = FIRE_SMOKE_CLASS_NAMES.get(str(class_id), f"Class {class_id}")

                                # Draw bounding box and label
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)  # Orange color for fire/smoke
                                text = f"{class_name}"
                                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                                # Check if detection is inside the ROI
                                point_to_check = ((x1 + x2) // 2, (y1 + y2) // 2)
                                is_inside_roi = cv2.pointPolygonTest(fireandsmoke_roi, point_to_check, False) >= 0

                                if is_inside_roi:
                                    detection_in_roi = True
                                    last_class_name = class_name 
                                    break  
                            
                            current_time = datetime.now()
                            if detection_in_roi:
                                consecutive_fire_smoke_count += 1
                                
                                # Check if 4 consecutive detections have occurred
                                if consecutive_fire_smoke_count >= 4:
                                    if current_time - fireandsmoke_alert_time[rtsp_id] >= timedelta(seconds=5):
                                        logger.info(f"{last_class_name} detected for 4 consecutive frames in stream {rtsp_id}")
                                        save_and_send_FireAndSmoke_data(rtsp_id, frame, last_class_name, logger=logger)
                                        fireandsmoke_alert_time[rtsp_id] = current_time
                                        consecutive_fire_smoke_count = 0 
                            else:
                                consecutive_fire_smoke_count = 0


                        if "waiting_time_in_roi" in [analytic["type"] for analytic in analytics]:
                            wait_time_dets = []
                            for box in yolo_results[0].boxes:
                                class_id = int(box.cls[0].cpu())
                                if class_id == 0:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                    score = float(box.conf[0].cpu().numpy())
                                    wait_time_dets.append([x1, y1, x2, y2, score])
                            
                            wait_time_dets = np.array(wait_time_dets, dtype=np.float32) if len(wait_time_dets) > 0 else np.empty((0, 5), dtype=np.float32)
                            tracked_objects = waiting_time_tracker.update(wait_time_dets)
                            current_time = datetime.now()
                            
                            for obj in tracked_objects:
                                x1, y1, x2, y2, person_id = map(int, obj)
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                
                                is_inside_roi = cv2.pointPolygonTest(waiting_time_roi, (cx, cy), False) >= 0
                                
                                if is_inside_roi:
                                    if person_id not in waiting_time_entry:
                                        waiting_time_entry[person_id] = current_time
                                    
                                    time_in_roi = (current_time - waiting_time_entry[person_id]).total_seconds()
                                    
                                    text = f"ID: {person_id}, Time: {round(time_in_roi, 1)}s"
                                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                else:
                                    if person_id in waiting_time_entry:
                                        time_in_roi = (current_time - waiting_time_entry[person_id]).total_seconds()
                                        
                                        text = f"ID: {person_id}, Time: {round(time_in_roi, 1)}s"
                                        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 0), -1)
                                        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                        
                                        if current_time - waiting_time_alert_time.get(rtsp_id, datetime.min) >= timedelta(seconds=5):
                                            logger.info(f"Person ID {person_id} exited ROI in stream {rtsp_id} after {round(time_in_roi, 2)} seconds")
                                            save_and_send_waiting_time_alert(rtsp_id, frame, person_id, time_in_roi,logger=logger)
                                            waiting_time_alert_time[rtsp_id] = current_time
                                        
                                        del waiting_time_entry[person_id]

                        if "person_waving_hand" in [analytic["type"] for analytic in analytics]:
                            pose_results = pose_model(frame,conf=0.3,iou=0.3, verbose=False)

                            # Prepare detections for SORT
                            wave_dets = []
                            for result in pose_results:
                                keypoints = result.keypoints.xy.cpu().numpy()
                                boxes = result.boxes.xyxy.cpu().numpy()

                                for idx, (keypoint, box) in enumerate(zip(keypoints, boxes)):
                                    x1, y1, x2, y2 = map(int, box)
                                    conf = result.boxes.conf[idx].cpu().numpy() if result.boxes.conf is not None else 0.5
                                    wave_dets.append([x1, y1, x2, y2, conf])
                            
                            wave_dets = np.array(wave_dets, dtype=np.float32) if len(wave_dets) > 0 else np.empty((0, 5), dtype=np.float32)
                            # Update SORT tracker with new detections
                            tracked_objects = waving_tracker.update(wave_dets)

                            # Loop through tracked objects
                            for obj in tracked_objects:
                                x1, y1, x2, y2, person_id = map(int, obj)

                                # Find matching keypoint for this tracked object
                                matched_keypoint = None
                                for result in pose_results:
                                    keypoints_list = result.keypoints.xy.cpu().numpy()
                                    boxes_list = result.boxes.xyxy.cpu().numpy()
                                    for kp, box in zip(keypoints_list, boxes_list):
                                        kpx1, kpy1, kpx2, kpy2 = map(int, box)
                                        if abs(kpx1 - x1) < 10 and abs(kpy1 - y1) < 10:  # simple IoU proxy
                                            matched_keypoint = kp
                                            break
                                    if matched_keypoint is not None:
                                        break

                                if matched_keypoint is None:
                                    continue

                                left_shoulder = matched_keypoint[5] if len(matched_keypoint) > 5 else None
                                right_shoulder = matched_keypoint[6] if len(matched_keypoint) > 6 else None
                                left_elbow = matched_keypoint[7] if len(matched_keypoint) > 7 else None
                                right_elbow = matched_keypoint[8] if len(matched_keypoint) > 8 else None
                                left_wrist = matched_keypoint[9] if len(matched_keypoint) > 9 else None
                                right_wrist = matched_keypoint[10] if len(matched_keypoint) > 10 else None

                                # Draw Bounding Box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # Draw Person ID Text
                                id_text = f"ID: {person_id}"
                                (text_width, text_height), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width + 10, y1), (0, 255, 0), -1)
                                cv2.putText(frame, id_text, (x1 + 5, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                                # Draw Keypoints
                                for pt in [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]:
                                    if pt is not None:
                                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

                                # ROI Check
                                point_to_check = ((x1 + x2) // 2, (y1 + y2) // 2)
                                in_waving_roi = cv2.pointPolygonTest(waving_roi, point_to_check, False) >= 0
                                if not in_waving_roi:
                                    continue

                                # Initialize history for this person
                                if person_id not in waving_keypoint_history:
                                    waving_keypoint_history[person_id] = {
                                        'positions': [],
                                        'start_time': None,
                                        'alert_sent': False
                                    }

                                wrist_positions = []
                                if left_wrist is not None:
                                    wrist_positions.append(left_wrist[0])
                                if right_wrist is not None:
                                    wrist_positions.append(right_wrist[0])

                                if wrist_positions:
                                    waving_keypoint_history[person_id]['positions'].extend(wrist_positions)
                                    waving_keypoint_history[person_id]['positions'] = waving_keypoint_history[person_id]['positions'][-30:]

                                positions = waving_keypoint_history[person_id]['positions']
                                if len(positions) >= 20:
                                    movement_range = max(positions) - min(positions)
                                    elbow_above = False
                                    if left_elbow is not None and left_shoulder is not None and left_elbow[1] < left_shoulder[1]:
                                        elbow_above = True
                                    elif right_elbow is not None and right_shoulder is not None and right_elbow[1] < right_shoulder[1]:
                                        elbow_above = True

                                    if movement_range > 50 and elbow_above:
                                        if waving_keypoint_history[person_id]['start_time'] is None:
                                            waving_keypoint_history[person_id]['start_time'] = time.time()

                                        if time.time() - waving_keypoint_history[person_id]['start_time'] >= 2 and not waving_keypoint_history[person_id]['alert_sent']:
                                            current_time = datetime.now()
                                            if current_time - waving_alert_time.get(person_id, datetime.min) >= timedelta(seconds=5):
                                                logger.info(f"Person Id {person_id} waving hand in stream {rtsp_id}")
                                                save_and_send_waving_hand_data(rtsp_id, frame, person_id,logger=logger)
                                                waving_alert_time[person_id] = current_time
                                                waving_keypoint_history[person_id]['alert_sent'] = True
                                                waving_keypoint_history[person_id]['positions'] = []  # Reset after alert
                                                waving_keypoint_history[person_id]['start_time'] = None
                                    else:
                                        waving_keypoint_history[person_id]['start_time'] = None

                        if "wrong_direction" in [analytic["type"] for analytic in analytics]:
                            expected_direction = metadata.get("direction", "Left to Right")
                            if len(person_bbox) == 0:
                                continue
                            if np.array(person_bbox).shape[1] != 5:
                                continue
                            
                            tracked_objects = wrong_direction_tracker.update(np.array(person_bbox))
                            
                            current_time = datetime.now()
                            for obj in tracked_objects:
                                bbox=obj[:4]
                                x1,y1,x2,y2=map(int,bbox)
                                person_id=obj[4]
                                cx = (x1 + x2) // 2
                                point_to_check = (cx, (y1 + y2) // 2)
                                is_inside_roi = cv2.pointPolygonTest(wrong_direction_roi, point_to_check, False) >= 0
                                
                                if not is_inside_roi:
                                    continue
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                cv2.putText(frame, f'ID: {person_id}', (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                if person_id not in wrong_direction_history:
                                    wrong_direction_history[person_id] = deque(maxlen=30)

                                wrong_direction_history[person_id].append(cx)
                                
                                detected_direction = "Stationary"
                                if len(wrong_direction_history[person_id]) >= 10:
                                    movement = wrong_direction_history[person_id][-1] - wrong_direction_history[person_id][0]
                                    detected_direction = "Right to Left" if movement < 0 else "Left to Right"
                                    
                                    if detected_direction != expected_direction:
                                        
                                        current_time = datetime.now()
                                        if current_time - wrong_direction_alert_time[rtsp_id] >= timedelta(seconds=5):
                                            logger.info(f"Wrong direction detected for person ID {person_id} in stream {rtsp_id}: Expected {expected_direction}, Detected {detected_direction}")
                                            save_and_send_wrong_direction_data(rtsp_id, frame, person_id, detected_direction,logger=logger)
                                            wrong_direction_alert_time[rtsp_id] = current_time

                                direction_text = f"Dir: {detected_direction}"
                                (text_w, text_h), _ = cv2.getTextSize(direction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(frame, (x1, y1 - text_h - 25), (x1 + text_w, y1 - 15), (0, 0, 0), -1)
                                cv2.putText(frame, direction_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        if "person_in_out_count" in [analytic["type"] for analytic in analytics]:
                            # Get analytics data
                            in_out_analytic = next((a for a in analytics if a["type"] == "person_in_out_count"), None)

                            if in_out_analytic:
                                entry_line_points = np.array(in_out_analytic.get("entry_line", []), dtype=np.int32)
                                exit_line_points = np.array(in_out_analytic.get("exit_line", []), dtype=np.int32)

                                # Get line types from metadata
                                entry_line_type = metadata.get("entry_line_type", "horizontal")  # From stream metadata
                                exit_line_type = metadata.get("exit_line_type", "horizontal")      # From stream metadata

                                # Compute reference values based on line type
                                if entry_line_type == 'horizontal':
                                    entry_ref = (entry_line_points[0][1] + entry_line_points[1][1]) // 2
                                    cv2.line(frame, (0, entry_ref), (frame.shape[1], entry_ref), (0, 255, 0), 2)
                                elif entry_line_type == 'vertical':
                                    entry_ref = (entry_line_points[0][0] + entry_line_points[1][0]) // 2
                                    cv2.line(frame, (entry_ref, 0), (entry_ref, frame.shape[0]), (0, 255, 0), 2)

                                if exit_line_type == 'horizontal':
                                    exit_ref = (exit_line_points[0][1] + exit_line_points[1][1]) // 2
                                    cv2.line(frame, (0, exit_ref), (frame.shape[1], exit_ref), (0, 0, 255), 2)
                                elif exit_line_type == 'vertical':
                                    exit_ref = (exit_line_points[0][0] + exit_line_points[1][0]) // 2
                                    cv2.line(frame, (exit_ref, 0), (exit_ref, frame.shape[0]), (0, 0, 255), 2)

                                # Prepare detection boxes for SORT
                                if len(person_bbox) == 0:
                                    continue
                                if np.array(person_bbox).shape[1] != 5:
                                    continue
                                
                                # Update SORT tracker
                                tracked_objects = in_out_tracker.update(np.array(person_bbox))

                                # Loop over tracked objects
                                for obj in tracked_objects:
                                    bbox=obj[:4]
                                    x1,y1,x2,y2=map(int,bbox)
                                    person_id=obj[4]
                                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point

                                    # Initialize state
                                    if person_id not in person_last_states:
                                        person_last_states[person_id] = {
                                            'crossed_entry': False,
                                            'crossed_exit': False
                                        }

                                    # Entry Detection
                                    if entry_line_type == 'horizontal' and cy < entry_ref and not person_last_states[person_id]['crossed_entry']:
                                        entry_count += 1
                                        person_last_states[person_id]['crossed_entry'] = True
                                    elif entry_line_type == 'vertical' and cx < entry_ref and not person_last_states[person_id]['crossed_entry']:
                                        entry_count += 1
                                        person_last_states[person_id]['crossed_entry'] = True

                                    # Exit Detection
                                    if exit_line_type == 'horizontal' and cy < exit_ref and not person_last_states[person_id]['crossed_exit']:
                                        exit_count += 1
                                        person_last_states[person_id]['crossed_exit'] = True
                                    elif exit_line_type == 'vertical' and cx < exit_ref and not person_last_states[person_id]['crossed_exit']:
                                        exit_count += 1
                                        person_last_states[person_id]['crossed_exit'] = True

                                    # Draw bounding box and ID
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                                    cv2.putText(frame, f'ID: {person_id}', (x1, y1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                # Draw count overlay
                                count_text = f"Entered: {entry_count} | Exited: {exit_count}"
                                cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                                # Optional: Send alert when counts change
                                if entry_count > 0 or exit_count > 0:
                                    print(f"Entry: {entry_count}, Exit: {exit_count}")
                                    logger.info(f"Person in out for Person_ID {person_id} in stream {rtsp_id}")
                                    save_and_send_in_out_data(rtsp_id,frame,entry_count,exit_count,logger=logger)

                        # cv2.polylines(frame, [crowd_formation_roi], isClosed=True, color=(0,255,255), thickness=2)
                        # cv2.imshow(f"{rtsp_id}",frame)
                        # if cv2.waitKey(1)==13:
                        #     cv2.destroyAllWindows()
                        #     break
                    except Exception as e:
                        print(f"error in detection is {e}")
                        
                    finally:
                        buff.unmap(map_info)
        except Exception as e:
            print(f"error processing frame  for stream {rtsp_id} is {e}")
            time.sleep(3)
        finally:
            print(f"Resetting pipeline to NULL state.")
            pipeline.set_state(Gst.State.NULL)
            del pipeline_dict[rtsp_id]
            cv2.destroyWindow(f"stream {rtsp_id}")
            continue
    print("**cleaning the resouces.**")
    
    del tracker
    del dir_tracker
    del wrong_direction_tracker,waving_tracker,waiting_time_tracker
    buff.unmap(map_info)
    

def zmq_listener(streams,analytics_dict,stream_metadata,executor):
    context=zmq.Context()
    listener_socket=context.socket(zmq.REP)
    listener_socket.bind(config["zmq"]["subscriber"]["address"])
    
    while True:
        try:
            response={"status":"error","message":"server error"}
            print("listener thread start.......-->>")

            message=listener_socket.recv_json()
            print(f"Recieved Json: {message}")
            action=message.get("action")
            data_list=message.get("data",[])

            with lock:
                if action=="add_device":
                    for data in data_list:
                        rtsp_id = data.get("id")
                        rtsp_url = data.get("url")
                        analytics = data.get("analytics",[])
                        name = data.get("name",f"Stream {rtsp_id}")
                        fps = data.get("fps",3)
                        username = data.get("username","")
                        password = data.get("password","")
                        loitering_threshold = data.get("loitering_threshold",30)
                        crowd_formation_threshold = data.get("crowd_formation_threshold",5)
                        crowd_formation_duration = data.get("crowd_formation_duration",10)
                        crowd_estimation_threshold = data.get("crowd_estimation_threshold",15)
                        crowd_estimation_duration = data.get("crowd_estimation_duration",10)
                        entry_line_type = data.get("entry_line_type")
                        exit_line_type = data.get("exit_line_type")
                        direction = data.get("direction")
                        crowd_dispersion_threshold = data.get("crowd_dispersion_threshold")
                        crowd_dispersion_duration = data.get("crowd_dispersion_duration")

                        if rtsp_id in streams:
                            print("already present")
                            response['status']='error'
                            response['message']=f"Stream with ID {rtsp_id} already present."
                        else:
                            streams[rtsp_id]=rtsp_url
                            analytics_dict[rtsp_id]=analytic
                            stream_metadata[rtsp_id]={
                                "name": name,
                                "fps": fps,
                                "username": username,
                                "password": password,
                                "loitering_threshold" : loitering_threshold,
                                "crowd_formation_threshold" : crowd_formation_threshold,
                                "crowd_formation_duration" : crowd_formation_duration,
                                "crowd_estimation_threshold": crowd_estimation_threshold,
                                "crowd_estimation_duration" : crowd_estimation_duration,
                                "entry_line_type" : entry_line_type,
                                "exit_line_type" : exit_line_type,
                                "direction" : direction, 
                                "crowd_dispersion_threshold" : crowd_dispersion_threshold,
                                "crowd_dispersion_duration" : crowd_dispersion_duration
                            }
                            stream_flag[rtsp_id]=True
                            print(f"stream with id {rtsp_id} thread starts ")

                            try:
                                start_stream(rtsp_id,rtsp_url,analytics_dict.get(rtsp_id,{}),stream_metadata.get(rtsp_id,{}),executor)
                                print(f"stream {rtsp_id} task submitted")
                                response['status']="success"
                                response['message']=f"Stream {rtsp_id} task submitted"
                            except Exception as e:
                                print(f"error in submitting stream {rtsp_id} is {e}")
                                response['status']="error"
                                response['message']=f"Error in submitting stream {rtsp_id} is {e}"
                            save_data_to_json('stream.json',streams,analytics_dict,stream_metadata)

                elif action=="delete_device":
                    print("entering into delete device code")

                    for data in data_list:
                        rtsp_id=int(data.get('id'))
                        print(f"i want to delete {rtsp_id}")
                        if rtsp_id in streams:
                            del streams[rtsp_id]
                            del analytics_dict[rtsp_id]
                            del stream_metadata[rtsp_id]

                            stop_stream(rtsp_id)

                            save_data_to_json('stream.json',streams,analytics_dict,stream_metadata)
                            print(f"Stream {rtsp_id} removed")
                            response['status']="success"
                            response['message']=f"Stream {rtsp_id} removed"
                        else:
                            print(f"no stream with id {rtsp_id}")
                            response['status']='error'
                            response['message']=f"no stream with id {rtsp_id}"
                
                elif action=="update_device":
                    for data in data_list:
                        rtsp_id=int(data.get('id'))
                        print(f"Updating the stream with id {rtsp_id}")

                        if rtsp_id in streams:
                            del streams[rtsp_id]
                            del analytics_dict[rtsp_id]
                            del stream_metadata[rtsp_id]

                            stop_stream(rtsp_id)
                            
                            rtsp_id=data.get('id')
                            rtsp_url=data.get('url')
                            name=data.get('name')
                            username=data.get('username')
                            password=data.get('password')
                            fps=data.get('fps')
                            analytic=data.get('analytics',[])
                            
                            streams[rtsp_id]=rtsp_url
                            analytics_dict[rtsp_id]=analytic
                            stream_metadata[rtsp_id]={
                                'name':name,
                                'username':username,
                                'password':password,
                                'fps':fps
                            }
                            stream_flag[rtsp_id]=True

                            save_data_to_json('stream.json',streams,analytics_dict,stream_metadata)

                            try:
                                print(f"starting updated stream {rtsp_id}")
                                start_stream(rtsp_id,rtsp_url,analytics_dict,stream_metadata.get(rtsp_id,{}),executor)
                                print(f"stream with id {rtsp_id} gets updated and restarted")
                                response['status']='success'
                                response['message']=f"stream with id {rtsp_id} gets updated and restarted"
                            except Exception as e:
                                print(f"error in restarting stream {rtsp_id}")
                                response['status']='error'
                                response['message']=f"error in restarting stream {rtsp_id}"

                elif action=="delete_all":
                    for rtsp_id in streams:
                        stop_stream(rtsp_id)
                        print(f"stopping stream with id :{rtsp_id}")

                    streams.clear()
                    analytics_dict.clear()
                    stream_metadata.clear()
                    
                    save_data_to_json('stream.json',streams,analytics_dict,stream_metadata)  

                    response['status']="success"
                    response['message']=f"All streams have been stopped and deleted from stream.json"
                    print(f"All streams have been stopped and deleted from stream.json")

                elif action=="add_analytic":
                    
                    for data in data_list:
                        rtsp_id =int(data.get('id'))
                        analytics=data.get('analytics',[])
                        
                        if not isinstance(analytics, list):
                            print(f"Invalid analytics format for {rtsp_id}")
                            continue

                        if rtsp_id in analytics_dict:
                            for new in analytics:
                                if not isinstance(new, dict):
                                    print(f"Invalid analytic entry for stream {rtsp_id}: {new}")
                                    continue
                                analytic_type=new.get("type")
                                
                                existing_analytics=analytics_dict[rtsp_id]
                                if any(analytic.get("type")==analytic_type for analytic in existing_analytics):
                                    response['status']="error"
                                    response['message']=f"For Stream {rtsp_id} , {analytic_type} already present"
                                    print(f"For Stream {rtsp_id},{analytic_type} already present")
                                    
                                else:
                                    existing_analytics.append(new)
                                    
                                    analytics_dict[rtsp_id]=existing_analytics

                            stop_stream(rtsp_id)

                            print(f"stopping stream {rtsp_id}")

                            analytics_dict[rtsp_id]=existing_analytics
                        
                            save_data_to_json("stream.json",streams,analytics_dict,stream_metadata)

                            try:
                                start_stream(rtsp_id,streams[rtsp_id],analytics_dict.get(rtsp_id,{}),stream_metadata.get(rtsp_id,{}),executor)
                                response['status']="error"
                                response['message']=f"starting stream with id:{rtsp_id} with added analytic"
                                print(f"starting stream with id:{rtsp_id} with added analytic")
                                
                            except Exception as e:
                                response['status']='error'
                                response['message']=f"error in starting {rtsp_id} is {e} with added analytic"
                                print(f"error in starting {rtsp_id} is {e} with added analytic")
                        else:
                            response['status']='error'
                            response['message']=f"no stream with id {rtsp_id} for adding analytic"
                            print(f"no stream with id {rtsp_id} for adding analytic")
                    

                elif action=="delete_analytic":
                    for data in data_list:
                        rtsp_id=int(data.get('id'))
                        analytic_type=data.get('analytic_type')

                        if rtsp_id in streams:
                            existing_analytics=analytics_dict.get(rtsp_id)
                            print(existing_analytics)
                            found=False
                            for analytic in existing_analytics:
                                if analytic.get("type")==analytic_type:
                                    existing_analytics.remove(analytic)
                                    found=True
                                    
                                    print(f"removing {analytic_type} analytics of stream {rtsp_id}")
                                    break

                            if found:
                                stop_stream(rtsp_id)
                                print(f"stopping stream with id {rtsp_id}")

                                analytics_dict[rtsp_id]=existing_analytics
                                
                                save_data_to_json("stream.json",streams,analytics_dict,stream_metadata)

                                try:
                                    start_stream(rtsp_id,streams[rtsp_id],analytics_dict.get(rtsp_id,{}),stream_metadata.get(rtsp_id,{}),executor)
                                    response['status']="success"
                                    response['message']=f"starting stream {rtsp_id} after deleting analytics"
                                    print(f"starting stream {rtsp_id} after deleting analytics")
                                except Exception as e:
                                    response['status']='error'
                                    response['message']=f"exception in starting {rtsp_id} is {e} after deleting analytics"
                                    print(f"exception in starting {rtsp_id} is {e} after deleting analytics")

                            else:
                                response['status']='error'
                                response['message']=f"analytic {analytic_type} is not present in stream {rtsp_id}"
                                print(f"analytic {analytic_type} is not present in stream {rtsp_id}")
                        
                        else:
                            response['status']='error'
                            response['message']=f"stream {rtsp_id} not found for deleting analytics"
                            print(f"stream {rtsp_id} not found for deleting analytics")
                    
                elif action=="update_analytic":

                    for data in data_list:
                        rtsp_id=int(data.get('id'))
                        update_analytics=data.get('analytics',[])

                        if rtsp_id in streams:
                            existing_analytics=analytics_dict.get(rtsp_id,[])
                            
                            updated_analytic=False
                            for analytic in update_analytics:
                                updated_type=analytic.get("type")
                                updated_roi=analytic.get("roi")
                                for existing_ai in existing_analytics:
                                    if existing_ai.get("type")==updated_type:
                                        existing_ai["roi"]=updated_roi
                                        updated_analytic=True
                                        print(f"updating {updated_type} analytics of stream {rtsp_id}")
                                        break

                            if updated_analytic:
                                stop_stream(rtsp_id)
                                print(f"stopping stream with id {rtsp_id}")

                                analytics_dict[rtsp_id]=existing_analytics

                                save_data_to_json("stream.json",streams,analytics_dict,stream_metadata)

                                try:
                                    start_stream(rtsp_id,streams[rtsp_id],analytics_dict.get(rtsp_id,{}),stream_metadata.get(rtsp_id,{}),executor)
                                    print(f"starting stream {rtsp_id} with updated analytics")
                                    response['status']="success"
                                    response['message']=f"starting stream {rtsp_id} with updated analytics"
                                except Exception as e:
                                    print(f"exception in starting {rtsp_id} is {e} after updating analytics")
                                    response['status']="error"
                                    response['message']=f"exception in starting {rtsp_id} is {e} after updating analytics"

                            else:
                                response['status']="error"
                                response['message']=f"Analytics of type(s) {', '.join([analytic.get('type') for analytic in update_analytics])} not found for stream {rtsp_id}. Cannot update."
                                print(f"Analytics of type(s) {', '.join([analytic.get('type') for analytic in update_analytics])} not found for stream {rtsp_id}. Cannot update.")
                        
                        else:
                            response['status']='error'
                            response['message']=f"stream {rtsp_id} not found.Cannot update"
                            print(f"stream {rtsp_id} not found.Cannot update")

                elif action=="get_all":
                    try:
                        all_streams_data=[]

                        for rtsp_id in streams:
                            stream_data={
                                "id":rtsp_id,
                                "url":streams[rtsp_id],
                                "analytics":analytics_dict.get(rtsp_id,[]),
                                "metadata":stream_metadata.get(rtsp_id,[])
                            }
                            all_streams_data.append(stream_data)

                        response['status']="success"
                        response['message']="All streams data retrieved successfully"
                        response['data']=all_streams_data
                        print("sent all stream data")

                    except Exception as e:
                        response['status']='error'
                        response['message']=f"error in retriving all streams data {e}"
                        print(f"error in retriving all streams data {e}")
        
        except Exception as e:
            response = {"status": "error", "message": f"Internal server error: {e}"}
            print(f"error in processing listener {e}")
        print("response sent***")
        listener_socket.send_json(response)
        

def main():
    try:
        streams,analytics_dict,stream_metadata=load_stream_json_file('stream.json')
        print(streams)
        
        with ThreadPoolExecutor(max_workers=len(streams)+20) as executor:
            for rtsp_id,rtsp_url in streams.items():
                print(f"stream {rtsp_id} starts..........**")
                analytics_data=analytics_dict.get(rtsp_id,[])
                metadata=stream_metadata.get(rtsp_id,{})
                start_stream(rtsp_id,rtsp_url,analytics_data,metadata,executor)

            # listener_thread=threading.Thread(target=zmq_listener, args=(streams,analytics_dict,stream_metadata,executor))
            # listener_thread.start()
            # listener_thread.join()

    except Exception as e:
        print(f"Error in main function: {e}")


if __name__=="__main__":
    main()