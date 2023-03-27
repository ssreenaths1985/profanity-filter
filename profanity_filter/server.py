from concurrent.futures import process
from shutil import Error
from flask import Flask,jsonify, request
from services.text_profanity_service import text_profanity_svc
from services.retraining_service import retraining
from flask_socketio import SocketIO
from flask_socketio import send, emit
from kafka import KafkaConsumer
import threading
from multiprocessing import Process
from json import loads, dumps
from kafka import KafkaProducer
import logging
import os

app = Flask(__name__)
socketio = SocketIO(app)

text_profanity_svc_obj = text_profanity_svc()
retraining_obj         = retraining()
cfg = loads(open('config/config.json').read())

@app.route('/checkProfanity', methods=['POST'])
def checkProfanity():
    try:
        response = text_profanity_svc_obj.infer(request.json)
    except Error as e:
        print(e)
        return {'code' : 500,'message' : str(e.message)} 
    print(response)
    return response

@app.route('/addProfaneWords',methods=['POST'])
def addProfaneWords():
    retraining_obj.add_words(request.json, text_profanity_svc_obj)
    return {'code' : 200, 'message' : 'success'}

@app.route('/addProfaneText',methods=['POST'])
def addProfaneText():
    retraining_obj.add_text(request.json, text_profanity_svc_obj)
    return {'code' : 200, 'message' : 'success'}

@app.route('/initiateModelTraining',methods=['POST'])
def initiateModelTraining():
    retraining_obj.train_model(text_profanity_svc_obj)
    return {'code' : 200, 'message' : 'success'}

@socketio.on('message')
def connection_event(sid, data):
    print('data received', sid)
    print(data)
    result = []
    data = str(data).strip()
    response = text_profanity_svc_obj.infer({'text' : data})
    send('processed_data', {'response': str(response)})
    print(result)
    pass

def consumer():
    #set configuration
    kafka_server = cfg["kafka_bootstrap_servers"]
    kafka_moderated_topic = cfg["kafka_moderated_topic"]
    kafka_flagged_topic = cfg["kafka_flagged_topic"]
    if os.environ.get('kafka_bootstrap_servers') is not None:
        kafka_server =  os.environ.get('kafka_bootstrap_servers')
    if os.environ.get('kafka_moderated_topic') is not None:
        kafka_moderated_topic = os.environ.get('kafka_moderated_topic')
    if os.environ.get('kafka_flagged_topic') is not None:
        kafka_flagged_topic = os.environ.get('kafka_flagged_topic')
    print("kafka server:" + kafka_server)
    print("moderation topic outgoing:" + kafka_moderated_topic)
    print("moderation topic incoming:" + kafka_flagged_topic)
    producer = KafkaProducer(value_serializer=lambda m: dumps(m).encode('ascii'),bootstrap_servers=[kafka_server])
    try:
        consumer = KafkaConsumer(
            kafka_flagged_topic,
            bootstrap_servers=[kafka_server],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='AI-Moderation',
            value_deserializer=lambda x: loads(x.decode('utf-8')))
        consumer.poll(timeout_ms=6000)
        print("listening to kafka messages")
        for msg in consumer:
            try:
                print("consumer received message for moderation")
                print('{}'.format(msg.value))
                consumer.commit(offsets=None)
                response = text_profanity_svc_obj.infer(msg.value)
                del response['possible_profanity_frequency']
                del response['performance']
                del response['code']
                del response['message']
                producer.send(kafka_moderated_topic, {'key' : msg.value['key'], 'payload' : response })
                print("moderated message sent to moderated.ai topic")
                print({'key' : msg.value['key'], 'payload' : response })
            except Exception as e:
                print("message handling failed")
                print(e)
    except Exception as e:
        print("kafka consumer failed")
        print(e)

#kafka_thread = threading.Thread(target=consumer)
kafka_process = Process(target=consumer)
if __name__ == "__main__":
    #app.run()
    #kafka_thread.start()
    kafka_process.start()
    print("kafka thread running, starting REST and socket server")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
