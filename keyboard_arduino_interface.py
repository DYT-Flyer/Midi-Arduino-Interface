import pygame
from pygame.locals import *
import math
from pygame import midi as pygame_midi
from pygame.midi import midis2events
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
import tensorflow as tf
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
import matlab

try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass 

class midi_classifier:
    def __init__(self, steps_per_second, number_of_notes, sample, modelpath,eng):
        pygame.init()
        pygame.font.init()
        pygame_midi.init()
        self.model = tf.keras.models.load_model(modelpath)
        self.midi_input_id = pygame_midi.get_default_input_id()
        self.midi_input = pygame_midi.Input(self.midi_input_id)
        self.sample = sample
        self.control = 1
        self.number_of_notes = number_of_notes
        self.steps_per_second = steps_per_second
        self.events = []
        self.eventpairs = []
        self.count = 0
        self.eng = eng
        self.pos = np.zeros([number_of_notes*steps_per_second*sample,2])
        self.c = 1
        
        count = 0
        for i in range(0,number_of_notes):
            for j in range(0,steps_per_second*sample):
                self.pos[count,0] = i
                self.pos[count,1] = j
                count += 1
        
        self.eng.init_figure(nargout=0)
        
        self.x = np
        sched = BackgroundScheduler()
        sched.add_job(self.update_piano_roll, 'interval', seconds=sample)
        sched.start()
        
        self.start_midi()
        sched.shutdown()
            
    def start_midi(self):
        while True:
            if not self.midi_input.poll():
                continue
            
            events = midis2events(self.midi_input.read(40), self.midi_input_id)
            for event in events:
                if self.control == 1:
                    self.control = 0
                    self.tbegin = event.timestamp
                self.events.append(event)
        
    def extract_events(self):
        #List is sequential
        #Thus: if first item in list has status: 128, skip
        #If Last item in list has status 144, skip
        
        eventpair = []
        eventpairs = []
        
        #Pairing begining and end notes
        control = 1
        
        while len(self.events)>0:
            #Skip if first note is remainder from previous
            current_event = self.events.pop(0)
            current_event = current_event
            if current_event.status == 128 and control == 1:
                continue            
            control = 0
            current_note = current_event.data1
            
            # matches note begin to note end
            for i in range(0,len(self.events)):
                if self.events[i].data1 == current_note:
                    eventpair.append(current_event)
                    eventpair.append(self.events.pop(i))
                    eventpairs.append(eventpair)
                    eventpair = []
                    break
                
        self.eventpairs = eventpairs

    def extract_time(self):
        self.upr = []
        for i in range(0,len(self.eventpairs)):
            if i == 0:
                self.tbegin = self.eventpairs[i][0].timestamp
                
            ts = self.eventpairs[i][0].timestamp
            te = self.eventpairs[i][1].timestamp
            current_note = self.eventpairs[i][0].data1
            tint = ((te - ts)/1000)*self.steps_per_second
            tc = ((ts-self.tbegin)/1000)*self.steps_per_second
            self.upr.append([current_note-48, math.floor(tc),math.ceil(tc+tint)])
    
         
    def update_piano_roll(self):
        self.extract_events()
        self.extract_time()
        self.piano_roll = np.zeros([self.number_of_notes,self.steps_per_second*self.sample])
        for note in self.upr:
            try:
                self.piano_roll[note[0],note[1]-self.count*self.steps_per_second:note[2]-self.count*self.steps_per_second] = 1
            except:
                print('over')
        
        self.piano_roll = np.expand_dims(self.piano_roll,axis=0)
        self.piano_roll = np.expand_dims(self.piano_roll,axis=3)
        pred = self.model.predict(self.piano_roll)
        temp = np.squeeze(self.piano_roll)
        
        
        eng.workspace['pr'] = matlab.double(temp.tolist())
        eng.update_figure(nargout=0)
        
        if sum(temp[1,:])>0:
            eng.turn_white_on(nargout=0)
        
        
        if sum(temp[3,:])>0:    
            eng.turn_green_on(nargout=0)
        
        if sum(temp[6,:])>0:
            eng.turn_red_on(nargout=0)
        
        if sum(temp[8,:])>0:
            eng.turn_blue_on(nargout=0)
        
        if sum(temp[10,:])>0:
            eng.turn_yellow_on(nargout=0)
        
        eng.update_title(nargout=0)
        self.count += 1

import matlab.engine

path = os.cwd() + '/model.hdf5'
future = matlab.engine.start_matlab(background=True)
eng = future.result()
eng.addpath(os.cwd()+'/matlab_code/')
eng.arduino_main(nargout=0)
midi = midi_classifier(5,25,5,path,eng)













































