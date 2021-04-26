import pygame
from pygame.locals import *
from datetime import datetime
import math
from pygame import midi as pygame_midi
from pygame.midi import midis2events
import numpy as np
import matplotlib.pyplot as plt
from apscheduler.schedulers.background import BackgroundScheduler
import tensorflow as tf
import pyfirmata

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
        
        count = 0
        for i in range(0,number_of_notes):
            for j in range(0,steps_per_second*sample):
                self.pos[count,0] = i
                self.pos[count,1] = j
                count += 1
                
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
    
    def plotpr(self):
        plt.scatter(self.pos[:,0],self.pos[:,1],c=self.piano_roll.flatten())
        plt.show()
        
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
            print(current_event.status)
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
        print('updating')
        self.extract_events()
        self.extract_time()
        self.piano_roll = np.zeros([self.number_of_notes,self.steps_per_second*self.sample])
        for note in self.upr:
            print(note)
            try:
                self.piano_roll[note[0],note[1]-self.count*self.steps_per_second:note[2]-self.count*self.steps_per_second] = 1
            except:
                print('over')
        
        self.piano_roll = np.expand_dims(self.piano_roll,axis=0)
        self.piano_roll = np.expand_dims(self.piano_roll,axis=3)
        print(self.piano_roll.shape)
        pred = self.model.predict(self.piano_roll)
        print(np.argmax(pred,1))
        if np.argmax(pred,1) == 3:
            eng.turn_on(nargout=0)
        else:
            eng.turn_off(nargout=0)
        print(bytes(self.count%4))
        self.count += 1

import matlab.engine
path = 'C:/Users/remove/Documents/GitHub/Midi-Arduino-Interface/model.hdf5'
future = matlab.engine.start_matlab(background=True)
eng = future.result()
eng.addpath("C:/Users/remove/Documents/GitHub/Midi-Arduino-Interface")
eng.arduino_main(nargout=0)
midi = midi_classifier(5,25,5,path,eng)













































