import pygame
from pygame.locals import *
from pygame import midi
from datetime import datetime
import math
import pygame
from pygame import midi as pygame_midi
from pygame.midi import midis2events
import numpy as np
import matplotlib.pyplot as plt
import time
from apscheduler.schedulers.background import BackgroundScheduler

def job_function():
    print("Hello World")

# Schedule job_function to be called every two hours

#3,4,5

# def printMIDIDeviceList():
#     for i in range(pygame.midi.get_count()):
#         print(pygame.midi.get_device_info(i), i)

# print("Found {} midi devices".format(pygame_midi.get_count()))
# for i in range(pygame_midi.get_count()):
#     print("#{}: {}".format(i, pygame_midi.get_device_info(i)))
# print("Using input #{}".format(midi_input_id))

class midi_classifier:
    def __init__(self, steps_per_second, number_of_notes, sample):
        pygame.init()
        pygame.font.init()
        pygame_midi.init()
        
        self.midi_input_id = pygame_midi.get_default_input_id()
        self.midi_input = pygame_midi.Input(self.midi_input_id)
        self.sample = sample
        self.control = 1
        self.number_of_notes = number_of_notes
        self.steps_per_second = steps_per_second
        self.events = []
        self.eventpairs = []
        self.count = 0
        
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
            
        
    def update_piano_roll(self):
        print('updating')
        self.extract_events()
        self.extract_time()
        self.piano_roll = np.zeros([self.number_of_notes,self.steps_per_second*self.sample])
        for note in self.upr:
            print(note)
            self.piano_roll[note[0],note[1]-self.count*self.steps_per_second:note[2]-self.count*self.steps_per_second] = 1
        #self.plotpr()
        self.count += 1
    
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
            print('-------------------')
            print(i)
            if i == 0:
                self.tbegin = self.eventpairs[i][0].timestamp
            
            ts = self.eventpairs[i][0].timestamp
            te = self.eventpairs[i][1].timestamp
            current_note = self.eventpairs[i][0].data1
            
            tint = math.floor(((te - ts)/1000)*self.steps_per_second)
            
            tc = math.floor((ts-self.tbegin)/1000)
            
            print(ts)
            print(te)
            print((te-ts)/1000)
            print(((te-ts)/1000)*self.steps_per_second)
            print('-')
            print(tc)
            print(tint)
            print(tc+tint)
            print(self.count)
            print((self.count-1)*self.sample*self.steps_per_second*1000)
            print(tc-(self.count-1)*self.sample*self.steps_per_second*1000)
            
            
            self.upr.append([current_note-48, tc, tint+tc])
        
#Status: 144-Push Down
#Status: 128-Let Up
#scheduler.run()

a = midi_classifier(5,25,5)

plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

sched = Scheduler()
sched.start()

sched.shutdown()

