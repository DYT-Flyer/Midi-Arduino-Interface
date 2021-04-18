import pygame
from pygame.locals import *
from pygame import midi
from datetime import datetime
import math
import pygame
from pygame import midi as pygame_midi
from pygame.midi import midis2events

from apscheduler.scheduler import Scheduler

#3,4,5

def printMIDIDeviceList():
    for i in range(pygame.midi.get_count()):
        print(pygame.midi.get_device_info(i), i)

# print("Found {} midi devices".format(pygame_midi.get_count()))
# for i in range(pygame_midi.get_count()):
#     print("#{}: {}".format(i, pygame_midi.get_device_info(i)))
# print("Using input #{}".format(midi_input_id))

tstart = datetime.now()

class midi_event:
    
    def __init__(self, steps_per_second, number_of_notes):
        pygame.init()
        pygame.font.init()
        pygame_midi.init()
        
        self.midi_input_id = pygame_midi.get_default_input_id()
        self.midi_input = pygame_midi.Input(self.midi_input_id)
        self.tbegin = datetime.now().hour*3600+datetime.now().minute*60+datetime.now().second
        
        sched.add_interval_job(self.update_piano_roll(), seconds = 5)
        
        self.number_of_notes = number_of_notes
        self.steps_per_second = steps_per_second
        self.eventpairs = []
        
    def update_piano_roll(self):
        self.extract_events()
        self.extract_time()
        self.piano_roll[:,0:19] = self.piano_roll[:,20:24]
        for note in self.upr:
            self.piano_roll[note[0],note[1]:note[2]] = 1

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
            te = self.eventpairs[i][1].time()
            ts = self.eventpairs[i][0].time()
            current_note = self.eventpairs[i][0].data1
            tint = math.floor((te - ts)*self.steps_per_second)
            self.upr.append([current_note, ts-self.tbegin, tint-self.tbegin])
        
#Status: 144-Push Down
#Status: 128-Let Up
    
while True:
    if not midi_input.poll():
        continue

    events = midis2events(midi_input.read(40), midi_input_id)
    for event in events:
        print(event)


    #scheduler.run()
    


plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)


now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

sched = Scheduler()
sched.start()

sched.shutdown()

