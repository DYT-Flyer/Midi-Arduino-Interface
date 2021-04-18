import pygame
from pygame.locals import *
from pygame import midi
from datetime import datetime
#from apscheduler.scheduler import Scheduler

#3,4,5

def printMIDIDeviceList():
    for i in range(pygame.midi.get_count()):
        print(pygame.midi.get_device_info(i), i)
            
import pygame
from pygame import midi as pygame_midi
from pygame.midi import midis2events

pygame.init()
pygame.font.init()
pygame_midi.init()

print("Found {} midi devices".format(pygame_midi.get_count()))

for i in range(pygame_midi.get_count()):
    print("#{}: {}".format(i, pygame_midi.get_device_info(i)))


midi_input_id = pygame_midi.get_default_input_id()
midi_input = pygame_midi.Input(midi_input_id)

print("Using input #{}".format(midi_input_id))



tstart = datetime.now()
tstart = tstart.hour*3600+tstart.minute*60+tstart.second

#sched.add_interval_job(update_piano_roll(), seconds = 5)

def update_piano_roll(piano_roll,new_notes):
    piano_roll[:,0:19] = piano_roll[:,20:24]
    piano_roll[:,20:24] = new_notes

def extract_events(events):
    #List is sequential
    #Thus: if first item in list has status: 128, skip
    #If Last item in list has status 144, skip
    
    eventpair = []
    eventpairs = []
    
    #Pairing begining and end notes
    control = 1
    while len(events)>0:
        #Skip if first note is remainder from previous
        current_event = events.pop(0)
        if current_event.status == 128 and control == 1:
            continue
        
        control = 0
        current_note = current_event.data1
        
        # matches note begin to note end
        for i in range(0,len(events)):
            if events[i].data1 == current_note:
                eventpair.append(current_event)
                eventpair.append(events.pop(i))
                eventpairs.append(eventpair)
                eventpair = []
                break
    
    return eventpairs

def extract_time(eventpairs):
    for i in range(0,len(eventpairs)):
        te = eventpairs[i][1].time()
        ts = eventpairs[i][0].time()
        tint = te - ts
        
        
            
            
    
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

